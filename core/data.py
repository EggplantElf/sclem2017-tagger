import numpy as np
from collections import deque, defaultdict
import gzip, cPickle
from enum import IntEnum

class Token(dict):

    def __init__(self, entries):
        tid_str = entries[0].split('_')[-1]
        if '.' in tid_str:
            self['tid'] = -1
            return
        self['tid'] = int(tid_str)
        self['word'] = entries[1].decode('utf8')
        self['lemma'] = entries[2].decode('utf8')
        self['utag'] = entries[3] # gold
        self['xtag'] = entries[4] # pred or extra tag
        self['morphstr'] = entries[5]
        self['morph'] = entries[5].split('|') # a list of key-value pairs in string
        self['head'] = -1 if entries[6] == '_' else int(entries[6])
        self['label'] = entries[7] # gold
        self['phead'] = -1 
        self['plabel'] = '_'
        self['stag'] = '_'
        self.idx = {} # (word, lemma, utag, xtag, label, morph, char)

    def to_str(self, pred=['head', 'label']):
        head = self['phead'] if 'head' in pred else self['head']
        label = self['plabel'] if 'label' in pred else self['label']
        utag = self['putag'] if 'utag' in pred else self['utag']
        xtag = self['pxtag'] if 'xtag' in pred else self['xtag']
        morphstr = self['pmorphstr'] if 'morphstr' in pred else self['morphstr']

        out = '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t_\t_' %\
                (self['tid'], self['word'].encode('utf8'), self['lemma'].encode('utf8'),\
                 utag, xtag, morphstr, head, label)
        if 'stag' in pred:
            out += '\t%s' % self['pstag']

        return out

class Root(Token):
    def __init__(self):
        self['tid'] = 0
        self['word'] = '<ROOT>'
        self['lemma'] = '<ROOT>'
        self['utag'] = '<ROOT>'
        self['xtag'] = '<ROOT>'
        self['morphstr'] = '_'
        self['morph'] = ['_']  
        self['head'] = -1
        self['label'] = '<ROOT>'
        self['stag'] = '<ROOT>'
        self.idx = {}


class Sentence(object):
    def __init__(self):
        self.tokens = [Root()]
        self.is_projective = False
        self.inorder = {}
        self.mpc = None
        self.null_tokens = 0
        self.restart = 0
        self.reversed = False
        self.lang = ''

    def add_token(self, token):
        if token['tid'] != -1:
            self.tokens.append(token)
        else:
            self.null_tokens += 1

    def to_str(self, pred):
        return '\n'.join(t.to_str(pred) for t in self.tokens[1:]) + '\n\n'

    def complete(self):
        self.gold_heads = [d['head'] for d in self.tokens]

        # easier for auxiliary tasks
        for token in self.tokens:
            token['lang'] = self.lang

        # prepare the non-projectivity preprocessing if not blinded
        if self.tokens[-1]['head'] != -1:
            if not self.check_proj():
                self.get_in_order()
        return self

    def check_proj(self):
        for d in range(1, len(self.tokens)):
            h = self.gold_heads[d]
            b, e = min(h, d), max(h, d)
            for j in range(b + 1, e):
                t = j
                while j != 0:
                    j = self.gold_heads[j]
                    if j < b or j > e:
                        self.is_projective = False
                        return False
                    elif j == b or j == e:
                        break
        self.is_projective = True
        return True


    def get_in_order(self, h=0):
        for l in self.get_left_deps(h):
            self.get_in_order(l)
        self.inorder[h] = len(self.inorder)
        for r in self.get_right_deps(h):
            self.get_in_order(r)

    def get_left_deps(self, h):
        return [d for d in range(1, h) if self.gold_heads[d] == h]

    def get_right_deps(self, h):
        return [d for d in range(h+1, len(self.tokens)) if self.gold_heads[d] == h]


    # overrides the predicted stags from input (if used)
    def get_gold_stag(self):
        directs = ['X'] * len(self.tokens)
        labels = ['X'] * len(self.tokens)
        has_lc = [False] * len(self.tokens)
        has_rc = [False] * len(self.tokens)

        for token in self.tokens[1:]:
            labels[token['tid']] = token['label']
            if token['head'] < token['tid']:
                directs[token['tid']] = 'L'
                has_rc[token['head']] = True
            else:
                directs[token['tid']] = 'R'
                has_lc[token['head']] = True
        for i in xrange(1, len(self.tokens)):
            if has_lc[i]: 
                if has_rc[i]:
                    cstr = '+L_R'
                else:
                    cstr = '+L'
            else:
                if has_rc[i]:
                    cstr = '+R'
                else:
                    cstr = ''
            self.tokens[i]['stag'] = '%s/%s%s' % (labels[i], directs[i], cstr)

    def reverse(self):
        # print ['%s<-%s' % (t['tid'], t['head']) for t in self.tokens]
        sent_len = len(self.tokens)
        for token in self.tokens[1:]:
            token['tid'] = sent_len - token['tid']
            if token['head'] > 0:
                token['head'] = sent_len - token['head']
            if token['phead'] > 0:
                token['phead'] = sent_len - token['phead']
        self.tokens = self.tokens[:1] + self.tokens[:0:-1]
        # print ['%s<-%s' % (t['tid'], t['head']) for t in self.tokens]
        self.reversed = True


def read_sentences(filename, reverse = False, first = None):
    i = 0
    sentence = Sentence()
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            if line:
                if line.startswith('#'):
                    sentence.lang = line.strip('#')
                else:
                    entries = line.split('\t')
                    if '-' not in entries[0] and '.' not in entries[0]:
                        sentence.add_token(Token(entries))
            elif len(sentence.tokens) > 1:
                if reverse:
                    sentence.reverse()
                yield sentence.complete()
                sentence = Sentence()
                i += 1
                if first and i >= first:
                    break


class State(object):

    def __init__(self, sent, prev_state = None, prev_y = None, prev_z = None,
                arcs = (), stk = None, bfr = None):
        self.sent = sent
        if not prev_state:
            sent.restart += 1
        self.prev_state = prev_state
        self.prev_y = prev_y
        self.prev_z = prev_z
        self.arcs = arcs
        self.fidx = ()
        self.mft = None
        self.prob = None
        self.step = self.prev_state.step + 1 if self.prev_state else 0
        self.stats = self.prev_state.stats if self.prev_state else defaultdict(int)
        if stk == None:
            self.stk = (0, )
            self.bfr = tuple(range(1, len(sent.tokens)))
            self.in_gold_path = True
        else:
            self.stk = stk
            self.bfr = bfr


    def show(self):
        return self.stk, self.bfr

    # caution, really changes stuff in sentence
    def to_str(self, system):
        for i, (d, h, l) in enumerate(self.arcs):
            token = self.sent.tokens[d]
            token['phead'] = h
            token['plabel'] = system.get_label(l)
        if self.sent.reversed:
            self.sent.reverse()
        return self.sent.to_str(pred=['head', 'label'])


    ##################################
    # helper functions to unify the behaviours of different state classes
    # could be merged into one function for efficiency
    def left_children(self, idx, num = 2):
        lefts = sorted([d for d, h, l in self.arcs if h == idx and d < h])
        if len(lefts) == 0:
            return (-1, -1)
        elif len(lefts) == 1:
            return (lefts[0], -1)
        else:
            return (lefts[0], lefts[1])

    def right_children(self, idx, num = 2):
        rights = sorted([d for d, h, l in self.arcs if h == idx and d > h], reverse = True)
        if len(rights) == 0:
            return (-1, -1)
        elif len(rights) == 1:
            return (rights[0], -1)
        else:
            return (rights[0], rights[1])

    def label(self, idx):
        for d, h, l in self.arcs:
            if d == idx:
                return l
        return 0

    def head(self, idx):
        for d, h, l in self.arcs:
            if d == idx:
                return h
        return None

    ##################################


    def finished(self):
        return len(self.stk) == 1 and len(self.bfr) == 0

    def valid(self, action):
        if action == 'SH':
            return len(self.bfr) > 0
        elif action == 'LA':
            return len(self.stk) > 2
        elif action == 'RA':
            return len(self.stk) > 1 and (self.stk[-2] != 0 or len(self.bfr) == 0) # extra
        elif action == 'LA2':
            return len(self.stk) > 3
        elif action == 'RA2':
            return len(self.stk) > 2 and (self.stk[-3] != 0 or len(self.bfr) == 0) # extra
        elif action == 'LA3':
            return len(self.stk) > 4
        elif action == 'RA3':
            return len(self.stk) > 3 and (self.stk[-4] != 0 or len(self.bfr) == 0) # extra
        elif action == 'SW':
            return len(self.stk) > 2 and self.stk[-2] < self.stk[-1]
        else:
            return False


    def perform(self, system, y, z = 0):
        action = system.get_action(y)

        if action == 'SH':
            stk = self.stk + self.bfr[:1]
            bfr = self.bfr[1:]
            arcs = self.arcs
        elif action == 'LA':
            stk = self.stk[:-2] + self.stk[-1:]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-2], self.stk[-1], z),)
        elif action == 'RA':
            stk = self.stk[:-1]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-1], self.stk[-2], z),)
        elif action == 'LA2':
            stk = self.stk[:-3] + self.stk[-2:]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-3], self.stk[-1], z),)
        elif action == 'RA2':
            stk = self.stk[:-1]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-1], self.stk[-3], z),)
        elif action == 'LA3':
            stk = self.stk[:-4] + self.stk[-3:]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-4], self.stk[-1], z),)
        elif action == 'RA3':
            stk = self.stk[:-1]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-1], self.stk[-4], z),)
        elif action == 'SW':
            stk = self.stk[:-2] + self.stk[-1:]
            bfr = self.stk[-2:-1] + self.bfr
            arcs = self.arcs
        else:
            raise Exception('No such move')
            
        return State(self.sent, self, y, z, arcs, stk, bfr)

    def get_pred_label(self, tidx):
        for (d, h, l) in self.arcs:
            if d == tidx:
                return l
        return 0


    def get_gold_label(self, system, y):
        action = system.get_action(y)

        if action in ['RA', 'RA2', 'RA3']:
            return system.get_label_idx(self.sent.tokens[self.stk[-1]]['label'])
        elif action == 'LA':
            return system.get_label_idx(self.sent.tokens[self.stk[-2]]['label'])
        elif action == 'LA2':
            return system.get_label_idx(self.sent.tokens[self.stk[-3]]['label'])
        elif action == 'LA3':
            return system.get_label_idx(self.sent.tokens[self.stk[-4]]['label'])
        else:
            return 0 # <UNK>

    def get_reg_tag(self, feat):
        if self.stk:
            return feat.get_idx(self.sent.tokens[self.stk[-1]]) # e.g. utag of s0
        else:
            return 0

class TaggerState:
    def __init__(self, sent, idx = 1):
        self.sent = sent
        self.idx = idx

    # WARN: initial state is also "finished"
    def finished(self):
        return self.idx == 1

    def next(self):
        if self.idx + 1 < len(self.sent.tokens):
            return TaggerState(self.sent, self.idx + 1)
        else:
            return TaggerState(self.sent, 1)

    def gold(self, feat):
        return self.sent.tokens[self.idx][feat]

class Extractor:
    def __init__(self, parser):
        self.parser = parser

    def extract(self, state):
        pos = [-1] * 26
        stk = state.stk
        bfr = state.bfr

        if len(bfr) > 0:
            pos[4] = bfr[0]
            if len(bfr) > 1:
                pos[5] = bfr[1]
                if len(bfr) > 2:
                    pos[6] = bfr[2]
                    if len(bfr) > 3:
                        pos[7] = bfr[3]

        if len(stk) > 0:
            pos[0] = stk[-1] # s0
            pos[8], pos[9] = state.left_children(pos[0]) # s0L0, s0L1
            pos[10], pos[11] = state.right_children(pos[0]) #s0R0, s0R1
            if pos[8] != -1:
                pos[12], _ = state.left_children(pos[8]) # s0L0L0
            if pos[10] != -1:
                pos[13], _ = state.right_children(pos[10]) # s0R0R0

            if len(stk) > 1:
                pos[1] = stk[-2] # s1
                pos[14], pos[15] = state.left_children(pos[1]) # s1L0, s1L1
                pos[16], pos[17] = state.right_children(pos[1]) # s1R0, s1R1
                if pos[14] != -1:
                    pos[18], _ = state.left_children(pos[12]) # s1L0L0
                if pos[16] != -1:
                    pos[19], _ = state.right_children(pos[14]) # s1R0R0

                if len(stk) > 2:
                    pos[2] = stk[-3] # s2

                    # extra to weiss:2015
                    pos[20], pos[21] = state.left_children(pos[2]) # s2L0, s2L1
                    pos[22], pos[23] = state.right_children(pos[2]) # s2R0, s2R1
                    if pos[20] != -1:
                        pos[24], _ = state.left_children(pos[20]) # s2L0L0
                    if pos[22] != -1:
                        pos[25], _ = state.right_children(pos[22]) # s2R0R0

                    if len(stk) > 3: # s3
                        pos[3] = stk[-4]

        # lidx = [(state.get_pred_label(tidx) if tidx != -1 else 0)  for tidx in pos]
        return pos

    # this is a new state class for tagger
    def extract_window(self, state, window = 15):
        context = window / 2
        return [-1] * (context - state.idx) \
                + range(max(0, state.idx-context), min(state.idx+context+1, len(state.sent.tokens))) \
                + [-1] * (state.idx+context+1-len(state.sent.tokens))



class DecoupledSystem:
    def __init__(self, name):
        self.name = name
        # initial label map, for compatibility of the master 
        self.labelmap = {'<PAD>': 0, '<UNK>': 1}
        self.labelmap_reverse = [l for l,i in sorted(self.labelmap.items(), key=lambda (k,v): v)]

        if name == 'ArcStandard':
            self.actions = ['LA', 'RA', 'SH'] # always the order prefered by the oracle
            self.idsh = 2
        elif name == 'Attardi':
            self.actions = ['LA', 'LA2', 'RA', 'RA2', 'SH']
            self.idsh = 4
        elif name == 'Attardi2':
            self.actions = ['LA', 'LA2', 'LA3', 'RA', 'RA2', 'RA3', 'SH']
            self.idsh = 6
        elif name == 'Swap':
            self.actions = ['LA', 'RA', 'SW', 'SH']
            self.idsh = 2
        else:
            raise Exception('No such system')

        self.action2idx = {t:i for i, t in enumerate(self.actions)}
        self.idx2action = {i:t for i, t in enumerate(self.actions)}
        self.num = len(self.action2idx)

    def register_labels(self, labelmap):
        self.labelmap = labelmap
        self.labelmap_reverse = [l for l,i in sorted(self.labelmap.items(), key=lambda (k,v): v)]

    def get_action(self, it):
        return self.idx2action[it]

    def get_label(self, il):
        return self.labelmap_reverse[il]

    def get_action_label(self, it, il):
        return self.idx2action[it], self.labelmap_reverse[il]

    def get_idx(self, action, label):
        return self.action2idx[action], self.labelmap.get(label, 1)

    def get_action_idx(self, action):
        return self.action2idx[action]

    def get_label_idx(self, label):
        return self.labelmap.get(label, 1)


    def valid_idx(self, state):
        return [y for y, action in enumerate(self.actions) if state.valid(action)]

    def valid_mask(self, state):
        '''return the mask of valid actions, with 1 for valid and 0 for invalid'''
        mask = np.zeros(self.num)
        mask[self.valid_idx(state)] = 1
        return mask

class ArcStandard(IntEnum):
    LA=0; RA=1; SH=1

class Attardi(IntEnum):
    LA=0; LA2=1; RA=2; RA2=3; SH=4

class Attardi2(IntEnum):
    LA=0; LA2=1; LA3=2; RA=3; RA2=4; RA3=5; SH=6

class Swap(IntEnum):
    LA=0; RA=1; SW=2; SH=3



