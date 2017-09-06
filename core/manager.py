from data import *
from util import *
from feat import *
from oracle import *
from collections import Counter
import gzip, cPickle
import os
import numpy as np
from itertools import izip


class DataManager(object):
    def __init__(self, args, train=True):
        self.args = args

        self.args.target_feats = filter(lambda x: x, self.args.target_feats.split(','))
        self.args.source_feats = filter(lambda x: x, self.args.source_feats.split(','))

        if not train:
            self.load_args()

        if self.args.tool == 'tagger':
            self.args.feat_shape = self.args.window_size
        else:
            self.args.feat_shape = 26


        if self.args.model_to:
            if not os.access(self.args.model_to, os.R_OK):
                os.makedirs(self.args.model_to)

            if not self.args.log:
                self.args.log = self.args.model_to + '.log'
            # if not self.args.out:
                # self.args.out = self.args.model_to + '.conllu'

        self.logger = Logger(self.args.log)
        self.system = DecoupledSystem(self.args.system)

        # TODO: put all useful and reusable args into config 
        self.get_config()
        
        print "Word embeddings size", self.config["word"]["emb_dim"]
        print "Freeze word embeddings", self.config["word"]["freeze"]

        if train:
            self.init()
        else:
            self.load()

        self.add_model_args()
        self.log_args()


    def save_feats(self, save_model = False):
        # save the feature maps, optionally feature embeddings for debugging (without target feature)
        stream = gzip.open(os.path.join(self.args.model_to, 'maps.gz'),'wb')
        for name in sorted(['label'] + self.args.target_feats + self.args.source_feats):
            cPickle.dump(self.feats[name].map,stream,-1)
        stream.close()
        if save_model:
            for name in sorted(self.args.source_feats):
                self.feats[name].save_model(self.args.model_to)

    def load_feats(self):
        self.feats = {}
        self.data = {}
        stream = gzip.open(os.path.join(self.args.model_from, 'maps.gz'),'rb')
        for name in sorted(['label'] + self.args.target_feats + self.args.source_feats):
            feat = Feature(name, self.config, self.args) 
            feat.map = cPickle.load(stream)
            feat.revmap = [k for k,v in sorted(feat.map.items(), key=lambda (k,v): v)]
            self.feats[name] = feat
            self.data[name] = feat.get_data(self.all_sents)
        stream.close()

    def save_args(self):
        stream = gzip.open(os.path.join(self.args.model_to, 'args.gz'),'wb')
        cPickle.dump(self.args,stream,-1)
        stream.close()

    # during test time, load (some of the important) args from the training time to ensure the parameters are correct
    def load_args(self):
        try:
            stream = gzip.open(os.path.join(self.args.model_from, 'args.gz'),'rb')
            old_args = cPickle.load(stream)
            stream.close()

            self.args.tool = old_args.tool
            self.args.system = old_args.system
            self.args.aux_tagger = old_args.aux_tagger
            self.args.source_feats = old_args.source_feats
            self.args.target_feats = old_args.target_feats
            self.args.embw_dim= old_args.embw_dim
            self.args.conv_dim = old_args.conv_dim
            self.args.reverse = old_args.reverse
            self.args.ngrams = old_args.ngrams
            self.args.context_ngrams = old_args.context_ngrams
            self.args.tagger_layers = old_args.tagger_layers
            self.args.multi_task = old_args.multi_task
            self.args.window_size = old_args.window_size
            self.args.embw_freeze = old_args.embw_freeze
            self.args.conv_layers = old_args.conv_layers
            self.args.lex = old_args.lex
            self.args.pos_emb = old_args.pos_emb
        except:
            print 'UNABLE TO LOAD OLD ARGS'

    def load(self):
        # self.load_args()
        self.system = DecoupledSystem(self.args.system)
        self.train_sents = []
        self.dev_sents = []
        # self.train_sents = list(read_sentences(self.args.train, self.args.reverse, self.args.first))\
                         # if self.args.train else []
        # self.dev_sents = list(read_sentences(self.args.dev, self.args.reverse))\
                         # if self.args.dev else []
        self.test_sents = list(read_sentences(self.args.test, self.args.reverse))\
                         if self.args.test else []
        self.all_sents = self.test_sents
        self.args.max_sent_len = max(len(s.tokens) for s in self.all_sents) + 1 # +1 for putting <PAD> at the end
        self.args.train_ends = len(self.train_sents)
        self.args.dev_ends = self.args.train_ends + len(self.dev_sents)
        self.args.test_ends = self.args.dev_ends + len(self.test_sents)
        print '# test sents:', len(self.test_sents)
        self.get_mask()
        # if self.args.train and 'stag' in self.args.target_feats:
        if 'stag' in self.args.target_feats:
            print 'use gold supertag as target'
            for sent in self.all_sents:
                sent.get_gold_stag()
        self.load_feats()
        # self.get_counts()
        # self.create_feats(['label'] + self.args.target_feats + self.args.source_feats)
        self.system.register_labels(self.feats['label'].map)



    def init(self):
        # data preprocessing: read sentences, get maps, lookup indices
        self.train_sents = list(read_sentences(self.args.train, self.args.reverse, self.args.first))\
                         if self.args.train else []
        self.dev_sents = list(read_sentences(self.args.dev, self.args.reverse))\
                         if self.args.dev else []
        self.test_sents = list(read_sentences(self.args.test, self.args.reverse))\
                         if self.args.test else []
        self.filter_train_sents()

        # if no test set, then use 10% of train as dev, and use dev as test
        if self.train_sents and not self.dev_sents and self.args.split:
            print 'Taking the last %d%% of train set as dev set' % (self.args.split * 100)
            split = int(len(self.train_sents) * (1 - self.args.split))
            self.dev_sents = self.train_sents[split:]
            self.train_sents = self.train_sents[:split]
        print '# train sents:', len(self.train_sents)
        print '# dev sents:', len(self.dev_sents)
        print '# test sents:', len(self.test_sents)

        # create data splits and mask
        self.all_sents = self.train_sents + self.dev_sents + self.test_sents
        self.args.train_ends = len(self.train_sents)
        self.args.dev_ends = self.args.train_ends + len(self.dev_sents)
        self.args.test_ends = self.args.dev_ends + len(self.test_sents)
        self.args.max_sent_len = max(len(s.tokens) for s in self.all_sents) + 1 # +1 for putting <PAD> at the end
        self.get_mask()

        if 'stag' in self.args.target_feats:
            print 'use gold supertag as target'
            for sent in self.all_sents:
                sent.get_gold_stag()
        self.get_counts()
        self.create_feats(['label'] + self.args.target_feats + self.args.source_feats)
        self.save_feats()
        self.system.register_labels(self.feats['label'].map)
        self.save_args()



    def filter_train_sents(self):
        print 'Filtering trainable sentences...'
        start = prev = len(self.train_sents)


        # remove too long training sentences
        self.train_sents = [sent for sent in self.train_sents if len(sent.tokens) < self.args.cut_len]
        self.logger.log('Removed %d sentences longer than %d' % (prev - len(self.train_sents), self.args.cut_len))

        # remove non-projective sentences if not using swap system
        if self.args.tool == 'parser' and self.args.system != 'Swap':
            oracle = Oracle(self.system)
            self.train_sents = [sent for sent in self.train_sents if oracle.can_parse(sent)]

        self.logger.log('Trainable sentences: %d / %d = %.2f%%' % (len(self.train_sents), start, 100.*len(self.train_sents)/prev))


    def add_model_args(self):
        self.args.idsh = self.system.idsh
        self.args.size = {k:len(f.map) for k, f in self.feats.items()}
        self.args.num_actions = len(self.system.actions)

        hidden_sizes = [int(s) for s in self.args.hidden_layer_sizes.split(',')]
        self.args.nh1 = hidden_sizes[0]
        self.args.nh2 = hidden_sizes[1]
        self.args.nh3 = self.system.num

        # for char
        self.args.ngrams = [int(n) for n in self.args.ngrams.split(',')] 
        self.args.context_ngrams = [int(n) for n in self.args.context_ngrams.split(',')] 
        # self.args.nconv = self.config['word']['emb_dim'] / len(self.args.ngrams)
        # items = self.args.dropout.split(',')
        # self.args.p1 = float(items[0])
        # self.args.p2 = float(items[1])

    def log_args(self):
        msg = ''
        if self.args.msg:
            msg += self.args.msg + '\n'
        msg += 'args:\n'
        msg += '  -tool: %s\n' % self.args.tool
        msg += '  -system: %s\n' % self.args.system
        msg += '  -train: %s\n' % self.args.train
        msg += '  -dev: %s\n' % self.args.dev
        msg += '  -test: %s\n' % self.args.test
        msg += '  -model_from: %s\n' % self.args.model_from
        msg += '  -model_to: %s\n' % self.args.model_to
        msg += '  -out: %s\n' % self.args.out
        msg += '  -embw: %s\n' % self.args.embw
        msg += '  -log: %s\n' % self.args.log
        msg += '  -num_steps: %d\n' % self.args.num_steps
        msg += '  -first: %d\n' % self.args.first
        msg += '  -batch_size: %d\n' % self.args.batch_size
        msg += '  -reg_rate: %s\n' % self.args.reg_rate
        msg += '  -learn_rate: %s\n' % self.args.learn_rate
        msg += '  -decay: %.2f\n' % self.args.decay
        msg += '  -momentum: %.2f\n' % self.args.momentum
        msg += '  -sigma: %.2f\n' % self.args.sigma
        msg += '  -seed: %s\n' % self.args.seed
        msg += '  -hidden_layer_sizes: %s\n' % self.args.hidden_layer_sizes
        msg += '  -dropout: %s\n' % self.args.dropout
        msg += '  -aux_ratio: %s\n' % self.args.aux_ratio
        msg += '  -ngrams: %s\n' % self.args.ngrams
        msg += '  -squeeze: %d\n' % self.args.squeeze
        msg += '  -l2norm: %s\n' % self.args.l2norm
        msg += '  -rescale: %s\n' % self.args.rescale
        msg += '  -source_feats: %s\n' % self.args.source_feats
        msg += '  -target_feats: %s\n' % self.args.target_feats

        self.logger.log(msg)


    def get_counts(self):
        self.counts = {name:Counter() for name in ['word', 'lemma', 'utag', 'xtag', 'stag',\
                                              'label', 'morph', 'char', 'morphstr']}

        for sent in self.train_sents:
            for token in sent.tokens:
                self.counts['word'][self.normalize_word(token['word'])] += 1
                self.counts['lemma'][token['lemma']] += 1
                self.counts['utag'][token['utag']] += 1                
                self.counts['xtag'][token['xtag']] += 1                
                self.counts['stag'][token['stag']] += 1                
                self.counts['label'][token['label']] += 1
                self.counts['morphstr'][token['morphstr']] += 1
                for char in token['word']:
                    self.counts['char'][char] += 1
                for morph in token['morph']:
                    self.counts['morph'][morph] += 1


    def normalize_word(self, w):
        if self.args.lower:
            return w.lower()
        else:
            return w

    # FIX: legacy, remove later
    def lookup_idx(self, sents):
        for sent in sents:
            for token in sent.tokens:
                token.idx['label'] = self.feats['label'].get_idx(token['label']) 


    def get_mask(self):
        self.mask = np.zeros((len(self.all_sents), self.args.max_sent_len))
        for i, sent in enumerate(self.all_sents):
            self.mask[i, :len(sent.tokens)] = 1

    def get_config(self):
        # TODO: use a json file
        # TODO: emb_file 
        # bundle more stuff here
        # separate parser config and tagger config
        self.config = \
         {'word': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': self.args.embw_dim, 'min_freq': 0, 'freeze': self.args.embw_freeze},
          'lemma': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': self.args.embw_dim, 'min_freq': 0, 'freeze': 0},
          'utag': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0},
          'xtag': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0},
          'stag': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0},
          'label': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0},
          'morphstr': {'feat_shape': [self.args.feat_shape], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0},
          'morph': {'feat_shape': [self.args.feat_shape, 16], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0},
          'char': {'feat_shape': [self.args.feat_shape, 32], 'mask_rate': 0, 'emb_dim': 32, 'min_freq': 0, 'freeze': 0}
         }

    def create_feats(self, fnames):
        self.feats = {}
        self.data = {}
        for name in fnames:
            # create feature object
            feat = Feature(name, self.config, self.args) 
            self.feats[name] = feat
            # add items into the map
            for item in sorted(self.counts[name]):
                if self.counts[name][item] > self.config[name]['min_freq']:
                    feat.add(item)
            # look up the map and create data
            self.data[name] = feat.get_data(self.all_sents)

    def read_data_once_more(self):
        self.feats["word"].set_data(self.all_sents)
