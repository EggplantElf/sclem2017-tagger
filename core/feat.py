import lasagne
import lasagne.layers as L
from lasagne.init import *
from lasagne.nonlinearities import *
import theano
import theano.tensor as T
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import os
import gzip, cPickle
from itertools import izip
from util import *

# TODO: config in the form of a dictionary
class Feature(object):
    def __init__(self, name, config, args):
        self.name = name
        self.config = config
        self.args = args
        if name == 'char':
            self.map = {'<PAD>': 0, '<UNK>': 1, '<MUL>':2, '<SOW>':3, '<EOW>':4}
        else:
            self.map = {'<PAD>': 0, '<UNK>': 1}
        self.revmap = [k for k,v in sorted(self.map.items(), key=lambda (k,v): v)]
        if name == 'char':
            self.data_shape = [self.args.max_sent_len, self.args.max_word_len]
        elif name == 'morph':
            self.data_shape = [self.args.max_sent_len, self.args.max_morph_len]
        else:
            self.data_shape = [self.args.max_sent_len]

        self.emb_layer = None
        self.is_source = True
        self.is_target = True


    # not really used, just for debugging
    def save_model(self, folder):
        print 'saving %s feat model' % self.name
        np.savez(os.path.join(folder, '%s.npz' % self.name), *L.get_all_param_values(self.emb_layer))

    def load_model(self, folder):
        print 'loading %s feat model' % self.name
        with np.load(os.path.join(folder, '%s.npz' % self.name)) as f:
            param_values = [f['arr_%d' %i] for i in range(len(f.files))]
            L.set_all_param_values(self.emb_layer, param_values)
    
    # TODO: sanity check, embeding dimension etc.
    def load_emb(self, emb_file):
        print 'Loading %s embedding...' % self.name
        W = self.emb.W.get_value()

        for line in open(emb_file):
            items = line.split()
            key = items[0]
            try:
               key = key.decode("utf-8")
            except:
               pass
            if key in self.map:
                value = np.array([float(n) for n in items[1:]])
                W[self.map[key]] = self.normalize_vector(value)
        self.emb.W.set_value(W)

    def add_emb(self, emp_file):
        print 'Loading additional %s embedding...' % self.name
        
        W = self.emb.W.get_value()
    
        print "Old shape", W.shape
        
        new_words = [ ]
        for line in open(emp_file):
            items = line.split()
            key = items[0].decode('utf8')
            
            if key not in self.map:
                self.add(key)
                
                value = np.array([np.float32(n) for n in items[1:]])
                new_words.append(self.normalize_vector(value))
                
        print "New words found:", len(new_words)
        W = np.concatenate((W, new_words))
        print "New shape", W.shape
        
        self.emb.W.set_value(W)

    # TODO: normalize on the global level, not on each vector
    def normalize_vector(self, v):
        if self.args.rescale:
            return ((v - v.mean()) / v.std()) * self.args.rescale
        elif self.args.l2norm:
            return v / (v ** 2).sum() * self.args.l2norm
        else:
            return v

    def save_emb(self, emb_file):
        with open(emb_file, 'w') as f:
            for ((w, i), e) in izip(sorted(self.map.items(), key = lambda k,v: v), 
                                    self.emb.W.get_value()):
                f.write('%s\t%s\n' % (w, '\t'.join(['%.8f'%d for d in e])))

    def add(self, item):
        if item not in self.map:
            self.map[item] = len(self.map)
            self.revmap.append(item)

    def get_idx(self, token):
        if self.name == 'char':
            return self.get_idxc(token['word'])
        elif self.name == 'morph':
            return self.get_idxm(token['morphstr'])
        elif self.name == 'word' and self.args.lower:
            return self.map.get(token['word'].lower(), 1)
        else:
            return self.map.get(token[self.name], 1)

    def get_idxm(self, morphstr):
        feat = np.zeros(self.args.max_morph_len, dtype='int64') # reserve 16 places for morphs
        for i, morph in enumerate(morphstr.split('|')):
            if i < self.args.max_morph_len:
                feat[i] = self.map.get(morph, 1)
        return feat

    def get_idxc(self, word):
        # exclude start and end symbol
        max_word_len = self.args.max_word_len - 2
        if len(word) <= max_word_len:
            # PAD on both sides
            return np.array([0]*((max_word_len-len(word)) / 2) \
                          + [3] \
                          + [self.map.get(c, 1) for c in word] \
                          + [4]\
                          + [0]*((max_word_len-len(word)+1) / 2))
        # for longer words, take the first and last N/2 chars and collapse the rest to 2 in the middle
        else:
            return np.array([3] \
                          + [self.map.get(c, 1) for c in word[:(max_word_len-1)/2]] \
                          + [2] \
                          + [self.map.get(c, 1) for c in word[-(max_word_len/2):]]\
                          + [4])


    # make a numpy matrix or tensor to store the indices of all the sentences
    # and only used by referring to the index of the sentence and token  
    # all data ends with 0, which will be used as <PAD> which is referred to by the parser feature 
    def get_data(self, sents):
        self.data_np = np.zeros([len(sents)] + self.data_shape, dtype = 'int64')
        print self.name, self.data_np.shape
        for i, sent in enumerate(sents):
            for j, token in enumerate(sent.tokens):
                self.data_np[i, j] = self.get_idx(token)
        self.data = theano.shared(self.data_np, name=self.name, borrow = True)
        return self.data


    def set_data(self, sents):
        self.data_np = np.zeros([len(sents)] + self.data_shape, dtype = 'int64')
        for i, sent in enumerate(sents):
            for j, token in enumerate(sent.tokens):
                self.data_np[i, j] = self.get_idx(token)
        self.data.set_value(self.data_np)

    # sidx and tidx are theano variables holding the sentence ids in the batch and token ids in the sentence
    # looks up the feature ids of the given sentences and tokens
    def get_emb_layer(self, sidx, tidx = None, avg = False):
        # do not create multiple emb_layer for the same feature
        # if self.emb_layer:
        #     return self.emb_layer

        if tidx is None:
            fidx = self.data[sidx] # (100, 161) or (100, 161, 16)
            fidx_layer = L.InputLayer(shape=[None] + self.data_shape, input_var=fidx)
        else:
            fidx = self.data[sidx.dimshuffle(0, 'x'), tidx] # (100, 26)
            fidx_layer = L.InputLayer(shape=[None]+self.config[self.name]['feat_shape'], input_var=fidx)
        self.emb_layer = self.get_emb_layer_from_idx(fidx_layer, avg)

        return self.emb_layer

    def get_emb_layer_from_idx(self, idx_layer, avg):
        suf = '_avg' if avg else ''
        # if not avg:
            # idx_layer = MaskLayer(idx_layer, self.config[self.name]['mask_rate'])
        self.emb = L.EmbeddingLayer(idx_layer, len(self.map), self.config[self.name]['emb_dim'],  
                                    W=Normal(self.args.init_std) if not avg else Constant(), name = 'e%s%s'%(self.name, suf))
                                    # W=HeNormal('relu') if not avg else Constant(), name = 'e%s%s'%(self.name, suf))
        self.emb.params[self.emb.W].remove('regularizable')
        if self.config[self.name]['freeze']:
            self.emb.params[self.emb.W].remove('trainable')

        # load embedding from external file if available
        if self.name == 'word' and self.args.train and self.args.embw:
            if not avg or self.config['word']['freeze']:
                try:
                    self.load_emb(self.args.embw)
                except:
                    print 'Not able to read pre-trained embeddings, use random initialization instead'


        add_layer = self.additional_layer(idx_layer, self.emb, avg)

        # add noise to embeddings as in Plank tagger
        add_layer = L.GaussianNoiseLayer(add_layer, 0.1)

        return add_layer

    def additional_layer(self, idx_layer, emb_layer, avg=False):
        suf = '_avg' if avg else ''
        if self.name == 'char':
            if self.args.char_model == 'cnn':
                lds = L.dimshuffle(emb_layer, (0, 3, 1, 2)) # (100, 16, 26, 32)
                ls = []
                for n in self.args.ngrams:
                    lconv = L.Conv2DLayer(lds, self.args.conv_dim, (1, n), untie_biases=False,
                                            # W=HeNormal('relu') if not avg else Constant(),
                                            W=GlorotNormal('relu') if not avg else Constant(),
                                            name = 'conv_%d'%n+suf) # (100, 64/4, 26, 32-n+1)
                    
                    lpool = L.MaxPool2DLayer(lconv, (1, self.args.max_word_len-n+1)) # (100, 64, 26, 1)
                    lpool = L.flatten(lpool, outdim=3) # (100, 16, 26)
                    lpool = L.dimshuffle(lpool, (0, 2, 1)) # (100, 26, 16)
                    ls.append(lpool)
                xc = L.concat(ls, axis=2, name='echar_concat') # (100, 26, 64)
                # additional 
                # xc = L.DenseLayer(xc, self.args.embw_dim, nonlinearity=None, name='echar_affine', num_leading_axes=2,
                                  # W=HeNormal() if not avg else Constant()) # (100, 26, 100)
                return xc
            elif self.args.char_model == 'lstm':
                ml = L.ExpressionLayer(idx_layer, lambda x: T.neq(x, 0)) # mask layer (100, 24, 32)
                ml = L.reshape(ml, (-1, self.args.max_word_len)) # (1500, 32)

                gate_params = L.recurrent.Gate(W_in=Orthogonal(), W_hid=Orthogonal())
                cell_params = L.recurrent.Gate(W_in=Orthogonal(), W_hid=Orthogonal(),
                                                W_cell=None, nonlinearity=tanh)

                lstm_in = L.reshape(emb_layer, (-1, self.args.max_word_len, self.config['char']['emb_dim'])) # (1500, 32, 16)
                lstm_f = L.LSTMLayer(lstm_in, 32, mask_input=ml, grad_clipping=10., learn_init=True,
                                     peepholes=False, precompute_input=True,
                                     ingate=gate_params,forgetgate=gate_params,cell=cell_params,outgate=gate_params,
                                     # unroll_scan=True,
                                     only_return_final=True, name='forward'+suf) # (1500, 32)
                lstm_b = L.LSTMLayer(lstm_in, 32, mask_input=ml, grad_clipping=10., learn_init=True, 
                                     peepholes=False, precompute_input=True,
                                     ingate=gate_params,forgetgate=gate_params,cell=cell_params,outgate=gate_params,
                                     # unroll_scan=True,
                                     only_return_final=True, backwards=True, name='backward'+suf) # (1500, 32)
                remove_reg(lstm_f)
                remove_reg(lstm_b)
                if avg:
                    set_zero(lstm_f)
                    set_zero(lstm_b)
                xc = L.concat([lstm_f, lstm_b], axis=1) # (1500, 64)
                if self.args.lstm_tagger:
                    xc = L.reshape(xc, (-1, self.args.max_sent_len, 64)) # (100, 161, 64)
                elif self.args.trans_tagger:
                    xc = L.reshape(xc, (-1, self.args.window_size, 64)) # (100, 15, 64)
                else:
                    xc = L.reshape(xc, (-1, 26, 64)) # (100, 26, 64)
                return xc


        elif self.name == 'morph':
            # idx (100, 26/161, 16)  emb (100, 26/161, 16, 32)
            if self.args.morph_model == 'max':
                xm = L.MaxPool2DLayer(emb_layer, (self.args.max_morph_len, 1)) # (100, 26/161, 1, 32)
                # xm = L.reshape(xm, (-1, 26, self.config['morph']['emb_dim'])) # (100, 26/161, 32)
                xm = L.flatten(xm, outdim=3) # (100, 26/161, 32)
                # xm = L.ExpressionLayer(emb_layer, lambda x: T.max(x, 2))
            elif self.args.morph_model == 'avg':
                mask = L.ExpressionLayer(idx_layer, lambda x: T.neq(x, 0)) # (100, 26, 16)
                mask = L.dimshuffle(mask, (0, 1, 2, 'x')) # (100, 26, 16, 1)
                mask = L.ExpressionLayer(mask, lambda x: T.extra_ops.repeat(x, self.config['morph']['emb_dim'], 3)) # (100, 26, 16, 1)
                xm = L.ElemwiseMergeLayer([emb_layer, mask], lambda x,m: T.sum(x*m, 2) / T.sum(m, 2)) # (100, 26, 32)
                # xm = L.reshape(xm, (-1, self.args.feat_shape, self.config['morph']['emb_dim'])) # (100, 26, 32)
            return xm
        else:
            return emb_layer


###################################
# Helpers

class MaskLayer(L.Layer):
    def __init__(self, incoming, p=0.01, **kwargs):
        super(MaskLayer, self).__init__(incoming, **kwargs)
        self.srng = RandomStreams(12345)
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            # 0 -> 0, 1+ -> 1 with probability of p
            return T.switch(input, T.switch(self.srng.binomial(n=1, p = self.p), 1, input), 0)



def clip(l, b = 1):
    """
    A very simple gradient clipping wrapper because stupid lasagne doens't support it 
    """
    return L.ExpressionLayer(l, lambda x: theano.gradient.grad_clip(x, -b, b))



