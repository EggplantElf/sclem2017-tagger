import lasagne
import lasagne.layers as L
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.regularization import *
from lasagne.objectives import *
import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
import os
from core.util import *


class TaggerModel(object):
    def __init__(self, manager):
        self.manager = manager
        self.args =  manager.args

    def save_params(self, folder):
        print 'saving model to %s' % folder
        # np.savez(os.path.join(folder, 'tagger.npz'), *L.get_all_param_values(self.taggers.values()))
        np.savez(os.path.join(folder, 'tagger_avg.npz'), *L.get_all_param_values(self.taggers_avg.values()))

    def load_params(self, folder):
        print 'reading model from %s' % folder
        # with np.load(os.path.join(folder, 'tagger.npz')) as f:
        #     param_values = [f['arr_%d' %i] for i in range(len(f.files))]
        #     L.set_all_param_values(self.taggers.values(), param_values)
        with np.load(os.path.join(folder, 'tagger_avg.npz')) as f:
            param_values = [f['arr_%d' %i] for i in range(len(f.files))]
            L.set_all_param_values(self.taggers_avg.values(), param_values)

    def log_params(self, logger):
        print
        msg = ''
        for p, v in zip(L.get_all_params(self.taggers.values()), L.get_all_param_values(self.taggers.values())):
            msg += '%s: shape = %s, mean = %.4f, std = %.4f, norm = %.2f\n' % (p.name, v.shape, v.mean(), v.std(), (v**2).sum())
        if logger:
            logger.log(msg)
        else:
            print msg


    def get_conv_input(self, sidx, tidx, avg = False):
        suf = '_avg' if avg else ''

        feat_embs = [self.manager.feats[name].get_emb_layer(sidx, tidx, avg=avg) for name in self.args.source_feats]
        
        # TODO: change the meaning 
        if self.args.lex == 'mix':
            concat_emb = L.ElemwiseSumLayer(feat_embs) # (100, 15, 256)
        else:
            concat_emb = L.concat(feat_embs, axis=2) # (100, 15, 256+100)


        pos = np.array([0]*(self.args.window_size/2) + [1] + [0]*(self.args.window_size/2)).astype(theano.config.floatX)
        post = theano.shared(pos[np.newaxis, :, np.newaxis], borrow=True) # (1, 15, 1)
        posl = L.InputLayer((None, self.args.window_size, 1), 
                            input_var=T.extra_ops.repeat(post, sidx.shape[0], axis=0)) # (100, 15, 1)
        conv_in = L.concat([concat_emb, posl], axis=2) # (100, 15, 256+1)

        if self.args.pos_emb:
            posint = L.flatten(L.ExpressionLayer(posl, lambda x: T.cast(x, 'int64'))) # (100, 15)
            pos_emb = L.EmbeddingLayer(posint, self.args.window_size, 8, name = 'epos'+suf, 
                                        W=Normal(0.01) if not avg else Constant()) # (100, 15, 8)
            pos_emb.params[pos_emb.W].remove('regularizable')
            conv_in = L.concat([concat_emb, posl, pos_emb], axis=2) # (100, 15, 256+1+8)

        # # squeeze
        # if self.args.squeeze:
        #     conv_in = L.DenseLayer(conv_in, num_units=self.args.squeeze, name='squeeze'+suf, num_leading_axes=2,
        #                     W=HeNormal('relu')) # (100, 15, 256)        

        conv_in = L.dimshuffle(conv_in, (0, 2, 1)) # (100, 256+1, 15)

        return conv_in

    def get_context(self, conv_in, avg = False):
        suf = '_avg' if avg else ''

        conv_out = []
        # for n in [2,3,4,5,6,7,8,9]:
        # for n in [2,3,4,5]:
        for n in self.args.context_ngrams:
            conv = conv_in
            for i in range(self.args.conv_layers):
                conv = L.Conv1DLayer(conv, 128, n, name = 'conv_window_%d(%d)%s' % (n, i, suf),
                                    # W=HeNormal('relu') if not avg else Constant()) # (100, 128, 15-n+1)
                                    W=GlorotNormal('relu') if not avg else Constant()) # (100, 128, 15-n+1)


            conv = L.MaxPool1DLayer(conv, self.args.window_size-(n-1)*self.args.conv_layers) # (100, 128, 1)
            conv = L.flatten(conv, 2) # (100, 128)
            conv_out.append(conv)

        x = L.concat(conv_out, axis=1) # (100, 1024)

        return x

    def get_taggers(self, sidx, tidx, avg = False):
        suf = '_avg' if avg else ''
        taggers = {}

        # hierachical: latter targets is higher
        # e.g. utag -> morphstr -> stag
        if self.args.multi_task == 'hierarchical':
            prev = None
            conv_in = self.get_conv_input(sidx, tidx, avg)
            x = self.get_context(conv_in, avg)
            for name in self.args.target_feats: 
                h0 = L.concat([x, prev]) if prev else x
                h1 = L.DenseLayer(h0, 512, name='hid-%s%s'%(name, suf), W=Constant() if avg else HeNormal('relu')) # (100, 512)
                prev = h1
                h1 = L.dropout(h1, self.args.dropout)
                taggers[name] = L.DenseLayer(h1, len(self.manager.feats[name].map), name='tagger-%s%s'%(name, suf), 
                                            W=Constant() if avg else HeNormal(), nonlinearity=softmax) # (100, 25)
        # only the final target is higher:
        # e.g. utag, morphstr, xtag -> stag
        elif self.args.multi_task == 'final':
            hids = [x]
            conv_in = self.get_conv_input(sidx, tidx, avg)
            x = self.get_context(conv_in, avg)
            for name in self.args.target_feats[:-1]:
                h1 = L.DenseLayer(x, 512, name='hid-%s%s'%(name, suf), W=Constant() if avg else HeNormal('relu')) # (100, 512)
                hids.append(h1) # before dropout
                h1 = L.dropout(h1, self.args.dropout)
                taggers[name] = L.DenseLayer(h1, len(self.manager.feats[name].map), name='tagger-%s'%name, 
                                            W=Constant() if avg else HeNormal(), nonlinearity=softmax) # (100, 25)
            name = self.args.target_feats[-1]
            h1 = L.concat(hids, axis = 1)
            h1 = L.dropout(h1, self.args.dropout)
            h2 = L.DenseLayer(h1, 512, name='hid-%s%s'%(name, suf), W=Constant() if avg else HeNormal('relu')) # (100, 512)
            h2 = L.dropout(h2, self.args.dropout)
            taggers[name] = L.DenseLayer(h2, len(self.manager.feats[name].map), name='tagger-%s'%name, 
                                        W=Constant() if avg else HeNormal(), nonlinearity=softmax) # (100, 25)
        # equal level (or just one task)
        elif self.args.multi_task == 'equal':
            conv_in = self.get_conv_input(sidx, tidx, avg)
            x = self.get_context(conv_in, avg)
            x = L.dropout(x, self.args.dropout)
            for name in self.args.target_feats: 
                for n in range(self.args.tagger_layers):
                    x = L.DenseLayer(x, 512, name='hid%d-%s%s'%(n,name, suf), W=Constant() if avg else GlorotNormal('relu')) # (100, 512)
                    x = L.dropout(x, self.args.dropout)
                taggers[name] = L.DenseLayer(x, len(self.manager.feats[name].map), name='tagger-%s'%name, 
                                            # W=Constant() if avg else HeNormal(), nonlinearity=softmax) # (100, 25)
                                            W=Constant() if avg else GlorotNormal(), nonlinearity=softmax) # (100, 25)

        # only share char model, equal level
        elif self.args.multi_task == 'char_only':
            conv_in = self.get_conv_input(sidx, tidx, avg)
            for name in self.args.target_feats: 
                x = self.get_context(conv_in, avg)
                x = L.dropout(x, self.args.dropout)

                h1 = L.DenseLayer(x, 512, name='hid-%s%s'%(name, suf), W=Constant() if avg else HeNormal('relu')) # (100, 512)
                h1 = L.dropout(h1, self.args.dropout)
                taggers[name] = L.DenseLayer(h1, len(self.manager.feats[name].map), name='tagger-%s'%name, 
                                            W=Constant() if avg else HeNormal(), nonlinearity=softmax) # (100, 25)





        # # clipping
        # for name in self.args.target_feats:
        #     taggers[name] = clip(taggers[name], 1)

        return taggers


    def build_graph(self):
        print 'building models...'
        sidx = T.lvector('sidx')    # sentence ids of the batch
        tidx = T.lmatrix('tidx')    # token ids in each sentence of the batch
        step = T.fscalar('step')

        self.targets = {}
        for name in self.args.target_feats:
            self.targets[name] = self.manager.feats[name].data[sidx.dimshuffle(0, 'x'), tidx][:, self.args.window_size/2] # (100, )

        lr = self.args.learn_rate * self.args.decay ** T.floor(step / 2000.)

        self.taggers  = self.get_taggers(sidx, tidx, avg=False)
        self.taggers_avg = self.get_taggers(sidx, tidx, avg=True)

        # averaged model for prediction
        preds_avg = []
        for name in self.args.target_feats:
            prob_avg = L.get_output(self.taggers_avg[name], deterministic=True) # (100, 25)
            pred_avg = T.argmax(prob_avg, axis=1) # (100, )
            preds_avg.append(pred_avg)
        self.tagger_predict_avg = theano.function([sidx, tidx], preds_avg, 
                                             on_unused_input='ignore', allow_input_downcast=True)

        # training
        if self.args.train:
            accs = []
            total_xent = theano.shared(np.array(0.).astype(theano.config.floatX))
            # total_xent = 0.
            for name in self.args.target_feats:
                prob = L.get_output(self.taggers[name], deterministic=False) # (100, 25)
                pred = T.argmax(prob, axis=1) # (100, )
                target = self.targets[name] # (100, )
                acc = T.mean(T.cast(T.eq(pred, target), theano.config.floatX))
                accs.append(acc)
                xent = categorical_crossentropy(prob, target)
                total_xent += T.mean(xent)
            reg = regularize_network_params(L.get_all_layers(self.taggers.values()), l2)
            cost = total_xent + self.args.reg_rate * reg

            params = L.get_all_params(self.taggers.values(), trainable=True) 
            avg_params = L.get_all_params(self.taggers_avg.values(), trainable=True)
            grads = T.grad(cost, params)

            updates = lasagne.updates.momentum(grads, params, lr, self.args.momentum)
            updates = apply_moving_average(params, avg_params, updates, step, 0.9999)


            self.train_taggers = theano.function([step, sidx, tidx],
                                                [cost] + accs,
                                                updates=updates, 
                                                on_unused_input='ignore',
                                                allow_input_downcast=True)




def apply_moving_average(params, avg_params, updates, step, decay):
    # assert params and avg_params are aligned
    weight = T.min([decay, step / (step + 1.)])
    avg_updates = []
    for p, a in zip(params, avg_params):
        avg_updates.append((a, a - (1. - weight) * (a - p)))
    return updates.items() + avg_updates


def set_zero(layer):
    for p in layer.get_params():
        v = p.get_value()
        p.set_value(np.zeros_like(v))


def remove_reg(layer):
    for p in layer.params:
        if 'regularizable' in layer.params[p]:
            layer.params[p].remove('regularizable')







