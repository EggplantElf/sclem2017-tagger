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
from util import *

# theano.config.compute_test_value = 'ignore'


class ParserModel(object):
    def __init__(self, manager):
        self.manager = manager
        self.args =  manager.args
        self.debug = {}



    def save_params(self, folder, num = 0):
        print 'saving model to %s' % folder
        # np.savez(os.path.join(folder, 'parser_%d.npz' % num), *L.get_all_param_values([self.actor, self.labeler]+self.taggers.values()))
        np.savez(os.path.join(folder, 'parser_%d_avg.npz' % num), *L.get_all_param_values([self.actor_avg, self.labeler_avg]+self.taggers_avg.values()))


    def load_params(self, folder, num = 0):
        print 'reading model from %s' % folder
        # with np.load(os.path.join(folder, 'parser_%d.npz' % num)) as f:
        #     param_values = [f['arr_%d' %i] for i in range(len(f.files))]
        #     L.set_all_param_values([self.actor, self.labeler]+self.taggers.values(), param_values)
        with np.load(os.path.join(folder, 'parser_%d_avg.npz' % num)) as f:
            param_values = [f['arr_%d' %i] for i in range(len(f.files))]
            L.set_all_param_values([self.actor_avg, self.labeler_avg]+self.taggers_avg.values(), param_values)

    def log_params(self, logger):
        print
        msg = ''
        for p, v in zip(L.get_all_params([self.actor, self.labeler]+self.taggers.values()), L.get_all_param_values([self.actor, self.labeler]+self.taggers.values())):
            msg += '%s: shape = %s, mean = %.4f, std = %.4f, norm = %.2f\n' \
                    % (p.name, v.shape, v.mean(), v.std(), (v**2).sum())
        if logger:
            logger.log(msg)
        else:
            print msg

    def get_actor(self, sidx, tidx, valid, avg = False):
        suf = '_avg' if avg else ''
        feat_embs = [self.manager.feats[name].get_emb_layer(sidx, tidx, avg=avg) for name in self.args.source_feats]


        x = L.concat(feat_embs, axis=2) # (100, 26, 256+32+32+...)
        if self.args.squeeze:
            x = L.DenseLayer(x, num_units=self.args.squeeze, name='h0'+suf, num_leading_axes=2,
                            W=HeNormal('relu')) # (100, 26, 256)

        x = L.flatten(x) # (100, 26*256)

        h1 = L.DenseLayer(x, num_units=self.args.nh1, name = 'h1'+suf,
                            W=HeNormal('relu')) # (100, 512)

        h1 = L.dropout(h1, self.args.dropout)

        taggers = {}
        if self.args.aux_tagger:
            hids = [h1]
            for name in self.args.target_feats:
                hid = L.DenseLayer(h1, 256, name='hid-%s%s'%(name, suf), W=HeNormal('relu')) # (100, 512)
                hids.append(hid)
                hid = L.dropout(hid, self.args.dropout)
                # h1 = L.dropout(h1, self.args.dropout)
                taggers[name] = L.DenseLayer(hid, len(self.manager.feats[name].map), name='tagger-%s'%name, 
                                            W=HeNormal(), nonlinearity=softmax) # (100, 25)
            h1 = L.concat(hids, axis=1)

        h2 = L.DenseLayer(h1, num_units=self.args.nh2, name = 'h2'+suf,
                            W=HeNormal('relu')) # (100, 256)

        h2 = L.dropout(h2, self.args.dropout)
        h3y = L.DenseLayer(h2, num_units=self.args.nh3, name = 'h3y'+suf, W=HeNormal(),
                            nonlinearity=softmax) # (100, 4) num of actions
        h3s = L.concat([h2, h3y], axis=1) # (100, 256+4+4), this way shouldn't output <UNK> if its not SHIFT
        h3z = L.DenseLayer(h2, num_units=self.args.size['label'], name = 'h3z'+suf, W=HeNormal(),
                            nonlinearity=softmax) # (100, 25) number of labels

        if avg:
            set_all_zero([h3y, h3z] + taggers.values())

        return h3y, h3z, taggers

    # normal parser without LSTM contexts
    def build_graph(self):
        print 'building models...'
        y_gold = T.lvector('y_gold')  # index of the correct action from oracle
        z_gold = T.lvector('z_gold')  # index of the correct label from oracle
        sidx = T.lvector('sidx')    # sentence ids of the batch
        tidx = T.lmatrix('tidx')    # token ids in each sentence of the batch
        valid = T.fmatrix('valid')  # valid action mask

        self.step = theano.shared(np.array(0.).astype(theano.config.floatX), name='step')


        lr = self.args.learn_rate * self.args.decay ** T.floor(self.step / 2000.)

        self.actor, self.labeler, self.taggers  = self.get_actor(sidx, tidx, valid, avg=False)
        self.actor_avg, self.labeler_avg, self.taggers_avg = self.get_actor(sidx, tidx, valid, avg=True)

        # averaged model for prediction
        actor_prob, labeler_prob = L.get_output([self.actor_avg, self.labeler_avg], deterministic=True)
        actor_rest = actor_prob * valid # mask the probabilities of invalid actions to 0
        actor_rest_normalized = actor_rest / T.sum(actor_rest, axis=1, keepdims=True)

        preds_avg = []
        if self.args.aux_tagger:
            for name in self.args.target_feats:
                prob_avg = L.get_output(self.taggers_avg[name], deterministic=True) # (100, 25)
                pred_avg = T.argmax(prob_avg, axis=1) # (100, )
                preds_avg.append(pred_avg)

        self.actor_predict_avg = theano.function([sidx, tidx, valid], 
                                             [actor_rest_normalized, labeler_prob] + preds_avg, 
                                             on_unused_input='ignore', allow_input_downcast=True)
        

        # training
        # only compile if in training mode (has training data)
        if self.args.train:
            # parser objectives
            y_prob, z_prob = L.get_output([self.actor, self.labeler], deterministic=False)
            y_xent = categorical_crossentropy(y_prob, y_gold)
            z_xent = categorical_crossentropy(z_prob, z_gold)

            y_pred = T.argmax(y_prob, 1)
            z_pred = T.argmax(z_prob, 1)
            z_mask = T.eq(y_pred, y_gold) & T.lt(y_gold, self.args.idsh)

            acc_y = T.mean(T.cast(T.eq(y_pred, y_gold), theano.config.floatX))
            acc_z = T.cast(T.sum(T.eq(z_pred, z_gold) * z_mask) + 1., theano.config.floatX)\
                        / T.cast(T.sum(z_mask) + 1., theano.config.floatX) 

            cost = T.mean(y_xent) + T.mean(z_xent * z_mask)

            params = L.get_all_params([self.actor, self.labeler] + self.taggers.values(), trainable = 'True') 
            avg_params = L.get_all_params([self.actor_avg, self.labeler_avg] + self.taggers_avg.values(), trainable = 'True')

            # accuracy of all auxiliary tasks
            acc_w = acc_y - acc_y
            # joint objective for aux tagger
            if self.args.aux_tagger:
                # tags of s0 are the targets
                for name in self.args.target_feats:
                    w_gold = self.manager.feats[name].data[sidx.dimshuffle(0, 'x'), tidx][:, 0] # (100, )
                    w_prob =  L.get_output(self.taggers[name], deterministic=False)
                    w_xent = categorical_crossentropy(w_prob, w_gold)
                    w_mask = T.neq(w_gold, 0)
                    cost += self.args.aux_ratio * T.mean(w_xent * w_mask)
                    
                    w_pred = T.argmax(w_prob, axis=1)
                    acc = T.cast(T.sum(T.eq(w_pred, w_gold) * w_mask) + 1., theano.config.floatX)\
                            / T.cast(T.sum(w_mask) + 1., theano.config.floatX) 
                    acc_w += acc / len(self.args.target_feats)

            reg = regularize_network_params(L.get_all_layers([self.actor, self.labeler] + self.taggers.values()), l2)
            cost += self.args.reg_rate * reg

            updates = lasagne.updates.momentum(cost, params, lr, self.args.momentum)
            updates = apply_moving_average(params, avg_params, updates, self.step, 0.9999)

            self.train_parser = theano.function([y_gold, z_gold, sidx, tidx, valid],
                                                [acc_y, acc_z, acc_w, cost],
                                                updates=updates, 
                                                on_unused_input='ignore',
                                                allow_input_downcast=True)




