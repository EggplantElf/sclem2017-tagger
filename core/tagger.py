import os, sys
from data import *
from util import *
from manager import *
from oracle import *
from tagger_model import *
from collections import defaultdict, OrderedDict, Counter
from itertools import izip
import numpy as np
from time import time

class Tagger(object):
    def __init__(self, manager):
        self.manager = manager
        self.args = manager.args
        self.system = manager.system
        self.logger = manager.logger
        self.extract = Extractor(self).extract_window
        self.model = TaggerModel(self.manager)
        t0 = time()
        self.model.build_graph()
        print 'time used for building graph: %d s' % (time() - t0)


    def predict(self, mode='dev'):
        assert mode in ['dev', 'test']
        print 'start tagging...'
        t0 = time()

        if mode == 'dev':
            batch_stream = BatchStream(range(self.args.train_ends, self.args.dev_ends), 
                                        min(100, len(self.manager.dev_sents)))
            all_states = {i:TaggerState(self.manager.all_sents[i]) for i in range(self.args.train_ends, self.args.dev_ends)}
        else:
            batch_stream = BatchStream(range(self.args.dev_ends, self.args.test_ends),
                                        min(100, len(self.manager.test_sents)))
            all_states = {i:TaggerState(self.manager.all_sents[i]) for i in range(self.args.dev_ends, self.args.test_ends)}

        for sidx_b in batch_stream:
            state_b = [all_states[sidx] for sidx in sidx_b]
            tidx_b = np.array([self.extract(state, self.args.window_size) for state in state_b]) # (100, 9)
            pred_bf = self.model.tagger_predict_avg(sidx_b, tidx_b)
            # get predictions
            for pred_b, name in izip(pred_bf, self.args.target_feats):
                for state, pred in izip(state_b, pred_b):
                    ptag = self.manager.feats[name].revmap[pred]
                    state.sent.tokens[state.idx]['p'+name] = ptag
                    if name == 'combi':
                        putag, pmorphstr = ptag.split('~')
                        state.sent.tokens[state.idx]['putag'] = putag
                        state.sent.tokens[state.idx]['pmorphstr'] = pmorphstr

                    # print state.sent.tokens[state.idx]['p'+name], state.sent.tokens[state.idx][name]

            # update sentence states
            for sidx, state in izip(sidx_b, state_b):
                new_state = state.next()
                all_states[sidx] = new_state
                if not new_state.finished():
                    batch_stream.add(sidx)

        evaluator = TaggerEvaluator(self.args.target_feats)

        sents = self.manager.dev_sents if mode == 'dev' else self.manager.test_sents
        for sent in sents:        
            evaluator.evaluate(sent)
        self.logger.log(evaluator.result())

        if mode == 'test' and self.args.out:
            with open(self.args.out, 'w') as f:
                for sent in sents:
                    f.write(sent.to_str(pred=self.args.target_feats))

        print 'time used for prediction: %d s' % (time() - t0)

        return evaluator.avg_acc(self.args.target_feats)

    # a wrapper for training just in case the gradient explodes
    def train(self, num_steps):
        for i in range(3):
            success = self.train_func(num_steps)
            if success: # no explosion
                return 
            else:
                print
                self.logger.log('REINITIALIZE THE PARSER')
                self.args.learn_rate /= 2.
                self.model.build_graph()
        self.logger.log('SOMETHING WRONG, GRADIENT EXPLODES 3 TIMES')


    def train_func(self, num_steps):
        oracle = Oracle(self.system)
        every = min(max(1000, num_steps / 10), 2000)
        run_accs = [0.0] * len(self.args.target_feats)  # running average accuracy of the most recent 10 batches
        best_acc, best_step = 0, 0
        no_improve = 0

        all_states = {i:TaggerState(self.manager.all_sents[i]) for i in range(self.args.train_ends)}

        # sample rate according to the sentence length, very important!
        batch_pool = BatchPool(range(self.args.train_ends), self.args.batch_size, 
                                lambda x: len(self.manager.all_sents[x].tokens))

        print 'mixing states...'
        for (step, sidx_b) in izip(xrange(2000), batch_pool):
            for sidx in sidx_b:
                all_states[sidx] = all_states[sidx].next()

        self.model.log_params(self.logger)

        # start real training
        for (step, sidx_b) in enumerate(batch_pool):
            if step > num_steps:
                acc = self.predict('dev')
                self.logger.log('FINAL STOP TAGGER, STEP = %d, ACC = %.2f' \
                                    % (step, acc))

                break
            state_b = [all_states[sidx] for sidx in sidx_b]
            tidx_b = np.array([self.extract(state, self.args.window_size) for state in state_b])

            for sidx, state in izip(sidx_b, state_b):
                all_states[sidx] = state.next()

            res = self.model.train_taggers(step, sidx_b, tidx_b)
            cost, accs = res[0], res[1:]

            # check for nan and restart
            if cost > 1000 or cost is np.nan:
                return False

            for i in range(len(accs)):
                run_accs[i] += 0.1 * (accs[i] - run_accs[i])

            acc_str = '\t'.join(['%s: %.2f' % (name, 100.*run_accs[i]) for i, name in enumerate(self.args.target_feats)])
            print 'step: %-6d cost: %.6f\t%s\r' % (step, cost, acc_str),
            sys.stdout.flush()

            if step and step % every == 0:
                print
                acc = self.predict('dev')
                if acc > best_acc: 
                    self.model.save_params(self.args.model_to)
                    best_acc, best_step = acc, step
                    no_improve = 0
                else:
                    no_improve += 1
                if self.args.stop_after and no_improve >= self.args.stop_after:
                    self.logger.log('EARLY STOP TAGGER, STEP = %d, ACC = %.2f' % (best_step, best_acc))
                    break
                self.model.log_params(self.logger)

        if self.args.test:
            print 'Testing...'
            self.model.load_params(self.args.model_to)
            acc = self.predict('test')
            self.logger.log('FINAL TEST: ACC = %.2f%%' % acc)

        return True