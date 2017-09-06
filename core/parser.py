import os, sys, gzip
from data import *
from util import *
from manager import *
from oracle import *
from parser_model import *
from collections import defaultdict, OrderedDict, Counter
from itertools import izip
import numpy as np
from time import time


class Parser(object):
    def __init__(self, manager):
        self.manager = manager
        self.args = manager.args
        self.system = manager.system
        self.logger = manager.logger
        self.extract = Extractor(self).extract
        self.model = ParserModel(self.manager)
        t0 = time()
        self.model.build_graph()
        print 'time used for building graph: %d s' % (time() - t0)

    def parse(self, mode='dev'):
        print 'start parsing...'
        t0 = time()
        assert mode in ['dev', 'test']
        self.stats = defaultdict(int)

        if mode == 'dev':
            batch_stream = BatchStream(range(self.args.train_ends, self.args.dev_ends), 
                                        min(100, len(self.manager.dev_sents)))
            all_states = {i:State(self.manager.all_sents[i]) for i in range(self.args.train_ends, self.args.dev_ends)}
        else:
            batch_stream = BatchStream(range(self.args.dev_ends, self.args.test_ends),
                                        min(100, len(self.manager.test_sents)))
            all_states = {i:State(self.manager.all_sents[i]) for i in range(self.args.dev_ends, self.args.test_ends)}

        for sidx_b in batch_stream:
            state_b = [all_states[sidx] for sidx in sidx_b]
            tidx_b = [self.extract(state) for state in state_b] # (100, 26)
            valid_b = np.array([self.system.valid_mask(state) for state in state_b]) # (100, 26), (100, 26)
            res = self.model.actor_predict_avg(sidx_b, tidx_b, valid_b) 
            yprob_b, zprob_b = res[0], res[1]

            for sidx, state, yprob, zprob in izip(sidx_b, state_b, yprob_b, zprob_b):
                yp = np.argmax(yprob)
                zp = np.argmax(zprob)
                self.stats[yp] += 1
                new_state = state.perform(self.system, yp, zp)
                all_states[sidx] = new_state
                if not new_state.finished():
                    batch_stream.add(sidx)

        evaluator = Evaluator(self.system)
        for state in all_states.values():
            evaluator.evaluate(state)

        if mode == 'test' and self.args.out:
            if self.args.out.endswith('.gz'):
                with gzip.open(self.args.out, 'wb') as f:
                    for sidx, state in sorted(all_states.items()):
                        f.write(state.to_str(self.system))
            else:
                with open(self.args.out, 'w') as f:
                    for sidx, state in sorted(all_states.items()):
                        f.write(state.to_str(self.system))

        self.logger.log(evaluator.result(mode))
        self.logger.log('STATS: '+ ', '.join('%s=%d'%(k,v) for (k,v) in self.stats.items()))
        print 'time used for prediction: %d s' % (time() - t0)
        return evaluator.uas(), evaluator.las()


    # TODO: restructure
    def train(self, num_steps):
        oracle = Oracle(self.system)
        every = min(max(1000, num_steps / 10), 2000)
        run_acc_y = 0.0 # running average accuracy of the most recent 10 batches
        run_acc_z = 0.0 # running average accuracy of the most recent 10 batches
        run_acc_w = 0.0 # running average accuracy of the most recent 10 batches

        best_uas = 0.0
        best_las = 0.0
        best_step = 0
        no_improve = 0

        all_states = {i:State(self.manager.all_sents[i]) for i in range(self.args.train_ends)}
        batch_pool = BatchPool(range(self.args.train_ends), self.args.batch_size, 
                                lambda x: len(self.manager.all_sents[x].tokens))

        # let the oracle mix the pool first, to avoid the initial batches are all at initial states
        print 'mixing states...'
        for (step, sidx_b) in izip(xrange(2000), batch_pool):
            for sidx in sidx_b:
                state = all_states[sidx]
                ygold = oracle.tell(state)
                zgold = state.get_gold_label(self.system, ygold)      
                new_state = state.perform(self.system, ygold, zgold)
                if not new_state.finished():
                    all_states[sidx] = new_state
                else:
                    all_states[sidx] = State(self.manager.all_sents[sidx])

        # start real training
        self.logger.log('START TRAINING')
        for (step, sidx_b) in enumerate(batch_pool):
            if step > num_steps:
                if self.manager.dev_sents:
                    uas, las = self.parse('dev')
                    self.logger.log('FINAL STOP PARSER, STEP = %d, UAS = %.2f, LAS = %.2f' \
                                        % (step, uas, las))
                else:
                    self.logger.log('FINAL STOP PARSER, NO DEV FILE')
                break

            state_b = [all_states[sidx] for sidx in sidx_b]
            tidx_b = [self.extract(state) for state in state_b] # (100, 26), (100, 26)
            valid_b = np.array([self.system.valid_mask(state) for state in state_b]) # (100, 26)

            ygold_b, zgold_b = [], []
            for sidx, state in izip(sidx_b, state_b):
                ygold = oracle.tell(state)
                zgold = state.get_gold_label(self.system, ygold)
                ygold_b.append(ygold)          
                zgold_b.append(zgold)
                new_state = state.perform(self.system, ygold, zgold)
                if not new_state.finished():
                    all_states[sidx] = new_state
                else:
                    all_states[sidx] = State(self.manager.all_sents[sidx])

            acc_y, acc_z, acc_w, loss = self.model.train_parser(ygold_b, zgold_b, \
                                                                      sidx_b, tidx_b, valid_b)

            run_acc_y += 0.1 * (acc_y - run_acc_y)
            run_acc_z += 0.1 * (acc_z - run_acc_z)
            run_acc_w += 0.1 * (acc_w - run_acc_w)
            print 'step: %d acc_y: %.2f acc_z: %.2f acc_w: %.2f loss: %.6f \r' \
                    % (step, run_acc_y*100, run_acc_z*100, run_acc_w*100, loss),
            sys.stdout.flush()

            # train auxiliary tagger for lstm parser
            if self.args.lstm_parser and self.args.aux_tagger:
                if step and step % int(1 / self.args.aux_ratio) == 0:
                    loss, acc = self.model.train_taggers(sidx_b)
                    print 'step: %d acc_y: %.2f acc_z: %.2f acc_w: %.2f loss: %.6f \r' \
                            % (step, run_acc_y*100, run_acc_z*100, acc*100, loss),
                    sys.stdout.flush()

            if step and step % every == 0:
                if self.manager.dev_sents:
                    print
                    uas, las = self.parse('dev')
                    self.logger.log('STEP = %d, UAS = %.2f, LAS = %.2f' \
                                        % (step, uas, las))
                    if las > best_las: 
                        self.model.save_params(self.args.model_to)
                        best_uas, best_las, best_step = uas, las, step
                        no_improve = 0
                    else:
                        no_improve += 1
                    if self.args.stop_after and no_improve >= self.args.stop_after:
                        # save the "not so good model"
                        self.model.save_params(self.args.model_to)
                        self.logger.log('EARLY STOP PARSER, STEP = %d, UAS = %.2f, LAS = %.2f' \
                                        % (step, uas, las))
                        break
                else:
                    self.model.save_params(self.args.model_to)
                print 
                self.model.log_params(self.logger)

        if self.manager.test_sents:
            self.model.load_params(self.args.model_to)
            uas, las = self.parse('test')
            self.logger.log('FINAL TEST: UAS = %.2f, LAS = %.2f' % (uas, las))












