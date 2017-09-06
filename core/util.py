import time
import random
from collections import deque
from itertools import izip
import numpy as np
import os
import lasagne.layers as L
import theano.tensor as T
import numpy as np
import theano

class Logger:
    def __init__(self, log_file = None):
        self.log_file = log_file
        if self.log_file:
            open(log_file, 'w+').close() # clear the previous content of the file

    def log(self, msg):
        print msg
        if self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
                f.write(msg + '\n\n')


class Evaluator(object):
    def __init__(self, system):
        self.total = 0
        self.heads = 0
        self.labels = 0
        self.system = system

    def evaluate(self, state):
        for d, h, l in state.arcs:
            self.total += 1
            if h == state.sent.gold_heads[d]:
                self.heads += 1
                if l == self.system.get_label_idx(state.sent.tokens[d]['label']):
                    self.labels += 1

    def result(self, msg = None):
        uas = 'UAS: %d / %d = %.2f%%' % (self.heads, self.total, self.uas())
        las = 'LAS: %d / %d = %.2f%%' % (self.labels, self.total, self.las())
        s = uas + '\n' + las 
        if msg:
            s = msg + '\n' + s
        return s

    def uas(self):
        return 100.0 * self.heads / self.total

    def las(self):
        return 100.0 * self.labels / self.total

class TaggerEvaluator(object):
    def __init__(self, targets = ['utag']):
        self.total = 0
        self.targets = targets[:]
        if 'combi' in self.targets:
            self.targets += ['utag', 'morphstr']
        self.correct = {k:0 for k in self.targets} 


    def evaluate(self, sent):
        for token in sent.tokens[1:]:
            self.total += 1
            for target in self.correct:
                self.correct[target] += int(token[target] == token['p'+target])

    def result(self, msg = ''):
        for target in self.targets:
            msg += '\n%s: %d / %d = %.2f%%' % (target, self.correct[target], self.total, self.acc(target))
        return msg

    def acc(self, target = 'utag'):
        return 100. * self.correct[target] / self.total

    def avg_acc(self, targets):
        return sum(self.acc(t) for t in targets) / len(targets)

    def max_acc(self):
        return 100. * max(self.correct.values()) / self.total

    def min_acc(self):
        return 100. * min(self.correct.values()) / self.total








class BatchStream(object):
    def __init__(self, init_data, batch_size, shuffle_every = 0):
        self.queue = deque(init_data)
        self.batch_size = batch_size
        self.count = 0
        self.shuffle_every = shuffle_every
        # print random.randint(0, 100)


    def __iter__(self):
        batch = []
        while self.queue:
            self.count += 1
            if self.shuffle_every and self.count > self.shuffle_every:
                random.shuffle(self.queue)
                self.count = 0

            batch.append(self.queue.popleft())
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            elif not self.queue and batch:
                yield batch
                batch = []

    def add(self, data):
        self.queue.append(data)

    def addleft(self, data):
        self.queue.appendleft(data)

    def addrand(self, data, left_prob = 0.5):
        if random.random() < left_prob:
            self.queue.appendleft(data)
        else:
            self.queue.append(data)



class BatchPool(object):
    def __init__(self, init_data, batch_size, prob_func = None):
        self.pool = init_data
        self.all_idx = np.arange(len(init_data))
        self.batch_size = batch_size
        if prob_func:
            z = np.array([prob_func(d) for d in init_data])
            self.prob = z.astype('float') / z.sum() # probability distribution
        else:
            self.prob = None
        self.sample_idx = None
        self.now = 0

    def __iter__(self):
        while True:
            self.sample_idx = np.random.choice(self.all_idx, min(self.batch_size, len(self.pool)), False, self.prob)
            yield [self.pool[i] for i in self.sample_idx] # pool is probably not a numpy array, thus no list indexing

    def add(self, new_data):
        self.pool[self.sample_idx[self.now]] = new_data
        self.now += 1
        if self.now == self.batch_size:
            self.now = 0



######################################
# helpers

def apply_moving_average(params, avg_params, updates, step, decay, update_step = True):
    # assert params and avg_params are aligned
    weight = T.min([decay, step / (step + 1.)])
    avg_updates = []
    for p, a in zip(params, avg_params):
        avg_updates.append((a, a - (1. - weight) * (a - p)))
    if update_step:
        return updates.items() + avg_updates + [(step, step + 1.)]
    else:
        return updates.items() + avg_updates

def set_zero(layer):
    for p in layer.get_params():
        v = p.get_value()
        p.set_value(np.zeros_like(v))

def set_all_zero(layer):
    for p in L.get_all_params(layer):
        v = p.get_value()
        p.set_value(np.zeros_like(v))

def remove_reg(layer):
    for p in layer.params:
        if 'regularizable' in layer.params[p]:
            layer.params[p].remove('regularizable')

def clip(l, b = 1):
    return L.ExpressionLayer(l, lambda x: theano.gradient.grad_clip(x, -b, b))


    