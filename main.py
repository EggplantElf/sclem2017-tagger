#! /usr/bin/python
from core.util import *
from core.parser import *
from core.tagger import *
from core.data import *
import argparse
import numpy as np
import random
from collections import defaultdict
from time import time
import os

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Greedy Parser/Tagger')
    args.add_argument('-msg',default='', help='simple message of the experiment')
    args.add_argument('-lex',default='char',choices=['word','char','concat', 'add', 'none'],help='type of lexical model')
    args.add_argument('-char_model',default='cnn',choices=['cnn','lstm'],help='char model, cnn or lstm ')
    args.add_argument('-morph_model',default='max',choices=['max','avg'],help='morph model: max pooling or attention')
    args.add_argument('-tool',default='tagger',choices=['tagger','parser'],help='tagger or parser')


    args.add_argument('-squeeze', default=256,type=int, help='squeeze the representation of each word (word, char, tag, label...), 0 for not using')

    args.add_argument('-train',default=None, help='training file')
    args.add_argument('-dev',default=None, help='dev file')
    args.add_argument('-test',default=None, help='test file')

    args.add_argument('-out',default=None, help='output prediction file')
    args.add_argument('-log',default=None, help='log file')
    args.add_argument('-num_steps',default=200000,type=int,help='number of training steps (batches)')
    args.add_argument('-restart',default=3,type=int,help='number of restart to find the best random initialization')
    args.add_argument('-num_restart_steps',default=5000,type=int,help='number of training steps before restart')

    args.add_argument('-model_to',default=None, help='folder to save model')
    args.add_argument('-model_from',default=None, help='continue training from model')
    args.add_argument('-system',default='Swap', help='transition system')
    args.add_argument('-batch_size', default=100, type=int, help='batch size')
    args.add_argument('-learn_rate', default=0.1, type=float, help='learning rate')
    args.add_argument('-reg_rate', default=1e-5, type=float, help='regularization rate')
    args.add_argument('-decay', default=0.96, type=float, help='decay of learning rate per 2000 steps')
    args.add_argument('-momentum', default=0.9, type=float, help='momentum of learning rate')
    args.add_argument('-dropout', default=0, type=float, help='dropout rate')
    args.add_argument('-sigma', default=0, type=float, help='std of gaussian noise for training')
    args.add_argument('-first', default=999999, type=int, help='read only the first N sentences, for easier debug')
    args.add_argument('-cut_len', default=200, type=int, help='use only sentences shorter than N for training')
    args.add_argument('-stop_after', default=3, type=int, help='stop training if no improvement in N epochs on dev set, 0 for not early stop')
    args.add_argument('-seed', default=None, type=int, help='random state seed')
    args.add_argument('-hidden_layer_sizes', default='512,256', help='comma separated hidden layer sizes for parser')

    args.add_argument('-embw',default=None, help='pretrained word embedding file')
    args.add_argument('-embw_dim',default=64, type=int, help='nr of dimentions of word embeddings')
    args.add_argument('-conv_dim',default=25, type=int, help='nr of dimentions of character conv filters')
    args.add_argument('-embw_freeze',default=0, type=int, help='freeze word embeddings')
    args.add_argument('-lower',action='store_true', help='use lowercase of word form, since some embeddings are lowercase')
    args.add_argument('-save_embw', action='store_true', help='save word embedding to txt file')
    args.add_argument('-rescale', default=None, type=float, help='rescale the std of pretrained embedding')
    args.add_argument('-l2norm', default=None, type=float, help='l2norm the std of pretrained embedding')

    args.add_argument('-max_word_len', default=32, type=int, help='maximum length of a word in character embedding')
    args.add_argument('-max_morph_len', default=16, type=int, help='maximum length of morphs')
    args.add_argument('-ngrams', default='3,5,7,9', help='comma separated number for N-grams for CNN character filter')
    args.add_argument('-context_ngrams', default='2,3,4,5', help='comma separated number for context N-grams for CNN tagger')


    ##########################
    # for parser
    args.add_argument('-reverse', action='store_true', help='parse from right to left')

    ##########################
    # for tagger
    args.add_argument('-target_feats', default='', help='target feature of the tagger')
    args.add_argument('-source_feats', default='word,char', help='target feature of the tagger')
    args.add_argument('-context', default='lstm', choices=['cnn','lstm'], help='context model')

    args.add_argument('-window_size', default=15, type=int, help='window size of the tagger')
    args.add_argument('-multi_task',default='equal',choices=['equal','hierarchical', 'final', 'char_only'], \
                        help='relations of layers in multi-tasking')

    args.add_argument('-num_parsers', default=1, type=int, help='number of parsers for blending')
    args.add_argument('-unlabeled', action='store_true', help='unlabeled parsing')
    args.add_argument('-aux_tagger', action='store_true', help='use auxiliary tagger to regularize the parser')
    args.add_argument('-aux_ratio', default=0.1, type=float, help='cost ratio of auxiliary tagger(s)')
    args.add_argument('-stack_lstm', action='store_true', help='stack two lstm layers')
    args.add_argument('-split', default=0.1, type=float, help='split training data for tunining if no dev')

    args.add_argument('-tagger_layers', default=1, type=int, help='number of hidden layers for tagger')

    args.add_argument('-conv_layers', default=1, type=int, help='number of convolution layers')
    args.add_argument('-init_std', default=0.01, type=float, help='initial std for feature embeddings')
    args.add_argument('-pos_emb', action='store_true', help='use position embeddings for tagger')



    args = args.parse_args()

    # set the seed for both native random and numpy.random
    if not args.seed:
        args.seed = np.random.randint(10000)
    args.rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    

    if args.model_from and args.test:
        manager = DataManager(args, False)
    elif args.model_to and args.train:
        manager = DataManager(args, True)
    else:
        print 'Either train with model_to or test with model_from!'
        exit(0)

    ##########################
    # tagger
    if args.tool == 'tagger':
        # test tagger
        if args.model_from and args.test:
            tagger = Tagger(manager)
            tagger.model.load_params(args.model_from)
            acc = tagger.predict('test')
            tagger.logger.log('FINAL TEST: ACC = %.2f' % acc)
        # train tagger
        else:
            tagger = Tagger(manager)
            tagger.train(args.num_steps)

    #########################
    # parser
    else:
        # test tagger
        if args.model_from and args.test:
            parser = Parser(manager)
            parser.model.load_params(args.model_from)
            uas, las = parser.parse('test')
            parser.logger.log('FINAL TEST: UAS = %.2f, LAS = %.2f' % (uas, las))

        # train parser
        else:
            parser = Parser(manager)
            parser.train(args.num_steps)
