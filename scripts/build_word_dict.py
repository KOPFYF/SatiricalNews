import os
import random
import pickle
import gzip
import logging
import numpy as np
from collections import Counter
# import spacy
# nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_sm', disable=['tagger','ner','textcat'])
# nlp.max_length=50000000

import argparse
import glob

import data_utilities as du

from functools import reduce

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

def build_dict(docs, max_words=5000000, dict_file=None):
    """
        :param docs: a doc is a list of sentences
        :return: dictionary of words
        """


    word_count = Counter()
    # char_count = Counter()
    for doc in docs:
        logging.info('processing %s' %(doc))
        docs = du.load_sent(doc)
        # sents = sum(sents, [])
        # text = nlp(' '.join(sents))
        # for sent in text.sents:
        for doc in docs:
            '''
            print(len(sent))
            raise Exception
            '''
            for sent in doc:
                for w in sent.split(' '):
                    # w = w.lemma_
                    # w = w.text
                    if w.isdigit():
                        w = "num@#!123"
                    word_count[w.lower()] += 1

    ls = word_count.most_common(max_words)
    # chars = char_count.most_common(80)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    # leave 0 to padding
    # leave 1 to UNK
    word_dict = {w[0]: index + 2 for (index, w) in enumerate(ls)}

    if dict_file:
        with open(dict_file, 'wb') as fid:
            pickle.dump(word_dict, fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--df', type=str, default=None)

    args = parser.parse_args()
    parent_dir = args.dir

    flist = ['fake/train.txt', 'fake/test.txt', 'fake/dev.txt']
    tlist = ['true/true_train_{}.txt'.format(i) for i in range(1, 7)] + \
            ['true/true_test_{}.txt'.format(i) for i in range(1,3)] + \
            ['true/true_validation_{}.txt'.format(i) for i in range(1,3)]

    # docs = glob.glob(os.path.join(parent_dir,fname))
    docs = [os.path.join(parent_dir, fname) for fname in (flist + tlist)]

    build_dict(docs, dict_file=args.df)

