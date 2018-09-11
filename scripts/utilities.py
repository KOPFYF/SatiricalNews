import random
import pickle
import gzip
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from collections import Counter
from graph import *
import spacy
nlp = spacy.load('en_core_web_sm')

import codecs


def build_dict(docs, max_words=500000, dict_file=None):
    """
        :param docs: a doc is a list of sentences
        :return: dictionary of words
        """

    def _dict_loader(dict_file):
        with open(dict_file) as f:
            dicts = f.readlines()
        return {d.strip(): index + 2 for (index, d) in enumerate(dicts)}

    if dict_file is not None:
        if len(dict_file) == 2:
            return _dict_loader(dict_file[0]), _dict_loader(dict_file[1])
        if len(dict_file) == 1:
            return _dict_loader(dict_file[0])

    word_count = Counter()
    # char_count = Counter()
    # for doc in docs:
    doc = nlp(' '.join(docs))
    for sent in doc.sents:
        for w in sent:
            # w = w.lemma_
            w = w.text
            # if w == "":
            #     continue
            if w.isdigit():
                w = "num@#!123"
            word_count[w.lower()] += 1

    ls = word_count.most_common(max_words)
    # chars = char_count.most_common(80)

    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    # leave 0 to padding
    # leave 1 to UNK
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def words2embedding(word_dict, dim, in_file=None):
    num_words = max(word_dict.values()) + 1
    embeddings = torch.FloatTensor(num_words, dim).uniform_()
    logging.warning("Embedding dimension: %d * %d" % (embeddings.shape[0],embeddings.shape[1]))

    if in_file is not None:
        logging.warning("loading embedding file: %s" % in_file)
        pre_trained = 0
        with codecs.open(in_file, encoding='utf') as f:
            l = f.readlines()
        for line in l:
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = torch.FloatTensor([float(x) for x in sp[1:]])
        logging.warning("pre-trained #: %d (%.2f%%)" % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings




if __name__ == '__main__':
    docs = ["President Donald Trump said North Korea's leader Kim Jong-un had vowed to destroy an engine test site, but did not specify which one."]
    w, c = build_dict(docs)
    print(w)
    print(c)
    # words2embedding(w, 100, in_file='/Users/feiyifan/Desktop/NLP/FEEL/glove_6B/glove.6B.100d.txt', init=lasagne.init.Uniform())
