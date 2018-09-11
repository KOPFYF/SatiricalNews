import data_utilities as du
import utilities as util
import os
import sys
import numpy as np
import logging
import lasagne
import arg_parser as ap
import srl_parse as sp
import spacy

nlp = spacy.load('en_core_web_sm')


def words2word(v,word_embed,word_dict):
    '''
    input: vertex: [v, range][0] -> v
    output: average embedding for this group

    '''
    embeddings = []
    for word in v.val.split():
    	if word.isdigit():
            word = "num@#!123"
    	embedding = word_embed[word_dict[word.lower()]]
    	embeddings.append(embedding)
    embeddings = np.array(embeddings)
    return {v: embeddings.mean(axis=0)}

    
# TODO: need to change the hash call in accordance with the modification to api

def test1(args):
    
    docs = []
    docs += du.load_sent('../datasets/bbcnews.txt')
    logging.info('docs: {}'.format(len(docs)))
    logging.info("building dictionary...")
    word_dict, char_dict = util.build_dict(docs)
    word_embed = util.words2embedding(word_dict, 100, args.embedding_file) 
    (args.word_vocab_size, args.word_embed_size) = word_embed.shape
  
    logging.info('docs: {}'.format(word_embed.shape)) # (119, 100) # Words: 117 -> 117
    print(word_dict)
    doc = ' '.join(docs[0])
    # with open('bbcnews.txt') as f:
    #     docs = f.read()
    # sp.build_graph(doc)
    vertice_map = sp.hash_vertex(doc)
    for vertice in vertice_map:
        print(words2word(vertice[0],word_embed,word_dict))


if __name__ == '__main__':
    args = ap.get_args()
    test1(args)
