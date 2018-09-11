import os, sys
import time 
import re
import spacy
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm', disable=['tagger','ner','textcat'])
# nlp.max_length = 1200000

from graph import *
from allennlp.predictors import Predictor
import data_utilities as du

import collections
import argparse
# import utilities as util
from pickle import load, dump

from pprint import pprint

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

class EventChainBuilder:
    offset = 0
    srl_predictor, coref_predictor = None, None
    word_dict = None
    predv, arg0v, arg1v, arg2v = None, None, None, None

    # the id of the verb in the current path
    v_id = -1

    corefs = None

    doc = None

    # important helpers
    # args_range = {}
    # path_list = []
    # list of all vertices
    vert_ls = []

    # create a hashtable of vertices (ling -> idx in vert_ls)
    # convenient for reusing vertices
    # so that the memory does not blow up for large datasets
    vert_map = {} 

    # get the maximum arg length
    max_length = 0

    v_idx = []
    arg0_idx = []
    arg1_idx = []
    arg2_idx = []

    event_chains = []

    debug = False
    use_v = False

    def __init__(self, wd=None, doc=None, debug=False, use_v=False, vfile=None, cfile=None, efile=None):
        if os.path.exists('/homes/du113/scratch/pretrained'):
            print('models already downloaded')
            self.srl_predictor = Predictor.from_path('/homes/du113/scratch/pretrained/srl-model-2018.05.25.tar.gz')
            self.coref_predictor = Predictor.from_path('/homes/du113/scratch/pretrained/coref-model-2018.02.05.tar.gz')
        else:
            print('downloading models...')
            self.srl_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
            self.coref_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")

        self.word_dict = wd 
        self.doc = doc
        self.debug = debug
        self.use_v = use_v

        '''
        logging.info('making path lists')
        self.make_path_list()

        if vfile:
            self.store_verts(vfile)
        
        logging.info('building coreference clusters')
        self.corefs = self.get_coref()
        logging.info('found %d clusters' % len(self.corefs))

        if cfile:
            self.store_corefs(cfile)

        logging.info('building event chains')
        self.make_event_chains()

        if efile:
            self.store_event_chains(efile)

        '''

    def store_verts(self, fname):
        logging.info('saving vertices to %s' % fname)
        params = {'vertices': self.vert_ls, \
                'vidx': self.v_idx, \
                'arg0idx': self.arg0_idx, \
                'arg1idx': self.arg1_idx, \
                'arg2idx': self.arg2_idx}

        with open(fname, 'wb') as fid:
            dump(params, fid)

    def load_verts(self, fname):
        logging.info('loading vertices from %s' % fname)
        with open(fname, 'rb') as fid:
            params = load(fid)
            self.vert_ls = params['vertices']
            self.v_idx = params['vidx']
            self.arg0_idx = params['arg0idx']
            self.arg1_idx = params['arg1idx']
            self.arg2_idx = params['arg2idx']

    def store_corefs(self, fname):
        logging.info('saving coreferences to %s' % fname)
        params = self.corefs

        with open(fname, 'wb') as fid:
            dump(params, fid)

    def load_corefs(self, fname):
        logging.info('loading coreferences from %s' % fname)

        with open(fname, 'rb') as fid:
            self.corefs = load(fid)

    '''
    def store_event_chains(self, fname):
        logging.info('saving event chains to %s' % fname)
        params = self.event_chains

        with open(fname, 'wb') as fid:
            dump(params, fid)
    '''
    def store_params(self, fname):
        logging.info('saving parameters to %s' % fname)
        params = {'vertices': self.vert_ls, \
                'vidx': self.v_idx, \
                'arg0idx': self.arg0_idx, \
                'arg1idx': self.arg1_idx, \
                'arg2idx': self.arg2_idx, \
                'echains': event_chains}

        with open(fname, 'wb') as fid:
            dump(params, fid)

    def make_path_list(self):
        # doc = nlp(self.doc)

        for i, sent in enumerate(self.doc):
            if i % 20 == 0:
                logging.info('processing sentence %d' % i)

            sent = nlp(sent)
            # self.path_list.append(self.build_path(sent))
            self.build_path(sent)

    def build_path(self, sent):
        # PARAMS: sent -- spacy sentence wrapper
        # just for debugging adn bookkeeping
        if self.debug:
            verb_dict = None

        if sent[0].text == '"' and sent[-1].text == '"':
            word_id_ls = []

            for w in sent[1:-1]:
                w = w.text
                if w.isdigit():
                    w = "num@#!123"
                word_id_ls.append(self.word_dict[w.lower()])
            self.arg1v = Vertex(word_id_ls, 'ARG1')

            # word_id_ls = []

            self.arg2v = None

            self.arg1v.set_range((self.offset+1, self.offset+len(sent)-2))   # inclusive bounds

            if self.debug:
                # for debugging
                self.arg1v.words = sent[1:-1]

            # record the idx of this arg1
            self.arg1_idx.append(len(self.vert_ls))
            # update the id of arg1, arg2
            # use the same v_id and arg0_id as the last sentence
            # nevermind
            # self.arg1_id = len(self.vert_ls)
            # self.arg2_id = -1

            # update the maximum arg length
            self.max_length = max(self.max_length, len(sent) - 2)
            # add to the vertex list
            self.vert_ls.append(self.arg1v)

        else:
            res = self.srl_predictor.predict(sentence=str(sent))
            # print(res)

            partitions = res['verbs']
            words = res['words']

            if not partitions:
                # self.offset + len(words)
                verb_dict = None

            elif len(partitions) == 1:
                # only 1 partition
                verb_dict = partitions[0]
            
            else:
                # otherwise, find the most-covering paritioning
                min_O = float('inf')
                min_id = 0

                for idx, verb_dict in enumerate(partitions):
                    num_O = collections.Counter(verb_dict["tags"])['O']
                    if num_O < min_O:
                        min_id = idx
                        min_O = num_O
                    # print('min_O:',min_O, 'min_id:',min_id)
                verb_dict = partitions[min_id]
                # print(verb_dict['description'])

            # print(verb_dict)
            '''
            verb_dict:
            {'description':'[ARG0 ... ] ..',
            'tags': ...., 'verb':'...'}
            '''
            begin, end = -1, -1
            last = ''
            
            # reset parameters
            self.predv, self.arg0v, self.arg1v, self.arg2v = None, None, None, None
            self.v_id = -1
            
            if not verb_dict:
                # make empty path
                return []

            # make vertex for each role in the sentence
            word_id_ls = []
            for i, t in enumerate(verb_dict['tags']):
                # print(i, t)
                if not t.startswith('I-'):
                    # print(end, begin)
                    if end < begin and i > 0:
                        end = i - 1

                        # for verb only, because for args they may have many corefs
                        # if the sentence has already appeared
                        # use the existing vertex instead of creating a new one
                        if last == 'V' and tuple(word_id_ls) in self.vert_map:
                            # get the vertex id (in vert_ls) from the hashtable
                            idx = self.vert_map[tuple(word_id_ls)]
                            # set the corresponding id to idx
                            self.v_id = idx
                            
                            # use the index to get the vertex
                            node = self.vert_ls[idx]
                            # should also set vertex reference to the correct ones
                            self.predv = node

                            if self.debug:
                                # for debugging
                                print('found',node,'at',idx)
                        else:
                            # otherwise, create a new vertex
                            node = Vertex(word_id_ls, last)

                            if last == 'ARG0':
                                self.arg0v = node
                                # record the idx of this arg0
                                self.arg0_idx.append(len(self.vert_ls))
                            elif last == 'V':
                                # if self.v_id == -1:
                                # indicating that no previous same verb was found
                                self.predv = node
                                # record the idx of this v
                                self.v_idx.append(len(self.vert_ls))
                                # update v_id
                                self.v_id = len(self.vert_ls)
                                # update vertices map
                                self.vert_map[tuple(word_id_ls)] = len(self.vert_ls)
                            
                            elif last == 'ARG1':
                                self.arg1v = node
                                # record the idx of this arg1
                                self.arg1_idx.append(len(self.vert_ls))
                            elif last == 'ARG2':
                                self.arg2v = node
                                # record the idx of this arg2
                                self.arg2_idx.append(len(self.vert_ls))

                            node.set_range((self.offset + begin, self.offset + end))
                            
                            if self.debug:
                                # for debugging
                                node.words = sent[begin:end+1]

                            # update max length
                            self.max_length = max(self.max_length, end - begin + 1)

                            # if node.type != 'V':
                            # add to the vertex list
                            self.vert_ls.append(node)

                        # reset word_ls
                        word_id_ls = []

                    if t == 'O':
                        continue
                    if t.startswith('B-'):
                        last = t.lstrip('B-')
                        # print('found head of', last)

                        if last not in ['V','ARG0','ARG1','ARG2']:
                            continue
                        begin = i

                if words[i].isdigit():
                    w = "num@#!123"
                else:
                    w = words[i].lower()
                word_id_ls.append(self.word_dict[w])
                    
        if self.use_v:
            res = [self.predv]
            res.append(self.arg0v)
            res.append(self.arg1v)
            res.append(self.arg2v)
        else:
            res = []
            res.append(self.v_id)
            res.append(self.arg0_idx[-1] if self.arg0v else -1)
            res.append(self.arg1_idx[-1] if self.arg1v else -1)
            res.append(self.arg2_idx[-1] if self.arg2v else -1)

        if self.use_v:
            if self.arg0v:
                self.arg0v.path = res
            if self.arg1v:
                self.arg1v.path = res
            if self.arg2v:
                self.arg2v.path = res
        else:
            assert len(res) > 0
            if self.arg0v:
                self.arg0v.idx_path = res
            if self.arg1v:
                self.arg1v.idx_path = res
            if self.arg2v:
                self.arg2v.idx_path = res

        # increase offset
        self.offset += len(sent)

        if self.debug:
            # for debugging
            if verb_dict:
                print(verb_dict['description'])
            print(res)

        # no need to return it
        # return res

    def get_coref(self):
        '''
        input: the document
        return:
        clusters_indexs: [[34, 41], [95, 98]] 

        '''
        self.doc = ' '.join(self.doc)
        results_coref = self.coref_predictor.predict(document=self.doc)

        '''
        if self.debug:
            # for debugging
            print(results_coref)
        '''

        return results_coref["clusters"]

    def make_event_chains(self, word_id=False):
        def bin_search(single_range):
            # vert_ls is a sorted list by the range of indices
            begin, end = 0, len(self.vert_ls) - 1

            # print(single_range)
            while begin <= end:
                # print(begin, end)
                # if begin == 0 and end == 2:
                #     raise Exception
                mid = (begin+end) // 2
                vert = self.vert_ls[mid]

                if vert.includes(single_range):
                    # we know that the query has to be contained in one of the list element
                    return vert

                if vert.before(single_range):
                    # the range of the query is bigger
                    begin = mid + 1

                elif vert.after(single_range):
                    # the pick is bigger, search in lowe half
                    end = mid - 1
                else:
                    return None

            # if got here, then the vertex is not found
            return None

        def sub_ent(vertex, head, r1, r2):
            # subtitute the word idx in r1 in vertex with the word idx in r2 in head
            if self.debug:
                # for debugging
                print('******')
                print('substituting', r1, 'in', vertex.debug())
                print('with', r2, 'in', head.debug())
            # starting point in vertex
            v_offset = vertex.range[0]
            
            # starting point in head
            h_offset = head.range[0]

            if self.debug:
                # for debugging
                print('original v', vertex.val)

            h_idx = head.val[r2[0]-h_offset:r2[1]-h_offset+1]   # this is the source indices
            if self.debug:
                # for debugging
                print('head', h_idx)

            ori_val = vertex.val

            vertex.val = ori_val[:r1[0]-v_offset] + h_idx    # this is the target
            if r1[1] < vertex.range[1]:
                vertex.val += ori_val[r1[1]-v_offset+1:]

            if self.debug:
                # for debugging
                print('final v', vertex.val)
                print('******')

        def get_wd_idx(vertex, r):
            v_offset = vertex.range[0]
            return tuple(vertex.val[r[0]-v_offset:r[1]-v_offset+1])

        # store the event chains for each type of arg
        '''
        echain_2d_arg0 = []
        echain_2d_arg1 = []
        echain_2d_arg2 = []
        '''
        # instead of using list, use hashtable to map word idx of the cluster head to event chain
        echain_2d_arg0 = {} 
        echain_2d_arg1 = {}
        echain_2d_arg2 = {}

        for i, cluster in enumerate(self.corefs):
            # single chains for one 'protagonist'
            if i % 10 == 0:
                logging.info('processing cluster %d' %i)

            echain_arg0 = []
            echain_arg1 = []
            echain_arg2 = []

            head = None
            head_range = None
            head_wd_idx = None
            
            for single_range in cluster:
                vertex = bin_search(single_range)
                # assuming that the vertex cannot be a verb

                if vertex:
                    # add the current event to the appropriate event chain
                    if vertex.type == 'ARG0':
                        chain = echain_arg0
                    elif vertex.type == 'ARG1':
                        chain = echain_arg1
                    elif vertex.type == 'ARG2':
                        chain = echain_arg2

                    if self.debug:
                        # for debugging
                        print('found', vertex)
                    # found the coreferencing vertex
                    if not head:
                        if self.debug:
                            print('this is cluster head')
                        # if it is the first one in this cluster, make it the head
                        head = vertex
                        head_range = single_range
                        head_wd_idx = get_wd_idx(head, head_range)
                    else:
                        sub_ent(vertex, head, single_range, head_range) 

                    event = vertex.idx_path
                    '''
                    if self.debug:
                        # for debugging
                        print(event)
                    '''

                    if not chain or event != chain[-1]:
                        if word_id:
                            event = [self.vert_ls[i].val if i != -1 else None for i in event]
                        chain.append(event)

            if echain_arg0 and head_wd_idx:
                echain_2d_arg0[head_wd_idx] = echain_arg0
            if echain_arg1 and head_wd_idx:
                echain_2d_arg1[head_wd_idx] = echain_arg1
            if echain_arg2 and head_wd_idx:
                echain_2d_arg2[head_wd_idx] = echain_arg2

        # return echain_2d_arg0, echain_2d_arg1, echain_2d_arg2
        self.event_chains = [echain_2d_arg0, echain_2d_arg1, echain_2d_arg2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dict',type=str,default=None)
    parser.add_argument('--doc', type=str, default=None)
    parser.add_argument('--save_v', type=str, default=None)
    parser.add_argument('--save_c', type=str, default=None)
    parser.add_argument('--save_e', type=str, default=None)

    args = parser.parse_args()

    # docs = du.load_sent(args.doc)
    with open(args.doc) as fid:
        docs = fid.readlines() 

    assert args.word_dict
    with open(args.word_dict, 'rb') as fid:
        word_dict = load(fid)
    
    ecb = EventChainBuilder(word_dict, docs, vfile=args.save_v, cfile=args.save_c, efile=args.save_e)
    # ecb.save_params(args.save)
