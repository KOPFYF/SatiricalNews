import os, sys
import resource
import gc
import time 
import re
import spacy
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm', disable=['tagger','ner','textcat'])
# nlp.max_length = 1200000

from graph import *
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import data_utilities as du

import collections
import argparse
# import utilities as util
from pickle import load, dump

import copy

from pprint import pprint

from multiprocessing.dummy import Pool as ThreadPool

import GPUtil

import logging
from glob import glob

from multiprocessing.dummy import Pool as ThreadPool 

logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                    format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

class EventChainBuilder:
    __slots__ = ('offset','predv', 'arg0v', 'arg1v', 'arg2v', \
            'srl_predictor', 'coref_predictor', \
            'word_dict', \
            'corefs', 'doc','v_id', 'vert_ls', 'vert_map', \
            'v_idx', 'arg0_idx', 'arg1_idx', 'arg2_idx', \
            'event_chains', 'debug')


    def __init__(self, wd=None, doc=None, srl=None, coref=None, debug=False):
        self.offset = 0
        self.predv, self.arg0v, self.arg1v, self.arg2v = \
                None, None, None, None
        
        self.srl_predictor = srl
        self.coref_predictor = coref

        self.word_dict = wd 
        self.corefs = None

        self.doc = doc
        # important helpers
        # the id of the verb in the current path
        self.v_id = -1

        # list of all vertices
        self.vert_ls = []

        # create a hashtable of vertices (ling -> idx in vert_ls)
        # convenient for reusing vertices
        # so that the memory does not blow up for large datasets
        self.vert_map = {} 

        # get the maximum arg length
        # max_length = 0

        self.v_idx = []
        self.arg0_idx = []
        self.arg1_idx = []
        self.arg2_idx = []

        self.event_chains = []
        self.debug = debug


    def reset(self):
        self.offset = 0
        
        self.predv, self.arg0v, self.arg1v, self.arg2v = \
                None, None, None, None
        self.corefs = None

        self.v_id = -1

        self.vert_ls = []

        self.vert_map = {} 

        self.v_idx = []
        self.arg0_idx = []
        self.arg1_idx = []
        self.arg2_idx = []

        self.event_chains = []


    def store_verts(self, fname):
        logging.warning('saving vertices to %s' % fname)
        params = {'vertices': self.vert_ls, \
                'vidx': self.v_idx, \
                'arg0idx': self.arg0_idx, \
                'arg1idx': self.arg1_idx, \
                'arg2idx': self.arg2_idx}

        with open(fname, 'wb') as fid:
            dump(params, fid)

    def load_verts(self, fname):
        logging.warning('loading vertices from %s' % fname)
        with open(fname, 'rb') as fid:
            params = load(fid)
            self.vert_ls = params['vertices']
            self.v_idx = params['vidx']
            self.arg0_idx = params['arg0idx']
            self.arg1_idx = params['arg1idx']
            self.arg2_idx = params['arg2idx']

    def store_corefs(self, fname):
        logging.warning('saving coreferences to %s' % fname)
        params = self.corefs

        with open(fname, 'wb') as fid:
            dump(params, fid)

    def load_corefs(self, fname):
        logging.warning('loading coreferences from %s' % fname)

        with open(fname, 'rb') as fid:
            self.corefs = load(fid)

    def store_params(self, fname):
        logging.warning('saving parameters to %s' % fname)
        params = {'vertices': self.vert_ls, \
                'vidx': self.v_idx, \
                'arg0idx': self.arg0_idx, \
                'arg1idx': self.arg1_idx, \
                'arg2idx': self.arg2_idx, \
                'echains': self.event_chains}

        with open(fname, 'wb') as fid:
            dump(params, fid)

    def make_path_list(self):
        # doc = nlp(self.doc)

        for i, sent in enumerate(self.doc):
            if i % 20 == 0:
                logging.warning('processing sentence %d' % i)

            sent = nlp(sent)
            # self.path_list.append(self.build_path(sent))
            self.build_path(sent)

    def build_path(self, sent):
        # PARAMS: sent -- spacy sentence wrapper
        # just for debugging adn bookkeeping
        if self.debug:
            verb_dict = None

        # if sent[0].text == '"' and sent[-1].text == '"':
        quotation = re.search(r'^\"([^\"]*)\"$', sent.text)
        if quotation:
            if self.debug:
                # for debugging
                print('content:', quotation.group(1))
            word_id_ls = []

            for w in sent[1:-1]:
                w = w.text
                if w.isdigit():
                    w = "num@#!123"
                word_id_ls.append(self.word_dict[w.lower()])
            self.arg1v = Vertex(word_id_ls, 'ARG1')

            self.arg2v = None

            self.arg1v.set_range((self.offset+1, self.offset+len(sent)-2))   # inclusive bounds

            if self.debug:
                # for debugging
                self.arg1v.words = sent[1:-1]

            # record the idx of this arg1
            self.arg1_idx.append(len(self.vert_ls))
            # update the id of arg1, arg2
            # use the same v_id and arg0_id as the last sentence

            # update the maximum arg length
            # self.max_length = max(self.max_length, len(sent) - 2)
            # add to the vertex list
            self.vert_ls.append(self.arg1v)

            self.formulate_path()

            # increase offset
            self.offset += len(sent)

        else:
            for s in sent.sents:
                if self.debug:
                    print(s)
                self.process_sent(s)
                    

    def formulate_path(self):
        res = []
        res.append(self.v_id)
        res.append(self.arg0_idx[-1] if self.arg0v else -1)
        res.append(self.arg1_idx[-1] if self.arg1v else -1)
        res.append(self.arg2_idx[-1] if self.arg2v else -1)

        assert len(res) > 0
        if self.arg0v:
            if self.debug:
                print('in arg0', res)
            self.arg0v.idx_path = res
        if self.arg1v:
            if self.debug:
                print('in arg1', res)
            self.arg1v.idx_path = res
        if self.arg2v:
            if self.debug:
                print('in arg2', res)
            self.arg2v.idx_path = res

        if self.debug:
            # for debugging
            print(res)


    def process_sent(self, sent):
        if self.debug:
            print(self.offset)
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
            verb_dict = partitions[min_id]

        if self.debug and verb_dict:
            # for debugging
            print(verb_dict['description'])
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
            # increase offset
            self.offset += len(sent)
            return

        # make vertex for each role in the sentence
        word_id_ls = []
        tags = verb_dict['tags']
        tags.append('END')  # special token to denote the end of tags
        for i, t in enumerate(tags):
            if self.debug:
                print(end, begin)
            if not t.startswith('I-'):
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
                            print(node.debug())

                        # update max length
                        # self.max_length = max(self.max_length, end - begin + 1)

                        # if node.type != 'V':
                        # add to the vertex list
                        self.vert_ls.append(node)

                    # reset word_ls
                    word_id_ls = []

                if t == 'O' or i == len(tags) - 1:
                    continue
                if t.startswith('B-'):
                    last = t.lstrip('B-')
                    if self.debug:
                        print('found head of', last)

                    if last not in ['V','ARG0','ARG1','ARG2']:
                        continue
                    begin = i

            if begin > end:        
                if words[i].isdigit():
                    w = "num@#!123"
                else:
                    w = words[i].lower()
                word_id_ls.append(self.word_dict[w])

        if self.debug:
            print(self.arg1v)

        self.formulate_path()

        # increase offset
        self.offset += len(sent)


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

        self.corefs = results_coref['clusters']
        # return results_coref["clusters"]

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

        def convert_range(v, r):
            # print(v.r_map, r)
            start, end = -1, -1

            for i, m in enumerate(v.r_map):
                if m == r[0] and start < 0:
                    start = i
                if m > r[0] and start < 0:
                    # already passed the value, use the last recorded
                    j = i - 1
                    while j >= 0 and v.r_map[j] == v.r_map[i-1]:
                        j -=1
                    start = j + 1

                if m <= r[1]:
                    end = i

            if start < 0:
                j = len(v.r_map) - 1
                while j >= 0 and v.r_map[j] == v.r_map[-1]:
                    j -=1
                start = j + 1

            return (start, end)

        def sub_ent(vertex, head, r1, r2):

            # subtitute the word idx in r1 in vertex with the word idx in r2 in head
            # in case r2 includes r1, don't substitute
            if r2[0] <= r1[0] and r2[1] >= r1[1]:
                return
            if self.debug:
                print(r1, r2)
            r1 = convert_range(vertex, r1)
            # if r1[0] == len(vertex.r_map) -1 or r1[1] < 0:
            if r1[1] < 0:
                # the string has already been altered, abort
                return
            r2 = convert_range(head, r2)
            # if r1[0] ==len(head.r_map) - 1 or r1[1] < 0:
            if r2[1] < 0:
                # the string has already been altered, abort
                return

            if self.debug:
                # for debugging
                print('******')
                print('substituting', r1, 'in', vertex.debug())
                print('with', r2, 'in', head.debug())

            if self.debug:
                # for debugging
                print('original v', vertex.val)

            # modified aug 16th, use the original val of the head
            # so that the range always stays consistent
            ''' old crap
            h_idx = head.val[r2[0]-h_offset:r2[1]-h_offset+1]   # this is the source indices
            '''
            h_idx = head.val[r2[0]:r2[1]+1]   # this is the source indices
            if self.debug:
                # for debugging
                print('head', h_idx)

            ''' old crap
            ori_val = vertex.ori_val

            vertex.val = ori_val[:r1[0]-v_offset] + h_idx    # this is the target
            if r1[1] < vertex.range[1]:
                vertex.val += ori_val[r1[1]-v_offset+1:]
            '''

            ori_val = vertex.val

            vertex.val = ori_val[:r1[0]] + h_idx    # this is the target
            if r1[1] < len(vertex.r_map) - 1:
                vertex.val += ori_val[r1[1]+1:]

            ori_map = vertex.r_map
            vertex.r_map = ori_map[:r1[0]]
            vertex.r_map += [ori_map[r1[0]] for _ in range(r2[0], r2[1]+1)]
            vertex.r_map += ori_map[r1[1]+1:]

            if self.debug:
                # for debugging
                print('final v', vertex.val)
                print('******')


        def get_wd_idx(vertex, r):
            # v_offset = vertex.range[0]
            '''
            start, end = -1, -1
            for i, pos in enumerate(vertex.r_map):
                if pos >= r[0] and start == -1:
                    start = i
                if pos <= r[1]:
                    end = i
            '''
            start, end = convert_range(vertex, r)
            try:
                # assert start != len(vertex.r_map)-1 and end != -1
                assert end != -1
            except Exception:
                print('start %d, end %d' % (start, end))
                print(r)
                print(vertex)
                print(vertex.r_map)
                raise Exception

            return tuple(vertex.val[start:end+1])

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
                logging.warning('processing cluster %d' %i)

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
                    else:
                        # then we do not include it
                        '''
                        logging.warning(vertex.type)
                        raise Exception
                        '''
                        continue

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
                        if self.debug:
                            print('idx:',head_wd_idx)
                    else:
                        sub_ent(vertex, head, single_range, head_range) 

                    # create a deep copy
                    event = copy.deepcopy(vertex.idx_path)
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


def last_step(args, doc, i):
    logging.warning('*************************')
    logging.warning('processing %i th document' % i)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.warning("Memory usage is: {0} MB".format(mem/1000))

    ecb = EventChainBuilder()

    root = args.save_dir
    filename = 'K170_{}'.format(i)

    ecb.doc = doc
    # gc.collect()

    vert_file = os.path.join(root, \
    '%s_vert.pkl' % (filename))
    coref_file = os.path.join(root, \
    '%s_coref.pkl' % (filename))
    params_file = os.path.join(root, \
    '%s_params.pkl' % (filename))
    '''
    vert_file = os.path.join(root, \
    '%d_vert_CNN_2.pkl' % (i))
    coref_file = os.path.join(root, \
    '%d_coref_CNN_2.pkl' % (i))
    params_file = os.path.join(root, \
    '%d_params_CNN_2.pkl' % (i))
    '''

    logging.warning('loading path lists')
    ecb.load_verts(vert_file)

    logging.warning('loading corefs')
    ecb.load_corefs(coref_file)

    logging.warning('building event chains')
    ecb.make_event_chains()
    if not ecb.debug:
        ecb.store_params(params_file)

def load_coref_predictor(gpu_id):
    coref_archive = load_archive('/homes/du113/scratch/pretrained/coref-model-2018.02.05.tar.gz', cuda_device=gpu_id)
    coref_predictor = Predictor.from_archive(coref_archive)
    return coref_predictor

def load_srl_predictor(gpu_id):
    srl_archive = load_archive('/homes/du113/scratch/pretrained/srl-model-2018.05.25.tar.gz', cuda_device=gpu_id)
    srl_predictor = Predictor.from_archive(srl_archive)
    return srl_predictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dict',type=str,default=None)
    parser.add_argument('--doc', type=str, default=None)
    parser.add_argument('--tdir', type=str, default=None)
    parser.add_argument('--fdir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    logging.warning(args)

    docs = []
    if args.doc:
        logging.warning('using docs')
        filename = args.doc
        docs = du.load_sent(filename)
    if args.tdir:
        logging.warning('using t dir')
        dir_ = args.tdir
        filenames = glob(os.path.join(dir_, 'true_*.txt'))
        for filename in filenames:
            docs += du.load_sent(filename)

    if args.fdir:
        logging.warning('using f dir')
        dir_ = args.fdir
        # filenames = glob(os.path.join(dir_, '.txt'))
        filenames = ['train.txt', 'test.txt', 'dev.txt']
        for filename in filenames:
            docs += du.load_sent(os.path.join(dir_, filename))

    '''
    with open(args.doc) as fid:
        docs = fid.readlines() 
    '''
    # print(docs)
    logging.warning('loaded %d documents' % len(docs))

    assert args.word_dict
    with open(args.word_dict, 'rb') as fid:
        word_dict = load(fid)

    gpus = GPUtil.getAvailable(order = 'memory', limit = 2, maxLoad = 1.0, maxMemory = 1.0)

    logging.warning("using gpu %d and %d" % (gpus[0], gpus[1]))
    
    srl_predictor = load_srl_predictor(int(gpus[1]))
    coref_predictor = load_coref_predictor(int(gpus[0]))

    ecb = EventChainBuilder(word_dict, srl=srl_predictor, coref=coref_predictor)
    # ecb = EventChainBuilder()
    ecb.debug = args.debug

    root = args.save_dir
    # filename = filename.split('/')[-1].split('.')[-2][:5]

    import time
    start = time.time()

    for i, doc in enumerate(docs):
        logging.warning('*************************')
        logging.warning('processing %i th document' % i)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.warning("Memory usage is: {0} MB".format(mem/1000))
        logging.warning("GPU usage:")
        GPUtil.showUtilization()
        '''
        if i == 179:
            # print(doc, file=open('179.txt', 'w'))
            continue # skip the longest articel
        '''

        ecb.doc = doc
        filename = 'K170_{}'.format(i)

        vert_filename = os.path.join(root, \
                '%s_vert.pkl' % (filename))
        coref_filename = os.path.join(root, \
                '%s_coref.pkl' % (filename))

        if os.path.exists(vert_filename):
            pass
        else:
            logging.warning('making path lists')
            ecb.make_path_list()
            ecb.store_verts(vert_filename)

        if os.path.exists(coref_filename):
            pass
        else:
            ecb.get_coref() 
            ecb.store_corefs(coref_filename)

        # logging.warning('building event chains')
        # ecb.make_event_chains()
        # reset ecb
        ecb.reset()
        gc.collect()

    pool = ThreadPool()
    pool.starmap(last_step, [(args, doc, i) for i , doc in enumerate(docs)])
    pool.close()
    pool.join()

    print('time used: %d' % (time.time() -start))

