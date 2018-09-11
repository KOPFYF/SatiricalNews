import os
import time 
import re
import spacy
nlp = spacy.load('en_core_web_sm')

from graph import *
from allennlp.predictors import Predictor
import data_utilities as du
import arg_parser as ap

import collections
import argparse
import utilities as util
import pdb
from pickle import load, dump


if os.path.exists('../pretrained'):
    print('models already downloaded')
    srl_predictor = Predictor.from_path('../pretrained/srl-model-2018.05.25.tar.gz')
    coref_predictor = Predictor.from_path('../pretrained/coref-model-2018.02.05.tar.gz')
else:
    print('downloading models...')
    srl_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    coref_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")

# load data

# args = ap.get_args()
# docs = []
# docs += du.load_sent('../datasets/bbcnews.txt')
# word_dict = util.build_dict(docs)
# inv_dict = util.build_inv_dict(word_dict)
# # print('word_dict:', word_dict)
# # print('inv_dict:', inv_dict)
# word_embed = util.words2embedding(word_dict, 100, args.embedding_file) 
# doc = ' '.join(docs)

def load_data(args):
    global word_dict, word_embed
    docs = []
    docs += du.load_sent('../datasets/bbcnews.txt') # BBC_news
    # docs += du.load_sent('../datasets/BBC_news.txt')
    word_dict = util.build_dict(docs)
    # inv_dict = util.build_inv_dict(word_dict)
    word_embed = util.words2embedding(word_dict, 100, args.embedding_file) 
    print('word_dict:', word_dict)
    with open('../datasets/word_dict', 'wb') as fid:
        dump(word_dict, fid)
    doc = ' '.join(docs)
    return doc



def get_srl(sent):
    # use pretrained AllenNLP model
    # results_srl = srl_predictor.predict(sentence=str(sent))
    # for verb_dict in results_srl["verbs"]:
    #     if 'ARG' in verb_dict['description']:
    #         return verb_dict

    res = srl_predictor.predict(sentence=str(sent))
    print(res)
    min_O = float('inf')
    min_id = 0

    if len(res["verbs"]) == 0:
        print('No verbs, need debug')
        return False

    # if len(res["verbs"]) == 1:
    #     verb_dict = res["verbs"][0]
    #     if 'ARG' in verb_dict['description']:
    #         return verb_dict

    for idx, verb_dict in enumerate(res["verbs"]):
        # print(verb_dict["tags"])
        num_O = collections.Counter(verb_dict["tags"])['O']
        if num_O < min_O:
            min_id = idx
            # print(res["verbs"][min_id])
        min_O = min(min_O,num_O)
        # print('min_O:',min_O, 'min_id:',min_id)
    verb_dict = res["verbs"][min_id]
    print(verb_dict['description'])
    if 'ARG' in verb_dict['description']:
        return verb_dict
    else:
        print('dead loop')

# Deal with sent with no verb.
# TODO: we might need a hashtable to store recurring verbs so that the space don't blow up.
# define global variable predv, arg0v, arg1v, arg2v
# global predv, arg0v, arg1v, arg2v = None, None, None, None
predv, arg0v, arg1v, arg2v = None, None, None, None

def build_path(sent):
    global predv, arg0v, arg1v, arg2v
    quotation = re.search(r"^(\")(.*)(\")$", str(sent))
    if quotation:
        # use the arg0v and predv in the last sentence
        arg1v = Vertex(quotation.group(1), 'ARG1')
        arg2v = None
        # print(predv, arg0v)
        
    else:
        srl = get_srl(sent)
        if srl: 
            # for debugging
            # print('description:',srl['description'])

            if re.search(r"(\[ARG0: )(.*?)(\])",srl['description']):
                arg0 = re.search(r"(\[ARG0: )(.*?)(\])",srl['description'])[2]
            else:
                arg0 = None

            if re.search(r"(\[ARG1: )(.*?)(\])",srl['description']):
                arg1 = re.search(r"(\[ARG1: )(.*?)(\])",srl['description'])[2]
            else:
                arg1 = None

            if re.search(r"(\[ARG2: )(.*?)(\])",srl['description']):
                arg2 = re.search(r"(\[ARG2: )(.*?)(\])",srl['description'])[2]
            else:
                arg2 = None


            # print('arg0:',arg0)
            # print('arg1:',arg1) 
            # mod: ARGM-DIS, ARGM-TMP, ARGM-ADV
            if re.search(r"(\[ARGM-.*?: )(.*?)(\])",srl['description']):
                mod = re.search(r"(\[ARGM-.*?: )(.*?)(\])",srl['description'])[2]
                # for debugging
                print('mod',mod)
            else:
                # print('debug:',srl['description'])
                mod = None
            verb = srl['verb']
            
            # convert verb to lemma_
            # verb = nlp(verb)[0].lemma_
            # for debugging
            # print('verb:',verb)
            
            # create 4 vertices
            predv = Vertex(verb,'PRED')
            util.vertex2wordidx(predv, word_dict, True)

            # if no arg0, then will not create vertex arg0v and edge arg0e
            if arg0:
                arg0v = Vertex(arg0, 'ARG0')
                # util.vertex2wordidx(arg0v, word_dict)
                # print('-------arg0v.word_idx:', arg0v.word_idx)

            else:
                arg0v = None

            if arg1:
                arg1v = Vertex(arg1, 'ARG1')
                # util.vertex2wordidx(arg1v, word_dict)
            else:
                arg1v = None

            if arg2:
                arg2v = Vertex(arg2, 'ARG2')
                # util.vertex2wordidx(arg2v, word_dict)
            else:
                arg2v = None
    # =======
            
    #         verb = srl['verb']
            
    #         # convert verb to lemma_
    #         # verb = nlp(verb)[0].lemma_
    #         # for debugging
    #         # print('verb:',verb)
            
    #         # create 4 vertices
    #         predv = Vertex(verb,'PRED')
    #         # if no arg0, then will not create vertex arg0v and edge arg0e
    #         arg0v = Vertex(arg0, 'ARG0') if arg0 else None
            
    #         arg1v = Vertex(arg1, 'ARG1') if arg1 else None
            
    #         arg2v = Vertex(arg2, 'ARG2') if arg2 else None

        # if mod:
        #     modv = Vertex(mod, 'MOD')

        else:
<<<<<<< HEAD
            # No verb
            predv, arg0v, arg1v, arg2v = None, None, None, None
=======
            arg1v = None

        if arg2:
            arg2v = Vertex(arg2, 'ARG2')
            # util.vertex2wordidx(arg2v, word_dict)
        else:
            arg2v = None

    # if mod:
    #     modv = Vertex(mod, 'MOD')
>>>>>>> master

    res = [predv]
    res.append(arg0v)
    res.append(arg1v)
    res.append(arg2v)

    if arg0v:
        arg0v.path = res
        # arg0v.arg_range1by1 = list(range(arg0v.arg_range[0],arg0v.arg_range[1]+1))
        util.vertex2wordidx(arg0v, word_dict, True) 
    if arg1v:
        arg1v.path = res
        # arg1v.arg_range1by1 = list(range(arg1v.arg_range[0],arg1v.arg_range[1]+1))
        util.vertex2wordidx(arg1v, word_dict, True) 
    if arg2v:
        arg2v.path = res
        util.vertex2wordidx(arg2v, word_dict, True) 
        # arg2v.arg_range1by1 = list(range(arg2v.arg_range[0],arg2v.arg_range[1]+1))
    # print('*********** in build path')
    # print(res)

    return res


def cluster_index2name(ls_2d,results_coref):
    '''
    ls_2d: [[34, 41], [95, 98]] 
    return cluster_name: ["North Korea 's leader Kim Jong - un", 'Kim Jong - un'] 

    '''
    word_ls = results_coref['document']
    clusters = results_coref['clusters']
    cluster_name = []
    for inter in ls_2d:
        inter_name = " ".join(word_ls[inter[0]:inter[1]+1])
        cluster_name.append(inter_name)
    return cluster_name


def args2index(arg,sent):
    # return [sent.find(arg),sent.find(arg)+len(arg)] sentence index
    # also return if arg == None
    # input: vertex, sent string
    v = arg
    if arg:
        arg = arg.val.split()
        sent = [token.text for token in sent]
        # print('in args2index')
        # print('arg:',arg,'\n')
        # print('sent:',sent,'\n')

        for i in range(len(sent)-len(arg)+1):
            quotation = False
            if sent[i] == arg[0] and sent[i+len(arg)-1] == arg[len(arg)-1]:
                # arg found! : [3, 3] ['carried']
                # print('arg found! :',[i,i+len(arg)-1],sent[i:i+len(arg)],'\n')
                # return [i,i+len(arg)-1], arg == None
                v.arg_range = [i,i+len(arg)-1]

                return True
        else:
            # quotation = True
            return False
                
    else:
        # when arg == None
        return False


def build_list(doc): 
    doc = nlp(doc)
    # pred_ls = []
    # arg0_ls = [] 
    # arg1_ls = []
    # arg2_ls = []
    pred_ls, arg0_ls, arg1_ls, arg2_ls = set(),set(),set(),set()

    for sent in doc.sents:
        doc_path = build_path(sent)
        # build pred vertex list
        predv = doc_path[0]
        # if predv.val not in pred_ls:
        #     pred_ls.append(predv)
        pred_ls.add(predv)
    
        for v in doc_path[1:]:
            if v:
                # check None vertex
                if v.type == 'ARG0':
                    arg0_ls.add(v)
                    # if v not in arg0_ls:
                    #     arg0_ls.append(v)
                elif v.type == 'ARG1':
                    arg1_ls.add(v)
                    # if v not in arg1_ls:
                    #     arg1_ls.append(v)
                elif v.type == 'ARG2':
                    arg2_ls.add(v)
                    # if v not in arg2_ls:
                    #     arg2_ls.append(v)
                else:
                    print('need debug')
    # print(pred_ls)
    # return list(pred_ls), list(arg0_ls), list(arg1_ls), list(arg2_ls) 
    return list(pred_ls), arg0_ls, arg1_ls, arg2_ls 



def hash_vertex(doc):
    '''
    return mappings from vertices to their index range 
    '''
    doc = nlp(doc)
    vertice_map = []
    pos = 0
    for sent in doc.sents:
        # for each sentence, get the 3 vertices
        doc_path = build_path(sent)
        # modified Aug 2
        # add the predicate vertex to the graph
        # graph.add_vertex(doc_path[0])
        for v in doc_path[1:]:
            if args2index(v,sent):
                v.arg_range = [v.arg_range[0]+pos,v.arg_range[1]+pos]
                vertice_map.append((v, v.arg_range))
                # vertice_map.append((v, [v.arg_range[0]+pos,v.arg_range[1]+pos]))
        pos = pos + len(sent)

    return vertice_map


def get_coref(doc):
    '''
    input: the document
    return: clusters_names: ["North Korea 's leader Kim Jong - un", 'Kim Jong - un'] 
    clusters_indexs: [[34, 41], [95, 98]] 

    '''
    results_coref = coref_predictor.predict(document=doc)
    # for debugging
    # print(results_coref)

    clusters_names = []
    clusters_indexs = []
    for index, cluster in enumerate(results_coref["clusters"]):
        name = cluster_index2name(cluster,results_coref)
        clusters_names.append(name)
        clusters_indexs.append(cluster)
        ## for debugging
        print(index, cluster,'\n')
        print(index, name,'\n')

    # clusters_ = zip(clusters_names,clusters_indexs)
    # return clusters_

    return clusters_names, clusters_indexs


def build_unique_list(doc, vert_ls, clusters_names, clusters_indexs):
    '''
    build unique list of arg0, arg1 and arg2
    return 2d list, while rows with different length
    for ex, if some arg0s in one group, they are added into the same row 

    '''
    # global vert_ls
    # global clusters_
    # vert_ls = hash_vertex(doc)
    pred_ls, arg0_ls, arg1_ls, arg2_ls = build_list(doc)
    # clusters_ = get_coref(doc)
    # clusters_names, clusters_indexs = get_coref(doc)
    arg0_2d_ls = []
    arg1_2d_ls = []
    arg2_2d_ls = []
    # for clusters_name,clusters_index in clusters_:
    for clusters_name,clusters_index in zip(clusters_names, clusters_indexs):
        group0_ls = []
        group1_ls = []
        group2_ls = []
        for single_range in clusters_index:         
            vertex = bin_search(single_range, vert_ls)
            if vertex:
                if vertex.type == 'ARG0':
                    group0_ls += [vertex]
                    # arg0_ls.remove(vertex)
                    arg0_ls.discard(vertex)
                elif vertex.type == 'ARG1':
                    group1_ls += [vertex]
                    # arg1_ls.remove(vertex)
                    arg1_ls.discard(vertex)
                elif vertex.type == 'ARG2':
                    group2_ls += [vertex]
                    # arg2_ls.remove(vertex)
                    arg2_ls.discard(vertex)
                else:
                    print('need debug')
                    break
        if group0_ls: 
            arg0_2d_ls.append(group0_ls)
        if group1_ls:
            arg1_2d_ls.append(group1_ls)
        if group2_ls:
            arg2_2d_ls.append(group2_ls)

    # add non-coref
    # The bug is that I only add coref-args into the unique list, but not the non-coref part like Vertex('Pyongyang')
    # The bug comes again with []. Need to cope with non-coref part
    arg0_ls, arg1_ls, arg2_ls = list(arg0_ls), list(arg1_ls), list(arg2_ls)
    for noncoref in arg0_ls:
        # print(noncoref,'debug')
        # util.vertex2wordidx(noncoref, word_dict, True) 
        # print('noncoref 0:',noncoref.word_idx)      
        arg0_2d_ls.append([noncoref])
    for noncoref in arg1_ls:
        # util.vertex2wordidx(noncoref, word_dict, True) 
        arg1_2d_ls.append([noncoref])
    for noncoref in arg2_ls:
        # util.vertex2wordidx(noncoref, word_dict, True) 
        arg2_2d_ls.append([noncoref])


    print('Unique_ls_arg0:','\n', arg0_2d_ls)
    print('Unique_ls_arg1:','\n', arg1_2d_ls)
    print('Unique_ls_arg2:','\n', arg2_2d_ls)
    print('Unique_ls_pred:','\n', pred_ls)
    return pred_ls, arg0_2d_ls, arg1_2d_ls, arg2_2d_ls


def args2dls2dic(ls_2d):
    '''
    input arg0_2d_ls
    return hash dict
    '''
    d = {}
    for idx, group in enumerate(ls_2d):
        for item in group:
            d[item] = idx
    return d


def build_args_dicts(doc, vert_ls, clusters_names, clusters_indexs):
    pred_ls, arg0_2d_ls, arg1_2d_ls, arg2_2d_ls = build_unique_list(doc, vert_ls, clusters_names, clusters_indexs)
    arg0_dict = args2dls2dic(arg0_2d_ls)
    arg1_dict = args2dls2dic(arg1_2d_ls)
    arg2_dict = args2dls2dic(arg2_2d_ls)
    return arg0_dict, arg1_dict, arg2_dict, pred_ls, arg0_2d_ls, arg1_2d_ls, arg2_2d_ls


def vertex2idx(vertex, pred_ls, arg0_dict, arg1_dict, arg2_dict):
    '''
    input vertex, return its idx

    idx 0 for None
    idx 1 for ...? Maybe unknown

    '''
    if vertex.type == 'PRED':
        return pred_ls.index(vertex) + 2
    elif vertex.type == 'ARG0':
        return arg0_dict[vertex] + 2   
    elif vertex.type == 'ARG1':
        return arg1_dict[vertex] + 2        
    elif vertex.type == 'ARG2':
        return arg2_dict[vertex] + 2       
    else:
        print('need debug')


def path2idx(path, pred_ls, arg0_dict, arg1_dict, arg2_dict):
    '''
    input vertex.path(may contain None)
    return vertex.idx_path
    '''
    idx_path = []
    for v in path:
        if v:
            idx_path.append(vertex2idx(v, pred_ls, arg0_dict, arg1_dict, arg2_dict))
        else:
            # idx 0 for None
            idx_path.append(0)
    return idx_path


def collapse(fix_range, arg_range, coref_range):
    '''
    coref range: [40,41]
    fix range: [9,10]
    arg range: [38,39,40,41,42]
    output: [38, 39, 9, 10, 42]


    fix_range: [8, 16]
    arg_range: [57]
    coref_range: [57, 57]
    output: [8, 16]

    '''
    # print('in collapse...')
    # print('fix_range:', fix_range)
    # print('arg_range:', arg_range)
    # print('coref_range:', coref_range)
    left, right = 0, 0
    for idx, n in enumerate(arg_range):
        if n == coref_range[0]:
            left = idx
            break

    for idx, n in enumerate(arg_range[::-1]):
        if n == coref_range[1]:
            right = len(arg_range) - idx
            break

    # print('left:', arg_range[:left], type(arg_range[:left]))
    # print('right:', arg_range[right:])
    # print('mid:', list(range(fix_range[0],fix_range[1]+1)))
    return arg_range[:left] + list(range(fix_range[0],fix_range[-1]+1)) + arg_range[right:]



def collapse_coref(doc, clusters_names, clusters_indexs, vert_ls):
    '''
    update self.arg_range by coref

    '''
    # vert_ls = hash_vertex(doc)
    # clusters_ = get_coref(doc)

    doc = nlp(doc)
    for clusters_name,clusters_index in zip(clusters_names, clusters_indexs):
        if len(clusters_name) > 1:
            fix_range = clusters_index[0]
            # print('fix_range:', fix_range)
            for item, coref_range in zip(clusters_name[1:], clusters_index[1:]):
                # print(item, coref_range)
                vertex = bin_search(coref_range, vert_ls)
                if vertex:
                    # update vertex.arg_range1by1 
                    print('vertex ',vertex)
                    left, right = vertex.arg_range[0], vertex.arg_range[1]
                    # print(left, right)
                    vertex.arg_range1by1 = collapse(fix_range, list(range(left, right+1)), coref_range)
                    print('vertex.arg_range1by1:', vertex.arg_range1by1)
                    tmp = []
                    for idx in vertex.arg_range1by1:
                        tmp.append(str(doc[idx]))
                    vertex.arg_str = " ".join(tmp)
                    util.vertex2wordidx(vertex, word_dict, False)
                    print('-------vertex.word_idx:', vertex.word_idx)

    print('collapse completed!')




def build_event_chain(doc, pred_ls, arg0_dict, arg1_dict, arg2_dict, vert_ls, clusters_names, clusters_indexs):
    '''
    build event chain based on coref
    return 2d event chain [[vertex.path]] while vertex.path is a event path(1d list)

    '''

    # vert_ls = hash_vertex(doc)
    # clusters_names, clusters_indexs = get_coref(doc)

    # Aug 13th
    collapse_coref(doc, clusters_names, clusters_indexs, vert_ls)
    
    echain_2d_ls_arg0 = []
    echain_2d_ls_arg1 = []
    echain_2d_ls_arg2 = []
    # pdb.set_trace()
    for clusters_name, clusters_index in zip(clusters_names, clusters_indexs):
        echain_ls_arg0 = []
        echain_ls_arg1 = []
        echain_ls_arg2 = []  
        # echain_ls_arg0 = set()
        # echain_ls_arg1 = set()
        # echain_ls_arg2 = set()   
        for single_range in clusters_index:         
            vertex = bin_search(single_range, vert_ls)
            if vertex:
                print('vertex:', vertex.val)
                print('vertex.word_idx:', vertex.word_idx)
                if vertex.type == 'ARG0':
                    # vertex.idx_path = path2idx(vertex.path, pred_ls, arg0_dict, arg1_dict, arg2_dict)
                    word_idx_path = []
                    for arg in vertex.path:
                        if arg: 
                            word_idx_path.append(arg.word_idx)
                        else:
                            word_idx_path.append([0])
                    # echain_ls_arg0.append(vertex.idx_path)

                    # Aug 13
                    echain_ls_arg0.append(word_idx_path)

                elif vertex.type == 'ARG1':
                    # vertex.idx_path = path2idx(vertex.path, pred_ls, arg0_dict, arg1_dict, arg2_dict)
                    word_idx_path = []
                    for arg in vertex.path: 
                        if arg:
                            word_idx_path.append(arg.word_idx)
                        else:
                            word_idx_path.append([0])
                    # echain_ls_arg1.append(vertex.idx_path)

                    # echain_ls_arg1.append(vertex.word_idx)
                    echain_ls_arg1.append(word_idx_path)

                elif vertex.type == 'ARG2':
                    # vertex.idx_path = path2idx(vertex.path, pred_ls, arg0_dict, arg1_dict, arg2_dict)
                    # echain_ls_arg2.append(vertex.idx_path)
                    word_idx_path = []
                    for arg in vertex.path: 
                        if arg:
                            word_idx_path.append(arg.word_idx)
                        else:
                            word_idx_path.append([0])
                    echain_ls_arg2.append(word_idx_path)

                    # echain_ls_arg2.append(vertex.word_idx)

                else:
                    print('need debug')
                    break
        # echain_ls_arg0 = list(echain_ls_arg0)
        # echain_ls_arg1 = list(echain_ls_arg1)
        # echain_ls_arg2 = list(echain_ls_arg2)

        if echain_ls_arg0:
            echain_2d_ls_arg0.append(echain_ls_arg0)
        if echain_ls_arg1:
            echain_2d_ls_arg1.append(echain_ls_arg1)
        if echain_ls_arg2:
            echain_2d_ls_arg2.append(echain_ls_arg2)
    print('echain_2d_ls_arg0:','\n', echain_2d_ls_arg0)
    print('echain_2d_ls_arg1:','\n', echain_2d_ls_arg1)
    print('echain_2d_ls_arg2:','\n', echain_2d_ls_arg2)

    
    return echain_2d_ls_arg0, echain_2d_ls_arg1, echain_2d_ls_arg2



def build_graph(doc):
    # graph = Graph()
    # vert_ls = hash_vertex(doc, graph)
    vert_ls = hash_vertex(doc)
    clusters_names, clusters_indexs = get_coref(doc)

    # transform the clusters from list of list of indices to list of list of names
    coref_2d_ls = []
    for clusters_name,clusters_index in zip(clusters_names, clusters_indexs):
        coref_ls = []
        for single_range in clusters_index:         
            # Binary search

            # add the resulting vertex to the list
            vertex = bin_search(single_range, vert_ls)
            # check : the coreference should be a vertex in the list
            # assert vertex is not None, "vertex not found"
            if vertex:
                coref_ls.append(vertex)

        coref_2d_ls.append(coref_ls)
    # print('coref_2d_ls:','\n', coref_2d_ls)

    # build 'coref' edges
    for coref_ls in coref_2d_ls:
        if len(coref_ls) > 1:
            for v in coref_ls[1:]:
                # coref_e = Edge(coref_ls[0],v, 'COREF')
                # coref_ls[0].add_edge(coref_e)

                # modified Aug 2nd
                # add reverse edge
                rev_coref_e = Edge(v, coref_ls[0], 'COREF')
                v.add_edge(rev_coref_e)

def rangeInRange(small,big):
    '''
    small: coref range
    big:   arg range

    '''
    return small[0]>= big[0] and small[1]<= big[1]


# TODO: test binary search
# TODO: maybe if the coref is in one vertex, we should not just think the entire thing as a coreferenced entity
# for example, "that he is happy" should not have the same embedding as the entity 'he' is referring to
# rather, "he" should have the same embedding
def bin_search(single_range, vert_ls):
    # vert_ls is a sorted list by the range of indices
    begin, end = 0, len(vert_ls) - 1

    # print(single_range)
    while begin <= end:
        # print(begin, end)
        # if begin == 0 and end == 2:
        #     raise Exception

        mid = (begin+end) // 2
        vert_range = vert_ls[mid][1]
        if rangeInRange(single_range, vert_range):
            # we know that the query has to be contained in one of the list element
            return vert_ls[mid][0]

        if single_range[0] > vert_range[1]:
            # the range of the query is bigger
            begin = mid + 1

        elif single_range[1] < vert_range[0]:
            # the pick is bigger, search in lowe half
            end = mid - 1
        else:
            return None

    # if got here, then the vertex is not found
    return None


def build_event_chain_sample(doc):
    vert_ls = hash_vertex(doc)
    clusters_names, clusters_indexs = get_coref(doc)

    # pred_ls, arg0_ls, arg1_ls, arg2_ls = build_list(doc)
    # print('arg0_ls:', arg0_ls)
    # print('arg1_ls:', arg1_ls)
    # print('arg2_ls:', arg2_ls)
    '''
    arg0_ls: [Vertex('Pyongyang'), Vertex('President Trump'), Vertex('US officials'), Vertex('the two leaders'), Vertex('President Donald Trump'), Vertex('North Korea'), Vertex('by US - based monitoring group 38 North')]
    '''
    # pred_ls, arg0_2d_ls, arg1_2d_ls, arg2_2d_ls = build_unique_list(doc)
    # len_ls = [len(pred_ls), len(arg0_2d_ls), len(arg1_2d_ls), len(arg2_2d_ls)]
    # print('pred_ls:', pred_ls, len(pred_ls))
    # print('arg0_2d_ls:', arg0_2d_ls, len(arg0_2d_ls))
    # print('arg1_2d_ls:', arg1_2d_ls, len(arg1_2d_ls))
    # print('arg2_2d_ls:', arg2_2d_ls, len(arg2_2d_ls))
    arg0_dict, arg1_dict, arg2_dict, pred_ls, arg0_2d_ls, arg1_2d_ls, arg2_2d_ls = build_args_dicts(doc, vert_ls, clusters_names, clusters_indexs)
    len_ls = [len(pred_ls), len(arg0_2d_ls), len(arg1_2d_ls), len(arg2_2d_ls)]
    '''
    pred: [Vertex('sign'), Vertex('maintain'), Vertex('suspect'), Vertex('see'), Vertex('criticise'), Vertex('carry'), Vertex('say')]

    arg0_dict: {Vertex('President Donald Trump'): 0, Vertex('President Trump'): 0, Vertex('North Korea'): 1, Vertex('Pyongyang'): 2, Vertex('US officials'): 3, Vertex('by US - based monitoring group 38 North'): 4, Vertex('the two leaders'): 5}
    arg1_dict: {Vertex('that Sohae is a satellite launch site'): 0, Vertex('that it has been used to test ballistic missiles'): 0, Vertex('that he was " very happy " with the progress in relations with North Korea'): 4, Vertex('North Korea 's leader Kim Jong - un had vowed to destroy an engine test site'): 4, Vertex('a deal to work towards the " complete denuclearisation of the Korean Peninsula "'): 3, Vertex('the deal'): 3, Vertex('a total of six nuclear tests , the most recent of which took place in September last year'): 5, Vertex('the Sohae station'): 6}
    arg2_dict: {Vertex('for a lack of details on when or how Pyongyang would renounce its nuclear weapons'): 0}
    '''
    # print('arg0_dict:', arg0_dict)
    # print('arg1_dict:', arg1_dict)
    # print('arg2_dict:', arg2_dict)

    echain_2d_ls_arg0, echain_2d_ls_arg1, echain_2d_ls_arg2 = build_event_chain(doc, pred_ls, arg0_dict, arg1_dict, arg2_dict, vert_ls, clusters_names, clusters_indexs)
    echain_ls = echain_2d_ls_arg0 + echain_2d_ls_arg1 + echain_2d_ls_arg2
    print('****** len_ls:',len_ls)
    print('****** echain_ls:',echain_ls)
    with open('../datasets/lenls', 'wb') as fid:
        dump(len_ls, fid)
    with open('../datasets/echainls', 'wb') as fid:
        dump(echain_ls,fid)

    return len_ls, echain_ls



if __name__ == '__main__':
    args = ap.get_args()

    # Aug 12th modifying, substitution  

    doc = load_data(args)
    build_event_chain_sample(doc)
