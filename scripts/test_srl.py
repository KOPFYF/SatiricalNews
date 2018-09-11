from event_chain import *
import data_utilities as du
# import utilities as util
from pprint import pprint

import time

# from srl_parse import build_path as sbp

def test0():
    docs = ["President Donald Trump said North Korea's leader Kim Jong-un had vowed to destroy an engine test site, but did not specify which one."] 
    wd = util.build_dict(docs)
    pprint(wd)

    ecb = EventChainBuilder(wd)

    # print(docs)
    for i, sent in enumerate(docs):
        print('processing sentence', i)

        res = ecb.build_path(nlp(sent))

        print(res)


def test1():

    docs = du.load_sent('../datasets/bbcnews.txt')

    wd = util.build_dict(docs)
    pprint(wd)
    doc = ' '.join(docs)

    ecb = EventChainBuilder(wd, doc, debug=True)

    # print(docs)
    '''
    for i, sent in enumerate(docs):
        print('processing sentence', i)

        res = ecb.build_path(nlp(sent))

        print(res)
    '''
    pprint(ecb.vert_ls)

    pprint(ecb.corefs)


def test2():

    docs = du.load_sent('../datasets/bbcsample1.txt')

    wd = util.build_dict(docs)
    pprint(wd)

    ecb = EventChainBuilder(wd)

    # print(docs)
    for i, sent in enumerate(docs):
        print('processing sentence', i)

        res = ecb.build_path(nlp(sent))

        print(res)

def test_coref():
    docs = du.load_sent('../datasets/bbcsample1.txt')

    wd = util.build_dict(docs)
    pprint(wd)

    doc = ' '.join(docs)

    # doc = nlp(doc)

    ecb = EventChainBuilder(wd, doc)

    # print(docs)
    res = ecb.get_coref()

    pprint(res)

    res = ecb.old_get_coref()

    pprint(list(res))

def test3():

    # docs = du.load_sent('../datasets/bbcnews.txt')
    docs = du.load_sent('../datasets/bbcsample1.txt')
    # docs = du.load_sent('../datasets/sample1.txt')
    # docs = du.load_sent('../datasets/sample2.txt')

    wd = util.build_dict(docs)
    # pprint(wd)
    # doc = ' '.join(docs)

    ecb = EventChainBuilder(wd, docs, debug=True, use_v=False)

    # pprint(ecb.vert_ls)
    for i, v in enumerate(ecb.vert_ls):
        print(i,'->',v)

    # pprint(ecb.corefs)
    print(ecb.v_idx)
    print(ecb.arg0_idx)
    print(ecb.arg1_idx)
    print(ecb.arg2_idx)

    e0, e1, e2 = ecb.make_event_chains()
    print('arg0 event chains:')
    pprint(e0)
    print('******************')
    print('arg1 event chains:')
    pprint(e1)
    print('******************')
    print('arg2 event chains:')
    pprint(e2)
    print('******************')

def test4():

    docs = du.load_sent('../datasets/bbcsample1.txt')

    wd = util.build_dict(docs)
    # pprint(wd)
    doc = ' '.join(docs)

    # ecb = EventChainBuilder(wd, doc, debug=True, use_v=True)
    ecb = EventChainBuilder(wd, doc, debug=True, use_v=False)

    pprint(ecb.vert_ls)

    pprint(ecb.corefs)

    print(ecb.v_idx)
    print(ecb.arg0_idx)
    print(ecb.arg1_idx)
    print(ecb.arg2_idx)

def test5():

    # docs = du.load_sent('../datasets/bbcnews.txt')
    docs = du.load_sent('../datasets/bbcsample2.txt')
    # docs = du.load_sent('../datasets/sample1.txt')
    # docs = du.load_sent('../datasets/sample2.txt')

    wd = util.build_dict(docs)
    # pprint(wd)
    # doc = ' '.join(docs)

    ecb = EventChainBuilder(wd, docs, debug=True, use_v=False)

    # pprint(ecb.vert_ls)
    for i, v in enumerate(ecb.vert_ls):
        print(i,'->',v)

    print(ecb.v_idx)
    print(ecb.arg0_idx)
    print(ecb.arg1_idx)
    print(ecb.arg2_idx)

    e0, e1, e2 = ecb.make_event_chains()
    print('******************')
    print('arg0 event chains:')
    pprint(e0)
    print('******************')
    print('arg1 event chains:')
    pprint(e1)
    print('******************')
    print('arg2 event chains:')
    pprint(e2)
    print('******************')

def test6():
    import os,pickle
    from allennlp.predictors import Predictor
    from pprint import pprint
    
    path = '/homes/du113/scratch/cnn-political-data'
    filename = os.path.join(path, 'cnn_dict.pkl')
    with open(filename, 'rb') as f:
        word_dict = pickle.load(f)

    srl = Predictor.from_path('/homes/du113/scratch/pretrained/srl-model-2018.05.25.tar.gz')
    coref = Predictor.from_path('/homes/du113/scratch/pretrained/coref-model-2018.02.05.tar.gz')

    ecb = EventChainBuilder(wd=word_dict, srl=srl)
    ecb.debug=True
    # sent = 'Giuliani, in his interview on Tuesday, continued to dispute the validity behind a full interview and said a question about potential obstruction of justice by Trump in private discussions with then-FBI Director James Comey would be a "perjury trap."'
    sent = 'some conservative criticism , however , Kavanaugh also wrote in his dissent that Supreme Court precedent " strongly suggests that the government has a compelling interest in facilitating access to contraception for the employees of these religious organizations . "'

    results_coref = coref.predict(document=sent)
    pprint(results_coref['clusters'])

    sent = nlp(sent)
    ecb.build_path(sent)

def test7():
    import os, pickle
    from pprint import pprint
    basedir = '/homes/du113/scratch/cnn-political-data'
    filename = 'CNN_2018_8_16.txt'
    filename = os.path.join(basedir, filename)

    word_dict = 'cnn_dict.pkl'
    word_dict = os.path.join(basedir, word_dict)

    docs = du.load_sent(filename)[0:1]
    '''
    with open(args.doc) as fid:
        docs = fid.readlines() 
    '''
    # print(docs)
    logging.warning('loaded %d documents' % len(docs))

    assert word_dict
    with open(word_dict, 'rb') as fid:
        wd = load(fid)
    
    # root = args.save_dir
    # filename = filename.split('/')[-1].split('.')[-2][:5]
    save_path = os.path.join(basedir, "test7.pkl")

    for i, doc in enumerate(docs):
        logging.warning('*************************')
        logging.warning('processing %i th document' % i)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.warning("Memory usage is: {0} MB".format(mem/1000))

        ecb = EventChainBuilder(wd, doc, True, False)
        gc.collect()

        ecb.debug = True

        logging.warning('making path lists')
        ecb.make_path_list()

        pprint(ecb.vert_ls)
        '''
        if not ecb.debug:
            """
            ecb.store_verts(os.path.join(root, \
                    '%d_vert_%s.pkl' % (i, filename)))
            """
            ecb.store_verts(save_path)
        '''
        
        '''
        logging.warning('building coreference clusters')
        ecb.get_coref()
        logging.warning('found %d clusters' % len(ecb.corefs))
        if not ecb.debug:
            ecb.store_corefs(os.path.join(root, \
                    '%d_coref_%s.pkl' % (i, filename)))
        '''

        '''
        logging.warning('building event chains')
        ecb.make_event_chains()
        if not ecb.debug:
            ecb.store_params(os.path.join(root, \
                    '%d_params_%s.pkl' % (i, filename)))
        '''



if __name__ == '__main__':
    test6()
    # test7()
    # test_coref()
