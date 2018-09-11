from event_chain import *
import data_utilities as du
# import utilities as util
from pprint import pprint

import time

def test7():
    import os, pickle
    basedir = '/homes/du113/scratch/cnn-political-data'
    filename = 'CNN_2018_8_16.txt'
    # filename = 'test_coref2.txt'
    filename = os.path.join(basedir, filename)

    word_dict = 'cnn_dict.pkl'
    word_dict = os.path.join(basedir, word_dict)

    docs = du.load_sent(filename)

    docs = docs[:1]
    '''
    with open(args.doc) as fid:
        docs = fid.readlines() 
    '''
    # print(docs)
    logging.warning('loaded %d documents' % len(docs))

    assert word_dict
    with open(word_dict, 'rb') as fid:
        wd = load(fid)
    
    reverse_dict = {v:k for k, v in wd.items()}

    def convert(ls):
        return ' '.join([reverse_dict[i] for i in ls])
    # root = args.save_dir
    filename = filename.split('/')[-1].split('.')[-2][:5]
    # save_path = os.path.join(basedir, "test7_2.pkl")

    for i, doc in enumerate(docs):
        logging.warning('*************************')
        logging.warning('processing %i th document' % i)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.warning("Memory usage is: {0} MB".format(mem/1000))

        ecb = EventChainBuilder(wd, doc, False, False)
        # gc.collect()

        ecb.debug = True

        '''
        logging.warning('making path lists')
        ecb.make_path_list()

        # pprint(ecb.vert_ls)
        for i, v in enumerate(ecb.vert_ls):
            print(i, v)
        if not ecb.debug:
            """
            ecb.store_verts(os.path.join(root, \
                    '%d_vert_%s.pkl' % (i, filename)))
            """
            ecb.store_verts(save_path)
        '''
        ecb.load_verts(os.path.join(basedir, \
                '%d_vert_%s.pkl' % (i, filename)))
        
        logging.warning('*************************')
        logging.warning('building coreference clusters')
        '''
        ecb.get_coref()
        logging.warning('found %d clusters' % len(ecb.corefs))
        print(ecb.corefs)
        if not ecb.debug:
            ecb.store_corefs(os.path.join(root, \
                    '%d_coref_%s.pkl' % (i, filename)))
        '''
        ecb.load_corefs(os.path.join(basedir, \
                '%d_coref_%s.pkl' % (i, filename)))

        logging.warning('building event chains')
        ecb.make_event_chains()

        '''
        if not ecb.debug:
            ecb.store_params(os.path.join(root, \
                    '%d_params_%s.pkl' % (i, filename)))
        '''

        from colors import Color
        for i, chain in enumerate(ecb.event_chains):
            print(Color.BOLD + ('ARG%d' % i) + Color.END)
            # print('ARG%d' % i)
            for k, v in chain.items():
                print('key:', k, convert(k), '\nchains:', v, '\n******')


if __name__ == '__main__':
    test7()
    # test_coref()
