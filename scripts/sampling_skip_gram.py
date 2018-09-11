import os
import random
import json
import numpy as np
# import srl_parse as sp
from pickle import load, dump
from graph import *

# make position and negative samples for each event
def make_sample_foreach_event(args, vertices_ls, neg_sample_rate, fname, skip_window):
    # args must include the followings:
    # 1. event_chains: 3d list
    #   for each event chain, it's id is the index in the list
    # 2. a0list: a list of all unique arg0's
    # 3. vlist: a list of all unique verbs
    # 4. a1list: a list of all unique arg1's
    # 5. a2list: a list of all unique arg2's
    # 6. modlist: a list of all unique modifiers

    '''
    event_chains: [[[7, 2, 2, 0]], [[5, 3, 9, 0]], [[8, 4, 6, 0], [8, 4, 6, 0], [8, 4, 10, 0]], [[4, 5, 7, 0]]]
    event_chain: [[7, 2, 2, 0]] or [[8, 4, 6, 0], [8, 4, 6, 0], [8, 4, 10, 0]]
    event: [7, 2, 2, 0]

    '''

    event_chains = args['echains']
    vrange = args['vrange']
    a0range = args['a0range']
    a1range = args['a1range']
    a2range = args['a2range']
    # modlist = args['modlist'] # don't have this for the moment
    
    # samples = []

    f = open(fname, 'a+', buffering=1024*4)   # buffer of 4 kb

    if os.stat(fname).st_size:
        # if the file contains data already
        # delete all the data inside
        f.seek(0)
        f.truncate()

    for ecidx, ec in enumerate(event_chains):
        # ecidx: the id of event chain 
        # ec: single chain
        # print('ec:', ec)
        # assert len(ec) > 0, "Event chain cannot be empty"
        if not ec:
            continue
       
        # prepare dictionary of 
        ec_len = len(ec)
        if ec_len >= 2 * skip_window + 1: # at least 3 here [0,1,2] 2 as center and 0,1 as left window
            # for qe in ec[skip_window : ec_len - skip_window]:
            for qe_idx in range(skip_window, ec_len - skip_window, skip_window):
                print('qe_idx:', qe_idx)
                qe = ec[qe_idx]  # be the center of the window
                qe_v, qe_a0, qe_a1, qe_a2 = qe
                # samples.append([])  # adding the second dimension
                # have more than 1 events in event chain
                # print(ec[qe_idx - skip_window : qe_idx] + ec[qe_idx + 1 : qe_idx + skip_window + 1])
                pos_idx = random.randint(qe_idx - skip_window, qe_idx + skip_window)
                # make a positive example
                pos = ec[pos_idx]
                ctx_v, ctx_a0, ctx_a1, ctx_a2 = pos

                for i in range(neg_sample_rate):
                    neg_v = random_sample(vlist, vertices_ls, exclude=(qe_v, ctx_v))
                    neg_a0 = random_sample(a0list, vertices_ls, exclude=(qe_a0, ctx_a0))
                    neg_a1 = random_sample(a1list, vertices_ls, exclude=(qe_a1, ctx_a1))
                    neg_a2 = random_sample(a2list, vertices_ls, exclude=(qe_a2, ctx_a2))
                    # neg_mod = random_sample(modlist, exclude=(qe_mod, ctx_mod))
                    # print('qe len:', len(qe)) # 4
                    # print('event len:', len(event)) # 4
                    res = qe + pos + [neg_v, neg_a0, neg_a1, neg_a2]
                    # print('res len:', len(res))
                    f.write('_'.join(map(str, res)))   # append a vector of length 15
                    f.write('\n')
        
        elif ec_len > 1: 
            # len(ec) <= 2
            qe_idx = np.random.randint(0, len(ec))
            qe = ec[qe_idx]
            qe_v, qe_a0, qe_a1, qe_a2 = qe

            for pos in ec[:qe_idx] + ec[qe_idx + 1:]:
                # eid: the id of this event inside current event_chain
                # make a positive example
                ctx_v, ctx_a0, ctx_a1, ctx_a2 = pos

                # samples[-1].append([])  # adding the third dimension

                for i in range(neg_sample_rate):
                    neg_v = random_sample(vlist, vertices_ls, exclude=(qe_v, ctx_v))
                    neg_a0 = random_sample(a0list, vertices_ls, exclude=(qe_a0, ctx_a0))
                    neg_a1 = random_sample(a1list, vertices_ls, exclude=(qe_a1, ctx_a1))
                    neg_a2 = random_sample(a2list, vertices_ls, exclude=(qe_a2, ctx_a2))
                    # neg_mod = random_sample(modlist, exclude=(qe_mod, ctx_mod))
                    # print('qe len:', len(qe)) # 4
                    # print('event len:', len(event)) # 4
                    res = qe + pos + [neg_v, neg_a0, neg_a1, neg_a2]
                    # print('res len:', len(res))

                    f.write('_'.join(map(str, res)))   # append a vector of length 15
                    f.write('\n')

    f.close()



def random_sample(idx_range, vertices_ls, exclude=None):
    if not exclude:
        return np.random.randint(0, idx_range)
    else:
        idx1, idx2 = exclude
        # print('idx1, idx2:', idx1, idx2 )
        for i in range(10):
            cand = np.random.randint(0, idx_range)
            cand = vertices_ls[cand].val
            if cand != idx1 and cand != idx2:
                # print('random sample, cand:',cand)
                return cand
    return 0


def get_max_arg_len(ls):
    max_len = 0
    for arg in ls:
        max_len = max(max_len, len(arg.val))
    return max_len


def padding(val, max_len):
    '''
    input: vertex.val
    output: padded val
    '''
    return val + [0]*(max_len-len(val))


if __name__ == '__main__':
    import collections
    # len_ls, echain_ls = sp.build_event_chain_sample()
    # vlist, \
    # a0list, \
    # a1list, \
    # a2list = len_ls
    # with open('../datasets/lenls', 'rb') as fid:
    #     len_ls = load(fid)
    # with open('../datasets/echainls', 'rb') as fid:
    #     echain_ls = load(fid)
    # BASE = '../datasets/dict/'
    # with open(BASE + 'cnn_dict.pkl', 'rb') as fid:
    #     word_dict = load(fid)
    # # print(word_dict.values())
    # reverse_dict = {i:w for w,i in word_dict.items()}

    # basedir = '/homes/du113/scratch/cnn-political-data'
    basedir = '/homes/du113/scratch/cnn-300-data'

    # with open('../datasets/all_final_params_CNN.pkl', 'rb') as fid:
    with open(os.path.join(basedir, 'aug_27_params_CNN.pkl'), 'rb') as fid:
        final_params_CNN = load(fid)
    # print(final_params_CNN)

    vertices_ls = final_params_CNN['vertices']

    max_len = get_max_arg_len(vertices_ls)
    print('max_len:', max_len)
    for vertex in vertices_ls:
        vertex.val = padding(vertex.val, max_len)

    
    echains = final_params_CNN['echains']


    echain1 = echains[0].values()
    echain2 = echains[1].values()
    echain3 = echains[2].values()
    echains_v = list(echain1) + list(echain2) + list(echain3)

    # for ec in echains_v:
    #     for event in ec:
    #         if len(event) == 0:
    #             print('debug')
    #             # break

    print('echain_v:', echains_v)

    echain_ls = []
    for echain in echains_v:
        tmp1 = []
        for event in echain:
            tmp2 = []
            for v in event:
                if v == -1:
                    padded_val = padding([], max_len)
                else:
                    padded_val = vertices_ls[v].val
                tmp2.append(padded_val)
            if len(tmp2) != 0:
                tmp1.append(tmp2)
        echain_ls.append(tmp1)

    # print('echain_ls:', echain_ls)

    vidx = final_params_CNN['vidx']
    arg0idx = final_params_CNN['arg0idx']
    arg1idx = final_params_CNN['arg1idx']
    arg2idx = final_params_CNN['arg2idx']

    vlist = len(vidx)
    a0list = len(arg0idx)
    a1list = len(arg1idx)
    a2list = len(arg2idx)

    # echain_ls = echains
    # print('echain_ls:', echain_ls)
    # vlist, \
    # a0list, \
    # a1list, \
    # a2list = len_ls

    args = collections.OrderedDict()
    args['echains'] = echain_ls
    args['vrange'] = vlist
    args['a0range'] = a0list
    args['a1range'] = a1list
    args['a2range'] = a2list
    # filename = '/Users/feiyifan/Desktop/NLP/FEEL/nlpstuff/datasets/train/samples.csv' 
    filename = os.path.join(basedir, 'aug_27_samples.csv')

    make_sample_foreach_event(args, vertices_ls, neg_sample_rate=5, fname = filename, skip_window = 2)


