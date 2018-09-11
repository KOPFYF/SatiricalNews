import os
import pickle
from colors import Color

BASE="/homes/du113/scratch/cnn-political-data"
WORD_DICT = os.path.join(BASE, 'cnn_dict.pkl')

with open(WORD_DICT, 'rb') as f:
    wd = pickle.load(f)

reverse_dict = {v:k for k, v in wd.items()}

def convert(ls):
    return ' '.join([reverse_dict[i] for i in ls])

fname = os.path.join(BASE, '0_vert_CNN_2.pkl')
# fname = os.path.join(BASE, 'bugfree_params_CNN.pkl')
# fname = os.path.join(BASE, '100_params_CNN_2.pkl')
# fname = os.path.join(BASE, '100_vert_CNN_2.pkl')
with open(fname, 'rb') as fid:
    params = pickle.load(fid)

print(len(params['vertices']))
for i, v in enumerate(params['vertices']):
    # if v.range == (135, 198):
    print(i, convert(v.val), v.type, v.range, v.r_map)
    # if v.idx_path[0] == 1300,1299,1301,-1]:
        # print(i, convert(v.val), v.type, v.range, v.r_map)

'''
for i, chain in enumerate(params['echains']):
    # print('ARG%d' % i)
    print(Color.BOLD + ('ARG%d' % i) + Color.END)
    for k, v in chain.items():
        print('key:', k, convert(k))
'''
"""
for i, chain in enumerate(params['echains']):
    print('ARG%d' % i)
    print(len(chain.items()))
"""
