import os
import pickle
import glob

from colors import Color

BASE = "/homes/du113/scratch/cnn-300-data"

WORD_DICT = os.path.join(BASE, "cnn_dict.pkl")

with open(WORD_DICT, 'rb') as f:
    wd = pickle.load(f)

reverse_dict = {v:k for k, v in wd.items()}

def convert(ls):
    return ' '.join([reverse_dict[i] for i in ls])


fnames = sorted(glob.glob(os.path.join(BASE, '*_params.pkl')))
# fnames = fnames[:2]

tot_len = 0
verts = []
vs = []
arg0s = []
arg1s = []
arg2s = []

echains = {}

for fname in fnames:
    print(fname)
    with open(fname, 'rb') as fid:
        params = pickle.load(fid)

    vertices = params['vertices']
    '''
    for j, v in enumerate(vertices):
        print(j+tot_len, v)
    '''

    v_idx = params['vidx']
    arg0_idx = params['arg0idx']
    arg1_idx = params['arg1idx']
    arg2_idx = params['arg2idx']

    # print(v_idx)
    for ls in [v_idx, arg0_idx, arg1_idx, arg2_idx]:
        for i, idx in enumerate(ls):
            ls[i] = idx + tot_len

    # print(v_idx)

    chains = params['echains']

    if echains:
        for i, chain in enumerate(chains):
            for k in chain.keys():
                events = chains[i][k]
                for j, event in enumerate(events):
                    # print(Color.RED + ('before %s' % str(event)) + Color.END)
                    for x, e in enumerate(event):
                        if e != -1:
                            chains[i][k][j][x] = e + tot_len
                    # print(Color.RED + ('after %s' % str(chains[i][k][j])) + Color.END)

            # print(Color.BOLD + ('ARG%d' % i) + Color.END)
            for k, v in chain.items():
                # print(convert(k), v, sep='=>')

                if k in echains[i]:
                    echains[i][k] += v
                else:
                    for newk in [k[1:], k[:-1]]:
                        if newk in echains[i]:
                            # in case there is a "'s"
                            echains[i][newk] += v
                            break
                    else:
                        echains[i][k] = v
    else:
        echains = chains

    verts += vertices
    vs += v_idx
    arg0s += arg0_idx
    arg1s += arg1_idx
    arg2s += arg2_idx
    tot_len += len(vertices)

    '''
    print(Color.BOLD + str(tot_len) + Color.END)
    print(Color.BOLD + '************************' + Color.END)
    '''
for i, chain in enumerate(echains):
    print(Color.BOLD + ('ARG%d' % i) + Color.END)
    for k, v in chain.items():
        print(convert(k), v, sep=Color.GREEN + ' => ' + Color.END)
'''
print(len(verts))
'''

new_params = {}
new_params['vertices'] = verts
new_params['vidx'] = vs
new_params['arg0idx'] = arg0s
new_params['arg1idx'] = arg1s
new_params['arg2idx'] = arg2s
new_params['echains'] = echains

with open(os.path.join(BASE, 'aug_27_params_CNN.pkl'), 'wb') as f:
    pickle.dump(new_params, f)
