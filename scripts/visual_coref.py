import pickle
from pprint import pprint

fname = '/homes/du113/scratch/cnn-political-data/0_coref_trump_headlines.pkl'
with open(fname, 'rb') as fid:
    corefs = pickle.load(fid)

pprint(corefs)

