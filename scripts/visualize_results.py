import torch
from collections import OrderedDict
from Model_baseline_ugly import FEELModel

basedir = '/homes/du113/scratch/cnn-political-data'

import os
params_file = 'final_params.pth.tar'
params_file = os.path.join(basedir, params_file)

old_state = torch.load(params_file)
old_state = old_state['state_dict']

new_state = OrderedDict()
for k, v in old_state.items():
    name = k[7:]
    new_state[name] = v

embeds = new_state['embeddings.weight']
torch.save(embeds, os.path.join(basedir, 'post_train_embed.pth.tar'))
'''
pretrained = 'pretrained_embed.pth.tar'
pretrained_embeds = torch.load(os.path.join(basedir, pretrained)).cpu()
# word_embed = words2embedding(word_dict, 100, args.embedding_file)
# net = FEELModel()
# net.load_state_dict(new_state)
# print(new_state.keys()

import pickle
with open(os.path.join(basedir,'cnn_dict.pkl'), 'rb') as fid:
    word_dict = pickle.load(fid)

trump_before = pretrained_embeds[word_dict['trump']]
trump_embeds = embeds[word_dict['trump']]
obama_before = pretrained_embeds[word_dict['obama']]
obama_embeds = embeds[word_dict['obama']]
president_embeds = embeds[word_dict['president']]
russia_embeds = embeds[word_dict['russia']]

trump_president = trump_embeds - president_embeds
obama_president = obama_embeds - president_embeds

trump_russia = trump_embeds - russia_embeds
obama_russia = obama_embeds - russia_embeds

def cosine(v1, v2):
    return (v1 * v2).sum() / (v1.norm(p=2) * v2.norm(p=2))

print('raw embeds')
# print(trump_embeds)
# print(obama_embeds)
print('dot before:', (trump_before * obama_before).sum())
print('cosine before:', cosine(trump_before, obama_before))
print('dot after:', (trump_embeds * obama_embeds).sum())
print('cosine after:', cosine(trump_embeds, obama_embeds))
print('*****************************')
'''
'''
print('distance to president')
print(trump_president)
print(obama_president)
print('dot:', (trump_president * obama_president).sum())
print('cos:', (trump_president * obama_president).sum() / (trump_president.norm(p=2) * obama_president.norm(p=2)))
print('*****************************')

print('distance to russia')
print(trump_russia)
print(obama_russia)
print((trump_russia * obama_russia).sum() / (trump_russia.norm(p=2) * obama_russia.norm(p=2)))
# print(trump_russia - obama_russia)
'''
