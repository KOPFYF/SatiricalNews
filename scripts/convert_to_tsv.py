import torch
import os

basedir = '/homes/du113/scratch/cnn-political-data'
filename = 'pretrained_embed.pth.tar'

# with open(os.path.join(basedir, filename),
embeds = torch.load(os.path.join(basedir, filename))

pretensor = embeds.cpu()# .numpy().tolist()
# print(pretensor)

'''
with open(os.path.join(basedir, filename.split('.')[0] + '.tsv'), 'a') as f:
    for i, row in enumerate(pretensor):
        # f.write('\t'.join(map(str, row)))
        print(row, sep='\t', file=f)
'''

filename = 'post_train_embed.pth.tar'

# with open(os.path.join(basedir, filename),
embeds = torch.load(os.path.join(basedir, filename))

posttensor = embeds.cpu() #.numpy().tolist()
# print(posttensor)

'''
with open(os.path.join(basedir, filename.split('.')[0] + '.tsv'), 'a') as f:
    for i, row in enumerate(posttensor):
        # f.write('\t'.join(map(str, row)))
        print(row, sep='\t', file=f)
'''
print(pretensor - posttensor)



