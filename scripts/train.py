import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from pickle import load, dump

from dataload import DataFromFile

import argparse
import logging
import codecs

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

def words2embedding(word_dict, dim, in_file=None):
    num_words = max(word_dict.values()) + 1
    embeddings = torch.FloatTensor(num_words, dim).uniform_()
    logging.warning("Embedding dimension: %d * %d" % (embeddings.shape[0], embeddings.shape[1]))

    if in_file is not None:
        logging.warning("loading embedding file: %s" % in_file)
        pre_trained = 0
        with codecs.open(in_file, encoding='utf') as f:
            l = f.readlines()
        for line in l:
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = torch.FloatTensor([float(x) for x in sp[1:]])
        logging.warning("pre-trained #: %d (%.2f%%)" % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings

class FEELModel(nn.Module):
    def __init__(self, vocab_size=0, embedding_dim=0, pretrained=False, embeddings=None, delta=1.0):
        super(FEELModel, self).__init__()
        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.embeddings.weight.requires_grad = True

        self.delta = delta

    def forward(self, inputs):
        # query, pos, neg = inputs    # one hot vectors, each has shape (batch_size)
        query, pos, neg = inputs    # event vectors each has shape (batch_size x max_word_length)
        
        q_embeds = self.embeddings(query).mean(dim=1)  # initial result would be 3d tensor (batch_size x max_word_length x embedding_dim) 
        p_embeds = self.embeddings(pos).mean(dim=1)     # batch x embedding_dim
        n_embeds = self.embeddings(neg).mean(dim=1)
        
        # max(0, delta - ev*ev' + ev*ev'')
        # the result should be of dimensions (batch_size)
        # return F.relu(self.delta - (q_embeds * p_embeds).sum(dim=1) + (q_embeds * n_embeds).sum(dim=1))
        a = (q_embeds*p_embeds).view(q_embeds.shape[0],-1).sum(1)
        b = (q_embeds*n_embeds).view(q_embeds.shape[0],-1).sum(1)
        return F.relu(self.delta - a + b)


def gen_dataloaders(args):
    batch_size = args.batch_size
    logging.warning('batch size: {}'.format(batch_size))

    train_set = DataFromFile(data_path = args.train_path, transforms=torch.LongTensor)
    logging.warning('#batch: {}'.format(len(train_set)))
    
    trainloader = torch.utils.data.DataLoader(dataset=train_set, \
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin)

    if not args.test_path:
        return trainloader, None

    test_set = DataFromFile(data_path = args.test_path)
    
    testloader = torch.utils.data.DataLoader(dataset=test_set, \
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin)
    
    return trainloader, testloader


def my_loss_fn(output):
    return output.sum() / output.data.nelement()

# TODO: need to expand events to list of word id's (fixed length)
def train(trainloader, args):
    device = 'cuda' if args.cuda else 'cpu'

    losses = []

    BASE = '/homes/du113/scratch/satire-models/'
    with open(BASE + 'word_dict', 'rb') as fid:
        word_dict = load(fid)

    # load word embedding
    word_embed = words2embedding(word_dict, 100, args.embedding_file) 

    # print(word_embed.shape)

    model = FEELModel(pretrained=True, embeddings=word_embed)

    if device == 'cuda':
        model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr) 

    for e in range(args.epochs):
        logging.warning('{}th epoch'.format(e))
        total_loss = 0
        for i, batch in enumerate(trainloader):
            # each tensor has size (batchsize x 5)
            # most probably a 2 dimensional tensor
            query, pos, neg = batch
            # print(query.shape)
            # print(pos.shape)
            # print(neg.shape)

            '''
            q_a0, q_v, q_a1, q_a2, q_m = query[:,0], query[:,1], \
                    query[:,2], query[:,3], query[:,4]
            p_a0, p_v, p_a1, p_a2, p_m = pos[:,0], pos[:,1], \
                    pos[:,2], pos[:,3], pos[:,4]
            n_a0, n_v, n_a1, n_a2, n_m = neg[:,0], neg[:,1], \
                    neg[:,2], neg[:,3], neg[:,4]
            q_a0, q_v, q_a1, q_a2, q_m = Variable(q_a0).to(device), \
                    Variable(q_v).to(device), \
                    Variable(q_a1).to(device), \
                    Variable(q_a2).to(device), \
                    Variable(q_m).to(device)
            p_a0, p_v, p_a1, p_a2, p_m = Variable(p_a0).to(device), \
                    Variable(p_v).to(device), \
                    Variable(p_a1).to(device), \
                    Variable(p_a2).to(device), \
                    Variable(p_m).to(device)
            n_a0, n_v, n_a1, n_a2, n_m = Variable(n_a0).to(device), \
                    Variable(n_v).to(device), \
                    Variable(n_a1).to(device), \
                    Variable(n_a2).to(device), \
                    Variable(n_m).to(device)
            model.zero_grad()
            
            output = model((q_v, p_v, n_v)) + model((q_v, q_a0, n_a0)) \
                    + model((q_v, q_a1, n_a1)) + model((q_v, q_a2, n_a2)) \
                    + model((q_v, q_m, n_m))
            '''
            q_v, q_a0, q_a1, q_a2 = query[:, 0, :], query[:, 1, :], \
                query[:, 2, :], query[:, 3, :]
            p_v, p_a0, p_a1, p_a2 = pos[:, 0, :], pos[:, 1, :], \
                pos[:, 2, :], pos[:, 3, :]
            n_v, n_a0, n_a1, n_a2 = neg[:, 0, :], neg[:, 1, :], \
                neg[:, 2, :], neg[:, 3, :]

            q_v, q_a0, q_a1, q_a2 = \
                    Variable(q_v).to(device), \
                    Variable(q_a0).to(device), \
                    Variable(q_a1).to(device), \
                    Variable(q_a2).to(device)

            p_v, p_a0, p_a1, p_a2 = \
                    Variable(p_v).to(device), \
                    Variable(p_a0).to(device), \
                    Variable(p_a1).to(device), \
                    Variable(p_a2).to(device)

            n_v, n_a0, n_a1, n_a2 = \
                    Variable(n_v).to(device), \
                    Variable(n_a0).to(device), \
                    Variable(n_a1).to(device), \
                    Variable(n_a2).to(device)

            model.zero_grad()
            
            output = model((q_v, p_v, n_v)) + model((q_v, q_a0, n_a0)) \
                    + model((q_v, q_a1, n_a1)) + model((q_v, q_a2, n_a2))

            loss = my_loss_fn(output)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            if i % 20 == 0:
                logging.warning('{}th iteration'.format(i))
        logging.warning('loss: {}'.format(total_loss))
        losses.append(total_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('-p', '--pin', type=bool, default=False)

    parser.add_argument('-c', '--cuda', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', type=int, default=1)

    parser.add_argument('--embedding_file', type=str, default=None)

    args = parser.parse_args()

    train_loader, test_loader = gen_dataloaders(args)
    train(train_loader, args)