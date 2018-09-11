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

import utilities as util
import data_utilities as du

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)


class FEELModel_intra(nn.Module):
    def __init__(self, vocab_size=0, embedding_dim=100, pretrained=False, embeddings=None, delta=1.0):
        super(FEELModel, self).__init__()
        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = True
        self.delta = delta

    def forward(self, inputs):
        # one hot vectors, each has shape (batch_size)
        query, pos, neg = inputs    # event vectors each has shape (batch_size x max_word_length)
        
        q_embeds = self.embeddings(query).mean(dim=1)  # initial result would be 3d tensor (batch_size x max_word_length x embedding_dim) 
        p_embeds = self.embeddings(pos).mean(dim=1)     # batch x embedding_dim
        n_embeds = self.embeddings(neg).mean(dim=1) # shape: (batchsize, max_word_length, emb) -> (batchsize, emb)
        
        # max(0, delta - ev*ev' + ev*ev'')
        # the result should be of dimensions (batch_size)
        # return F.relu(self.delta - (q_embeds * p_embeds).sum(dim=1) + (q_embeds * n_embeds).sum(dim=1))
        a = (q_embeds*p_embeds).view(q_embeds.shape[0],-1).sum(1)
        b = (q_embeds*n_embeds).view(q_embeds.shape[0],-1).sum(1)
        return F.relu(self.delta - a + b)


# TODO: Build Tree 
class FEELModel_inter(nn.Module):
    def __init__(self, vocab_size=0, in_dim=0, hidden_dim=50, embedding_dim=100, delta=1.0, out_dim=30):
        super(FEELModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # cat event from 3d to 2d then use this 2d matrix as input, then lower the dim
        self.wh = nn.Linear(4 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.out_dim)
        self.delta = delta


    # def forward(self, qevent_inputs, pevent_inputs, nevent_inputs, qtree, ptree, ntree):
    def forward(self, inputs):
        # inputs are avg emb from lower level
        # each event now has a 2d embedding, event[0] = predv
        query, pos, neg = inputs

        # torch.Size([2, 4, 3]) -> torch.Size([2, 12])
        # (batchsize = 2, hidden_dim = 4, arg.max_word_length = 3) -> (batchsize = 2, 4 x arg.max_word_length = 12)
        # more elegant to flat the event
        query = query.view(query.shape[0], 1, -1).squeeze(1)
        pos = pos.view(pos.shape[0], 1, -1).squeeze(1)
        neg = neg.view(neg.shape[0], 1, -1).squeeze(1)

        q_embeds = self.embeddings(query) # shape: (batchsize, 4 x arg.max_word_length, mem_dim)
        p_embeds = self.embeddings(pos)
        n_embeds = self.embeddings(neg)

        q_out = self.wp(F.sigmoid(self.wh(q_embeds))) # shape: (batchsize, 4 x arg.max_word_length, out_dim)
        p_out = self.wp(F.sigmoid(self.wh(p_embeds)))
        n_out = self.wp(F.sigmoid(self.wh(n_embeds)))

        a = (q_out*p_out).view(q_out.shape[0],-1).sum(1)
        b = (q_out*n_out).view(q_out.shape[0],-1).sum(1)
        
        # the result should be of dimensions (batch_size)
        return F.relu(self.delta - a + b)


def gen_dataloaders(args):
    batch_size = args.batch_size
    logging.warning('batch size: {}'.format(batch_size))

    train_set = DataFromFile(data_path=args.train_path,
                             transforms=torch.LongTensor)
    logging.warning('#batch: {}'.format(len(train_set)))

    trainloader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=args.pin)

    if not args.test_path:
        return trainloader, None

    test_set = DataFromFile(data_path=args.test_path)

    testloader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin)

    return trainloader, testloader


def my_loss_fn(output):
    return output.sum() / output.data.nelement()


def train(trainloader, args):
    device = 'cuda' if args.cuda else 'cpu'

    losses = []

    # TODO: concretize the embedding step

    # BASE = '/homes/du113/scratch/satire-models/'
    BASE = '../datasets/'
    with open(BASE + 'word_dict', 'rb') as fid:
        word_dict = load(fid)

    # load word embedding
    word_embed = util.words2embedding(word_dict, 100, args.embedding_file)
    # print(word_embed.shape)

    model_intra= FEELModel_intra(pretrained=True, embeddings=word_embed)
    model_inter= FEELModel_inter()

    if device == 'cuda':
        model_inter = nn.DataParallel(model_inter).cuda()
        model_intra = nn.DataParallel(model_intra).cuda()

    optimizer = optim.SGD(model_intra.parameters(), lr=args.lr)

    for e in range(args.epochs):
        logging.warning('{}th epoch'.format(e))
        total_loss = 0
        for i, batch in enumerate(trainloader):
            # each tensor has size (batchsize x 4 x arg.max_word_length)
            # most probably a 3 dimensional tensor
            query, pos, neg = batch

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

            # torch.Size([2, 4, 3]) -> torch.Size([2, 3])


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


            query, pos, neg = \
                Variable(query).to(device), \
                Variable(pos).to(device), \
                Variable(neg).to(device)

            model.zero_grad()

            output1 = model_intra((q_v, q_a0, n_a0))
            output2 = model_intra((q_v, q_a1, n_a1))
            output3 = model_intra((q_v, q_a2, n_a2))
            output_inter = model_inter((query, pos, neg))
            # query, pos, neg: 2d wordidx matrix -> average to 1d list emb
            # the output should be of dimensions (batch_size)
            output = output1 + output2 + output3 + output_inter

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

    # to visulize word embedding, use: https://projector.tensorflow.org/
