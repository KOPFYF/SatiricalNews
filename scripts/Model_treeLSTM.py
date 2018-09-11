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


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        # in_dim: input dimension, batch size
        # mem_dim: embedding demension
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ioux, self.iouh, self.fx, self.fh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def node_forward(self, inputs, child_c, child_h):
        # 
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        # input: (batchsize, 4 x arg.max_word_length, mem_dim)
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(
                1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(
                1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(
                child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, out_dim):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)  # mem_dim x hidden_dim
        self.wh = nn.Linear(self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.out_dim) # hidden_dim x out_dim

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        # abs_dist = torch.abs(torch.add(lvec, -rvec))
        # vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(mult_dist))
        out = self.wp(out)
        return out  # (batchsize, 4 x arg.max_word_length, out_dim)



class FEELModel_intra(nn.Module):
    def __init__(self, vocab_size=0, embedding_dim=0, pretrained=False, embeddings=None, delta=1.0):
        super(FEELModel_intra, self).__init__()
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
        n_embeds = self.embeddings(neg).mean(dim=1)
        
        # max(0, delta - ev*ev' + ev*ev'')
        # the result should be of dimensions (batch_size)
        a = (q_embeds*p_embeds).view(q_embeds.shape[0],-1).sum(1)
        b = (q_embeds*n_embeds).view(q_embeds.shape[0],-1).sum(1)
        return F.relu(self.delta - a + b)



# TODO: Build Tree 
class FEELModel_inter(nn.Module):
    def __init__(self, vocab_size=0, in_dim=0, hidden_dim=0, embedding_dim=0, delta=1.0, out_dim=2):
        super(FEELModel_inter, self).__init__()
        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.delta = delta


    # def forward(self, qevent_inputs, pevent_inputs, nevent_inputs, qtree, ptree, ntree):
    def forward(self, inputs, qtree, ptree, ntree):
        # batchsize x 4 x arg.max_word_length
        query, pos, neg = inputs

        # batchsize x (4 * arg.max_word_length)
        # query = torch.cat((query[:, 0, :], query[:, 1, :], query[:, 2, :], query[:, 3, :]), 1)
        # pos = torch.cat((pos[:, 0, :], pos[:, 1, :], pos[:, 2, :], pos[:, 3, :]), 1)
        # neg = torch.cat((neg[:, 0, :], neg[:, 1, :], neg[:, 2, :], neg[:, 3, :]), 1)

        # more elegant to flat the event
        query = query.view(query.shape[0], 1, -1).squeeze(1)
        pos = pos.view(pos.shape[0], 1, -1).squeeze(1)
        neg = neg.view(neg.shape[0], 1, -1).squeeze(1)

        # batchsize x (4 * arg.max_word_length) then ... 
        q_embeds = self.embeddings(query) # shape: (batchsize, 4 x arg.max_word_length, mem_dim)
        p_embeds = self.embeddings(pos)
        n_embeds = self.embeddings(neg)

        # Aug 14th, input 3 tree and 3 embeddings and we can get loss for each pos&neg pair 
        # Build tree like this:
        #                      root -> predv
        #                       /   |    \
        # three children: arg0v, arg1v, arg2v, (Modv)
        qstate, qhidden = self.childsumtreelstm(qtree, q_embeds)
        pstate, phidden = self.childsumtreelstm(ptree, p_embeds)
        nstate, nhidden = self.childsumtreelstm(ntree, n_embeds)
        output1 = self.similarity(qstate, pstate) # shape: (batchsize, 4 x arg.max_word_length, out_dim)
        output2 = self.similarity(qstate, nstate) # output event embedding as (4 x arg.max_word_length, out_dim)
        a = output1.view(qstate.shape[0],-1).sum(1)
        b = output2.view(qstate.shape[0],-1).sum(1)
        
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

    model_inter= FEELModel_inter()
    model_intra= FEELModel_intra(pretrained=True, embeddings=word_embed)

    if device == 'cuda':
        model_inter = nn.DataParallel(model_inter).cuda()
        model_intra = nn.DataParallel(model_intra).cuda()

    # TODO; how to merge 2 opt
    optimizer = optim.SGD(model_intra.parameters(), lr=args.lr)

    for e in range(args.epochs):
        logging.warning('{}th epoch'.format(e))
        total_loss = 0
        for i, batch in enumerate(trainloader):
            # each tensor has size (batchsize x 4 x arg.max_word_length)
            # most probably a 3 dimensional tensor
            # query, pos, neg = batch

            query_zip, pos_zip, neg_zip = batch

            # TODO: query should contain both vertex id(query[0]) and word_idx_list(query[1])
            qtree = du.build_tree(query_zip[0])
            ptree = du.build_tree(pos_zip[0])
            ntree = du.build_tree(neg_zip[0])

            query, pos, neg = query_zip[1], pos_zip[1], neg_zip[1]
            # print(query.shape) 2d
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

            output1 = model_intra((q_v, q_a0, n_a0))
            output2 = model_intra((q_v, q_a1, n_a1))
            output3 = model_intra((q_v, q_a2, n_a2))
            output_inter = model_inter(query, pos, neg, qtree, ptree, ntree)
            # query, pos, neg: 2d wordidx matrix -> average to 1d list emb

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
