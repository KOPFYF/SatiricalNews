import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from pickle import load, dump

import numpy as np
from tensorboardX import SummaryWriter

from dataload import DataFromFile

import argparse
import logging
import codecs

import shutil

from baseline import FEELModel
from test_baseline import TestModel

scratch = '/homes/du113/scratch'
tbrun = os.path.join(scratch, 'tensorboard/runs')
BASE = '/homes/du113/scratch/cnn-political-data'
# FORMAT = '%(asctime)-15s %(message)s'
# logging.basicConfig(format=FORMAT)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")

def words2embedding(word_dict, dim, in_file=None):
    num_words = max(word_dict.values()) + 1
    embeddings = torch.FloatTensor(num_words, dim).uniform_()
    logging.info("Embedding dimension: %d * %d" % (embeddings.shape[0], embeddings.shape[1]))

    if in_file is not None:
        logging.info("loading embedding file: %s" % in_file)
        pre_trained = 0
        with codecs.open(in_file, encoding='utf') as f:
            l = f.readlines()
        for line in l:
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = torch.FloatTensor([float(x) for x in sp[1:]])
        logging.info("pre-trained #: %d (%.2f%%)" % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def gen_dataloaders(args):
    batch_size = args.batch_size
    logging.info('batch size: {}'.format(batch_size))
    logging.info('trainpath: {}\ttestpath: {}'.format(args.train_path, args.test_path))

    if args.train_path:
        train_set = DataFromFile(data_path=args.train_path,
                                transforms=torch.LongTensor)
        logging.info('#batch: {}'.format(len(train_set)))

        trainloader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=args.pin)
    else:
        trainloader = None

    if args.test_path:
        test_set = DataFromFile(data_path=args.test_path, transforms=torch.LongTensor)
        logging.info('#batch: {}'.format(len(test_set)))

        testloader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=args.pin)
    else:
        testloader = None

    return trainloader, testloader


def my_loss_fn(output):
    return output.sum() / output.data.nelement()



def distance_abs(lvec, rvec):
    # mult_dist = torch.mul(lvec, rvec)
    abs_dist = torch.abs(torch.add(lvec, -rvec))
    # vec_dist = torch.cat((mult_dist, abs_dist), 1)
    return abs_dist.sum(dim=0)


def distance_mul(lvec, rvec):
    mult_dist = torch.mul(lvec, rvec)
    # abs_dist = torch.abs(torch.add(lvec, -rvec))
    # vec_dist = torch.cat((mult_dist, abs_dist), 1)
    return mult_dist.sum(dim=0)


def load_embeds(args):
    print('load word dict...')
    with open(os.path.join(BASE,'cnn_dict.pkl'), 'rb') as fid:
        word_dict = load(fid)
    print('loading completed!')

    # load word embedding
    word_embed = words2embedding(word_dict, 100, args.embedding_file)
    print('word_embed.shape:', word_embed.shape)
    
    return word_embed


def train(loaders, args, word_embed):
    writer = SummaryWriter(os.path.join(tbrun, 'aug_21'))

    trainloader, testloader = loaders

    label = get_meta_data()
    writer.add_embedding(word_embed, metadata=label)

    torch.save(word_embed.clone(), os.path.join(BASE, 'aug_21_pretrained_embed.pth.tar'))
    device = 'cuda' if args.cuda else 'cpu'

    model = FEELModel(pretrained=True, delta=1.0, embeddings=word_embed)
    model = model.double()

    if device == 'cuda':
        model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_loss = 1000000

    n_data = len(trainloader)

    for e in range(args.epochs):
        logging.info('{}th epoch'.format(e))
        # for every 5 epochs, test on test data
        if e % 5 == 0:
            test_accr = test(testloader, args, model.module.embeddings.weight)
            writer.add_scalar('test/accuracy', test_accr, e) 

        total_loss = 0
        for i, batch in enumerate(trainloader):
            # each tensor has size (batchsize x 4 x arg.max_word_length)
            # most probably a 3 dimensional tensor
            query, pos, neg = batch # (batchsize x 4 x arg.max_word_length)

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
            # ugly input:
            inputs = (q_v, q_a0, n_a0), (q_v, q_a1, n_a1), \
                    (q_v, q_a2, n_a2), (query, pos, neg)
            output = model(inputs)

            loss = my_loss_fn(output)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            if i % 20 == 0:
                # logging.info('{}th iteration'.format(i))
                writer.add_scalar('loss/batch', loss.item(), e*n_data+i)

        writer.add_scalar('loss/total', total_loss, e)

        is_best = (total_loss <= best_loss)
        best_loss = min(total_loss, best_loss)

        save_checkpoint({
        'epoch': e,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict()
        }, is_best)

    model = model.cpu()
    torch.save(model.state_dict(),'final_params.pth.tar')

    writer.add_embedding(model.embeddings.weight, metadata=label)

    torch.save(model.embeddings.weight, os.path.join(BASE, 'aug_21_post_train_embed.pth.tar'))
    # return model.embeddings.weight
    writer.close()


def test(testloader, args, word_embed):
    device = 'cuda' if args.cuda else 'cpu'

    model = TestModel(embeddings=word_embed)
    model = model.double()

    total, correct = 0, 0

    if device == 'cuda':
        model = nn.DataParallel(model).cuda()

    model = model.eval()

    for i, batch in enumerate(testloader):
        # each tensor has size (batchsize x 4 x arg.max_word_length)
        # most probably a 3 dimensional tensor
        query, pos, neg = batch # (batchsize x 4 x arg.max_word_length)

        q_v, q_a0, q_a1, q_a2 = query[:, 0, :], query[:, 1, :], \
            query[:, 2, :], query[:, 3, :]
        n_v, n_a0, n_a1, n_a2 = neg[:, 0, :], neg[:, 1, :], \
            neg[:, 2, :], neg[:, 3, :]

        # torch.Size([2, 4, 3]) -> torch.Size([2, 3])

        q_v, q_a0, q_a1, q_a2 = \
            Variable(q_v).to(device), \
            Variable(q_a0).to(device), \
            Variable(q_a1).to(device), \
            Variable(q_a2).to(device)

        n_v, n_a0, n_a1, n_a2 = \
            Variable(n_v).to(device), \
            Variable(n_a0).to(device), \
            Variable(n_a1).to(device), \
            Variable(n_a2).to(device)

        inputs = (q_v, q_a0, n_v, n_a0, n_a1, n_a2)

        output = model(inputs)

        total += args.batch_size
        correct += output.numel() - output.nonzero().shape[0]

    # print('testing accuracy: %.2f' % (correct / total))
    return correct/total


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    basedir = '/homes/du113/scratch/cnn-political-data'
    filename = os.path.join(basedir,filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(basedir,'model_best.pth.tar'))


def get_meta_data():
    label = ['PAD','UNK']
    with open(os.path.join(BASE,'cnn_dict.pkl'), 'rb') as f:
        labels = sorted(load(f).items(), key=lambda x: x[1])
        label += ['"'+l[0]+'"' for l in labels]
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('-p', '--pin', type=bool, default=False)

    # parser.add_argument('-c', '--cuda', type=bool, default=False)
    parser.add_argument('-c', '--cuda', action='store_true')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-e', '--epochs', type=int, default=10)

    parser.add_argument('--embedding_file', type=str, default=None)

    args = parser.parse_args()

    loaders = gen_dataloaders(args)
    # print(train_loader)
    embeds = load_embeds(args)

    new_embeds = train(loaders, args, embeds)

    # to visulize word embedding, use: https://projector.tensorflow.org/

if __name__ == '__main__':
    main()
    '''
    python Model_baseline_ugly.py --embedding '/Users/feiyifan/Desktop/NLP/FEEL/glove_6B/glove.6B.100d.txt' --train_path '/Users/feiyifan/Desktop/NLP/FEEL/nlpstuff/datasets/train/samples.csv' -e 30
    Python —embedding ‘/homes/fei8/scratch/glove100d.txt’ --train_path '/homes/fei8/scratch/FEEL/nlpstuff/datasets/train/samples.csv' -e 30

    '''
