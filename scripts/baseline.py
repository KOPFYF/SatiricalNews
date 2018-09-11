import os
import sys
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

import data_utilities as du

import shutil

import matplotlib.pyplot as plt

class FEELModel(nn.Module):
    # inputs = (q_v, q_a0, n_a0), (q_v, q_a1, n_a1), (q_v, q_a2, n_a2), (query, pos, neg)
    def __init__(self, vocab_size=0, embedding_dim=100, pretrained=False, embeddings=None, delta=0.0, in_dim=0, hidden_dim=50, out_dim=30):
        super(FEELModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

        if pretrained:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = True

        self.wh = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.out_dim)

        self.delta = delta

    def forward(self, inputs):
        # one hot vectors, each has shape (batch_size)
        (q_v, q_a0, n_a0), (q_v, q_a1, n_a1), (q_v,
                                               q_a2, n_a2), (query, pos, neg) = inputs

        # initial result would be 3d tensor (batch_size x max_word_length x embedding_dim)
        q_v_embeds = self.embeddings(q_v).mean(dim=1)
        q_a0_embeds = self.embeddings(q_a0).mean(dim=1)    
        # shape: (batchsize, max_word_length, emb) -> (batchsize, emb)
        n_a0_embeds = self.embeddings(n_a0).mean(dim=1)

        q_a1_embeds = self.embeddings(q_a1).mean(dim=1)     
        n_a1_embeds = self.embeddings(n_a1).mean(dim=1)

        q_a2_embeds = self.embeddings(q_a2).mean(dim=1)     
        n_a2_embeds = self.embeddings(n_a2).mean(dim=1)

        # the result should be of dimensions (batch_size)
        loss0 = F.relu(self.delta - (q_v_embeds * q_a0_embeds).view(q_v_embeds.shape[0], -1).sum(1) + \
            (q_v_embeds * n_a0_embeds).view(q_v_embeds.shape[0], -1).sum(1))

        loss1 = F.relu(self.delta - (q_v_embeds * q_a1_embeds).view(q_v_embeds.shape[0], -1).sum(1) + \
            (q_v_embeds * n_a1_embeds).view(q_v_embeds.shape[0], -1).sum(1))

        loss2 = F.relu(self.delta - (q_v_embeds * q_a2_embeds).view(q_v_embeds.shape[0], -1).sum(1) + \
            (q_v_embeds * n_a2_embeds).view(q_v_embeds.shape[0], -1).sum(1))

        # shape: (batchsize, 4 x arg.max_word_length) flat and cat
        query = query.view(query.shape[0], 1, -1).squeeze(1)
        pos = pos.view(pos.shape[0], 1, -1).squeeze(1)
        neg = neg.view(neg.shape[0], 1, -1).squeeze(1)

        # shape: (batchsize, 4 x arg.max_word_length, mem_dim)
        q_embeds = self.embeddings(query)
        p_embeds = self.embeddings(pos)
        n_embeds = self.embeddings(neg)

        # shape: (batchsize, 4 x arg.max_word_length, out_dim)
        q_out = self.wp(F.sigmoid(self.wh(q_embeds)))
        p_out = self.wp(F.sigmoid(self.wh(p_embeds)))
        n_out = self.wp(F.sigmoid(self.wh(n_embeds)))

        # the result should be of dimensions (batch_size)
        loss_inter = F.relu(self.delta - (q_out * p_out).view(q_out.shape[0], -1).sum(1) + \
            (q_out * n_out).view(q_out.shape[0], -1).sum(1))

        loss = loss0 + loss1 + loss2 + loss_inter

        return loss

