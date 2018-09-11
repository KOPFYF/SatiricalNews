import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import logging

class TestModel(nn.Module):
    def __init__(self, embeddings=None):
        super(TestModel, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad = False

    def forward(self, inputs):
        pred, c1, c2, c3, c4, c5 = inputs

        # initial result would be 3d tensor (batch_size x max_word_length x embedding_dim)
        pred_embeds = self.embeddings(pred).mean(dim=1)
        c1_embeds = self.embeddings(c1).mean(dim=1)
        c2_embeds = self.embeddings(c2).mean(dim=1)
        c3_embeds = self.embeddings(c3).mean(dim=1)
        c4_embeds = self.embeddings(c4).mean(dim=1)
        c5_embeds = self.embeddings(c5).mean(dim=1)
        # each of the tensors now is a 2d tensor (batch_size x embedding_size)

        # c1 is the correct answer, and the rest are wrong answers
        stacked_pred_embeds = torch.stack([pred_embeds for _ in range(5)], dim=1)
        # now it is (batch_size x 5 x embedding_dim)
        stacked_choices = torch.stack([c1_embeds,c2_embeds,c3_embeds,c4_embeds,c5_embeds], dim=1)

        result = (stacked_pred_embeds * stacked_choices)
        result = result.view(result.shape[0], 5, -1).sum(2)
        # now result is (batch_size x 5)
        result = torch.argmax(result, dim=1)

        # the result has dimension (batch_size,)
        return result

