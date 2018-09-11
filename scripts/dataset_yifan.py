import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree


# Dataset class for SICK dataset
class FEELDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(FEELDataset, self).__init__()