import os
import torch
from torch.utils.data.dataset import Dataset
from pprint import pprint

class DataFromFile(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path

        assert data_path is not None and os.path.exists(data_path), 'No file to read data from'

        if data_path:
            # will be read in from txt file
            with open(data_path, 'r') as datafile:
                self.data = datafile.readlines() 

        self.transforms = transforms

    def __getitem__(self, index):
        item = self.data[index]
        # pprint('item:{}'.format(str(item)))
        # item would be a comma-delimited string of length
        # split, map to ints
        item = item.strip().split('_')  # now it is a list of ints
        # pprint('item:{}'.format(str(item)))
        # print(len(item))

        # no modifiers
        query = item[:4]
        # pprint(query)
        query = [list(map(int, q.lstrip('[').rstrip(']').split(','))) for q in query]
        # print(len(query), len(query[0]))

        pos = item[4:8]
        pos = [list(map(int, p.lstrip('[').rstrip(']').split(','))) for p in pos]
        # print(len(pos), len(pos[0]))
        neg = item[8:]
        neg = [list(map(int, n.lstrip('[').rstrip(']').split(','))) for n in neg]
        # print(len(neg), len(neg[0]))
        # pprint(neg)

        if self.transforms:
            query, pos, neg = self.transforms(query), self.transforms(pos), self.transforms(neg)

        return query, pos, neg

    def __len__(self):
        return len(self.data)

