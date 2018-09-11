import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import data_utilities as du

query = torch.randint(10,(2, 4, 3))
print(query)
# query = torch.cat((query[:, 0, :], query[:, 1, :], query[:, 2, :], query[:, 3, :]), 1)
x = query[:, 0, :]
print(query.shape)
print(x.shape)
print('x:',x)


embedding = nn.Embedding(10, 5) # emd_size = 3
# i = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
i = torch.LongTensor(x.data.numpy())
a = embedding(i)
print('shape i', i.shape)
print('shape a', a.shape)


a = torch.randint(10,(16,4,100))
qtree = du.build_tree(a)
print(qtree)
print(qtree.num_children)



# d = a[0][0][0]
# print(d)
# e = d + c
# print('e:', e)
# print(c)
# # d = (a*b).sum(dim=1)
# e = torch.mul(a, b)
# f = (a * b).view(a.shape[0],-1).sum(1)
# print(d.shape[0])
# print(e.shape)
# print(e)
# print('f:',f)

# g = a.view(2, 1, -1).squeeze(1)
# h = b.view(2, 1, -1).squeeze(1)
# print('a:', a)
# print(g*h)
