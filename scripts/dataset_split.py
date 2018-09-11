import os
basedir = '/homes/du113/scratch/cnn-political-data'

filename = 'aug_20_samples.csv'
with open(os.path.join(basedir, filename)) as fid:
    data = fid.readlines()

data_size = len(data)
import random
train = random.sample(data, (3*data_size)//4)
test = list(filter(lambda x: x not in train, data))

print('%d train data and %d test data' % (len(train), len(test)))

train_file = 'aug_20_train_samples.csv'
test_file = 'aug_20_test_samples.csv'

with open(os.path.join(basedir, train_file), 'a') as tf:
    for td in train:
        tf.write(td)

with open(os.path.join(basedir, test_file), 'a') as tf:
    for td in test:
        tf.write(td)

