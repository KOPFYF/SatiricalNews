import time
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

with open('/homes/du113/scratch/pretrained/0_BBCNews.txt') as f:
    sents = f.readlines()

archive = load_archive('/homes/du113/scratch/pretrained/srl-model-2018.05.25.tar.gz', cuda_device=0)
srl_predictor = Predictor.from_archive(archive)
# print(srl_predictor.predict.__code__.co_varnames)

start = time.time()
for i, sent in enumerate(sents):
    res = srl_predictor.predict(sentence=sent)

    if i % 10 == 9:
        print('parsed 10 sents')

print('used %d seconds' % (time.time() - start))
