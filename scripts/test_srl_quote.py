from textacy.extract import direct_quotations
import spacy
nlp = spacy.load('en_core_web_sm')

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# sent = '"It was all me at the beginning, squeezing the lemonade at my stand, but then my parents designed some nice stickers for the cups," says Mikaila.'

sent = 'Omarosa Manigault Newman, the former "Apprentice" contestant who became a White House aide, says in her new book that President Donald Trump is a "racist" who used the N-word — and tried to silence her with money and legal threats.'

doc = nlp(sent)

# print(next(direct_quotations(doc)))

srl_predictor = Predictor.from_archive(load_archive('/homes/du113/scratch/pretrained/srl-model-2018.05.25.tar.gz',
    cuda_device=[1,2]))

results_srl = srl_predictor.predict(sentence=str(sent))

print(results_srl)

print(type(result_srl['words'][0]))
