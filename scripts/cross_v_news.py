import spacy
nlp = spacy.load('en_core_web_sm')

with open('/homes/du113/scratch/cnn-political-data/test_coref2.txt')as f:
    news = f.readlines()

doc = ' '.join([n.strip() for n in news])

doc = nlp(doc)

offset = 0
for sent in doc.sents:
    for word in sent:
        print(word.text, 'at', offset)
        offset += 1
    
