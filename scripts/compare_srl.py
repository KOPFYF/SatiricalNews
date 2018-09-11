from event_chain import *
import data_utilities as du
import utilities as util
from pprint import pprint

import time
####### test 3 compare ######
docs = du.load_sent('../datasets/bbcsample1.txt')
word_dict = util.build_dict(docs)

print('models already downloaded')
srl_predictor = Predictor.from_path('../pretrained/srl-model-2018.05.25.tar.gz')

def test3():
    global word_dict, srl_predictor

    # docs = du.load_sent('../datasets/bbcsample1.txt')

    # wd = util.build_dict(docs)
    # pprint(wd)

    # print(docs)
    print('using ecb **********')
    start = time.time()

    ecb = EventChainBuilder(word_dict)

    for i, sent in enumerate(docs):
        print('processing sentence', i)
        sent = nlp(sent)

        res = ecb.build_path(sent)

        print(res)
        print()
    timespan = time.time() - start
    print('used', timespan)

    vertice_map = []

    pos = 0

    print('using hashvertex **********')
    start = time.time()
    print('models already downloaded')
    srl_predictor = Predictor.from_path('../pretrained/srl-model-2018.05.25.tar.gz')

    for i, sent in enumerate(docs):
        print('processing sentence', i)
        sent = nlp(sent)

        doc_path = build_path(sent)
        print(doc_path)
        print()
        
        for v in doc_path[1:]:
            if args2index(v,sent):
                # vertice_map.append((v, [arg_range[0]+pos,arg_range[1]+pos]))
                vertice_map.append((v, [v.arg_range[0]+pos,v.arg_range[1]+pos]))
        pos = pos + len(sent)

        # print()
    timespan = time.time() - start
    print('used', timespan)


def get_srl(sent):
    global word_dict, srl_predictor
    res = srl_predictor.predict(sentence=str(sent))
    min_O = float('inf')
    min_id = 0

    if len(res["verbs"]) == 0:
        pass
    if len(res["verbs"]) == 1:
        verb_dict = res["verbs"][0]
        if 'ARG' in verb_dict['description']:
            return verb_dict

    for idx, verb_dict in enumerate(res["verbs"]):
        # print(verb_dict["tags"])
        num_O = collections.Counter(verb_dict["tags"])['O']
        if num_O < min_O:
            min_id = idx
        min_O = min(min_O,num_O)
    verb_dict = res["verbs"][min_id]
    if 'ARG' in verb_dict['description']:
        return verb_dict
    else:
        pass

# TODO: we might need a hashtable to store recurring verbs so that the space don't blow up.
# define global variable predv, arg0v, arg1v, arg2v
# global predv, arg0v, arg1v, arg2v = None, None, None, None
predv, arg0v, arg1v, arg2v = None, None, None, None

def build_path(sent):
    global word_dict, srl_predictor
    global predv, arg0v, arg1v, arg2v
    quotation = re.search(r"^(\")(.*)(\")$", str(sent))
    if quotation:
        # use the arg0v and predv in the last sentence
        arg1v = Vertex(quotation.group(1), 'ARG1')
        arg2v = None
        
    else:
        srl = get_srl(sent) 

        # for debugging
        # print('description:',srl['description'])

        if re.search(r"(\[ARG0: )(.*?)(\])",srl['description']):
            arg0 = re.search(r"(\[ARG0: )(.*?)(\])",srl['description'])[2]
        else:
            arg0 = None

        if re.search(r"(\[ARG1: )(.*?)(\])",srl['description']):
            arg1 = re.search(r"(\[ARG1: )(.*?)(\])",srl['description'])[2]
        else:
            arg1 = None

        if re.search(r"(\[ARG2: )(.*?)(\])",srl['description']):
            arg2 = re.search(r"(\[ARG2: )(.*?)(\])",srl['description'])[2]
        else:
            arg2 = None


        # print('arg0:',arg0)
        # print('arg1:',arg1) 
        # mod: ARGM-DIS, ARGM-TMP, ARGM-ADV
        if re.search(r"(\[ARGM-.*?: )(.*?)(\])",srl['description']):
            mod = re.search(r"(\[ARGM-.*?: )(.*?)(\])",srl['description'])[2]
            # for debugging
        else:
            # print('debug:',srl['description'])
            mod = None
        verb = srl['verb']
        
        # convert verb to lemma_
        # verb = nlp(verb)[0].lemma_
        # for debugging
        # print('verb:',verb)
        
        # create 4 vertices
        predv = Vertex(verb,'PRED')
        util.vertex2wordidx(predv, word_dict)

        # if no arg0, then will not create vertex arg0v and edge arg0e
        if arg0:
            arg0v = Vertex(arg0, 'ARG0')
            util.vertex2wordidx(arg0v, word_dict)

        else:
            arg0v = None

        if arg1:
            arg1v = Vertex(arg1, 'ARG1')
            util.vertex2wordidx(arg1v, word_dict)
        else:
            arg1v = None

        if arg2:
            arg2v = Vertex(arg2, 'ARG2')
            util.vertex2wordidx(arg2v, word_dict)
        else:
            arg2v = None

    # if mod:
    #     modv = Vertex(mod, 'MOD')

    res = [predv]
    res.append(arg0v)
    res.append(arg1v)
    res.append(arg2v)

    if arg0v:
        arg0v.path = res
    if arg1v:
        arg1v.path = res
    if arg2v:
        arg2v.path = res
    return res


def args2index(arg,sent):
    global word_dict, srl_predictor
    # return [sent.find(arg),sent.find(arg)+len(arg)] sentence index
    # also return if arg == None
    # input: vertex, sent string
    v = arg
    if arg:
        arg = arg.val.split()
        sent = [token.text for token in sent]

        for i in range(len(sent)-len(arg)+1):
            quotation = False
            if sent[i] == arg[0] and sent[i+len(arg)-1] == arg[len(arg)-1]:
                # arg found! : [3, 3] ['carried']
                # print('arg found! :',[i,i+len(arg)-1],sent[i:i+len(arg)],'\n')
                # return [i,i+len(arg)-1], arg == None
                v.arg_range = [i,i+len(arg)-1]

                return True
        else:
            # quotation = True
            return False
                
    else:
        # when arg == None
        # return [-1, -1], arg == None
        return False

if __name__ == '__main__':
    test3()
