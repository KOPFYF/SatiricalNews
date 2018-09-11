import os
from tree import Tree
import torch

homedic = os.getcwd()


def build_tree(event):
    '''
    input: a batch of events [prev, arg0, arg1, arg2], a list of vertex id

    shape: (batchsize x 4 x arg.max_word_length)

    '''
    # prev, arg0, arg1, arg2 = event

    tree = Tree()
    tree.idx = 0
    ev_len = event.shape[1]
    id_ls = list(range(ev_len))
    tree.parent = 0
    for arg in id_ls[1:]:
        argtree = Tree()
        argtree.idx = arg
        tree.add_child(argtree)
    return tree


def print_tree(tree, level):
    indent = ''
    for i in range(level):
        indent += '| '
    line = indent + str(tree.idx)
    print(line)
    for i in range(tree.num_children):
        print_tree(tree.children[i], level+1)


def load_doc(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    return [l.strip() for l in lines] # delete space


def load_sent(file_name):
    docs = []
    doc = []
    with open(file_name) as f:
        lines = f.readlines()
    for line in lines:
        if line == "******\n":
            docs.append(doc)
            # docs += doc
            doc = []
        else:
            doc.append(line.strip())
    return docs



def list2doc(docs):
    """
        convert a list of sentences to a single document
        """
    doc_docs = []
    s = ''
    for doc in docs:
        if isinstance(doc, tuple):
            doc = doc[0]
        else:
            doc = doc
        for sent in doc:
            s += sent + ' '
        doc_docs.append(s)
        s = ''
    return doc_docs


def list2file(docs, file):
    with open(file, 'a') as f:
        for doc in docs:
            f.write(doc[0])
            f.write('\n')


if __name__ == '__main__':
    print()


