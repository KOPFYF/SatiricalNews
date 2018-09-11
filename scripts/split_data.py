import os

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


if __name__ == '__main__':
    srcfile = '/homes/du113/scratch/11k-data/uk/true/BBC_2018_7_25.txt'
    dest = '/homes/du113/scratch/real-bbc'
    
    all_docs = load_sent(srcfile)
    small_doc = []

    for i, doc in enumerate(all_docs):
        small_doc += doc

        if i % 150 == 149:
            with open(os.path.join(dest, ('%d_BBCNews.txt' % (i // 150))), 'w') as fid:
                fid.write('\n'.join(small_doc))
            small_doc = []

    if small_doc:
        with open(os.path.join(dest, ('%d_BBCNews.txt' % (i // 150))), 'w') as fid:
            fid.write('\n'.join(small_doc))
