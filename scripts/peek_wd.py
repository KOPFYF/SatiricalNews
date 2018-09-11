import pickle
WORD_DICT="/homes/du113/scratch/170k/word_dict.pkl"

with open(WORD_DICT, 'rb') as f:
    wd = pickle.load(f)

# reverse_dict = {v:k for k, v in wd.items()}
print(wd.keys())

'''
print(reverse_dict[301])
print(reverse_dict[28])
print(reverse_dict[119])
print(reverse_dict[14])
print(reverse_dict[11])
print(reverse_dict[301])
print(reverse_dict[54])
print(reverse_dict[635])
print(reverse_dict[158])
print(reverse_dict[885])
print(reverse_dict[16])
print(reverse_dict[25])
'''
'''
print('before')
print(reverse_dict[17478])
print(reverse_dict[65832])
print(reverse_dict[16])
print(reverse_dict[16722])
print('after')
print(reverse_dict[2])
print(reverse_dict[35641])
'''
