import copy
from event_chain_threads import EventChainBuilder as ECB1
from event_chain import EventChainBuilder as ECB2

def test():
    '''
    ecb1 = ECB1()

    ecb1.load_verts("/homes/du113/scratch/test-11k/bbc0_1_verts.pkl")
    ecb1.load_corefs("/homes/du113/scratch/test-11k/bbc0_1_corefs.pkl")
    # print(ecb1.event_chains)

    ecb1.make_event_chains()
    with open('echain1_out.txt', 'a') as f:
        for chains in ecb1.event_chains:
            for item in chains.items():
                print(item, file=f)
    
    '''
    ecb2 = ECB2()

    ecb2.load_verts("/homes/du113/scratch/test-11k/bbc0_1_verts.pkl")
    for i, v in enumerate(ecb2.vert_ls):
        ecb2.vert_ls[i].r_map = list(range(v.range[0], v.range[1]+1))

    ecb2.load_corefs("/homes/du113/scratch/test-11k/bbc0_1_corefs.pkl")

    # ecb2.debug = True
    ecb2.make_event_chains()
    with open('echain3_out.txt', 'a') as f:
        for chains in ecb2.event_chains:
            for item in chains.items():
                print(item, file=f)


test()
