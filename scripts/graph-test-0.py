from srl_parse import *

def test_path_0():
    with open('../datasets/trump_uk.txt') as f:
        lines = f.readlines()

    # discard the last line
    lines = lines[:-1]

    # for each line, run thru build_path()
    for line in lines:
        res = build_path(line)
        
        # print(res)

        # res[0] should be a predicate
        assert res[0].degrees() == 2, "degree of predicate is incorrect"

        assert res[1].degrees() == 1
        assert res[1].edges[0].type == 'PRED'
        assert res[2].degrees() == 1
        assert res[2].edges[0].type == 'PRED'

        print('test 0 passed')

def test_get_coref():
    with open('../datasets/dummy-test0.txt') as f:
        doc = f.read()
    res = get_coref(doc)
    print(list(res))
    

if __name__ == '__main__':
    test_path_0()
    # test_get_coref()
