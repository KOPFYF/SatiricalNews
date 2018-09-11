from graph import Vertex

def convert_range(v, r):
    print(v.r_map, r)
    start, end = -1, -1

    for i, m in enumerate(v.r_map):
        if m == r[0] and start < 0:
            start = i
        if m <= r[1]:
            end = i

    print(start, end)
    
    assert start >= 0 and end >= 0
    return (start, end)

def sub_ent(vertex, head, r1, r2):
    r1 = convert_range(vertex, r1)
    r2 = convert_range(head, r2)
    print('******')
    print('substituting', r1, 'in', vertex.debug())
    print('with', r2, 'in', head.debug())
    # starting point in vertex
    # v_offset = vertex.range[0]
    
    # starting point in head
    # h_offset = head.range[0]

    print('original v', vertex.val)

    # modified aug 16th, use the original val of the head
    # so that the range always stays consistent
    h_idx = head.val[r2[0]:r2[1]+1]   # this is the source indices
    print('head', h_idx)

    ori_val = vertex.val

    vertex.val = ori_val[:r1[0]] + h_idx    # this is the target
    if r1[1] < len(vertex.r_map) - 1:
        vertex.val += ori_val[r1[1]+1:]

    print('final v', vertex.val)
    print('******')

    ori_map = vertex.r_map
    vertex.r_map = ori_map[:r1[0]]
    print(vertex.r_map)
    vertex.r_map += [ori_map[r1[0]] for _ in range(r2[0], r2[1]+1)]
    print(vertex.r_map)
    vertex.r_map += ori_map[r1[1]+1:]
    print(vertex.r_map)


def create_vertices():
    v1 = Vertex([12,23],'')
    v1.words = 'Mike Pence'
    v1.set_range((10,11))
    # v1.ori_val = copy.deepcopy(v1.val)
    v1.r_map = list(range(10,12))

    v2 = Vertex([18,50,79,28],'')
    v2.words = 'First Lady Melane Trump'
    v2.set_range((18,21))
    # v2.ori_val = copy.deepcopy(v2.val)
    v2.r_map = list(range(18,22))

    v3 = Vertex([1,2,3,4],'')
    v3.words = 'He bought her flowers'
    v3.set_range((50,53))
    # v3.ori_val = copy.deepcopy(v3.val)
    v3.r_map = list(range(50,54))

    v4 = Vertex([101,95,81,27,58,66,71],'')
    v4.words = 'but this was seen by donald trump'
    v4.set_range((201,207))
    # v4.ori_val = copy.deepcopy(v3.val)
    v4.r_map = list(range(201,208))

    return v1, v2, v3, v4

def test():
    v1, v2, v3, v4 = create_vertices()

    sub_ent(v3, v1, [50,50], [10,11])

    sub_ent(v3, v2, [52,52], [18,21])

    sub_ent(v4, v3, [202,202], [50,53])
test()
