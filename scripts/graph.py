class Graph:
    def __init__(self):
        self.vertices = [] # list of all the vertices in the graph

    def add_vertex(self, v):
        self.vertices.append(v) # add a vertex to the list


class Vertex:
    words = ''  # for debugging

    def __init__(self, x, type):
        self.val = x # list of word indices 
        # self.edges = [] # list of adjacent edges
        self.type = type
        # self.path = [] # path contains that vertex
        self.idx_path = [] # path contains that vertex idx
        # self.word_idx = [] 
        self.range = None

        self.r_map = None 

    def is_(self, s):
        return self.val == s

    def set_range(self, r):
        self.range = r
        self.r_map = list(range(r[0], \
                r[1]+1))

    def __gt__(self, other):
        return self.range[0] > other.range[1]

    def __lt__(self, other):
        return self.range[1] < other.range[0]

    def includes(self, r):
        return r[0]>= self.range[0] and r[1]<= self.range[1]

    def before(self, r):
        return self.range[1] < r[0]

    def after(self, r):
        return self.range[0] > r[1]

    def __repr__(self):
        '''
        return "{}('{}',[{}],type={})".format(self.__class__.__name__, self.val, str(self.range),self.type)
        '''
        return self.debug()

    def debug(self):
        return "{}({},'{}',[{}],type={})".format(self.__class__.__name__, self.val, self.words, str(self.range),self.type)

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return hash(self.range)


class Edge:
    def __init__(self, src, dest, type):
        '''
        @params:
            - src: the source vertex
            - dest: the destination vertex
            - type: the type of the edge (string)
        '''
        self.src = src
        self.dest = dest
        self.type = type

    def __repr__(self):
        return "{}('{}':{}=>{})".format(self.__class__.__name__, self.type, str(self.src), str(self.dest))
