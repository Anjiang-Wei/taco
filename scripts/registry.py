class SparseTensor:
    def __init__(self, name, dims, nnz):
        self.name = name
        self.order = len(dims)
        self.dims = dims
        self.nnz = nnz

class SparseTensorRegistry:
    def __init__(self):
        self.tensors = []
    def add(self, tensor):
        self.tensors.append(tensor)
    def addAll(self, tensors):
        self.tensors += tensors
    def getAllNames(self, filterFunc=None):
        if filterFunc is not None:
            return [t.name for t in self.tensors if filterFunc(t)]
        else:
            return [t.name for t in self.tensors]
    def getAll(self, filterFunc=None):
        if filterFunc is not None:
            return [t for t in self.tensors if filterFunc(t)]
        else:
            return self.tensors
    def getByName(self, name):
        filtered = list(filter(lambda tensor: tensor.name == name, self.tensors))
        assert(len(filtered) == 1)
        return filtered[0]
    @staticmethod
    def initialize():
        registry = SparseTensorRegistry()
        # TODO (rohany): Register all of the sparse tensors that we are considering.
        registry.addAll([
            # 3-tensors. Only the nell-2 and patents tensor can be loaded in
            # CTF. The other tensors are too large dimension-wise.
            SparseTensor("amazon-reviews", [4821207, 1774269, 1805187], 1741809018),
            SparseTensor("freebase_music", [23344784, 23344784, 166], 99546551),
            SparseTensor("freebase_sampled", [38955429, 38955429, 532], 139920771),
            SparseTensor("nell-1", [2902330, 2143368, 25495389], 143599552),
            SparseTensor("nell-2", [12092, 9184, 28818], 76879419),
            # TODO (rohany): We'll want to load patents into a DDS format for some benchmarks.
            SparseTensor("patents", [46, 239172, 239172], 3596640708),
            SparseTensor("reddit-2015", [8211298, 176962, 8116559], 4687474081),
            # Matrices.
            SparseTensor("arabic-2005", [22744080, 22744080], 639999458),
            SparseTensor("it-2004", [41291594, 41291494], 1150725436),
            SparseTensor("kmer_A2a", [170728175, 170728175], 360585172),
            SparseTensor("kmer_V1r", [214005017, 214005017], 465410904),
            # This tensor is too large (dimension-wise) to be represented in
            # PETSc and Trilinos with 32-bit indexing.
            # SparseTensor("mawi_201512020330", [226196185, 226196185], 480047894),
            SparseTensor("mycielskian19", [393215, 393215], 903194710),
            # nlpkkt200 is too small to be an interesting problem.
            # SparseTensor("nlpkkt200", [16240000, 16240000], 440225632),
            SparseTensor("nlpkkt240", [27993600, 27993600], 760648352),
            SparseTensor("sk-2005", [50636154, 50636154], 1949412601),
            SparseTensor("twitter7", [41652230, 41652230], 1468365182),
            SparseTensor("uk-2005", [39459925, 39459925], 936364282),
            SparseTensor("webbase-2001", [118142155, 118142155], 1019903190),
        ])
        return registry
