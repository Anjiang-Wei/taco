from enum import Enum, auto
class BenchmarkKind(Enum):
    SpMV = auto()
    SpMSpV = auto()
    SpMM = auto()
    SDDMM = auto()
    SpAdd3 = auto()
    SpTTV = auto()
    SpMTTKRP = auto()
    SpInnerProd = auto()

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [str(b) for b in BenchmarkKind]

    @staticmethod
    def getByName(name):
        filtered = list(filter(lambda bk: str(bk) == name, list(BenchmarkKind)))
        assert(len(filtered) == 1)
        return filtered[0]

