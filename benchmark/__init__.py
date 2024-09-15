from .beir import BEIRBenchmarkData
from .lotte import LoTTEBenchmarkData

class BenchmarkData:
    def __new__(cls, collection, dataset, split):
        if collection == "lotte":
            return LoTTEBenchmarkData(collection, dataset, split)
        elif collection == "beir":
            return BEIRBenchmarkData(collection, dataset, split)
        assert False