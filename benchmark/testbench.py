import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

from benchmark import BenchmarkData

def run_plaid_evaluation(collection, dataset, split, k, nbits=2):
    config = ColBERTConfig(
        root="ColBERTv2/PLAID",
        nbits=nbits,
    )
    data = BenchmarkData(collection=collection, dataset=dataset, split=split)
    with Run().context(RunConfig(nranks=0, experiment="eval")):
        searcher = Searcher(index=f"{dataset}.split={split}.nbits={nbits}", config=config)
        rankings = searcher.search_all(data.queries, k=k)
    metrics = data.submit(rankings, k=k)

    return metrics