import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tqdm import tqdm

from colbert.data import Queries, Ranking
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

from benchmark import BenchmarkData
from benchmark.tracker import ExecutionTracker

PLAID_STEPS = ["Query Encoding", "Candidate Generation", "Filtering", "Decompress Residuals", "Scoring", "Sorting"]

def run_plaid_evaluation(collection, dataset, split, k, nbits=2, show_progress=True):
    config = ColBERTConfig(
        root="ColBERTv2/PLAID",
        nbits=nbits,
    )
    
    data = BenchmarkData(collection=collection, dataset=dataset, split=split)
    tracker = ExecutionTracker(name="ColBERTv2/PLAID", steps=PLAID_STEPS)

    rankings = dict()
    with Run().context(RunConfig(nranks=0, experiment="eval")):
        searcher = Searcher(index=f"{dataset}.split={split}.nbits={nbits}", config=config)
        for qidx, qtext in tqdm(data.queries, disable=not show_progress):
            tracker.next_iteration()
            rankings[qidx] = searcher.search(qtext, k=k, tracker=tracker)
            tracker.end_iteration()

    for qidx in rankings:
        rankings[qidx] = list(zip(*rankings[qidx]))
    rankings = Ranking(data=rankings)

    metrics = data.submit(rankings, k=k)

    return tracker, metrics