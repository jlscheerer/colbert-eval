import os
import json
from collections import defaultdict

import pytrec_eval

from colbert.data import Queries

BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]

def eval_metrics_beir(qrels, dict_rankings):
    K_VALUES = [5, 10, 50, 100]
    METRIC_NAMES = ['ndcg_cut', 'map_cut', 'recall']

    measurements = []
    for metric_name in METRIC_NAMES:
        measurements.append(
            f"{metric_name}." + ",".join([str(k) for k in K_VALUES])
        )
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measurements)
    final_scores = evaluator.evaluate(dict_rankings)

    final_metrics = dict()
    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = 0.0

    for query_id in final_scores.keys():
        for metric_name in METRIC_NAMES:
            for k in K_VALUES:
                final_metrics[f"{metric_name}@{k}"] += final_scores[query_id][
                    f"{metric_name}_{k}"
                ]

    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = round(
                final_metrics[f"{metric_name}@{k}"] / len(final_scores), 5
            )

    return final_metrics

class BEIRBenchmarkData:
    def __init__(self, collection, dataset, split):
        assert collection == "beir"
        
        questions_path = os.path.join(BEIR_COLLECTION_PATH, f"{dataset}/questions.{split}.tsv")
        self.queries, self.qidx_to_id = BEIRBenchmarkData._load_and_remap_queries(questions_path)

        collection_map_path = os.path.join(BEIR_COLLECTION_PATH, f"{dataset}/collection_map.json")
        self.collection_map = BEIRBenchmarkData._load_collection_map(collection_map_path)

        qrels_path = os.path.join(BEIR_COLLECTION_PATH, f"{dataset}/qrels.{split}.json")
        self.qrels = BEIRBenchmarkData._load_qrels(qrels_path)
    
    @staticmethod
    def _load_and_remap_queries(filename):
        queries, idx_to_id = dict(), dict()
        with open(filename, "r") as file:
            for qidx, line in enumerate(file.readlines()):
                qid, query, *_ = line.strip().split('\t')
                queries[qidx] = query
                idx_to_id[qidx] = qid
        return Queries(data=queries), idx_to_id

    @staticmethod
    def _load_collection_map(filename):
        with open(filename, "r") as file:
            collection_map = json.loads(file.read())
        return collection_map

    @staticmethod
    def _load_qrels(filename):
        with open(filename, "r") as file:
            qrels = json.loads(file.read())
        return qrels
    
    def _apply_collection_map(self, rankings):
        results = []
        for collection_doc_id, position, score in rankings:
            results.append((self.collection_map[str(collection_doc_id)], score))
        return results
    
    def submit(self, rankings, k=100):
        if not isinstance(rankings, dict):
            rankings = rankings.data
        dict_rankings = defaultdict(dict)
        for qidx, results in rankings.items():
            assert len(results) <= k
            for docid, score in self._apply_collection_map(results):
                dict_rankings[self.qidx_to_id[qidx]][str(docid)] = score
        return eval_metrics_beir(self.qrels, dict(dict_rankings))

