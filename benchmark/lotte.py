import os
import jsonlines

from colbert.data import Queries

LOTTE_COLLECTION_PATH = "/lfs/1/scheerer/datasets/lotte/lotte/"

class LoTTEBenchmarkData:
    def __init__(self, collection, dataset, split):
        assert collection == "lotte"
        self.queries = Queries(os.path.join(LOTTE_COLLECTION_PATH, dataset, split, "questions.search.tsv"))
        self.qas = LoTTEBenchmarkData._load_lotte_qas(os.path.join(LOTTE_COLLECTION_PATH, dataset, split, "qas.search.jsonl"))

    @staticmethod
    def _load_lotte_qas(filename):
        num_lines = 0
        qas = dict()
        with jsonlines.open(filename) as reader:
            for obj in reader:
                qas[obj["qid"]] = set(obj["answer_pids"])
                num_lines += 1
        assert len(qas) == num_lines
        return qas
    
    @staticmethod
    def _success_at_k_lotte(expected, rankings, k):
        num_total_qids, success = 0, 0
        for qid, answer_pids in expected.items():
            num_total_qids += 1
            if qid not in rankings.data:
                print(f"WARNING: qid {qid} not found in {rankings}!", file=sys.stderr)
                continue
            qid_rankings = set(map(lambda x: x[0], rankings.data[qid][:k]))
            if len(qid_rankings.intersection(answer_pids)) > 0:
                success += 1
        return success / num_total_qids
    
    @staticmethod
    def _recall_at_k_lotte(expected, rankings, k):
        avg, num_relevant = 0, 0
        for qid, answer_pids in expected.items():
            if qid not in rankings.data:
                print(f"WARNING: qid {qid} not found in {rankings}!", file=sys.stderr)
                continue
            relevant_count = len(answer_pids)
            if relevant_count == 0:
                continue
            num_relevant += 1
            qid_rankings = set(map(lambda x: x[0], rankings.data[qid][:k]))
            correct_count = len(answer_pids & qid_rankings)
            avg += correct_count / relevant_count
        return avg / num_relevant
    
    def submit(self, rankings, k=100):
        K_VALUES = [5, 10, 100, 1000]
        final_metrics = dict()
        for k in K_VALUES:
            final_metrics[f"success@{k}"] = LoTTEBenchmarkData._success_at_k_lotte(expected=self.qas, rankings=ranking, k=k)
            final_metrics[f"recall@{k}"] = LoTTEBenchmarkData._recall_at_k_lotte(expected=self.qas, rankings=ranking, k=k)
        return final_metrics