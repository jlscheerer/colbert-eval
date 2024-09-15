import os
os.environ["LD_LIBRARY_PATH"] = f"/future/u/scheerer/miniconda3/envs/gcc/include/:/usr/local/cuda-12.2/lib64"

import argparse

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

BEIR_COLLECTION_PATH = "/lfs/1/scheerer/datasets/beir/datasets"
LOTTE_COLLECTION_PATH = "/lfs/1/scheerer/datasets/lotte/lotte/"

checkpoint = "/future/u/scheerer/home/models/colbertv2.0"

def _index_collection(dataset, split, nbits, collection_path):
    with Run().context(RunConfig(nranks=4, experiment="eval")):
        config = ColBERTConfig(
            root="ColBERTv2/PLAID",
            nbits=nbits,
            doc_maxlen=300,
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=f"{dataset}.split={split}.nbits={nbits}", collection=collection_path)

def _index_beir_collection(dataset, split, nbits):
    collection_path = os.path.join(BEIR_COLLECTION_PATH, f"{dataset}/collection.tsv")
    _index_collection(dataset, split, nbits, collection_path)

def _index_lotte_collection(dataset, split, nbits):
    collection_path = os.path.join(LOTTE_COLLECTION_PATH, f"{dataset}/{split}/collection.tsv")
    _index_collection(dataset, split, nbits, collection_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="collection_indexer")

    parser.add_argument("-c", "--collection", choices=["beir", "lotte"])
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-s", "--split", choices=["train", "test", "dev"])
    parser.add_argument("-n", "--nbits", type=int, choices=[1, 2, 4, 8])
    args = parser.parse_args()

    if args.collection == "beir":
        _index_beir_collection(args.dataset, args.split, args.nbits)
    elif args.collection == "lotte":
        _index_lotte_collection(args.dataset, args.split, args.nbits)
