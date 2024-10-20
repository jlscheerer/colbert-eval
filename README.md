# Baseline Evaluation of ColBERTv2/PLAID

> We build on the code provided by [Stanford Future Data Systems](https://github.com/stanford-futuredata/ColBERT) to evaluate ColBERTv2/PLAID. This evaluation serves as the baseline for the _highly optimized_ [XTR/WARP](https://github.com/jlscheerer/xtr-warp) retrieval engine.

## Installation

colbert-eval requires Python 3.8+, PyTorch 1.9+ and Tensorflow 2.8.2 and uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.
We evaluate XTR using the [`XTR_base` checkpoint](https://huggingface.co/google/xtr-base-en) provided on Hugging Face.

It is strongly recommended to create a [conda environment](https://docs.anaconda.com/anaconda/install/linux/#installation) using the commands below. We include the corresponding environment file (`conda_env_cpu.yml`).

```sh
conda activate colbert
```

## Building Indexes

- To obtain the necessary datasets, follow the instructions in the [main repository](https://github.com/jlscheerer/xtr-warp).
- Open `index_collection.py` and adjust the following variables: `BEIR_COLLECTION_PATH`, `LOTTE_COLLECTION_PATH` and `checkpoint`
- Finally, run the indexing script:
```sh
python index_collection.py -c COLLECTION -d DATASET -s SPLIT -n 2
```
