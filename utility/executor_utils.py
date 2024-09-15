import os
import json
import os
import io
import sys
import copy
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout


BEIR_DATASETS = ["nfcorpus", "scifact", "scidocs", "fiqa", "webis-touche2020", "quora"]
LOTTE_DATASETS = ["lifestyle", "writing", "recreation", "technology", "science", "pooled"]

def _make_config(collection, dataset, document_top_k, split="test", nbits=2, num_threads=1):
    assert collection in ["beir", "lotte"]
    if collection == "beir":
        assert dataset in BEIR_DATASETS
    else:
        assert dataset in LOTTE_DATASETS
    assert nbits in [2, 4]
    assert isinstance(document_top_k, int)
    assert split in ["dev", "test"]
    return {
        "collection": collection,
        "dataset": dataset,
        "document_top_k": document_top_k,
        "split": split,
        "nbits": nbits,
        "num_threads": num_threads
    }

def _expand_configs(datasets, document_top_ks, split="test", nbits=2, num_threads=None):
    if num_threads is None:
        num_threads = 1
    if not isinstance(nbits, list):
        nbits = [nbits]
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(document_top_ks, list):
        document_top_ks = [document_top_ks]
    if not isinstance(num_threads, list):
        num_threads = [num_threads]
    configs = []
    for collection_dataset in datasets:
        collection, dataset = collection_dataset.split(".")
        for document_top_k in document_top_ks:
            for nbit in nbits:
                for threads in num_threads:
                    configs.append(_make_config(collection=collection, dataset=dataset, nbits=nbit,
                                                document_top_k=document_top_k, split=split, num_threads=threads))
    return configs

def _get(config, key):
    if key in config:
        return config[key]
    return None

def _expand_configs_file(configuration_file):
    configs = configuration_file["configurations"]
    return _expand_configs(datasets=_get(configs, "datasets"), document_top_ks=_get(configs, "document_top_k"),
                           split=_get(configs, "datasplit"), nbits=_get(configs, "nbits"), num_threads=_get(configs, "num_threads"))

def _write_results(results_file, data):
    with open(results_file, "w") as file:
        file.write(json.dumps(data, indent=3))

def load_configuration(filename, overwrite=False):
    with open(filename, "r") as file:
        config_file = json.loads(file.read())
    name = config_file["name"]
    type_ = config_file["type"]
    params = _get(config_file, "parameters") or {}
    configs = _expand_configs_file(config_file)

    os.makedirs("experiments/results", exist_ok=True)
    results_file = os.path.join("experiments/results", f"{name}.json")
    assert not os.path.exists(results_file) or overwrite

    _write_results(results_file, [])
    return results_file, type_, params, configs

def _init_proc(env_vars):
    for key, value in env_vars.items():
        os.environ[key] = value

def _prepare_result(result):
    if "_update" not in result:
        return result
    # NOTE Used to update parameters determined at runtime.
    result = copy.deepcopy(result)
    update = result["_update"]
    for key, value in update.items():
        current = result["provenance"][key]
        assert current is None or current == value
        result["provenance"][key] = value
    del result["_update"]
    return result

def _execute_configs_parallel(configs, callback, type_, params, results_file, max_workers):
    env_vars = dict(os.environ)
    progress = tqdm(total=len(configs))
    results = []
    with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_proc, initargs=(env_vars,)
        ) as executor, redirect_stdout(
            io.StringIO()
        ) as rd_stdout:
        futures = {executor.submit(callback, config, params): config for config in configs}
        for future in as_completed(futures.keys()):
            result = future.result()
            config = futures[future]

            result["provenance"] = config
            result["provenance"]["type"] = type_
            result["provenance"]["parameters"] = params
            results.append(_prepare_result(result))
            _write_results(results_file=results_file, data=results)
            
            sys.stdout = sys.__stdout__
            sys.stdout = rd_stdout
            progress.update(1)
    progress.close()

def _execute_configs_sequential(configs, callback, type_, params, results_file):
    results = []
    for config in tqdm(configs):
        result = callback(config, params)
        result["provenance"] = config
        result["provenance"]["type"] = type_
        result["provenance"]["parameters"] = params
        results.append(_prepare_result(result))
        _write_results(results_file=results_file, data=results)

def execute_configs(exec_info, configs, results_file, type_, params, max_workers):
    exec_info = exec_info[type_]
    callback, parallelizable = exec_info["callback"], exec_info["parallelizable"]
    if parallelizable:
        _execute_configs_parallel(configs, callback, type_, params, results_file, max_workers=max_workers)
    else:
        _execute_configs_sequential(configs, callback, type_, params, results_file)

def read_subprocess_inputs():
    data = json.loads(input())
    return data["config"], data["params"]

def publish_subprocess_results(results):
    print("")
    print(json.dumps(results))
    print("#> Done")

def spawn_and_execute(script, config, params):
    process = subprocess.run(
        ["python", script],
        input=json.dumps({"config": config, "params": params}),
        stdout=subprocess.PIPE, 
        bufsize=1,
        text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()},
        cwd=os.getcwd()
    )
    response = process.stdout.strip().split("\n")
    if response[-1] != "#> Done" or process.returncode != 0:
        print(process.stderr, file=sys.stderr)
    assert response[-1] == "#> Done"
    return json.loads(response[-2])