import argparse
import psutil
import os
from dotenv import load_dotenv

from utility.executor_utils import load_configuration, execute_configs, spawn_and_execute

def latency(config, params):
    NUM_RUNS = params.get("num_runs", 3)
    assert NUM_RUNS > 0
    results = []
    for _ in range(NUM_RUNS):
        results.append(spawn_and_execute("utility/latency_runner.py", config, params))
    metrics = results[0]["metrics"]
    assert all(x["metrics"] == metrics for x in results)
    return {
        "metrics": metrics,
        "tracker": [x["tracker"] for x in results]
    }

def metrics(config, params):
    run = spawn_and_execute("utility/latency_runner.py", config, params)
    return {
        "metrics": run["metrics"],
    }

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog='XTR/WARP Experiment [Executor/Platform]')
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers or psutil.cpu_count(logical=False)
    OVERWRITE = args.overwrite
    results_file, type_, params, configs = load_configuration(args.config, overwrite=OVERWRITE)

    EXEC_INFO = {
        "latency": {"callback": latency, "parallelizable": False},
        "metrics": {"callback": metrics, "parallelizable": True}
    }
    execute_configs(EXEC_INFO, configs, results_file=results_file, type_=type_,
                    params=params, max_workers=MAX_WORKERS)
