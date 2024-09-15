import os
# Enforces CPU-only execution of torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import psutil

from utility.executor_utils import read_subprocess_inputs, publish_subprocess_results

if __name__ == "__main__":
    config, params = read_subprocess_inputs()

    num_threads = config["num_threads"]

    proc = psutil.Process()
    if "cpu_affinity" in params:
        # Set the cpu_affinity, e.g., [0, 1] for CPUs #0 and #1
        # Reference: https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_affinity
        proc.cpu_affinity(params["cpu_affinity"])

    # Configure environment to ensure *correct* number of threads.
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"]= str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_AFFINITY"] = "disabled"

    import torch
    torch.set_num_threads(num_threads)

    from benchmark.testbench import run_plaid_evaluation
    tracker, metrics = run_plaid_evaluation(collection=config["collection"], dataset=config["dataset"],
                                            split=config["split"], k=config["document_top_k"], nbits=config["nbits"])

    publish_subprocess_results({
        "tracker": tracker.as_dict(),
        "metrics": metrics,
    })