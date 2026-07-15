import os
import time

import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from src.inference.inference import PolarizationPredictor

def run_benchmark(predictor, features, W):
    i = 0
    n_windows = len(features) - W
    times = []
    cuda = predictor.device.type == 'cuda'
    for _ in tqdm.tqdm(range(n_windows), desc="Inferencing"):
        window = features[i:i+W]
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predictor.predict(window)
        if cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        i += 1

    inference_time_mean = np.mean(times)
    inference_time_std = np.std(times)
    inference_time_median = np.median(times)

    print(f"Inference time mean: {inference_time_mean:.6f} s")
    print(f"Inference time std: {inference_time_std:.6f} s")
    print(f"Inference time median: {inference_time_median:.6f} s")

    return times


def main():
    model_path = "results/best_model_MAMBA.pt"
    predictor_gpu = PolarizationPredictor(model_path, device="cuda")
    predictor_cpu = PolarizationPredictor(model_path, device="cpu")
    W = predictor_gpu.window_size


    data_path = "data/chicago_loop/txp_1551.5_pax_1556.5_fiber_loop.mat"

    if data_path is None:
            print(f"Dataset does not exist for specific data path")
            exit(1)
            
    if not os.path.exists(data_path):
        print(f"File not found at {data_path}, exiting...")
        exit(1)
    else:
        print(f"Loading data from {data_path}")
        mat_data = sio.loadmat(data_path)


    s1_txp = mat_data['s1_txp'].flatten()
    s2_txp = mat_data['s2_txp'].flatten()
    s3_txp = mat_data['s3_txp'].flatten()

    features = np.column_stack([s1_txp, s2_txp, s3_txp])

    times_cpu = run_benchmark(predictor_cpu, features, W)
    times_gpu = run_benchmark(predictor_gpu, features, W)


    _, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.plot(times_gpu[1:], 'r.', alpha=0.5, label="GPU")
    ax1.plot(times_cpu[1:], 'b.', alpha=0.5, label="CPU")
    ax1.axhline(np.median(times_gpu[1:]), color='r', linestyle='--', label=f"Median: {np.median(times_gpu[1:]):.6f} s")
    ax1.axhline(np.median(times_cpu[1:]), color='b', linestyle='--', label=f"Median: {np.median(times_cpu[1:]):.6f} s")
    ax1.set_title("Inference Time")
    ax1.set_xlabel("Inference Number")
    ax1.set_ylabel("Time (s)")
    ax1.grid(True)
    ax1.legend()
    ax2.hist(times_gpu[1:], bins=50, alpha=0.5, label="GPU")
    ax2.hist(times_cpu[1:], bins=50, alpha=0.5, label="CPU")
    ax2.set_title("Inference Time Distribution")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)
    ax2.legend()
    plt.show()
    plt.savefig("inference_time.png")


if __name__ == "__main__":
    main()
