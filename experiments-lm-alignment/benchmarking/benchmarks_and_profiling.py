# %%
import os
from statistics import mean
import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import triton
import triton.language as tl
# from edtrace import text, link, image
# from lecture_util import get_local_url

import torch

def cuda_if_available(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def get_max_memory_usage(func):
    """Measure how much memmory calling `func` uses."""
    if not torch.cuda.is_available():
        return 0  # Can't measure it without GPUs!

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    func()
    return torch.cuda.max_memory_allocated()

def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    # Return a function to perform the operation
    return lambda : operation(x, y)

# %%

def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `func` by running it `num_trials`.  Return the average time."""
    # Warmup: first times might be slower due to compilation, etc.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = [] 
    for trial in range(num_trials):  # Do it multiple times to capture variance
        # Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # Start timing
        run()  # Actually perform computation
        end_event.record()  # End timing
        torch.cuda.synchronize()  # Wait for CUDA threads to finish
        times.append((start_event.elapsed_time(end_event)))  
    mean_time = mean(times)   
    return mean_time


# Benchmark matrix multiplication
matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
result = benchmark(matmul)  
# See how timing scales with dimension
results = {}
for dim in [256, 512, 1024, 2048, 4096, 8192]:
        results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))  

# %%
import matplotlib.pyplot as plt
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xlabel('Dimension (dim x dim)')
plt.ylabel('Average Time (ms)')
plt.title('Matrix Multiplication Time vs Dimension')
plt.xscale('log', base=2)
plt.yscale('log', base=10)
plt.grid(True, which="both", ls="--")
plt.show()  