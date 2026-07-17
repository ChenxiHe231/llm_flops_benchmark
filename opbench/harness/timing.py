"""CUDA-graph timing harness (from bench_glm5_*.py).

time_callable: warmup -> capture NUM_RUNS iters into one CUDA graph -> time a
single replay / NUM_RUNS. Removes python/launch overhead. Falls back to plain
event timing if graph capture fails (some allocation patterns).
"""
import torch

NUM_WARMUP = 5
NUM_RUNS = 20


def time_callable(fn, num_warmup=NUM_WARMUP, num_runs=NUM_RUNS):
    """Return avg ms per call. fn() runs one kernel invocation."""
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(num_runs):
                fn()
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / num_runs
        del graph
        return avg_ms
    except Exception:
        # Fallback: non-graph event timing (still excludes python loop via sync).
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(num_runs):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_runs
