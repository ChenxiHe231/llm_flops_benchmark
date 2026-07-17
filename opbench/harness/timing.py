"""Timing harnesses for opbench.

Two methods:
  - time_callable   : CUDA-graph, warm-L2, mean. Fast sanity (original opbench).
  - time_cold_l2    : cold-L2 flush + inputs cloned per iter + median CUDA-event.
                      Mirrors kernel-harness (testbench/harness/timing.py) so numbers
                      are comparable to the best-kernels reward bench. This is the
                      more honest metric for memory-bound kernels (weights are cold
                      in real serving, not warm-cached from a prior identical call).
"""
import statistics
import torch

NUM_WARMUP = 5
NUM_RUNS = 20


def time_callable(fn, num_warmup=NUM_WARMUP, num_runs=NUM_RUNS):
    """CUDA-graph warm-L2 mean ms per call. fn() runs one kernel invocation."""
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
        # Fallback (graph capture failed): brackets the whole python loop with CUDA
        # events, so per-launch dispatch overhead IS included; small-kernel latency
        # overestimated vs the graph path.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(num_runs):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_runs


# ── kernel-harness-style cold-L2 timing ──────────────────────────────────────
def _flush_buffer(device):
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    return torch.empty(int(l2 * 2), dtype=torch.int8, device=device)


def clone_inputs(inputs: dict) -> dict:
    """Fresh copies of tensor values (prevents cross-iteration cache reuse);
    non-tensors pass through unchanged. Same idea as kernel-harness clone_args."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}


def time_cold_l2(run_fn, base_inputs, warmup=10, rep=100, device="cuda"):
    """Median per-call ms, kernel-harness methodology:
      - inputs cloned fresh each iteration (no cross-iter reuse)
      - L2 flushed before each timed call (cold cache)
      - CUDA-event timed, median over `rep` reps
    run_fn(inputs_dict) executes ONE kernel invocation on the given inputs.
    Flush + clone happen OUTSIDE the timed region.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    buf = _flush_buffer(dev)

    torch.cuda.synchronize()
    for _ in range(warmup):
        ins = clone_inputs(base_inputs)
        buf.zero_()
        run_fn(ins)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        ins = clone_inputs(base_inputs)
        buf.zero_()
        torch.cuda.synchronize()
        start.record()
        run_fn(ins)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)
