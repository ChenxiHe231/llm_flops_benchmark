#!/usr/bin/env python
"""Latency: real backend (baseline) vs candidate, single operator.

Usage:  python latency.py --op dsa_attn --M 32 [--timing kh|graph]
Default timing = kh (kernel-harness style: cold-L2 flush + inputs cloned per
iter + median CUDA-event), comparable to the best-kernels reward bench.
--timing graph = original CUDA-graph warm-L2 mean (fast sanity).
If no tasks/{op}/{phase}/impl.py exists, candidate == reference (baseline only).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import specs
from harness.timing import time_cold_l2, time_callable
from harness.loader import load_candidate


def _time(run_fn, inputs, mode):
    if mode == "graph":
        return time_callable(lambda: run_fn(inputs))
    return time_cold_l2(run_fn, inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=specs.ALL_OPS)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--S", type=int, default=specs.DEFAULT_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--timing", choices=["kh", "graph"], default="kh",
                    help="kh = kernel-harness cold-L2 median (default); graph = CUDA-graph warm mean")
    args = ap.parse_args()

    phase = specs.infer_phase(args.M)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    inputs = specs.build_inputs(args.op, phase, args.M, args.S, device, args.seed)
    run_cand, src = load_candidate(args.op, phase)

    ref_ms = _time(lambda ins: specs.reference(args.op, phase, ins), inputs, args.timing)
    print(f"op={args.op} phase={phase} M={args.M} S={args.S} timing={args.timing}")
    print(f"backend (reference): {ref_ms:.4f} ms")
    if src != "reference":
        cand_ms = _time(lambda ins: run_cand(ins), inputs, args.timing)
        speedup = ref_ms / cand_ms if cand_ms > 0 else float("nan")
        print(f"candidate ({src}): {cand_ms:.4f} ms")
        print(f"speedup vs backend: {speedup:.3f}x")
    else:
        print("candidate: (none — showing backend baseline only)")


if __name__ == "__main__":
    main()
