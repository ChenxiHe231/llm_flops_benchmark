#!/usr/bin/env python
"""Latency: CUDA-graph timing of the real backend (baseline) vs candidate.

Usage:  python latency.py --op dsa_attn --M 32
If no tasks/{op}/{phase}/impl.py exists, candidate == reference (baseline only).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import specs
from harness.timing import time_callable
from harness.loader import load_candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=specs.ALL_OPS)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--S", type=int, default=specs.DEFAULT_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    phase = specs.infer_phase(args.M)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    inputs = specs.build_inputs(args.op, phase, args.M, args.S, device, args.seed)
    run_cand, src = load_candidate(args.op, phase)

    ref_ms = time_callable(lambda: specs.reference(args.op, phase, inputs))
    print(f"op={args.op} phase={phase} M={args.M} S={args.S}")
    print(f"backend (reference): {ref_ms:.4f} ms")
    if src != "reference":
        cand_ms = time_callable(lambda: run_cand(inputs))
        speedup = ref_ms / cand_ms if cand_ms > 0 else float("nan")
        print(f"candidate ({src}): {cand_ms:.4f} ms")
        print(f"speedup vs backend: {speedup:.3f}x")
    else:
        print("candidate: (none — showing backend baseline only)")


if __name__ == "__main__":
    main()
