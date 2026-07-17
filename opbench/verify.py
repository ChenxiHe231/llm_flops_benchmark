#!/usr/bin/env python
"""Correctness: run real backend + candidate on IDENTICAL frozen inputs, cosine.

Usage:  python verify.py --op fused_qkv_a --M 4096 [--seed 0]
Phase is inferred from M (>=1024 prefill, else decode).
Pass if cosine >= threshold (0.999 for gemm/moe/score, 0.99 for bmm/mla).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import specs, compare
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
    meta = specs.op_meta(args.op, phase)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    inputs = specs.build_inputs(args.op, phase, args.M, args.S, device, args.seed)
    run_cand, src = load_candidate(args.op, phase)

    ref_out = specs.reference(args.op, phase, inputs)
    cand_out = run_cand(inputs)
    cos = compare.compare(ref_out, cand_out, meta["output_kind"], inputs)

    ok = cos >= meta["threshold"]
    print(f"op={args.op} phase={phase} M={args.M} S={args.S} kind={meta['output_kind']}")
    print(f"candidate={src}")
    print(f"cosine={cos:.6f}  threshold={meta['threshold']}  -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
