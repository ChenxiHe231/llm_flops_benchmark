#!/usr/bin/env python
"""MFU + bandwidth utilization on B200.

Usage:  python mfu.py --op q_b --M 4096
Reports latency, TFLOP/s, MFU%, GB/s, BW-util%, arithmetic intensity (FLOP/B),
and flags the meaningful metric (compute-bound -> MFU, memory-bound -> BW-util)
via the roofline ridge point.

B200 peaks (dense; HGX B200 180GB 1000W):
  FP8 e4m3 tensor core = 4.5 PFLOP/s   (NOT the 9 PF sparse number)
  BF16 tensor core     = 2.25 PFLOP/s
  HBM3e bandwidth      = 7.7 TB/s
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import specs
from harness.timing import time_callable
from harness.loader import load_candidate

PEAK_FP8 = 4.5e15
PEAK_BF16 = 2.25e15
PEAK_HBM = 7.7e12


def report(tag, ms, flops, mem_bytes, peak_flops):
    sec = ms * 1e-3
    achieved_flops = flops / sec
    achieved_bw = mem_bytes / sec
    mfu = achieved_flops / peak_flops
    bwu = achieved_bw / PEAK_HBM
    intensity = flops / mem_bytes
    ridge = peak_flops / PEAK_HBM
    bound = "compute" if intensity >= ridge else "memory"
    print(f"  [{tag}] {ms:.4f} ms | {achieved_flops/1e12:8.1f} TFLOP/s  MFU={mfu*100:5.1f}% "
          f"| {achieved_bw/1e9:8.1f} GB/s  BW={bwu*100:5.1f}% "
          f"| FLOP/B={intensity:7.1f} (ridge={ridge:.0f}, {bound}-bound)")
    headline = "MFU" if bound == "compute" else "BW-util"
    print(f"        -> {bound}-bound: headline metric = {headline}"
          + (" (small-M decode: also latency-bound, low MFU expected)"
             if bound == "memory" else ""))


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

    flops = specs.flops(args.op, phase, args.M, args.S)
    mem_bytes = specs.bytes_(args.op, phase, args.M, args.S)
    peak = PEAK_FP8 if meta["peak_dtype"] == "fp8" else PEAK_BF16

    print(f"op={args.op} phase={phase} M={args.M} S={args.S} "
          f"peak_dtype={meta['peak_dtype']} (peak={peak/1e15:.2f} PF, HBM={PEAK_HBM/1e12:.1f} TB/s)")
    print(f"  FLOPs={flops/1e9:.2f} G   mem={mem_bytes/1e6:.2f} MB")

    ref_ms = time_callable(lambda: specs.reference(args.op, phase, inputs))
    report("backend", ref_ms, flops, mem_bytes, peak)
    if src != "reference":
        cand_ms = time_callable(lambda: run_cand(inputs))
        report(f"candidate", cand_ms, flops, mem_bytes, peak)


if __name__ == "__main__":
    main()
