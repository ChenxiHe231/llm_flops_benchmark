#!/usr/bin/env python
"""allLatency: full-layer latency, like bench_glm5_{decode,prefill}.py but with
candidate kernels swapped in where available.

Runs ALL opbench operators for one phase (phase inferred from M), times each with
the backend AND with the candidate (if tasks/{op}/{phase}/impl.py exists), prints a
per-op table sorted by latency + the layer TOTAL, and the end-to-end delta from
swapping in candidates.

Usage:  python allLatency.py --M 32                  # decode, all candidates active
        python allLatency.py --M 32 --op o_proj      # only o_proj uses candidate, rest backend
        python allLatency.py --M 4096 --timing graph --rep 30

--op OP : if given, ONLY that op uses its candidate (the rest use backend) — lets
          you isolate one candidate's end-to-end impact. If omitted, every op that
          has a candidate uses it.

Timing default = kh (kernel-harness cold-L2 + median), comparable to the
best-kernels reward bench. Coverage note: these are opbench's 12 ops; the real
DSA layer also has index_weights_proj (bf16 gemm), not modeled here — so the
TOTAL is the sum over these 12, not a byte-perfect full-layer figure.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import specs
from harness.timing import time_cold_l2, time_callable
from harness.loader import load_candidate

CAT = {"gemm": "GEMM", "bmm": "BMM", "moe": "MoE", "mla": "MLA", "score": "Score"}


def _time(run_fn, inputs, mode, rep):
    if mode == "graph":
        return time_callable(lambda: run_fn(inputs))
    return time_cold_l2(run_fn, inputs, rep=rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--op", choices=specs.ALL_OPS, default=None,
                    help="if set, ONLY this op uses its candidate; rest use backend")
    ap.add_argument("--S", type=int, default=specs.DEFAULT_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--timing", choices=["kh", "graph"], default="kh")
    ap.add_argument("--rep", type=int, default=50, help="reps for kh timing (default 50)")
    args = ap.parse_args()

    phase = specs.infer_phase(args.M)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print("=" * 92)
    print(f"GLM-5 {phase.upper()} layer latency (candidates swapped in)  "
          f"M={args.M} S={args.S} timing={args.timing}"
          + (f" rep={args.rep}" if args.timing == "kh" else "")
          + (f"  [only --op {args.op} uses candidate]" if args.op else ""))
    print("=" * 92)
    print(f"  {'op':<18s} {'cat':<6s} {'backend(ms)':>12s} {'candidate(ms)':>14s} "
          f"{'used':>9s} {'speedup':>9s}")
    print("  " + "-" * 88)

    rows = []
    for op in specs.ALL_OPS:
        fam = specs.family(op)
        inputs = specs.build_inputs(op, phase, args.M, args.S, device, args.seed)
        run_cand, src = load_candidate(op, phase)
        # --op given: only that op may use its candidate; force backend for the rest.
        use_candidate = (src != "reference") and (args.op is None or op == args.op)
        try:
            back_ms = _time(lambda ins: specs.reference(op, phase, ins), inputs, args.timing, args.rep)
        except Exception as e:
            print(f"  {op:<18s} {CAT[fam]:<6s}   backend FAILED: {e}")
            continue
        if use_candidate:
            try:
                cand_ms = _time(lambda ins: run_cand(ins), inputs, args.timing, args.rep)
            except Exception as e:
                cand_ms = None
                print(f"  {op:<18s} candidate FAILED: {e}")
        else:
            cand_ms = None
        rows.append(dict(op=op, cat=CAT[fam], back=back_ms, cand=cand_ms,
                         has_cand=use_candidate))

    # sort by backend latency descending (like bench)
    rows.sort(key=lambda r: r["back"], reverse=True)
    total_back = 0.0
    total_used = 0.0  # candidate where available, else backend
    for r in rows:
        used_ms = r["cand"] if (r["cand"] is not None) else r["back"]
        total_back += r["back"]
        total_used += used_ms
        cand_str = f"{r['cand']:.4f}" if r["cand"] is not None else "-"
        used_tag = "cand" if r["has_cand"] and r["cand"] is not None else "backend"
        spd = (r["back"] / r["cand"]) if (r["cand"] and r["cand"] > 0) else None
        spd_str = f"{spd:.2f}x" if spd else "-"
        print(f"  {r['op']:<18s} {r['cat']:<6s} {r['back']:>12.4f} {cand_str:>14s} "
              f"{used_tag:>9s} {spd_str:>9s}")

    print("  " + "-" * 88)
    n_cand = sum(1 for r in rows if r["has_cand"] and r["cand"] is not None)
    print(f"  layer TOTAL (all backend):        {total_back:>10.4f} ms")
    print(f"  layer TOTAL (candidates swapped): {total_used:>10.4f} ms   "
          f"({n_cand} candidate(s) active)")
    if total_used > 0 and total_back > 0:
        delta = (total_back - total_used) / total_back * 100
        print(f"  end-to-end layer speedup:         {total_back/total_used:.4f}x  "
              f"({delta:+.2f}%)")
    print("=" * 92)
    print("note: 12 opbench ops; index_weights_proj (bf16) not modeled. "
          "Verify candidate correctness with verify.py before trusting a speedup.")


if __name__ == "__main__":
    main()
