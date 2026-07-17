"""TEMPLATE for a candidate operator implementation.

Copy to  tasks/{operator}/{phase}/impl.py  and implement `run`.

Contract:
  - `run(inputs)` receives the SAME frozen input dict the real backend gets
    (already quantized to fp8 + scales). DO NOT re-quantize, re-seed, or build
    new random data — that would break the fair comparison.
  - Return the raw output tensor(s), in the same form the backend returns:
      gemm/bmm/moe  -> a single tensor
      dsa_attn      -> a tuple (output, max_logits, lse)  (harness compares output)
      index_score   -> the logits tensor
  - The harness applies masking (masked_m for MoE, ks/ke or seqlens for logits)
    before cosine — just return the full tensor.

Input dict keys per family (see harness/specs.py build_inputs):
  gemm : x_fp8[rows,K], x_scale(TMA-aligned), w_fp8[N,K], w_scale, rows, N, device
  bmm  : A_fp8[64,M,K], B_fp8[64,K,N], A_scale[1], B_scale[1]
  moe  : x_fp8[E,em,K], x_scale, w_fp8[E,N,K], w_scale, masked_m[E], expected_m, E, N, device
  mla  : q[M,64,576], kv[S,1,576], indices[M,1,2048], sm_scale, d_v
  score(prefill): q_fp8[M,32,128], k_fp8[S,128], k_scale[S], weights[M,32], ks[M], ke[M]
  score(decode) : q_fp8[M,1,32,128], kv_cache_fp8[..,64,1,132], weights, seqlens,
                  block_tables, schedule_metadata, max_seq_len

Then:
  python verify.py  --op {operator} --M {M}   # cosine >= threshold ?
  python latency.py --op {operator} --M {M}   # speedup vs backend
  python mfu.py     --op {operator} --M {M}   # MFU / bandwidth util
"""
import torch


def run(inputs: dict):
    raise NotImplementedError("copy this file to tasks/{op}/{phase}/impl.py and implement run()")
