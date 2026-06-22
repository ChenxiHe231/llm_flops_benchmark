"""
Unified Benchmark: GLM-5 DECODE phase operators — matching sglang decode operator selection.

Decode characteristics:
  - M = batch_size (each request contributes 1 token, no seq dim)
  - hidden_states shape: [batch_size, hidden_dim] (2D, not 3D)
  - S = KV context length (total cached tokens per request)

Operator selection (sglang decode path):
  Attention:
    - fused_qkv_a_proj:   dsv3_fused_a_gemm (BF16 fused, batch<=16) or DeepGEMM fp8_gemm_nt
    - q_b_proj:            DeepGEMM fp8_gemm_nt
    - absorbed_W_UK:       sgl_kernel.bmm_fp8  (cuBLAS FP8, per-tensor)
    - absorbed_W_UV:       sgl_kernel.bmm_fp8
    - o_proj:              DeepGEMM fp8_gemm_nt
  MLA Decode Attention:
    - flash_mla_with_kvcache (paged KV, FP8 or BF16 decode kernel)
  DSA Indexer:
    - index_k_proj (wk):          DeepGEMM fp8_gemm_nt
    - index_q_upproj (wq_b):      DeepGEMM fp8_gemm_nt
    - index_weights_proj:          DeepGEMM bf16_gemm_nt  (BF16 in, F32 out)
    - index_score:                 deep_gemm.fp8_paged_mqa_logits (paged, decode-specific)
  MoE:
    - gate/up/down:               deep_gemm.fp8_m_grouped_gemm_nt_masked

All timing uses CUDA Graph capture + replay.
Final summary table sorted by avg_ms descending.
"""

import time
import os
import random

import torch
import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
from sgl_kernel import bmm_fp8

# ══════════════════════════════════════════════════════════════════════════════
# GLM-5 Model Parameters
# ══════════════════════════════════════════════════════════════════════════════
HIDDEN_SIZE = 6144
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 256
V_HEAD_DIM = 256
NUM_HEADS = 64
D_QK = 576
D_V = 512
TOPK = 2048

INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128

MOE_INTERMEDIATE_SIZE = 2048
N_EXPERT = 8
NUM_EXPERTS_PER_TOK = 8

FUSED_QKV_A_OUT = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 2624

# ── Configurable inputs ──
# M = batch_size (decode: 1 token per request)
M_LIST = [1, 4, 8, 16, 32, 64]
S_LIST = [65536]       # KV context length

BLOCK_SIZE_KV = 64     # paged KV cache block size

NUM_WARMUP = 5
NUM_RUNS = 20


# ══════════════════════════════════════════════════════════════════════════════
# FP8 Quantization Utilities
# ══════════════════════════════════════════════════════════════════════════════

def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, k = x.shape
    x_view = x.view(m, k // 128, 128)
    x_amax = x_view.abs().float().amax(dim=-1)
    x_scale = (x_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    x_fp8 = (x_view.float() / x_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k), x_scale


def per_block_cast_to_fp8(w: torch.Tensor):
    assert w.dim() == 2 and w.shape[1] % 128 == 0
    n, k = w.shape
    n_ceil = (n + 127) // 128 * 128
    w_padded = torch.zeros(n_ceil, k, dtype=w.dtype, device=w.device) if n < n_ceil else w
    if n < n_ceil:
        w_padded[:n] = w
    w_view = w_padded.view(n_ceil // 128, 128, k // 128, 128)
    w_amax = w_view.abs().float().amax(dim=(1, 3))
    w_scale = (w_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    w_fp8 = (w_view.float() / w_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.view(n_ceil, k)[:n].contiguous(), w_scale


def cast_to_fp8_per_tensor(x: torch.Tensor):
    amax = x.abs().float().amax()
    scale = (amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale.view(1).to(x.device)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Primitives
# ══════════════════════════════════════════════════════════════════════════════

def _cuda_graph_bench(run_fn):
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run_fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(NUM_RUNS):
            run_fn()
    torch.cuda.synchronize()

    for _ in range(NUM_WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / NUM_RUNS
    del graph
    return avg_ms


# ── DeepGEMM FP8 GEMM ──
def bench_deepgemm_fp8(M, K, N, device):
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = per_token_cast_to_fp8(x_bf16)
    x_scale = get_mn_major_tma_aligned_tensor(x_scale)
    del x_bf16
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_fp8, w_scale = per_block_cast_to_fp8(w_bf16)
    del w_bf16
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    avg_ms = _cuda_graph_bench(lambda: deep_gemm.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out))
    del x_fp8, x_scale, w_fp8, w_scale, out
    return avg_ms


# ── DeepGEMM BF16 GEMM (for indexer weights_proj) ──
def bench_deepgemm_bf16(M, K, N, device):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    out = torch.empty(M, N, dtype=torch.float32, device=device)

    avg_ms = _cuda_graph_bench(lambda: deep_gemm.bf16_gemm_nt(x, w, out))
    del x, w, out
    return avg_ms


# ── sgl_kernel.bmm_fp8 ──
def bench_sgl_bmm_fp8(batch, M, K, N, device):
    A_bf16 = torch.randn(batch, M, K, dtype=torch.bfloat16, device=device)
    B_bf16 = torch.randn(batch, K, N, dtype=torch.bfloat16, device=device)
    A_fp8, A_scale = cast_to_fp8_per_tensor(A_bf16)
    B_fp8, B_scale = cast_to_fp8_per_tensor(B_bf16)
    A_fp8 = A_fp8.view(batch, M, K)
    B_fp8 = B_fp8.view(batch, K, N)
    del A_bf16, B_bf16

    avg_ms = _cuda_graph_bench(lambda: bmm_fp8(A_fp8, B_fp8, A_scale, B_scale, torch.bfloat16))
    del A_fp8, A_scale, B_fp8, B_scale
    return avg_ms


# ── MLA Decode: flash_mla_with_kvcache (paged) ──
def bench_mla_decode(M, S, device):
    """Decode MLA attention using flash_mla_with_kvcache with paged KV cache."""
    h_q = NUM_HEADS
    h_kv = 1
    d_qk = D_QK
    d_v = D_V

    num_blocks_per_seq = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = num_blocks_per_seq * M

    # q: [batch, s_q=1, h_q, d_qk]
    q = torch.randn(M, 1, h_q, d_qk, dtype=torch.bfloat16, device=device)
    # k_cache: [total_blocks, block_size, h_kv, d_qk]
    k_cache = torch.randn(total_blocks, BLOCK_SIZE_KV, h_kv, d_qk, dtype=torch.bfloat16, device=device)
    # block_table: [batch, num_blocks_per_seq]
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(M, num_blocks_per_seq)
    # cache_seqlens: [batch]
    cache_seqlens = torch.full((M,), S, dtype=torch.int32, device=device)

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 1 * h_q // h_kv, h_kv)

    def run():
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, d_v,
            tile_scheduler_metadata, num_splits, causal=False,
        )

    avg_ms = _cuda_graph_bench(run)
    del q, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits
    return avg_ms


# ── DSA Indexer score: deep_gemm.fp8_paged_mqa_logits (decode, paged) ──
def bench_index_score_paged(M, S, device):
    """Decode indexer score using deep_gemm.fp8_paged_mqa_logits."""
    BLOCK_SIZE = 128
    num_blocks_per_seq = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = num_blocks_per_seq * M

    # q_fp8: [M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM] — unsqueeze next_n dim
    q_bf16 = torch.randn(M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    q_view = q_bf16.view(M * INDEX_N_HEADS, INDEX_HEAD_DIM // BLOCK_SIZE, BLOCK_SIZE)
    q_amax = q_view.abs().float().amax(dim=-1)
    q_scale = (q_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    q_fp8 = (q_view.float() / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    q_fp8 = q_fp8.view(M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM)
    del q_bf16, q_view, q_amax, q_scale

    # kv_cache_fp8: [total_blocks, block_size_kv=64, h_kv=1, head_dim_with_sf=132]
    # head_dim_with_sf = 128 + 4 (scale factors) = 132
    head_dim_with_sf = 132
    kv_cache_fp8 = torch.randn(total_blocks, BLOCK_SIZE_KV, 1, head_dim_with_sf,
                                dtype=torch.bfloat16, device=device).to(torch.uint8).view(
                                    total_blocks, BLOCK_SIZE_KV, 1, head_dim_with_sf)

    # weights: [M, INDEX_N_HEADS]
    weights = torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)

    # seqlens: [M]
    seqlens = torch.full((M,), S, dtype=torch.int32, device=device)

    # block_tables: [M, num_blocks_per_seq]
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=device).view(M, num_blocks_per_seq)

    max_seq_len = num_blocks_per_seq * BLOCK_SIZE_KV
    sm_count = deep_gemm.get_num_sms()
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(seqlens, BLOCK_SIZE_KV, sm_count)

    def run():
        deep_gemm.fp8_paged_mqa_logits(
            q_fp8, kv_cache_fp8, weights, seqlens, block_tables,
            schedule_metadata, max_seq_len, clean_logits=False,
        )

    avg_ms = _cuda_graph_bench(run)
    del q_fp8, kv_cache_fp8, weights, seqlens, block_tables, schedule_metadata
    return avg_ms


# ── MoE Grouped GEMM (masked, decode) ──
def bench_moe_grouped_masked(M, K, N, device):
    total_m = M * NUM_EXPERTS_PER_TOK
    expected_m = (total_m + N_EXPERT - 1) // N_EXPERT
    expected_m = ((expected_m + 127) // 128) * 128

    x_bf16 = torch.randn(N_EXPERT, expected_m, K, dtype=torch.bfloat16, device=device)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(N_EXPERT, expected_m, K // 128, dtype=torch.float32, device=device)
    for i in range(N_EXPERT):
        x_fp8[i], x_scale[i] = per_token_cast_to_fp8(x_bf16[i])
    del x_bf16

    w_bf16 = torch.randn(N_EXPERT, N, K, dtype=torch.bfloat16, device=device)
    n_ceil = (N + 127) // 128 * 128
    w_fp8 = torch.empty(N_EXPERT, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty(N_EXPERT, n_ceil // 128, K // 128, dtype=torch.float32, device=device)
    for i in range(N_EXPERT):
        w_fp8[i], w_scale[i] = per_block_cast_to_fp8(w_bf16[i])
    del w_bf16

    out = torch.empty(N_EXPERT, expected_m, N, dtype=torch.bfloat16, device=device)
    counts = [0] * N_EXPERT
    for _ in range(total_m):
        counts[random.randint(0, N_EXPERT - 1)] += 1
    masked_m = torch.tensor(counts, dtype=torch.int32, device=device)

    avg_ms = _cuda_graph_bench(lambda: deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (x_fp8, x_scale), (w_fp8, w_scale), out, masked_m, expected_m))
    del x_fp8, x_scale, w_fp8, w_scale, out, masked_m
    return avg_ms


# ══════════════════════════════════════════════════════════════════════════════
# Operator Definitions (Decode)
# ══════════════════════════════════════════════════════════════════════════════

def get_all_operators_decode(M, S):
    ops = []

    # ── Attention GEMM ──
    ops.append(("fused_qkv_a_proj", "Attention",
                lambda dev: bench_deepgemm_fp8(M, HIDDEN_SIZE, FUSED_QKV_A_OUT, dev),
                f"[{M}, {HIDDEN_SIZE}]×[{HIDDEN_SIZE}, {FUSED_QKV_A_OUT}]  (DeepGEMM FP8)"))
    ops.append(("q_b_proj", "Attention",
                lambda dev: bench_deepgemm_fp8(M, Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM, dev),
                f"[{M}, {Q_LORA_RANK}]×[{Q_LORA_RANK}, {NUM_HEADS * QK_HEAD_DIM}]  (DeepGEMM FP8)"))
    ops.append(("absorbed_W_UK", "Attention",
                lambda dev: bench_sgl_bmm_fp8(NUM_HEADS, M, QK_NOPE_HEAD_DIM, KV_LORA_RANK, dev),
                f"bmm_fp8 [{NUM_HEADS}, {M}, {QK_NOPE_HEAD_DIM}]×[{QK_NOPE_HEAD_DIM}, {KV_LORA_RANK}]"))
    ops.append(("absorbed_W_UV", "Attention",
                lambda dev: bench_sgl_bmm_fp8(NUM_HEADS, M, KV_LORA_RANK, V_HEAD_DIM, dev),
                f"bmm_fp8 [{NUM_HEADS}, {M}, {KV_LORA_RANK}]×[{KV_LORA_RANK}, {V_HEAD_DIM}]"))
    ops.append(("o_proj", "Attention",
                lambda dev: bench_deepgemm_fp8(M, NUM_HEADS * V_HEAD_DIM, HIDDEN_SIZE, dev),
                f"[{M}, {NUM_HEADS * V_HEAD_DIM}]×[{NUM_HEADS * V_HEAD_DIM}, {HIDDEN_SIZE}]  (DeepGEMM FP8)"))

    # ── MLA Decode Attention (paged KV cache) ──
    ops.append(("mla_decode_attn", "MLA",
                lambda dev, _M=M, _S=S: bench_mla_decode(_M, _S, dev),
                f"flash_mla_with_kvcache batch={M} s_q=1 s_kv={S} paged"))

    # ── DSA Indexer ──
    ops.append(("index_k_proj", "DSA Indexer",
                lambda dev: bench_deepgemm_fp8(M, HIDDEN_SIZE, INDEX_HEAD_DIM, dev),
                f"[{M}, {HIDDEN_SIZE}]×[{HIDDEN_SIZE}, {INDEX_HEAD_DIM}]  (DeepGEMM FP8)"))
    ops.append(("index_q_upproj", "DSA Indexer",
                lambda dev: bench_deepgemm_fp8(M, Q_LORA_RANK, INDEX_N_HEADS * INDEX_HEAD_DIM, dev),
                f"[{M}, {Q_LORA_RANK}]×[{Q_LORA_RANK}, {INDEX_N_HEADS * INDEX_HEAD_DIM}]  (DeepGEMM FP8)"))
    ops.append(("index_weights_proj", "DSA Indexer",
                lambda dev: bench_deepgemm_bf16(M, HIDDEN_SIZE, INDEX_N_HEADS, dev),
                f"[{M}, {HIDDEN_SIZE}]×[{HIDDEN_SIZE}, {INDEX_N_HEADS}]  (DeepGEMM BF16->F32)"))
    ops.append(("index_score", "DSA Indexer",
                lambda dev, _M=M, _S=S: bench_index_score_paged(_M, _S, dev),
                f"fp8_paged_mqa_logits batch={M} s_kv={S} (paged decode)"))

    # ── MoE Grouped GEMM (masked) ──
    ops.append(("moe_gate_proj", "MoE",
                lambda dev: bench_moe_grouped_masked(M, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dev),
                f"masked [{N_EXPERT}x, {HIDDEN_SIZE}]×[{HIDDEN_SIZE}, {MOE_INTERMEDIATE_SIZE}]"))
    ops.append(("moe_up_proj", "MoE",
                lambda dev: bench_moe_grouped_masked(M, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dev),
                f"masked [{N_EXPERT}x, {HIDDEN_SIZE}]×[{HIDDEN_SIZE}, {MOE_INTERMEDIATE_SIZE}]"))
    ops.append(("moe_down_proj", "MoE",
                lambda dev: bench_moe_grouped_masked(M, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, dev),
                f"masked [{N_EXPERT}x, {MOE_INTERMEDIATE_SIZE}]×[{MOE_INTERMEDIATE_SIZE}, {HIDDEN_SIZE}]"))

    return ops


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 130)
    print("GLM-5 DECODE Phase Unified Benchmark  [CUDA Graph, sglang-aligned]")
    print("=" * 130)
    print(f"Model:   hidden={HIDDEN_SIZE}, q_lora={Q_LORA_RANK}, kv_lora={KV_LORA_RANK}, heads={NUM_HEADS}")
    print(f"         qk_nope={QK_NOPE_HEAD_DIM}, qk_rope={QK_ROPE_HEAD_DIM}, v_head={V_HEAD_DIM}, topk={TOPK}")
    print(f"         index_heads={INDEX_N_HEADS}, index_dim={INDEX_HEAD_DIM}")
    print(f"         moe_inter={MOE_INTERMEDIATE_SIZE}, n_expert={N_EXPERT}, top_k={NUM_EXPERTS_PER_TOK}")
    print(f"M list:  {M_LIST}  (batch_size, 1 token per request)")
    print(f"S list:  {S_LIST}  (KV context length)")
    print(f"Bench:   {NUM_WARMUP} warmup + {NUM_RUNS} graph replays")
    print("=" * 130)

    all_results = []

    for M in M_LIST:
        for S in S_LIST:
            print(f"\n{'='*130}")
            print(f"  batch={M}, S={S}")
            print(f"{'='*130}")
            print(f"  {'name':<24s} {'category':<14s} {'shape':>60s} {'avg(ms)':>12s}")
            print(f"  {'-'*114}")

            ops = get_all_operators_decode(M, S)
            for op_name, category, bench_fn, shape_str in ops:
                torch.cuda.empty_cache()
                try:
                    avg_ms = bench_fn(device)
                    print(f"  {op_name:<24s} {category:<14s} {shape_str:>60s} {avg_ms:>12.4f}")
                    all_results.append({
                        "name": op_name, "category": category,
                        "M": M, "S": S, "shape": shape_str, "avg_ms": avg_ms,
                    })
                except Exception as e:
                    print(f"  {op_name:<24s} {category:<14s} {shape_str:>60s}   FAILED: {e}")
                    all_results.append({
                        "name": op_name, "category": category,
                        "M": M, "S": S, "shape": shape_str, "avg_ms": 0, "error": str(e),
                    })
                time.sleep(0.1)

    # ── Summary: sorted by avg_ms descending ──
    for M in M_LIST:
        for S in S_LIST:
            subset = [r for r in all_results if r["M"] == M and r["S"] == S and r["avg_ms"] > 0]
            subset.sort(key=lambda r: r["avg_ms"], reverse=True)
            total_ms = sum(r["avg_ms"] for r in subset)

            print(f"\n{'='*130}")
            print(f"  Summary (batch={M}, S={S}) — sorted by time descending — Total: {total_ms:.3f} ms")
            print(f"{'='*130}")
            print(f"  {'rank':<6s} {'name':<24s} {'category':<14s} {'avg(ms)':>12s} {'pct':>8s} {'cumulative':>12s}")
            print(f"  {'-'*80}")

            cum_ms = 0.0
            for rank, r in enumerate(subset, 1):
                cum_ms += r["avg_ms"]
                pct = r["avg_ms"] / total_ms * 100 if total_ms > 0 else 0
                cum_pct = cum_ms / total_ms * 100 if total_ms > 0 else 0
                print(f"  {rank:<6d} {r['name']:<24s} {r['category']:<14s} "
                      f"{r['avg_ms']:>12.4f} {pct:>7.1f}% {cum_pct:>11.1f}%")

    # CSV
    csv_path = "glm5_decode_perf.csv"
    with open(csv_path, "w") as f:
        f.write("name,category,batch,S,shape,avg_ms\n")
        for r in all_results:
            f.write(f"{r['name']},{r['category']},{r['M']},{r['S']},\"{r['shape']}\",{r['avg_ms']:.4f}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

