"""Central spec registry for the 12 GLM-5 operators.

For each (operator, phase) this provides:
  build_inputs -> frozen input dict (seeded, quantized once)
  reference    -> calls the real backend on those inputs, returns output tensor(s)
  flops/bytes  -> for MFU / bandwidth-util
  op_meta      -> family, output_kind, peak_dtype, cosine threshold

Phase is inferred from M (prefill M in {1024,2048,4096}, decode M in {16,32}).
Input construction mirrors bench_glm5_{prefill,decode}.py verbatim.
"""
import random
import torch
import deep_gemm
from sgl_kernel import bmm_fp8
from sgl_kernel.flash_mla import flash_mla_sparse_fwd

from .quant import (
    per_token_cast_to_fp8, per_block_cast_to_fp8, cast_to_fp8_per_tensor,
    get_mn_major_tma_aligned_tensor,
)
# Canonical UE8M0 quant (Blackwell fp8_gemm_nt/moe require pow-2 scales; this
# rounds the scale to a power of two BEFORE quantizing the data, so fp8 data and
# scale stay consistent — the official deep_gemm helper).
from deep_gemm.utils.math import (
    per_token_cast_to_fp8 as dg_per_token_cast,
    per_block_cast_to_fp8 as dg_per_block_cast,
)

# ── model constants (GLM-5) ──
NUM_HEADS = 64
D_QK = 576
D_V = 512
TOPK = 2048
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
N_EXPERT = 8          # single-GPU shard
EXPERTS_PER_TOK = 8
BLOCK_SIZE_KV = 64

PREFILL_M = (1024, 2048, 4096)
DECODE_M = (16, 32)
DEFAULT_S = 65536

# ── operator tables ──
GEMM_OPS = {
    "fused_qkv_a":    dict(K=6144,  N=2624,  rows="M"),
    "q_b":            dict(K=2048,  N=16384, rows="M"),
    "o_proj":         dict(K=16384, N=6144,  rows="M"),
    "index_q_upproj": dict(K=2048,  N=4096,  rows="M"),
    "index_k":        dict(K=6144,  N=128,   rows="S_or_M"),  # prefill rows=S, decode rows=M
}
BMM_OPS = {
    "absorbed_W_UK": dict(K=192, N=512),
    "absorbed_W_UV": dict(K=512, N=256),
}
MOE_OPS = {
    "moe_gate": dict(K=6144, N=2048),
    "moe_up":   dict(K=6144, N=2048),
    "moe_down": dict(K=2048, N=6144),
}
MLA_OPS = ("dsa_attn",)
SCORE_OPS = ("index_score",)

ALL_OPS = list(GEMM_OPS) + list(BMM_OPS) + list(MOE_OPS) + list(MLA_OPS) + list(SCORE_OPS)


def infer_phase(M: int) -> str:
    return "prefill" if M >= 1024 else "decode"


def family(op: str) -> str:
    if op in GEMM_OPS:  return "gemm"
    if op in BMM_OPS:   return "bmm"
    if op in MOE_OPS:   return "moe"
    if op in MLA_OPS:   return "mla"
    if op in SCORE_OPS: return "score"
    raise KeyError(f"unknown op {op}")


def _round128(x): return ((x + 127) // 128) * 128


def op_meta(op: str, phase: str) -> dict:
    fam = family(op)
    if fam in ("gemm", "bmm"):
        kind = "dense"
    elif fam == "moe":
        kind = "masked_grouped"
    elif fam == "mla":
        kind = "mla_sparse"
    else:  # score
        kind = "logits_ksrange" if phase == "prefill" else "logits_paged"
    peak_dtype = "bf16" if fam == "mla" else "fp8"
    threshold = 0.99 if fam in ("bmm", "mla") else 0.999
    return dict(family=fam, output_kind=kind, peak_dtype=peak_dtype, threshold=threshold)


# ══════════════════════════════════════════════════════════════════════════
# Input builders (frozen, seeded). Same dict feeds backend AND candidate.
# ══════════════════════════════════════════════════════════════════════════
def build_inputs(op: str, phase: str, M: int, S: int, device, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    fam = family(op)
    if fam == "gemm":   return _build_gemm(op, phase, M, S, device)
    if fam == "bmm":    return _build_bmm(op, M, device)
    if fam == "moe":    return _build_moe(op, M, device)
    if fam == "mla":    return _build_mla(M, S, device)
    return _build_score(phase, M, S, device)


def _build_gemm(op, phase, M, S, device):
    cfg = GEMM_OPS[op]
    K, N = cfg["K"], cfg["N"]
    rows = S if (cfg["rows"] == "S_or_M" and phase == "prefill") else M
    x_bf16 = torch.randn(rows, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = dg_per_token_cast(x_bf16, use_ue8m0=True)
    x_scale = get_mn_major_tma_aligned_tensor(x_scale)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_fp8, w_scale = dg_per_block_cast(w_bf16, use_ue8m0=True)
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
                rows=rows, N=N, device=device)


def _build_bmm(op, M, device):
    cfg = BMM_OPS[op]
    K, N = cfg["K"], cfg["N"]
    batch = NUM_HEADS
    A_bf16 = torch.randn(batch, M, K, dtype=torch.bfloat16, device=device)
    B_bf16 = torch.randn(batch, K, N, dtype=torch.bfloat16, device=device)
    A_fp8, A_scale = cast_to_fp8_per_tensor(A_bf16)
    B_fp8, B_scale = cast_to_fp8_per_tensor(B_bf16)
    A_fp8 = A_fp8.view(batch, M, K)
    B_fp8 = B_fp8.view(batch, K, N)
    return dict(A_fp8=A_fp8, B_fp8=B_fp8, A_scale=A_scale, B_scale=B_scale)


def _build_moe(op, M, device):
    cfg = MOE_OPS[op]
    K, N = cfg["K"], cfg["N"]
    E = N_EXPERT
    total_m = M * EXPERTS_PER_TOK
    expected_m = _round128((total_m + E - 1) // E)
    x_bf16 = torch.randn(E, expected_m, K, dtype=torch.bfloat16, device=device)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(E, expected_m, K // 128, dtype=torch.float32, device=device)
    for i in range(E):
        x_fp8[i], x_scale[i] = dg_per_token_cast(x_bf16[i], use_ue8m0=True)
    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)
    n_ceil = _round128(N)
    w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty(E, n_ceil // 128, K // 128, dtype=torch.float32, device=device)
    for i in range(E):
        w_fp8[i], w_scale[i] = dg_per_block_cast(w_bf16[i], use_ue8m0=True)
    counts = [0] * E
    for _ in range(total_m):
        counts[random.randint(0, E - 1)] += 1
    masked_m = torch.tensor(counts, dtype=torch.int32, device=device)
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
                masked_m=masked_m, expected_m=expected_m, E=E, N=N, device=device)


def _build_mla(M, S, device):
    q = torch.randn(M, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(S, 1, D_QK, dtype=torch.bfloat16, device=device)
    topk_actual = min(TOPK, S)
    indices = torch.stack([
        torch.randperm(S, device=device)[:topk_actual] for _ in range(M)
    ]).view(M, 1, topk_actual).to(torch.int32)
    return dict(q=q, kv=kv, indices=indices, sm_scale=D_QK ** -0.5, d_v=D_V)


def _build_score(phase, M, S, device):
    if phase == "prefill":
        q_bf16 = torch.randn(M, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
        q_view = q_bf16.view(M, INDEX_N_HEADS, INDEX_HEAD_DIM // 128, 128)
        q_scale = (q_view.abs().float().amax(dim=-1) / 448.0).clamp(min=1e-12)
        q_fp8 = (q_view.float() / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(
            M, INDEX_N_HEADS, INDEX_HEAD_DIM)
        k_bf16 = torch.randn(S, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
        k_view = k_bf16.view(S, INDEX_HEAD_DIM // 128, 128)
        k_scale = (k_view.abs().float().amax(dim=-1) / 448.0).clamp(min=1e-12)
        k_fp8 = (k_view.float() / k_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(S, INDEX_HEAD_DIM)
        k_scale = k_scale.squeeze(-1)
        weights = torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)
        ks = torch.zeros(M, dtype=torch.int32, device=device)
        ke = torch.full((M,), S, dtype=torch.int32, device=device)
        return dict(q_fp8=q_fp8, k_fp8=k_fp8, k_scale=k_scale, weights=weights, ks=ks, ke=ke)
    # decode: paged
    nbps = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = nbps * M
    q_bf16 = torch.randn(M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    q_view = q_bf16.view(M * INDEX_N_HEADS, INDEX_HEAD_DIM // 128, 128)
    q_scale = (q_view.abs().float().amax(dim=-1) / 448.0).clamp(min=1e-12)
    q_fp8 = (q_view.float() / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(
        M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM)
    head_dim_with_sf = 132
    # 128 fp8 data bytes + 4 scale bytes (one f32). Build from REAL fp8-quantized
    # data (finite) + scale=1.0, so logits are finite (random uint8 would yield
    # fp8-NaN bytes and NaN logits). Both ref & candidate consume identical bytes.
    data = torch.randn(total_blocks, BLOCK_SIZE_KV, 1, 128, dtype=torch.bfloat16,
                       device=device).to(torch.float8_e4m3fn).view(torch.uint8)
    sf = torch.ones(total_blocks, BLOCK_SIZE_KV, 1, dtype=torch.float32,
                    device=device).view(torch.uint8).reshape(total_blocks, BLOCK_SIZE_KV, 1, 4)
    kv_cache_fp8 = torch.cat([data, sf], dim=-1).contiguous()  # [total_blocks,64,1,132] uint8
    weights = torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)
    seqlens = torch.full((M, 1), S, dtype=torch.int32, device=device)  # 2D: [batch, next_n]
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=device).view(M, nbps)
    max_seq_len = nbps * BLOCK_SIZE_KV
    sm_count = deep_gemm.get_num_sms()
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(seqlens, BLOCK_SIZE_KV, sm_count)
    return dict(q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8, weights=weights, seqlens=seqlens,
                block_tables=block_tables, schedule_metadata=schedule_metadata,
                max_seq_len=max_seq_len)


# ══════════════════════════════════════════════════════════════════════════
# Reference: the real backend call (ground truth). Also the default candidate.
# ══════════════════════════════════════════════════════════════════════════
def reference(op: str, phase: str, inputs: dict):
    fam = family(op)
    if fam == "gemm":
        out = torch.empty(inputs["rows"], inputs["N"], dtype=torch.bfloat16, device=inputs["device"])
        deep_gemm.fp8_gemm_nt((inputs["x_fp8"], inputs["x_scale"]),
                              (inputs["w_fp8"], inputs["w_scale"]), out)
        return out
    if fam == "bmm":
        return bmm_fp8(inputs["A_fp8"], inputs["B_fp8"],
                       inputs["A_scale"], inputs["B_scale"], torch.bfloat16)
    if fam == "moe":
        out = torch.empty(inputs["E"], inputs["x_fp8"].shape[1], inputs["N"],
                          dtype=torch.bfloat16, device=inputs["device"])
        deep_gemm.fp8_m_grouped_gemm_nt_masked((inputs["x_fp8"], inputs["x_scale"]),
                                               (inputs["w_fp8"], inputs["w_scale"]),
                                               out, inputs["masked_m"], inputs["expected_m"])
        return out
    if fam == "mla":
        return flash_mla_sparse_fwd(inputs["q"], inputs["kv"], inputs["indices"],
                                    inputs["sm_scale"], inputs["d_v"])
    # score
    if phase == "prefill":
        return deep_gemm.fp8_mqa_logits(inputs["q_fp8"], (inputs["k_fp8"], inputs["k_scale"]),
                                        inputs["weights"], inputs["ks"], inputs["ke"],
                                        clean_logits=False)
    return deep_gemm.fp8_paged_mqa_logits(inputs["q_fp8"], inputs["kv_cache_fp8"],
                                          inputs["weights"], inputs["seqlens"],
                                          inputs["block_tables"], inputs["schedule_metadata"],
                                          inputs["max_seq_len"], clean_logits=False)


# ══════════════════════════════════════════════════════════════════════════
# FLOP / byte models (for MFU & bandwidth-util). dtype-aware: fp8=1B bf16=2B f32=4B.
# ══════════════════════════════════════════════════════════════════════════
def flops(op: str, phase: str, M: int, S: int) -> float:
    fam = family(op)
    if fam == "gemm":
        cfg = GEMM_OPS[op]
        rows = S if (cfg["rows"] == "S_or_M" and phase == "prefill") else M
        return 2.0 * rows * cfg["N"] * cfg["K"]
    if fam == "bmm":
        cfg = BMM_OPS[op]
        return 2.0 * NUM_HEADS * M * cfg["N"] * cfg["K"]
    if fam == "moe":
        cfg = MOE_OPS[op]
        eff = M * EXPERTS_PER_TOK  # sum(masked_m) valid rows
        return 2.0 * eff * cfg["N"] * cfg["K"]
    if fam == "mla":
        sq = M
        return 2.0 * NUM_HEADS * sq * TOPK * (D_QK + D_V)
    # score: 2 * heads * M * S * head_dim
    return 2.0 * INDEX_N_HEADS * M * S * INDEX_HEAD_DIM


def bytes_(op: str, phase: str, M: int, S: int) -> float:
    fam = family(op)
    if fam == "gemm":
        cfg = GEMM_OPS[op]
        rows = S if (cfg["rows"] == "S_or_M" and phase == "prefill") else M
        K, N = cfg["K"], cfg["N"]
        return rows * K * 1 + N * K * 1 + rows * N * 2
    if fam == "bmm":
        cfg = BMM_OPS[op]
        K, N = cfg["K"], cfg["N"]
        return NUM_HEADS * (M * K * 1 + N * K * 1 + M * N * 2)
    if fam == "moe":
        cfg = MOE_OPS[op]
        K, N = cfg["K"], cfg["N"]
        eff = M * EXPERTS_PER_TOK
        return eff * K * 1 + N_EXPERT * N * K * 1 + eff * N * 2
    if fam == "mla":
        sq = M
        kv_tokens = min(TOPK * sq, S)
        return 2.0 * (NUM_HEADS * sq * D_QK + kv_tokens * D_QK + NUM_HEADS * sq * D_V)
    # score
    q_bytes = M * INDEX_N_HEADS * INDEX_HEAD_DIM * 1
    logits_bytes = M * S * 4  # fp32 logits write
    if phase == "prefill":
        k_bytes = S * INDEX_HEAD_DIM * 1
        return q_bytes + k_bytes + logits_bytes
    nbps = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    kv_bytes = (nbps * M) * BLOCK_SIZE_KV * 132 * 1  # paged cache read (per-request)
    return q_bytes + kv_bytes + logits_bytes
