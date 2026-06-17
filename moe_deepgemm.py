"""
Benchmark: DeepGEMM MoE grouped GEMM (FP8) with GLM-5 parameters.

GLM-5 MoE config (https://www.modelscope.cn/models/ZhipuAI/GLM-5/file/view/master/config.json):
  hidden_size:           6144
  moe_intermediate_size: 2048
  n_routed_experts:      256
  n_shared_experts:      1
  num_experts_per_tok:   8
  hidden_act:            silu

MoE FFN per expert (SwiGLU):
  gate_proj: [moe_intermediate_size, hidden_size] = [2048, 6144]  (N=2048, K=6144)
  up_proj:   [moe_intermediate_size, hidden_size] = [2048, 6144]  (N=2048, K=6144)
  down_proj: [hidden_size, moe_intermediate_size] = [6144, 2048]  (N=6144, K=2048)

DeepGEMM API:
  deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
      (x_fp8, x_scale),    # input:  x_fp8 [total_m, K] fp8, x_scale [total_m, K//128] fp32
      (w_fp8, w_scale),     # weight: w_fp8 [N_EXPERT, N, K] fp8, w_scale [N_EXPERT, N_ceil//128, K//128] fp32
      out,                  # output: [total_m, N] bf16
      m_indices,            # [total_m] int32, expert id for each row
  )

Benchmark:
  Fixed TOTAL_TOKENS, randomly shuffled into N_EXPERT experts (non-uniform distribution).
  Multiple random distributions are tested (NUM_DISTRIBUTIONS rounds).
"""

import time
import os
import random

import torch
import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

# ── GLM-5 MoE parameters ──
HIDDEN_SIZE = 6144
MOE_INTERMEDIATE_SIZE = 2048
N_EXPERT = 8
NUM_EXPERTS_PER_TOK = 8

# Total tokens to distribute across N_EXPERT experts
TOTAL_TOKENS = 16 * N_EXPERT  # e.g. 524288

# Number of random distributions to test
NUM_DISTRIBUTIONS = 5

NUM_WARMUP = 5
NUM_RUNS = 20


def generate_random_m_per_expert(total_tokens: int, n_expert: int) -> list:
    """
    Randomly distribute total_tokens across n_expert experts.
    Uses a random shuffle approach: assign each token to a random expert.
    """
    counts = [0] * n_expert
    for _ in range(total_tokens):
        counts[random.randint(0, n_expert - 1)] += 1
    return counts


def per_token_cast_to_fp8(x: torch.Tensor):
    """Per-token (per-row) FP8 quantization with block scaling (block_size=128 along K)."""
    assert x.dim() == 2
    m, k = x.shape
    assert k % 128 == 0
    x_view = x.view(m, k // 128, 128)
    x_amax = x_view.abs().float().amax(dim=-1)
    x_scale = (x_amax / torch.finfo(torch.float8_e4m3fn).max).float()
    x_scale = x_scale.clamp(min=1e-12)
    x_fp8 = (x_view.float() / x_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k), x_scale


def per_block_cast_to_fp8(w: torch.Tensor):
    """Per-block FP8 quantization for weight (block_size=128 along both N and K)."""
    assert w.dim() == 2
    n, k = w.shape
    assert k % 128 == 0
    n_ceil = (n + 127) // 128 * 128
    if n < n_ceil:
        w_padded = torch.zeros(n_ceil, k, dtype=w.dtype, device=w.device)
        w_padded[:n] = w
    else:
        w_padded = w
    w_view = w_padded.view(n_ceil // 128, 128, k // 128, 128)
    w_amax = w_view.abs().float().amax(dim=(1, 3))
    w_scale = (w_amax / torch.finfo(torch.float8_e4m3fn).max).float()
    w_scale = w_scale.clamp(min=1e-12)
    w_fp8 = (w_view.float() / w_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.view(n_ceil, k)[:n].contiguous(), w_scale


def make_grouped_gemm_tensors(m_per_expert: list, K: int, N: int, device: torch.device):
    """
    Create contiguous-layout tensors for MoE grouped GEMM.
    m_per_expert: list of length N_EXPERT, each element is the token count for that expert.
    """
    num_groups = N_EXPERT
    assert len(m_per_expert) == num_groups

    alignment = 128
    aligned_m = [(((m + alignment - 1) // alignment) * alignment) if m > 0 else 0
                 for m in m_per_expert]
    total_m = sum(aligned_m)

    x_bf16 = torch.randn(total_m, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = per_token_cast_to_fp8(x_bf16)
    x_scale = get_mn_major_tma_aligned_tensor(x_scale)

    w_bf16 = torch.randn(num_groups, N, K, dtype=torch.bfloat16, device=device)
    n_ceil = (N + 127) // 128 * 128
    w_fp8 = torch.empty(num_groups, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty(num_groups, n_ceil // 128, K // 128, dtype=torch.float32, device=device)
    for i in range(num_groups):
        w_fp8[i], w_scale[i] = per_block_cast_to_fp8(w_bf16[i])

    out = torch.empty(total_m, N, dtype=torch.bfloat16, device=device)

    m_indices_list = []
    for i, m in enumerate(aligned_m):
        if m > 0:
            m_indices_list.append(torch.full((m,), i, dtype=torch.int32, device=device))
    m_indices = torch.cat(m_indices_list)

    return (x_fp8, x_scale), (w_fp8, w_scale), out, m_indices, total_m


def bench_grouped_gemm(m_per_expert: list, K: int, N: int, device: torch.device):
    """Benchmark one grouped GEMM configuration."""
    (x_fp8, x_scale), (w_fp8, w_scale), out, m_indices, total_m = \
        make_grouped_gemm_tensors(m_per_expert, K, N, device)

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (x_fp8, x_scale), (w_fp8, w_scale), out, m_indices
        )

    torch.cuda.synchronize()

    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]

    for i in range(NUM_RUNS):
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    flops = 2.0 * total_m * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12

    mem_bytes = (total_m * K * 1
                 + N_EXPERT * N * K * 1
                 + total_m * N * 2)
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12

    flops_per_byte = flops / mem_bytes if mem_bytes > 0 else 0

    return avg_ms, min_ms, max_ms, tflops, tbps, flops_per_byte, total_m


def format_distribution(m_per_expert: list) -> str:
    """Summarize the distribution: min/max/avg/std and top-5 largest experts."""
    import statistics
    avg = statistics.mean(m_per_expert)
    std = statistics.stdev(m_per_expert) if len(m_per_expert) > 1 else 0
    sorted_m = sorted(m_per_expert, reverse=True)
    top5 = sorted_m[:5]
    nonzero = sum(1 for m in m_per_expert if m > 0)
    return (f"min={min(m_per_expert)}, max={max(m_per_expert)}, "
            f"avg={avg:.0f}, std={std:.0f}, nonzero={nonzero}/{len(m_per_expert)}, "
            f"top5={top5}")


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_runs = int(os.environ.get("NUM_RUNS", NUM_RUNS))
    num_warmup = int(os.environ.get("NUM_WARMUP", NUM_WARMUP))
    num_distributions = int(os.environ.get("NUM_DISTRIBUTIONS", NUM_DISTRIBUTIONS))

    print("=" * 110)
    print("GLM-5 MoE DeepGEMM Grouped GEMM (FP8) Benchmark")
    print("=" * 110)
    print(f"Model:    hidden={HIDDEN_SIZE}, moe_inter={MOE_INTERMEDIATE_SIZE}, "
          f"experts={N_EXPERT}, top_k={NUM_EXPERTS_PER_TOK}")
    print(f"Total:    {TOTAL_TOKENS} tokens, {num_distributions} random distributions")
    print(f"Bench:    {num_warmup} warmup + {num_runs} runs per case")
    print(f"Projs:    gate/up: K={HIDDEN_SIZE}, N={MOE_INTERMEDIATE_SIZE}")
    print(f"          down:    K={MOE_INTERMEDIATE_SIZE}, N={HIDDEN_SIZE}")
    print("=" * 110)

    proj_configs = [
        ("gate_proj", HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
        ("up_proj",   HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
        ("down_proj", MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
    ]

    # Generate random distributions
    distributions = []
    for i in range(num_distributions):
        m_per_expert = generate_random_m_per_expert(TOTAL_TOKENS, N_EXPERT)
        distributions.append(m_per_expert)

    all_results = []

    for proj_name, K, N in proj_configs:
        print(f"\n{'='*80}")
        print(f"  {proj_name}: K={K}, N={N}")
        print(f"{'='*80}")

        for dist_idx, m_per_expert in enumerate(distributions):
            dist_info = format_distribution(m_per_expert)
            print(f"\n  [dist {dist_idx}] {dist_info}")
            print(f"  M_PER_EXPERT = {m_per_expert}")

            torch.cuda.empty_cache()

            try:
                avg_ms, min_ms, max_ms, tflops, tbps, fpb, total_m = \
                    bench_grouped_gemm(m_per_expert, K, N, device)
                print(f"  -> total_m={total_m}, avg={avg_ms:.3f}ms, min={min_ms:.3f}ms, "
                      f"max={max_ms:.3f}ms, {tflops:.1f} TFlops, {tbps:.3f} TB/s, {fpb:.1f} FLOP/B")
                all_results.append({
                    "proj": proj_name, "dist_idx": dist_idx,
                    "total_m": total_m, "K": K, "N": N,
                    "m_min": min(m_per_expert), "m_max": max(m_per_expert),
                    "m_avg": sum(m_per_expert) // N_EXPERT,
                    "avg_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms,
                    "tflops": tflops, "tbps": tbps, "flops_per_byte": fpb,
                    "m_per_expert": m_per_expert,
                })
            except Exception as e:
                print(f"  -> FAILED: {e}")
                all_results.append({
                    "proj": proj_name, "dist_idx": dist_idx,
                    "total_m": 0, "K": K, "N": N,
                    "m_min": 0, "m_max": 0, "m_avg": 0,
                    "avg_ms": 0, "min_ms": 0, "max_ms": 0,
                    "tflops": 0, "tbps": 0, "flops_per_byte": 0,
                    "m_per_expert": m_per_expert, "error": str(e),
                })

            time.sleep(0.3)

    # CSV output
    csv_path = "glm5_moe_deepgemm_perf.csv"
    with open(csv_path, "w") as f:
        f.write("proj,dist_idx,total_m,K,N,m_min,m_max,m_avg,avg_ms,min_ms,max_ms,tflops,tbps,flops_per_byte,m_per_expert\n")
        for r in all_results:
            m_str = "|".join(str(x) for x in r["m_per_expert"])
            f.write(f"{r['proj']},{r['dist_idx']},{r['total_m']},{r['K']},{r['N']},"
                    f"{r['m_min']},{r['m_max']},{r['m_avg']},"
                    f"{r['avg_ms']:.4f},{r['min_ms']:.4f},{r['max_ms']:.4f},"
                    f"{r['tflops']:.2f},{r['tbps']:.4f},{r['flops_per_byte']:.2f},\"{m_str}\"\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
