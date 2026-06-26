"""
Benchmark: DeepEP dispatch + token permute for GLM-5 MoE prefill.

Tests the full MoE dispatch pipeline:
  1. get_dispatch_layout (token permute / routing metadata)
  2. DeepEP all-to-all dispatch (send tokens to expert-owning GPUs)
  3. DeepEP all-to-all combine (gather results back)

Configuration:
  - N_NODE nodes × 8 GPUs/node = total_ranks GPUs
  - 256 experts, each GPU owns 256/total_ranks experts
  - M tokens per GPU (prefill)
  - top_k = 8 experts per token

Load balancing scenarios:
  - balanced (EPLB): uniform routing across all experts
  - mild/medium/heavy: increasing skew

Usage:
  # Multi-node (4 nodes):
  python bench_glm5_deepep_dispatch.py --nnodes 4 --node-rank $MY_RANK \
    --master-addr $MASTER_IP --m-per-gpu 4096

  # Single node 8 GPUs:
  python bench_glm5_deepep_dispatch.py --nnodes 1 --m-per-gpu 4096
  
  # 单节点 8 卡
  python bench_glm5_deepep_dispatch.py --nnodes 1 --m-per-gpu 4096

  # 4 节点，每个节点执行（指定各自的 node-rank）
  # Node 0:
  python bench_glm5_deepep_dispatch.py --nnodes 2 --node-rank 0 --master-addr 10.227.27.245 --m-per-gpu 4096
  # Node 1:
  python bench_glm5_deepep_dispatch.py --nnodes 4 --node-rank 1 --master-addr 10.0.0.1 --m-per-gpu 4096

  # 只测 balanced 场景
  python bench_glm5_deepep_dispatch.py --nnodes 1 --m-per-gpu 4096 --scenario balanced

  # 自定义参数
  python bench_glm5_deepep_dispatch.py --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 \
    --master-port 29500 --m-per-gpu 8192 --num-sms 32
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import deep_ep

# ── GLM-5 MoE parameters ──
HIDDEN_SIZE = 6144
NUM_EXPERTS = 256
NUM_TOPK = 8
NUM_TOPK_GROUPS = 4  # n_group from config (grouped topk routing)

# Tokens per GPU to benchmark
M_PER_GPU_LIST = [512, 1024, 2048, 4096, 8192, 16384]

NUM_WARMUP = 10
NUM_RUNS = 30

# ══════════════════════════════════════════════════════════════════════════════
# Init distributed (spawn-based, like sglang tuning_deepep.py)
# ══════════════════════════════════════════════════════════════════════════════

def init_dist(local_rank, num_local_ranks, args):
    ip = args.master_addr
    port = args.master_port
    num_nodes = args.nnodes
    node_rank = args.node_rank
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_nodes * num_local_ranks))),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Token routing generation with configurable skew
# ══════════════════════════════════════════════════════════════════════════════

def generate_routing_uniform(num_tokens, num_experts, num_topk, device):
    """EPLB: perfectly uniform routing — each expert gets ~equal tokens."""
    scores = torch.ones((num_tokens, num_experts), dtype=torch.float32, device=device)
    # Add small noise to break ties
    scores += torch.randn_like(scores) * 0.01
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=device) / num_topk
    return topk_idx, topk_weights, scores


def generate_routing_skewed(num_tokens, num_experts, num_topk, skew_std, device):
    """Skewed routing: some experts are more popular.
    skew_std controls the imbalance — higher = more skewed.
      0.0 = uniform
      1.0 = mild skew
      3.0 = heavy skew (a few experts dominate)
    """
    # Expert popularity: log-normal distribution
    expert_popularity = torch.randn(num_experts, device=device) * skew_std
    expert_popularity = expert_popularity.exp()
    expert_popularity = expert_popularity / expert_popularity.sum() * num_experts

    # Scores = base popularity + per-token noise
    scores = expert_popularity.unsqueeze(0).expand(num_tokens, -1).clone()
    scores += torch.randn((num_tokens, num_experts), device=device) * 0.5
    scores = scores.abs() + 1e-6

    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=device) / num_topk
    return topk_idx, topk_weights, scores


def create_grouped_scores(scores, group_idx, num_groups):
    """Mask scores by selected groups (for grouped topk routing)."""
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def inplace_unique(x, num_slots):
    """Deduplicate rank indices per token (from DeepEP utils)."""
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def per_token_cast_to_fp8(x):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def bench_fn(fn, num_warmups=NUM_WARMUP, num_tests=NUM_RUNS):
    """Benchmark with L2 flush."""
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(num_warmups):
        fn()

    cache.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


def run_dispatch_bench(
    rank, world_size, num_nodes, num_local_ranks,
    buffer, group, config,
    num_tokens, hidden, topk_idx, topk_weights, scores,
    use_fp8, scenario_name,
):
    """Run one dispatch+combine benchmark for a given routing scenario."""
    num_experts = NUM_EXPERTS
    num_ranks = world_size

    # Prepare input
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_input = per_token_cast_to_fp8(x) if use_fp8 else x

    # Grouped routing (like DeepSeek-V3 / GLM-5)
    num_topk_groups = min(num_nodes, NUM_TOPK_GROUPS)
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx_grouped = torch.topk(masked_scores, NUM_TOPK, dim=-1, largest=True, sorted=False).indices

    # Compute dispatch layout
    rank_idx = topk_idx_grouped // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx_grouped == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx_grouped, num_experts)

    # ── Bench get_dispatch_layout ──
    t_layout, _, _ = bench_fn(lambda: buffer.get_dispatch_layout(topk_idx_grouped, num_experts))

    # ── Bench dispatch ──
    dispatch_args = {
        "x": x_input,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "topk_idx": topk_idx_grouped,
        "topk_weights": topk_weights,
        "config": config,
    }
    # Warm up and get handle for combine
    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, _ = \
        buffer.dispatch(**dispatch_args)

    t_dispatch, _, _ = bench_fn(lambda: buffer.dispatch(**dispatch_args))

    # ── Bench combine ──
    recv_x_for_combine = recv_x if not isinstance(recv_x, tuple) else \
        (recv_x[0].to(torch.float32).view(recv_x[0].size(0), -1, 128) * recv_x[1].view(recv_x[0].size(0), -1, 1)).view(recv_x[0].shape).to(torch.bfloat16)
    if isinstance(recv_x_for_combine, tuple):
        recv_x_for_combine = recv_x_for_combine[0]

    combine_args = {
        "x": recv_x_for_combine if not isinstance(recv_x, tuple) else recv_x_for_combine,
        "handle": handle,
        "topk_weights": recv_topk_weights,
        "config": config,
    }
    t_combine, _, _ = bench_fn(lambda: buffer.combine(**combine_args))

    # ── Stats ──
    recv_count = recv_x_for_combine.shape[0] if not isinstance(recv_x, tuple) else recv_x[0].shape[0]
    expert_counts = num_tokens_per_expert.cpu().tolist()
    experts_per_rank = num_experts // num_ranks
    local_expert_counts = expert_counts[rank * experts_per_rank:(rank + 1) * experts_per_rank]

    result = {
        "scenario": scenario_name,
        "rank": rank,
        "num_tokens": num_tokens,
        "recv_tokens": recv_count,
        "layout_ms": t_layout * 1000,
        "dispatch_ms": t_dispatch * 1000,
        "combine_ms": t_combine * 1000,
        "total_ms": (t_layout + t_dispatch + t_combine) * 1000,
        "expert_count_min": min(local_expert_counts) if local_expert_counts else 0,
        "expert_count_max": max(local_expert_counts) if local_expert_counts else 0,
        "expert_count_avg": sum(local_expert_counts) / len(local_expert_counts) if local_expert_counts else 0,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Worker (runs on each GPU)
# ══════════════════════════════════════════════════════════════════════════════

def worker(local_rank, num_local_ranks, args):
    num_nodes = args.nnodes
    rank, world_size, group = init_dist(local_rank, num_local_ranks, args)

    torch.manual_seed(rank)
    device = torch.device("cuda", local_rank)

    # Create DeepEP buffer
    # low_latency_mode=True for single-node (num_ranks <= 8, NVLink only)
    # low_latency_mode=False for multi-node (num_ranks > 8, uses RDMA)
    num_qps_per_rank = args.num_sms // 2
    use_low_latency = (world_size <= 8)
    buffer = deep_ep.Buffer(
        group,
        int(1e9),  # RDMA buffer
        int(1e9),  # NVL buffer
        low_latency_mode=use_low_latency,
        num_qps_per_rank=num_qps_per_rank,
    )

    nvl_buffer_size = 720 if world_size in (144, 160) else 512
    rdma_buffer_size = 128
    config = deep_ep.Config(args.num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    hidden = args.hidden
    num_experts = NUM_EXPERTS

    # Scenarios: (name, skew_std)
    scenarios = {
        "balanced": 0.0,
        "mild":     1.0,
        "medium":   2.0,
        "heavy":    3.0,
    }
    if args.scenario != "all":
        scenarios = {args.scenario: scenarios[args.scenario]}

    if rank == 0:
        print("=" * 120)
        print("GLM-5 DeepEP Dispatch Benchmark (Prefill)")
        print("=" * 120)
        print(f"Cluster:   {num_nodes} nodes × {num_local_ranks} GPUs = {world_size} ranks")
        print(f"Model:     hidden={hidden}, experts={num_experts}, top_k={NUM_TOPK}")
        print(f"M/GPU:     {M_PER_GPU_LIST}")
        print(f"Per GPU:   experts_per_gpu={num_experts // world_size}")
        print(f"FP8:       {args.use_fp8}")
        print(f"DeepEP:    num_sms={args.num_sms}, low_latency={use_low_latency}")
        print(f"Scenarios: {list(scenarios.keys())}")
        print("=" * 120, flush=True)

    all_results = []

    for num_tokens in M_PER_GPU_LIST:
        for scenario_name, skew_std in scenarios.items():
            dist.barrier(group=group)

            if skew_std == 0.0:
                topk_idx, topk_weights, scores = generate_routing_uniform(
                    num_tokens, num_experts, NUM_TOPK, device)
            else:
                topk_idx, topk_weights, scores = generate_routing_skewed(
                    num_tokens, num_experts, NUM_TOPK, skew_std, device)

            result = run_dispatch_bench(
                rank, world_size, num_nodes, num_local_ranks,
                buffer, group, config,
                num_tokens, hidden, topk_idx, topk_weights, scores,
                args.use_fp8, scenario_name,
            )
            all_results.append(result)

            if rank == 0:
                print(f"\n--- M={num_tokens}, Scenario: {scenario_name} (skew_std={skew_std}) ---")
                print(f"  layout:   {result['layout_ms']:.3f} ms")
                print(f"  dispatch: {result['dispatch_ms']:.3f} ms")
                print(f"  combine:  {result['combine_ms']:.3f} ms")
                print(f"  total:    {result['total_ms']:.3f} ms")
                print(f"  recv_tokens: {result['recv_tokens']}")
                print(f"  local expert load: min={result['expert_count_min']}, "
                      f"max={result['expert_count_max']}, avg={result['expert_count_avg']:.0f}", flush=True)

    # ── Summary table (rank 0 only) ──
    if rank == 0:
        print(f"\n{'='*120}")
        print("  Summary: DeepEP dispatch latency by M and scenario")
        print(f"{'='*120}")
        print(f"  {'M':>8s} {'scenario':<12s} {'layout(ms)':>12s} {'dispatch(ms)':>14s} {'combine(ms)':>13s} "
              f"{'total(ms)':>12s} {'recv_tok':>10s} {'exp_min':>9s} {'exp_max':>9s} {'exp_avg':>9s}")
        print(f"  {'-'*104}")
        for r in all_results:
            print(f"  {r['num_tokens']:>8d} {r['scenario']:<12s} {r['layout_ms']:>12.3f} {r['dispatch_ms']:>14.3f} "
                  f"{r['combine_ms']:>13.3f} {r['total_ms']:>12.3f} {r['recv_tokens']:>10d} "
                  f"{r['expert_count_min']:>9d} {r['expert_count_max']:>9d} {r['expert_count_avg']:>9.0f}")

        # CSV
        csv_path = "glm5_deepep_dispatch_perf.csv"
        with open(csv_path, "w") as f:
            f.write("scenario,rank,num_tokens,recv_tokens,layout_ms,dispatch_ms,combine_ms,total_ms,"
                    "expert_count_min,expert_count_max,expert_count_avg\n")
            for r in all_results:
                f.write(f"{r['scenario']},{r['rank']},{r['num_tokens']},{r['recv_tokens']},"
                        f"{r['layout_ms']:.4f},{r['dispatch_ms']:.4f},{r['combine_ms']:.4f},{r['total_ms']:.4f},"
                        f"{r['expert_count_min']},{r['expert_count_max']},{r['expert_count_avg']:.1f}\n")
        print(f"\nResults saved to {csv_path}", flush=True)

    dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Main (spawn-based)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--use-fp8", action="store_true", default=True)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "balanced", "mild", "medium", "heavy"])
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    num_local_ranks = 8
    print(f"Starting with {args}", flush=True)

    torch.multiprocessing.spawn(
        worker,
        args=(num_local_ranks, args),
        nprocs=num_local_ranks,
    )

