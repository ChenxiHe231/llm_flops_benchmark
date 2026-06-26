# GLM-5 Operator Performance Benchmarks

GLM-5 模型各算子的 CUDA 性能测试工具集，基于 DeepGEMM、sgl_kernel、FlashMLA 等底层库，使用 CUDA Graph 精确计时。

## 依赖

- SGLang镜像sglang:v0.5.10，下载地址：https://hub.docker.com/r/lmsysorg/sglang/tags

## 模型参数

所有脚本使用统一的 GLM-5 模型配置：

| 参数 | 值 |
|------|-----|
| hidden_size | 6144 |
| q_lora_rank | 2048 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim | 64 |
| num_attention_heads | 64 |
| v_head_dim | 256 |
| index_n_heads | 32 |
| index_head_dim | 128 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 256 (全量) / 8 (单卡) |
| num_experts_per_tok | 8 |

---

## 测试脚本

### 1. bench_glm5_prefill.py — Prefill 阶段全算子性能

测试 sglang prefill 路径下的所有 GLM-5 算子，包括 Attention GEMM、MLA、DSA Indexer、MoE。

**覆盖算子：**
- Attention: `fused_qkv_a_proj`, `q_b_proj`, `absorbed_W_UK`, `absorbed_W_UV`, `o_proj`
- MLA: `flash_mla_with_kvcache`（paged KV cache）
- DSA Indexer: `index_k_proj`, `index_q_upproj`, `index_weights_proj`, `index_score`（fp8_mqa_logits）
- MoE: `gate_proj`, `up_proj`, `down_proj`（fp8_m_grouped_gemm_nt_masked）

**参数：**
- `M`（输入 token 数）：默认 [1024, 2048, 4096]
- `S`（KV 上下文长度）：默认 [65536]

**运行：**
```bash
python bench_glm5_prefill.py
```

**输出：** 各算子延迟（ms），按耗时降序排列的 summary 表，CSV 保存到 `glm5_unified_perf.csv`。

---

### 2. bench_glm5_decode.py — Decode 阶段全算子性能

测试 sglang decode 路径下的所有算子。与 prefill 的区别：M 为 batch_size（每请求 1 token），MLA 使用 decode kernel，DSA Indexer 使用 `fp8_paged_mqa_logits`。

**覆盖算子：** 同 prefill，但使用 decode 版本的 MLA 和 indexer score。

**参数：**
- `M`（batch_size）：默认 [1, 4, 8, 16, 32, 64]
- `S`（KV 上下文长度）：默认 [65536]

**运行：**
```bash
python bench_glm5_decode.py
```

**输出：** CSV 保存到 `glm5_decode_perf.csv`。

---

### 3. bench_glm5_deepep.py — DeepEP All-to-All 通信性能

测试 MoE 的 expert parallel 通信开销：`get_dispatch_layout` + `dispatch`（发送 token 到专家所在 GPU）+ `combine`（收集结果）。

**支持多种负载均衡场景：**
- `balanced`：EPLB 均匀路由
- `mild` / `medium` / `heavy`：递增的路由倾斜

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--nnodes` | 节点数 | 1 |
| `--node-rank` | 当前节点编号 | 0 |
| `--master-addr` | 主节点 IP | 127.0.0.1 |
| `--master-port` | 端口 | 29500 |
| `--hidden` | hidden dim | 6144 |
| `--num-sms` | DeepEP 使用的 SM 数 | 24 |
| `--use-fp8` | 启用 FP8 传输 | True |
| `--scenario` | 测试场景 | all |

**运行：**
```bash
# 单节点 8 卡
python bench_glm5_deepep.py --nnodes 1 --m-per-gpu 4096

# 多节点（每个节点上执行，指定各自的 node-rank）
# Node 0:
python bench_glm5_deepep.py --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 --m-per-gpu 4096
# Node 1:
python bench_glm5_deepep.py --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 --m-per-gpu 4096

# 只测 balanced 场景
python bench_glm5_deepep.py --nnodes 1 --scenario balanced
```

**输出：** layout / dispatch / combine 延迟，expert 负载统计，CSV 保存到 `glm5_deepep_dispatch_perf.csv`。

---

### 4. dsa_flashmla.py — FlashMLA Sparse Prefill 性能

测试 DSA（Dynamic Sparse Attention）中的 sparse prefill 算子 `flash_mla_sparse_fwd`，模拟不同的 KV cache 命中率。

**场景：**
- 总上下文 65536 tokens
- KV cache 命中率从 0% 到 90%（命中部分不需要 sparse prefill）
- `s_q = 65536 * (1 - hit_rate)`

**运行：**
```bash
python dsa_flashmla.py
```

**输出：** 各命中率下的延迟（ms）、TFlops、TB/s、计算访存比，CSV 保存到 `glm5_sparse_prefill_perf.csv`。

---

### 5. dsa_indexer.py — DSA Indexer GEMM (cuBLAS FP8) 性能

单独测试 DSA Indexer 的 4 个 GEMM 算子，使用 `torch._scaled_mm`（cuBLAS FP8）。

**覆盖算子：**
- `index_k_proj`：[S, 6144] × [6144, 128]
- `index_q_upproj`：[M, 2048] × [2048, 4096]
- `index_weights_proj`：[M, 6144] × [6144, 32]
- `index_score`：[32×M, 128] × [128, S]

**参数：**
- `M`：默认 [16, 256, 512, 1024]
- `S`：默认 [65536, 131072, 262144]

**运行：**
```bash
python dsa_indexer.py
```

**输出：** 延迟、TFlops、TB/s、计算访存比，CSV 保存到 `glm5_dsa_indexer_perf.csv`。

---

### 6. dsa_projection.py — Attention GEMM/BMM (DeepGEMM FP8) 性能

单独测试 MLA attention 中的 6 个 GEMM/BMM 算子，使用 DeepGEMM FP8。

**覆盖算子：**
- `q_a_proj`：GEMM [M, 6144] × [6144, 2048]
- `q_b_proj`：GEMM [M, 2048] × [2048, 16384]
- `absorbed_W_UK`：BMM batch=64, [M, 192] × [192, 512]
- `kv_a_proj`：GEMM [M, 6144] × [6144, 576]
- `absorbed_W_UV`：BMM batch=64, [M, 512] × [512, 256]
- `o_proj`：GEMM [M, 16384] × [16384, 6144]

**参数：**
- `M`：默认 [1024, 4096, 16384, 65536]

**运行：**
```bash
python dsa_projection.py
```

**输出：** 延迟、TFlops、TB/s，CSV 保存到 `glm5_attention_gemm_perf.csv`。

---

### 7. moe_deepgemm.py — MoE Grouped GEMM (DeepGEMM FP8) 性能

单独测试 MoE FFN 的 grouped GEMM（contiguous layout），使用 `deep_gemm.m_grouped_fp8_gemm_nt_contiguous`。测试多种随机 token 分布。

**覆盖算子：**
- `gate_proj`：K=6144, N=2048
- `up_proj`：K=6144, N=2048
- `down_proj`：K=2048, N=6144

**参数：**
- `TOTAL_TOKENS`：默认 128（16 × 8 experts）
- `NUM_DISTRIBUTIONS`：默认 5 种随机分布
- 可通过环境变量覆盖：`NUM_RUNS`、`NUM_WARMUP`、`NUM_DISTRIBUTIONS`

**运行：**
```bash
python moe_deepgemm.py

# 自定义
NUM_RUNS=50 NUM_DISTRIBUTIONS=10 python moe_deepgemm.py
```

**输出：** 各分布下的延迟、TFlops、TB/s，CSV 保存到 `glm5_moe_deepgemm_perf.csv`。

---

## 脚本关系

| 脚本 | 定位 | 适用场景 |
|------|------|----------|
| `bench_glm5_prefill.py` | 端到端 prefill | 评估单层 prefill 总耗时和瓶颈 |
| `bench_glm5_decode.py` | 端到端 decode | 评估单层 decode 总耗时和瓶颈 |
| `bench_glm5_deepep.py` | 通信 | 评估 MoE EP 通信开销 |
| `dsa_flashmla.py` | 单算子 | 评估 sparse attention 随命中率的变化 |
| `dsa_indexer.py` | 单算子 | 评估 DSA indexer 各 GEMM（cuBLAS） |
| `dsa_projection.py` | 单算子 | 评估 attention GEMM/BMM（DeepGEMM） |
| `moe_deepgemm.py` | 单算子 | 评估 MoE grouped GEMM 不同分布下性能 |
