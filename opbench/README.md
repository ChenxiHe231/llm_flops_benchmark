# opbench — GLM-5 算子级 baseline / 验证框架

对 12 个 GLM-5 算子,按 **(算子 × 相位 × M) = task** 提供:单算子的正确性(cosine)/ latency / MFU,以及全层端到端 latency。
每个脚本传入 `--op --M`(`allLatency` 只需 `--M`),相位从 M 自动推断(prefill M∈{1024,2048,4096} / decode M∈{16,32}),S 默认 65536。

## 四个脚本

| 脚本 | 作用 | 粒度 |
|---|---|---|
| `verify.py` | 真实后端 vs candidate,同一份 frozen 输入,cosine ≥ 阈值 | 单算子 |
| `latency.py` | 真实后端 baseline(有 candidate 则给 speedup) | 单算子 |
| `mfu.py` | MFU + 带宽利用率(自动标 compute/memory-bound) | 单算子 |
| `allLatency.py` | 全层所有算子跑一遍、按耗时排序 + 层总延迟,有 candidate 的换 candidate | 全层端到端 |

## 12 算子 → 真实后端

| 算子 | 后端 |
|---|---|
| fused_qkv_a, q_b, o_proj, index_k, index_q_upproj | `deep_gemm.fp8_gemm_nt` |
| absorbed_W_UK, absorbed_W_UV | `sgl_kernel.bmm_fp8` |
| moe_gate, moe_up, moe_down | `deep_gemm.fp8_m_grouped_gemm_nt_masked` |
| dsa_attn | `sgl_kernel.flash_mla_sparse_fwd` |
| index_score | prefill `fp8_mqa_logits` / decode `fp8_paged_mqa_logits` |

> **覆盖范围说明**:这 12 个算子 **不等于** 完整 DSA layer。未建模:`index_weights_proj`(bf16 `deep_gemm.bf16_gemm_nt`)以及小 batch(M≤16)下 `fused_qkv_a` 走的 BF16 融合 `dsv3_fused_a_gemm` 路径。这与 `bench_glm5_*.py` 的口径一致。

## 依赖

需要一个装好 `torch` / `deep_gemm` / `sgl_kernel`(含 `flash_mla`)的 Python 环境(如 SGLang 镜像里的环境),且有一块可见 GPU(Blackwell/B200,因为 deep_gemm 的 fp8 GEMM 走 UE8M0)。

## 用法

```bash
# 用你自己装了 deep_gemm/sgl_kernel/flash_mla 的 python
cd opbench

# 正确性:真实后端 vs candidate,同一份 frozen 输入,cosine >= 阈值
python verify.py  --op fused_qkv_a --M 4096

# latency:真实后端 baseline(有 candidate 则一并给 speedup)
python latency.py --op dsa_attn --M 32

# MFU + 带宽利用率(自动标 compute/memory-bound)
python mfu.py     --op q_b --M 4096

# 全层端到端:所有算子跑一遍,有 candidate 的换 candidate,出层总延迟 + 端到端 delta
python allLatency.py --M 32                 # decode 层,所有 candidate 生效
python allLatency.py --M 4096               # prefill 层
python allLatency.py --M 32 --op o_proj     # 只 o_proj 用 candidate,其余 backend(隔离单个 candidate 的影响)

# 指定 GPU:加 --device cuda:N(默认 cuda:0)
```

## 计时方式(默认对齐 kernel-harness)

`latency.py` / `allLatency.py` 有两种计时,`--timing` 选择:

- **`kh`(默认)** —— kernel-harness 口径:**每轮 flush L2 + 重新 clone 输入 + median CUDA-event**。cold-cache,更贴近真实 serving(权重在层间已被冲出 L2,每次都要从 HBM 冷读),数字可和 best-kernels reward bench 比。cupti 装了会自动升级到设备端 kernel 时间。
- **`graph`** —— 原始 CUDA-graph、warm-L2、mean。快,但会**美化 memory-bound 算子**(复用了缓存的权重,隐藏真实 HBM 流量)。仅作快速 sanity。

> 同一个 candidate 两种口径可能给出**相反**结论:例如 o_proj 的 compiled_dims 变体在 `graph` 下 1.07x(看着快),在 `kh` 下 ~0.98x(其实略慢)——那个"加速"只存在于缓存命中的假象里。评分/排名一律用 `kh`。

## candidate 怎么接入

把 `tasks/_template_impl.py` 拷到 `tasks/{算子}/{相位}/impl.py`,实现 `run(inputs)->out`。
- harness **拥有输入生成**:同一份量化好的 fp8 输入同时喂真实后端和 candidate(公平对比结构性保证)。
- candidate **禁止**重新量化 / 重新 seed / 造新随机数。
- **不写 impl.py 时,脚本自动测真实后端 baseline**(verify 得 cosine≈1)。
- `tasks/{算子}/{相位}/` 下的 candidate 被 `.gitignore` 排除(属于具体接入产物,不进框架仓库);模板 `tasks/_template_impl.py` 保留。
- candidate 的实现(别处生成的 kernel)常需薄适配:拆 `inputs` dict → 位置参数,必要时把 opbench 的 float32 UE8M0 scale 重打包成 kernel 需要的格式(见已有 `tasks/*/impl.py` 例子)。

## 阈值(分层)

- GEMM / MoE / index_score → **cosine ≥ 0.999**(对齐 DeepGEMM 官方测试)
- bmm_fp8 / flash_mla → **cosine ≥ 0.99**(对齐 sglang 测试)

## 量化(UE8M0 必需)

- `deep_gemm.fp8_gemm_nt` 和 `fp8_m_grouped_gemm_nt_masked` 在这台 B200 的 deep_gemm build 上**强制要求 UE8M0(2 的幂)scale**(否则设备断言 `smxx_layout.cuh:232`)。**bench_glm5_{decode,prefill}.py 原样跑也会崩这个断言**(它们用普通 fp32 scale)。
- harness 对这两条路径用官方 `deep_gemm.utils.math.per_token_cast_to_fp8(x, use_ue8m0=True)` / `per_block_cast_to_fp8(w, use_ue8m0=True)`——scale 先 round 成 2 的幂再量化数据,fp8 数据与 scale 一致。
- `bmm_fp8`(cuBLAS per-tensor)、`flash_mla`(bf16)、`index_score`(mqa_logits)**不需要** UE8M0,按普通 e4m3。
- index_score decode 的 paged KV cache 用**真实 fp8 量化数据 + 内嵌 scale** 构造 132 布局(不是随机 uint8,否则 fp8-NaN 字节使 logits/cosine 变 NaN);seqlens/context_lens 必须 **2D `[M,1]`**(metadata 和 kernel 都断言 dim==2)。

## B200 峰值(dense)

`FP8=4.5 PF/s, BF16=2.25 PF/s, HBM=7.7 TB/s`。小 M 的 decode GEMM 天然 latency/memory-bound,低 MFU 正常。
