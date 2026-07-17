"""FP8 quantization helpers (verbatim from bench_glm5_decode.py) + TMA-align import.

All three quant recipes produce plain e4m3 blockwise/per-tensor scales (the
"dequant multiplier" convention: scale = amax / fp8_max, kernel multiplies back).
This matches the existing bench口径 (NOT UE8M0 — a deliberate choice).
"""
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # re-export

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def per_token_cast_to_fp8(x: torch.Tensor):
    """Activation quant: per-token, block_size=128 along K.
    Returns (x_fp8 [m,k] e4m3, x_scale [m, k//128] f32)."""
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, k = x.shape
    x_view = x.view(m, k // 128, 128)
    x_amax = x_view.abs().float().amax(dim=-1)
    x_scale = (x_amax / FP8_MAX).float().clamp(min=1e-12)
    x_fp8 = (x_view.float() / x_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k), x_scale


def per_block_cast_to_fp8(w: torch.Tensor):
    """Weight quant: 128x128 blocks (N padded up to 128).
    Returns (w_fp8 [n,k] e4m3, w_scale [ceil(n/128), k//128] f32)."""
    assert w.dim() == 2 and w.shape[1] % 128 == 0
    n, k = w.shape
    n_ceil = (n + 127) // 128 * 128
    w_padded = torch.zeros(n_ceil, k, dtype=w.dtype, device=w.device) if n < n_ceil else w
    if n < n_ceil:
        w_padded[:n] = w
    w_view = w_padded.view(n_ceil // 128, 128, k // 128, 128)
    w_amax = w_view.abs().float().amax(dim=(1, 3))
    w_scale = (w_amax / FP8_MAX).float().clamp(min=1e-12)
    w_fp8 = (w_view.float() / w_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.view(n_ceil, k)[:n].contiguous(), w_scale


def cast_to_fp8_per_tensor(x: torch.Tensor):
    """Per-tensor quant (for bmm_fp8). Returns (x_fp8 same-shape e4m3, scale [1] f32)."""
    amax = x.abs().float().amax()
    scale = (amax / FP8_MAX).float().clamp(min=1e-12)
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale.view(1).to(x.device)
