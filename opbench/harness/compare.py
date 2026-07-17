"""Cosine comparison with per-output-kind masking.

output_kind determines how the raw backend/candidate output is reduced to a
comparable vector before cosine:
  dense          -> flatten everything
  masked_grouped -> keep only masked_m[e] valid rows per expert, then flatten
  mla_sparse     -> take main output (tuple[0]), flatten
  logits_ksrange -> zero out positions outside [ks,ke) in BOTH, flatten
  logits_paged   -> zero out positions >= seqlens[row] in BOTH, flatten
"""
import torch


def _main(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item()


def prepare(out, kind: str, inputs: dict):
    """Return a flat comparable tensor from a raw backend/candidate output."""
    out = _main(out)
    if kind == "dense":
        return out.reshape(-1)
    if kind == "mla_sparse":
        return out.reshape(-1)
    if kind == "masked_grouped":
        masked_m = inputs["masked_m"]
        E = out.shape[0]
        parts = [out[e, : int(masked_m[e].item())].reshape(-1)
                 for e in range(E) if int(masked_m[e].item()) > 0]
        return torch.cat(parts) if parts else out.reshape(-1)
    if kind == "logits_ksrange":
        M, S = out.shape[-2], out.shape[-1]
        ks = inputs["ks"].view(-1, 1)
        ke = inputs["ke"].view(-1, 1)
        col = torch.arange(S, device=out.device).view(1, -1)
        mask = (col >= ks) & (col < ke)
        return (out * mask).reshape(-1)
    if kind == "logits_paged":
        S = out.shape[-1]
        seqlens = inputs["seqlens"].view(-1, 1)
        col = torch.arange(S, device=out.device).view(1, -1)
        mask = col < seqlens
        return (out[..., :S] * mask).reshape(-1)
    raise ValueError(f"unknown output_kind {kind}")


def compare(ref_out, cand_out, kind: str, inputs: dict) -> float:
    """Cosine similarity between reference and candidate after masking."""
    ref = prepare(ref_out, kind, inputs)
    cand = prepare(cand_out, kind, inputs)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch ref{tuple(ref.shape)} vs cand{tuple(cand.shape)}")
    return cosine(ref, cand)
