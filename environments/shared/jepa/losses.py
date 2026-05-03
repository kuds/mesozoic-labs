"""Loss + EMA helpers for JEPA pretraining.

We follow the V-JEPA 2 recipe:

- Per-token L2-normalize / LayerNorm the *target* embeddings before the
  loss, which empirically prevents representation collapse better than
  raw L2 on absolute embeddings.
- Smooth-L1 (Huber) on the (predicted - target) residual.
- EMA target encoder, momentum schedule cosine-rampable from m_start to
  m_end across training.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def jepa_masked_loss(
    predicted_targets: torch.Tensor,
    actual_targets: torch.Tensor,
    *,
    normalize_targets: bool = True,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 loss between predictor output and target encoder output.

    Both inputs must have shape ``(B, M, D)`` -- one prediction per
    masked target token. We mean-reduce over batch, target tokens, and
    embedding dim, in that order, to match the reference implementation.

    Parameters
    ----------
    predicted_targets: ``(B, M, D)`` predictor output.
    actual_targets: ``(B, M, D)`` target-encoder output at the same
        token positions.
    normalize_targets: if True (default), apply LayerNorm-style
        per-token normalization (zero mean, unit variance over the D
        embedding dim) to the targets before the loss. Predicted
        embeddings are left unnormalized so the predictor must learn to
        match the normalized scale.
    beta: smooth-L1 transition point (Huber delta).

    Returns
    -------
    Scalar loss tensor.
    """
    if predicted_targets.shape != actual_targets.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(predicted_targets.shape)} vs "
            f"target {tuple(actual_targets.shape)}"
        )

    target = actual_targets.detach()
    if normalize_targets:
        target = F.layer_norm(target, normalized_shape=(target.size(-1),))

    return F.smooth_l1_loss(predicted_targets, target, beta=beta, reduction="mean")


@torch.no_grad()
def ema_update(
    target_module: nn.Module,
    online_module: nn.Module,
    momentum: float,
) -> None:
    """In-place EMA update: ``target = momentum * target + (1-momentum) * online``.

    Works for any pair of modules with identical parameter shapes.
    Buffers (e.g. positional sin/cos tables) are *copied* unchanged.
    """
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum}")
    for p_t, p_o in zip(target_module.parameters(), online_module.parameters(), strict=True):
        p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
    for b_t, b_o in zip(target_module.buffers(), online_module.buffers(), strict=True):
        b_t.copy_(b_o)


def cosine_momentum_schedule(
    step: int,
    total_steps: int,
    *,
    m_start: float = 0.996,
    m_end: float = 1.0,
) -> float:
    """Cosine ramp from ``m_start`` to ``m_end`` over ``total_steps``.

    Returns ``m_start`` at step 0 and ``m_end`` at the final step. Outside
    the range it clamps so wandering callers do not crash on edge cases.
    """
    if total_steps <= 0:
        return m_end
    progress = max(0.0, min(1.0, step / total_steps))
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return m_start + (m_end - m_start) * cosine


def freeze_module(module: nn.Module) -> None:
    """Set ``requires_grad=False`` for every parameter."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module, *, names: Iterable[str] | None = None) -> None:
    """Set ``requires_grad=True`` for every parameter (or a named subset)."""
    if names is None:
        for p in module.parameters():
            p.requires_grad = True
        return
    name_set = set(names)
    for n, p in module.named_parameters():
        if any(n.startswith(prefix) for prefix in name_set):
            p.requires_grad = True
