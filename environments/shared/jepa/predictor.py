"""Narrow transformer predictor for JEPA masked-target reconstruction.

Given the *context* tokens produced by the (online) context encoder and
the positional locations of the *target* tokens, the predictor outputs
embeddings for the target locations. The training loss compares those
predictions against embeddings produced by the *EMA target* encoder on
the unmasked image.

Architecture follows the I-JEPA paper:
- predictor dim < encoder dim (we default to encoder_dim // 2);
- context tokens are passed through with their pos-embedding;
- a learned mask token (one per spatial position, broadcast across
  batch and time) is added at the target locations;
- the entire concatenated token list is run through 2 transformer
  blocks; outputs at target positions are extracted and projected back
  to encoder dim for the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import _Block  # type: ignore[attr-defined]


@dataclass
class JEPAPredictorConfig:
    """Configuration record for :class:`JEPAPredictor`."""

    encoder_dim: int = 192
    predictor_dim: int = 96
    depth: int = 2
    heads: int = 3
    mlp_ratio: float = 4.0
    n_total_tokens: int = 11 * 11  # one position per (frame, patch) slot


class JEPAPredictor(nn.Module):
    """Narrow transformer that maps context tokens -> target-token embeddings."""

    def __init__(self, config: JEPAPredictorConfig) -> None:
        super().__init__()
        self.config = config
        d = config.predictor_dim

        self.context_proj = nn.Linear(config.encoder_dim, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Learned absolute positional embedding, shared between context and
        # target tokens. Size = total possible token positions.
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_total_tokens, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _Block(d, config.heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, config.encoder_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target embeddings.

        Parameters
        ----------
        context_tokens: ``(B, K, encoder_dim)`` -- output of the context
            encoder on the K visible patches.
        context_indices: ``(B, K)`` -- absolute token positions of the
            context patches in ``[0, n_total_tokens)``.
        target_indices: ``(B, M)`` -- absolute token positions of the
            target patches.

        Returns
        -------
        ``(B, M, encoder_dim)`` -- predicted target embeddings.
        """
        b, k, _ = context_tokens.shape
        m = target_indices.size(1)

        d = self.config.predictor_dim
        ctx = self.context_proj(context_tokens) + _gather_pos(
            self.pos_embed, context_indices
        )
        tgt = self.mask_token.expand(b, m, d) + _gather_pos(self.pos_embed, target_indices)

        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        target_out = x[:, k:, :]
        return self.out_proj(target_out)


def _gather_pos(pos_embed: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather positional embeddings at the given token indices.

    pos_embed: ``(1, N, D)``.
    indices: ``(B, K)`` long tensor.
    Returns ``(B, K, D)``.
    """
    b, k = indices.shape
    d = pos_embed.size(-1)
    expanded = pos_embed.expand(b, -1, -1)
    gather_idx = indices.unsqueeze(-1).expand(b, k, d)
    return torch.gather(expanded, dim=1, index=gather_idx)


# `F` re-exported so dependent modules can find it without
# re-importing torch.nn.functional separately.
_ = F
