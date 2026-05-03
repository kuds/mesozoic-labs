"""Small ViT encoder for JEPA pretraining and downstream RL feature use.

Design choices
--------------
- ViT-Tiny defaults (depth=4, dim=192, heads=3, mlp_ratio=4) -- about
  4-5 M parameters at 84x84 with patch size 8. Fits in a Colab T4 with
  batch 64 and 4-frame video stacks at FP32.
- Sin/cos absolute positional embedding plus a single learned scalar
  per patch axis. We keep things simple and rely on the predictor to
  tell context vs. target tokens via positional embedding alone.
- Optional temporal dimension: when ``T > 1``, a single learned
  positional embedding is added per frame index, and the encoder
  treats the input as ``(B, T, C, H, W)``. The output is per-token
  per-frame ``(B, T, N, D)``.

Heavy ``torch`` imports happen at class-definition time. Modules in
``__init__.py`` keep this file optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------


def _sincos_2d_pos_embed(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """Standard sin/cos 2D positional embedding (DeiT/MAE-style).

    Returns a tensor of shape ``(grid_h * grid_w, dim)``.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos, got {dim}")
    h_pos = torch.arange(grid_h, dtype=torch.float32)
    w_pos = torch.arange(grid_w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(h_pos, w_pos, indexing="ij")
    embed_h = _sincos_1d(grid_y.flatten(), dim // 2)
    embed_w = _sincos_1d(grid_x.flatten(), dim // 2)
    return torch.cat([embed_h, embed_w], dim=1)  # (N, dim)


def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    omega = torch.arange(half, dtype=torch.float32) / float(half)
    omega = 1.0 / (10000.0 ** omega)
    out = positions[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


# ---------------------------------------------------------------------------
# Transformer block (compact, uses native scaled_dot_product_attention)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _MHSA(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHSA(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------


@dataclass
class ViTEncoderConfig:
    """Configuration record for :class:`ViTEncoder`."""

    image_size: int = 84
    patch_size: int = 8
    in_channels: int = 3
    dim: int = 192
    depth: int = 4
    heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    temporal_frames: int = 1  # 1 == still-image I-JEPA; >1 == V-JEPA-lite
    use_cls_token: bool = False

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} not divisible by patch_size {self.patch_size}"
            )
        return self.image_size // self.patch_size

    @property
    def n_patches_per_frame(self) -> int:
        g = self.grid_size
        return g * g

    @property
    def n_tokens(self) -> int:
        return self.n_patches_per_frame * max(1, self.temporal_frames)


class ViTEncoder(nn.Module):
    """ViT encoder used for both context and target encoders in JEPA.

    Forward returns per-token embeddings ``(B, N, D)`` where
    ``N = grid * grid * T`` (T = ``temporal_frames``, typically 1).
    """

    def __init__(self, config: ViTEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        # 2D spatial pos embedding -- registered as a buffer (not learned).
        spatial = _sincos_2d_pos_embed(config.grid_size, config.grid_size, config.dim)
        self.register_buffer("pos_embed_spatial", spatial, persistent=False)

        # Temporal pos embedding (learned, tiny).
        if config.temporal_frames > 1:
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(config.temporal_frames, config.dim)
            )
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            self.register_parameter("pos_embed_temporal", None)

        # Optional CLS token -- defaults off to keep the embedding pure
        # token-mean for downstream RL.
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        self.blocks = nn.ModuleList(
            [
                _Block(config.dim, config.heads, mlp_ratio=config.mlp_ratio, dropout=config.dropout)
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.dim)

        self.apply(_init_weights)

    @property
    def embed_dim(self) -> int:
        return self.config.dim

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Project pixels to patch tokens with positional embedding added.

        Input
        -----
        images: ``(B, C, H, W)`` for a single frame or ``(B, T, C, H, W)``
            for a temporal stack.

        Output
        ------
        tokens: ``(B, N, D)`` where ``N = T * grid * grid``.
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)  # add T axis
        if images.dim() != 5:
            raise ValueError(f"expected 4D or 5D input, got shape {tuple(images.shape)}")
        b, t, c, h, w = images.shape
        if t != max(1, self.config.temporal_frames):
            raise ValueError(
                f"expected temporal dim {self.config.temporal_frames}, got {t}"
            )
        # Patch embed each frame independently.
        flat = images.reshape(b * t, c, h, w)
        feats = self.patch_embed(flat)  # (B*T, D, gh, gw)
        feats = feats.flatten(2).transpose(1, 2)  # (B*T, N_per_frame, D)
        feats = feats + self.pos_embed_spatial[None]
        feats = feats.reshape(b, t, -1, feats.size(-1))
        if self.pos_embed_temporal is not None:
            feats = feats + self.pos_embed_temporal[None, :, None, :]
        feats = feats.reshape(b, -1, feats.size(-1))  # (B, T*N_per_frame, D)
        return feats

    def forward(
        self,
        images: torch.Tensor,
        keep_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode pixels.

        Parameters
        ----------
        images: pixel tensor in ``[0, 1]``, shape ``(B, C, H, W)`` or
            ``(B, T, C, H, W)``.
        keep_indices: ``(B, K)`` long tensor of token indices to retain
            *after* patch embedding. If provided, gathers the K context
            tokens before the transformer stack runs (the I-JEPA
            "context encoder" path). If None, all tokens are processed.
        """
        tokens = self.patchify(images)
        if keep_indices is not None:
            if keep_indices.dim() != 2:
                raise ValueError("keep_indices must be (B, K)")
            b, _, d = tokens.shape
            k = keep_indices.size(1)
            gather_idx = keep_indices.unsqueeze(-1).expand(b, k, d)
            tokens = torch.gather(tokens, dim=1, index=gather_idx)

        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens

    @torch.no_grad()
    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience: return a single ``(B, D)`` embedding by mean-pooling tokens.

        This is the embedding consumed by the SB3 feature extractor
        downstream.
        """
        out = self.forward(images)
        if self.cls_token is not None:
            return out[:, 0]
        return out.mean(dim=1)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def parameter_count(module: nn.Module) -> int:
    """Helper: number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def vit_tiny_84(temporal_frames: int = 1) -> ViTEncoder:
    """ViT-Tiny preset: 4 blocks, 192 dim, 84x84, patch 8 (~5 M params)."""
    return ViTEncoder(ViTEncoderConfig(temporal_frames=temporal_frames))


def vit_small_84(temporal_frames: int = 1) -> ViTEncoder:
    """ViT-Small preset: 6 blocks, 384 dim (~22 M params; larger run only)."""
    return ViTEncoder(
        ViTEncoderConfig(
            dim=384,
            depth=6,
            heads=6,
            temporal_frames=temporal_frames,
        )
    )


# Silence "imported but unused" for math when type checkers are strict.
_ = math
