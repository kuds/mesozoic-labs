"""I-JEPA-style multi-block mask generation, in pure NumPy.

The masks here select *patch indices* (not pixels) along the flattened
spatial grid produced by a ViT patch embedding. Two mask sets are
returned per sample:

- ``context_indices``: patches the context encoder sees.
- ``target_indices``: a list of disjoint blocks the predictor must
  reconstruct in embedding space.

The default settings follow the published I-JEPA recipe scaled down for
small grids: 4 target blocks each covering ~25% of the patches, with a
context that is the complement (so each sample has full coverage).

Pure NumPy keeps the module importable without ``torch``; we deliberately
do not depend on PyTorch here so unit tests can run on a fresh
checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class BlockMaskConfig:
    """Configuration for a single block-mask sample."""

    grid_h: int
    grid_w: int
    n_target_blocks: int = 4
    target_scale: Tuple[float, float] = (0.15, 0.20)
    target_aspect: Tuple[float, float] = (0.75, 1.5)
    min_context_patches: int = 8
    max_attempts: int = 32


def sample_block_mask(
    cfg: BlockMaskConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sample one (context, [target_blocks]) split for a single image.

    Returns
    -------
    context_indices: int64 array of patch indices in ``[0, H*W)`` for the
        context encoder to attend to. Always nonempty.
    target_blocks: list of ``n_target_blocks`` int64 arrays. Blocks are
        disjoint from each other and from the context. Some blocks may
        be empty if the rejection sampler runs out of room.
    """
    h, w = cfg.grid_h, cfg.grid_w
    grid = np.zeros((h, w), dtype=np.int8)  # 0=free, 1=target
    target_blocks: List[np.ndarray] = []

    for _ in range(cfg.n_target_blocks):
        block = _sample_one_block(grid, cfg, rng)
        if block is None:
            target_blocks.append(np.empty(0, dtype=np.int64))
            continue
        rows, cols = block
        flat = (rows * w + cols).astype(np.int64)
        grid[rows, cols] = 1
        target_blocks.append(flat)

    free_mask = (grid == 0).reshape(-1)
    context_indices = np.where(free_mask)[0].astype(np.int64)

    # Fall-back: if the targets ate everything (only possible with very
    # aggressive scales), force a single random patch back into the
    # context so the encoder has *something* to attend to.
    if context_indices.size < cfg.min_context_patches:
        deficit = cfg.min_context_patches - int(context_indices.size)
        # Grab `deficit` patches from the largest target block.
        donor_idx = int(np.argmax([b.size for b in target_blocks])) if target_blocks else 0
        donor = target_blocks[donor_idx]
        if donor.size > 0:
            take = donor[: min(deficit, donor.size)]
            target_blocks[donor_idx] = donor[take.size :]
            context_indices = np.sort(np.concatenate([context_indices, take]))

    return context_indices, target_blocks


def _sample_one_block(
    grid: np.ndarray,
    cfg: BlockMaskConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray] | None:
    h, w = grid.shape
    n_patches = h * w

    for _ in range(cfg.max_attempts):
        scale = rng.uniform(*cfg.target_scale)
        aspect = rng.uniform(*cfg.target_aspect)
        n_block = max(1, int(round(n_patches * scale)))
        bh = max(1, int(round(np.sqrt(n_block / aspect))))
        bw = max(1, int(round(np.sqrt(n_block * aspect))))
        bh = min(bh, h)
        bw = min(bw, w)
        if bh <= 0 or bw <= 0:
            continue
        top = int(rng.integers(0, h - bh + 1))
        left = int(rng.integers(0, w - bw + 1))
        rows = np.arange(top, top + bh)[:, None].repeat(bw, axis=1)
        cols = np.arange(left, left + bw)[None, :].repeat(bh, axis=0)
        if grid[rows, cols].any():
            continue  # overlaps an existing target
        return rows.reshape(-1), cols.reshape(-1)
    return None


class MultiBlockMaskGenerator:
    """Batched wrapper around :func:`sample_block_mask`.

    Use a single ``MultiBlockMaskGenerator`` per data loader worker so
    the seeds stay reproducible.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_target_blocks: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.20),
        target_aspect: Tuple[float, float] = (0.75, 1.5),
        seed: int | None = None,
    ) -> None:
        self.cfg = BlockMaskConfig(
            grid_h=grid_h,
            grid_w=grid_w,
            n_target_blocks=n_target_blocks,
            target_scale=target_scale,
            target_aspect=target_aspect,
        )
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        batch_size: int,
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        contexts: List[np.ndarray] = []
        targets: List[List[np.ndarray]] = []
        for _ in range(batch_size):
            ctx, tgt = sample_block_mask(self.cfg, self._rng)
            contexts.append(ctx)
            targets.append(tgt)
        return contexts, targets
