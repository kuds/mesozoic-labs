"""Frame dataset for JEPA pretraining.

Reads .npz shards produced by ``scripts/collect_jepa_data.py``. Each shard
holds a single array ``frames`` with shape ``(N, H, W, C)`` and ``uint8``
dtype, optionally accompanied by ``actions`` ``(N, A)`` and ``episode_ids``
``(N,)`` for reproducibility / V-JEPA-2-AC follow-on work.

The dataset is index-mapped lazily so that 1M+ frames can be served
without ever loading every shard into memory at once.

Implementation notes
--------------------
- ``torch`` and ``numpy`` are both required (numpy alone is enough for
  shard reading; torch is needed to return tensors).
- Temporal stacks of length ``T`` are sampled by drawing a starting
  index and taking ``T`` consecutive frames *within the same episode*
  (we use the ``episode_ids`` array to avoid crossing resets). If
  ``episode_ids`` is missing (single-shard random-policy dump), we fall
  back to plain consecutive indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ShardInfo:
    path: Path
    n_frames: int
    has_episode_ids: bool


class JEPAFrameDataset(Dataset):
    """Lazy shard-backed dataset of RGB frames for JEPA pretraining.

    Parameters
    ----------
    shard_paths: list of ``.npz`` file paths.
    temporal_frames: ``T``. Each ``__getitem__`` returns a tensor with
        shape ``(T, C, H, W)`` (or ``(C, H, W)`` if ``T==1``).
    frame_stride: spacing between consecutive frames in a clip (1 = every
        frame; 2 = every other frame, doubling the temporal receptive
        field for the same T).
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        temporal_frames: int = 1,
        frame_stride: int = 1,
    ) -> None:
        if temporal_frames < 1:
            raise ValueError("temporal_frames must be >= 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self._shards: List[ShardInfo] = []
        self._shard_handles: dict[int, np.lib.npyio.NpzFile] = {}
        # Index map: i-th valid clip start -> (shard_idx, local_start_idx).
        self._index_map: List[tuple[int, int]] = []
        self.temporal_frames = temporal_frames
        self.frame_stride = frame_stride

        for path in shard_paths:
            info = self._inspect(Path(path))
            self._shards.append(info)
            n_clips = max(0, info.n_frames - (temporal_frames - 1) * frame_stride)
            for start in range(n_clips):
                if info.has_episode_ids and not self._all_same_episode(
                    info, start
                ):
                    continue
                self._index_map.append((len(self._shards) - 1, start))
        if not self._index_map:
            raise ValueError(
                f"no valid clips found across {len(shard_paths)} shard(s) "
                f"with temporal_frames={temporal_frames}, stride={frame_stride}"
            )

    # -- shard inspection ------------------------------------------------

    def _inspect(self, path: Path) -> ShardInfo:
        with np.load(path, allow_pickle=False) as npz:
            if "frames" not in npz.files:
                raise ValueError(f"shard {path} missing 'frames' array")
            frames = npz["frames"]
            n = int(frames.shape[0])
            has_eps = "episode_ids" in npz.files
        return ShardInfo(path=path, n_frames=n, has_episode_ids=has_eps)

    def _open(self, shard_idx: int) -> np.lib.npyio.NpzFile:
        if shard_idx not in self._shard_handles:
            self._shard_handles[shard_idx] = np.load(
                self._shards[shard_idx].path, allow_pickle=False
            )
        return self._shard_handles[shard_idx]

    def _all_same_episode(self, info: ShardInfo, start: int) -> bool:
        npz = self._open(self._shards.index(info))
        eps = npz["episode_ids"]
        end = start + (self.temporal_frames - 1) * self.frame_stride
        if end >= eps.shape[0]:
            return False
        first = eps[start]
        for k in range(1, self.temporal_frames):
            if eps[start + k * self.frame_stride] != first:
                return False
        return True

    # -- Dataset API -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx, start = self._index_map[idx]
        npz = self._open(shard_idx)
        frames = npz["frames"]
        clip = np.stack(
            [
                frames[start + k * self.frame_stride]
                for k in range(self.temporal_frames)
            ],
            axis=0,
        )  # (T, H, W, C) uint8
        # NHWC uint8 -> NCHW float32 in [0, 1].
        clip_t = torch.from_numpy(clip).float() / 255.0
        clip_t = clip_t.permute(0, 3, 1, 2).contiguous()
        if self.temporal_frames == 1:
            return clip_t.squeeze(0)
        return clip_t

    def close(self) -> None:
        for h in self._shard_handles.values():
            h.close()
        self._shard_handles.clear()


def discover_shards(directory: str | Path, glob: str = "*.npz") -> list[Path]:
    """List shard paths in a directory, sorted by name for reproducibility."""
    return sorted(Path(directory).glob(glob))
