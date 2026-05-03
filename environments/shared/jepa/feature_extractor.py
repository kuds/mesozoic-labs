"""SB3 feature extractor that wraps a (frozen or fine-tuned) JEPA encoder.

Plumbing: SB3's ``MlpPolicy`` calls ``features_extractor(obs)`` and feeds
the resulting tensor into the actor / critic MLPs. We override the
extractor to:

1. read the ``"pixels"`` slot from a Dict observation (shape
   ``(B, H, W, C)`` uint8 by default);
2. permute to ``(B, C, H, W)`` and rescale to ``[0, 1]``;
3. run the JEPA encoder, producing per-token embeddings; mean-pool to a
   single ``(B, D)`` vector;
4. optionally concatenate the proprioceptive ``"state"`` vector (kept by
   the wrapper alongside pixels) so the policy gets the best of both.

The encoder weights are loaded from a JEPA pretraining checkpoint and
are frozen by default. Setting ``trainable_encoder=True`` enables
end-to-end fine-tuning -- but the SB3 optimizer will treat the encoder
parameters at the same learning rate as the policy MLP, which is
typically too high; pair this with a lower base LR or per-param-group
LR scaling in the user's training script.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError as e:  # pragma: no cover - the [train] extra provides SB3
    raise ImportError(
        "JEPAFeatureExtractor requires stable-baselines3. Install with "
        "`pip install -e .[train,jepa]`."
    ) from e

from .encoder import ViTEncoder, ViTEncoderConfig
from .losses import freeze_module

logger = logging.getLogger(__name__)


class JEPAFeatureExtractor(BaseFeaturesExtractor):
    """Frozen-or-trainable JEPA encoder as an SB3 feature extractor.

    Parameters
    ----------
    observation_space: must be ``gym.spaces.Dict`` with at least a
        ``"pixels"`` key. ``"state"`` is optional but recommended.
    encoder_config: ``ViTEncoderConfig`` matching the pretraining run.
    checkpoint_path: path to a torch checkpoint produced by
        ``train_jepa.py`` (containing a ``"context_encoder"`` state
        dict). May be ``None`` to start from a fresh init -- useful for
        ablation runs.
    trainable_encoder: if False (default), encoder is frozen and put in
        ``eval()`` mode permanently.
    include_state: if True (default) and a ``"state"`` key is present,
        concatenate the raw state vector to the pooled embedding before
        returning to the policy.
    pool: ``"mean"`` (default) or ``"cls"``. ``"cls"`` requires the
        encoder to have been built with ``use_cls_token=True``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        encoder_config: ViTEncoderConfig,
        checkpoint_path: str | Path | None = None,
        trainable_encoder: bool = False,
        include_state: bool = True,
        pool: str = "mean",
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(
                "JEPAFeatureExtractor requires a Dict observation space "
                "(use PixelObservationWrapper)."
            )
        if "pixels" not in observation_space.spaces:
            raise ValueError("observation_space must include a 'pixels' key.")

        self._has_state = include_state and "state" in observation_space.spaces
        if self._has_state:
            state_dim = int(np.prod(observation_space["state"].shape))
        else:
            state_dim = 0

        features_dim = encoder_config.dim + state_dim
        super().__init__(observation_space, features_dim=features_dim)

        if pool not in {"mean", "cls"}:
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")
        if pool == "cls" and not encoder_config.use_cls_token:
            raise ValueError("pool='cls' requires encoder_config.use_cls_token=True")
        self._pool = pool
        self._trainable = bool(trainable_encoder)

        self.encoder = ViTEncoder(encoder_config)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if not self._trainable:
            freeze_module(self.encoder)
            self.encoder.eval()

        self._pixels_shape = observation_space["pixels"].shape
        self._channels_last = self._pixels_shape[-1] in {1, 3} and len(self._pixels_shape) == 3
        self._uint8_input = observation_space["pixels"].dtype == np.uint8

    # -- forward ---------------------------------------------------------

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        pixels = observations["pixels"]
        pixels = self._prepare_pixels(pixels)
        if self._trainable:
            tokens = self.encoder(pixels)
        else:
            with torch.no_grad():
                tokens = self.encoder(pixels)

        if self._pool == "cls":
            embed = tokens[:, 0]
        else:
            embed = tokens.mean(dim=1)

        if self._has_state:
            state = observations["state"]
            if state.dim() > 2:
                state = state.flatten(1)
            embed = torch.cat([embed, state], dim=1)
        return embed

    def _prepare_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if self._channels_last and pixels.dim() == 4:
            # SB3 may stack frames along channels; here we assume one frame.
            pixels = pixels.permute(0, 3, 1, 2)
        if self._uint8_input or pixels.dtype == torch.uint8:
            pixels = pixels.to(torch.float32) / 255.0
        return pixels

    # -- checkpoint helpers ---------------------------------------------

    def _load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location="cpu")
        # Accept either a flat state dict or the structured dict produced
        # by train_jepa.py.
        state = ckpt.get("context_encoder", ckpt)
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint structure at {path}")
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            logger.warning("JEPA checkpoint missing keys: %s", missing[:8])
        if unexpected:
            logger.warning("JEPA checkpoint unexpected keys: %s", unexpected[:8])

    # -- diagnostics -----------------------------------------------------

    def parameter_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        return {"encoder_total": total, "encoder_trainable": trainable}


# Re-export the nn alias so SB3 can resolve the module from policy_kwargs.
_ = nn
