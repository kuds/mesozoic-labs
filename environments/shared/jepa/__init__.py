"""JEPA (Joint Embedding Predictive Architecture) for dinosaur RL.

Pretraining + downstream-RL utilities for I-JEPA / V-JEPA-lite-style
visual representation learning on Mesozoic Labs environments.

Public API
----------
- :class:`encoder.ViTEncoder` -- patch-token transformer encoder.
- :class:`predictor.JEPAPredictor` -- masked-target predictor.
- :func:`losses.jepa_masked_loss` -- normalized smooth-L1 loss.
- :func:`losses.ema_update` -- target-encoder momentum update.
- :class:`masking.MultiBlockMaskGenerator` -- I-JEPA-style block masking.
- :class:`dataset.JEPAFrameDataset` -- shard-backed frame loader.
- :class:`feature_extractor.JEPAFeatureExtractor` -- SB3 wrapper.

Heavy components import ``torch`` lazily so that the module can be
imported without the ``[jepa]`` extra installed (e.g. by code that only
wants to register a CLI flag).
"""

from __future__ import annotations

# Re-export pure-Python helpers eagerly (they only need numpy).
from .masking import MultiBlockMaskGenerator, sample_block_mask  # noqa: F401

__all__ = [
    "MultiBlockMaskGenerator",
    "sample_block_mask",
]
