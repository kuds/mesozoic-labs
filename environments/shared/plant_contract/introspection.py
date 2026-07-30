"""Small MuJoCo model-introspection helpers shared by the layer payloads."""

from __future__ import annotations

from typing import Any, Sequence

import mujoco
import numpy as np


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def _fields(source: Any, names: Sequence[str], indices: np.ndarray | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in names:
        value = np.asarray(getattr(source, name))
        if indices is not None:
            value = value[indices]
        payload[name] = value
    return payload
