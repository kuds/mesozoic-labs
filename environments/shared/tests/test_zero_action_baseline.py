"""Tests for the zero-action baseline diagnostic.

``score()``'s return value is serialised straight into the baseline records the
training notebook writes to Google Drive, so it has to survive a strict JSON
round-trip.  It did not: a species whose statue never reaches the horizon
produced ``float("nan")``, and ``json.dumps`` renders that as a bare ``NaN``
literal, which RFC 8259 does not permit.  Brachiosaurus hits that case today.
"""

import json

import numpy as np
import pytest

from environments.shared.scripts.zero_action_baseline import score


class _StubEnv:
    """Minimal env stand-in: ``score`` only needs these four members."""

    def __init__(self, lengths, horizon=1000, reset_noise_scale=0.1):
        self.max_episode_steps = horizon
        self.reset_noise_scale = reset_noise_scale
        self.action_space = type("Box", (), {"shape": (3,)})()
        self._lengths = list(lengths)
        self._step = 0
        self._episode = -1

    def reset(self, seed=None):
        self._episode += 1
        self._step = 0

    def step(self, action):
        self._step += 1
        done = self._step >= self._lengths[self._episode]
        truncated = done and self._lengths[self._episode] >= self.max_episode_steps
        info = {} if truncated else {"termination_reason": "fallen"}
        return None, 1.0, done and not truncated, truncated, info


def _strict_json_round_trip(payload):
    """Serialise and parse with bare NaN/Infinity literals rejected."""

    def _reject(constant):
        raise ValueError(f"bare {constant} literal is not valid JSON")

    return json.loads(json.dumps(payload), parse_constant=_reject)


class TestScoreIsJsonSerialisable:
    def test_no_standing_episodes_serialises_strictly(self):
        """The regression: nothing reaches the horizon, so there is no standing score."""
        result = score(_StubEnv([100, 250, 400]), episodes=3, seed=0)

        assert result["n_standing"] == 0
        assert result["reward_mean_standing"] is None, (
            "must be None, not NaN — json.dumps renders NaN as a bare literal that "
            "strict JSON parsers reject, and this dict is written to Drive"
        )
        assert result["reward_std_standing"] is None

        parsed = _strict_json_round_trip(result)
        assert parsed["reward_mean_standing"] is None

    def test_some_standing_episodes_report_a_number(self):
        result = score(_StubEnv([100, 1000, 1000]), episodes=3, seed=0)

        assert result["n_standing"] == 2
        assert result["reward_mean_standing"] == pytest.approx(1000.0)
        assert result["reward_std_standing"] == pytest.approx(0.0)
        assert result["full_horizon_share"] == pytest.approx(2 / 3)

        parsed = _strict_json_round_trip(result)
        assert parsed["reward_mean_standing"] == pytest.approx(1000.0)

    def test_standing_score_exceeds_the_unconditional_mean(self):
        """The property the number exists for: surviving episodes are worth more."""
        result = score(_StubEnv([100, 200, 1000, 1000]), episodes=4, seed=0)

        assert result["reward_mean_standing"] > result["reward_mean"], (
            "a per-step reward makes full-horizon episodes score above the mix, "
            "which is why a gate calibrated on the unconditional mean under-binds"
        )

    def test_every_value_is_json_native(self):
        """No numpy scalars: they serialise inconsistently across versions."""
        result = score(_StubEnv([100, 1000]), episodes=2, seed=0)
        for key, value in result.items():
            assert not isinstance(value, np.generic), f"{key} is a numpy scalar"
