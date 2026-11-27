"""Each walk's stand block is its stance's reward, line for line (decision D-D62).

A walk's ``[commands.stand_weights]`` is paid in place of its ``[env]`` weights on every step under an exactly zero
command, so that zero means stand.  The rule every committed walk follows: at a zero command each reward weight is
the stance's (``stance.toml`` ``[env]``, or the constructor default the stance runs at), and so is each setting of a
term (a tolerance, a reference, a force floor), except where this module lists the difference with its reason.  A
weight both share is paid anyway and is not restated; one the walk's ``[env]`` holds at another value is restated in
the stand block, and the stand block swaps weights only (``BaseDinoEnv._stand_block``), so a stand term's settings
are declared in the walk's ``[env]``, each at the term's weight 0 there.

The tests hold that rule from both sides, so neither the stance nor the walk can drift without one failing:

* each weight at a zero command is the stance's, or a listed difference whose values really differ;
* the stand block restates no weight the walk already pays;
* each stand weight and each stand-term setting names the ``stance.toml`` line it was copied from
  (``# stance.toml:N``), and that line holds the same key at the same value;
* every setting is the stance's, or a listed difference;
* the stand-term settings move no reward on a cruise step (their terms are at weight 0 there), and do move the
  reward on a stand step (the block runs them).
"""

from __future__ import annotations

import inspect
import re
import tomllib
from functools import cache
from typing import Any

import numpy as np
import pytest

from environments.shared.command_frame import COMMAND_ENV_KEYS
from environments.shared.command_tracking_panel import reset_command_family
from environments.shared.config import build_env, load_stage_config
from environments.shared.curriculum.command_tracking_gate import CRUISE_FAMILY, STAND_FAMILY
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_names import model_ids
from environments.shared.species_registry import get_species_config
from environments.shared.task_fingerprint import NON_TASK_ENV_PARAMS

MODEL_IDS = model_ids(backend="stable-baselines3")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
ALLOSAURUS_FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")

#: The non-``*_weight`` names the stand block swaps (``BaseDinoEnv._STAND_WEIGHT_NAMES``).
STAND_WEIGHT_NAMES = frozenset({"alive_bonus", "support_conditioned_alive_fraction"})

_IDLE = (
    "the idle penalty is never paid at a live zero command (BaseDinoEnv._speed_terms, D-D62 (3)), so a stand step "
    "pays the stance's 0 with no stand weight (the block may name it only at 0)"
)
_UNREAD = "read by no reward since D-D17 retired the MJX trainer, its only reader; it stays hashed"
#: Reward weights a zero command pays at another value than the stance's, on purpose.
WEIGHT_DIFFERENCES: dict[ModelId, dict[str, str]] = {
    VELOCIRAPTOR_MONGOLIENSIS: {},
    TYRANNOSAURUS_REX: {},
    BRACHIOSAURUS_ALTITHORAX: {"idle_penalty_weight": _IDLE},
    DIBOTHROSUCHUS_ELAPHROS: {"idle_penalty_weight": _IDLE, "foot_contact_weight": _UNREAD},
    COMPSOGNATHUS_LONGIPES: {},
    COMPSOGNATHUS_LONGIPES_ROBOT: {},
    ALLOSAURUS_FRAGILIS: {},
}

_FALL = "a terminal penalty, not a weight the stand block swaps: a stand episode that falls pays the walk's"
_FORWARD_ONLY = "scales only the forward and backward terms, which are 0 on every walk step, stand or cruise"
_TERMINATION = "a termination on every step, cruise steps included; the walk keeps its own"
#: Settings (constructor parameters that are not stand-block weights) the walk holds at another value than the
#: stance's, on purpose.  The stand block swaps weights only, so each of these is also what a stand step runs.
SETTING_DIFFERENCES: dict[ModelId, dict[str, str]] = {
    VELOCIRAPTOR_MONGOLIENSIS: {"fall_penalty": _FALL, "forward_vel_max": _FORWARD_ONLY},
    TYRANNOSAURUS_REX: {
        "fall_penalty": _FALL,
        "forward_vel_max": _FORWARD_ONLY,
        "nosedive_termination_threshold": _TERMINATION,
        "action_penalty_source": (
            "selects the action the energy, smoothness, jerk and saturation terms read on every step: the stance's "
            "raw action would move the energy and smoothness terms the walk keeps on cruise steps"
        ),
        "height_target_tolerance": (
            "shapes the height term the walk keeps on cruise steps (height_weight 0.3): the stance's tolerance "
            "would move it"
        ),
        "foot_load_balance_airborne_penalty": (
            "a penalty paid whatever foot_load_balance_weight is (reward_functions.reward_foot_load_balance), so "
            "in [env] it would price every airborne cruise step, and it is not a weight the stand block swaps"
        ),
    },
    BRACHIOSAURUS_ALTITHORAX: {
        "forward_vel_max": _FORWARD_ONLY,
        "natural_pitch": "the pitch reference of the posture and nosedive terms the walk keeps on cruise steps",
        "idle_velocity_threshold": "the idle penalty's speed scale; the idle penalty is never paid at a zero command",
        "healthy_z_range": _TERMINATION,
    },
    DIBOTHROSUCHUS_ELAPHROS: {
        "fall_penalty": _FALL,
        "forward_vel_max": _FORWARD_ONLY,
        "nosedive_termination_threshold": _TERMINATION,
        "idle_velocity_threshold": "the idle penalty's speed scale; the idle penalty is never paid at a zero command",
        "healthy_z_range": _TERMINATION,
        "foot_contact_gate": _UNREAD,
    },
    COMPSOGNATHUS_LONGIPES: {},
    COMPSOGNATHUS_LONGIPES_ROBOT: {},
    ALLOSAURUS_FRAGILIS: {"max_episode_steps": "the walk's horizon, 30 s, which the eight-metre course needs"},
}

#: A walk line that cites the stance line it copies: ``key = value  # stance.toml:N ...``.
_CITATION = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[^#]+?)\s*#\s*stance\.toml:(?P<line>\d+)\b")


def _ids(model_id: ModelId) -> str:
    return f"{model_id.species}-{model_id.embodiment}"


def _model_dir(model_id: ModelId) -> Any:
    return model_id.path(REPOSITORY_ROOT / "configs")


@cache
def _defaults(model_id: ModelId) -> dict[str, Any]:
    """The env class's constructor defaults, task parameters only."""
    parameters = inspect.signature(get_species_config(model_id).env_class).parameters
    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and name not in NON_TASK_ENV_PARAMS
        and name not in COMMAND_ENV_KEYS
    }


@cache
def _effective(model_id: ModelId, stage: str) -> dict[str, Any]:
    """The stage's effective constructor kwargs: the defaults overlaid with its loaded ``[env]``."""
    env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
    return {**_defaults(model_id), **{k: v for k, v in env_kwargs.items() if k not in COMMAND_ENV_KEYS}}


def _stand(model_id: ModelId) -> dict[str, float]:
    config: DirectionCommandConfig = load_stage_config(model_id, "locomotion")["env_kwargs"]["command_config"]
    return config.stand_weight_map()


def _is_weight(model_id: ModelId, name: str) -> bool:
    """Whether the stand block may swap *name*: a numeric reward weight (``BaseDinoEnv._check_stand_weights``)."""
    default = _defaults(model_id)[name]
    numeric = isinstance(default, (int, float)) and not isinstance(default, bool)
    return numeric and (name.endswith("_weight") or name in STAND_WEIGHT_NAMES)


def _citations(model_id: ModelId) -> dict[str, dict[str, tuple[str, int]]]:
    """The walk's cited lines by table: ``{"env": {key: (value, line)}, "stand_weights": {...}}``."""
    tables: dict[str, dict[str, tuple[str, int]]] = {"env": {}, "stand_weights": {}}
    table = None
    for line in (_model_dir(model_id) / "locomotion.toml").read_text().splitlines():
        if line.startswith("["):
            table = {"[env]": "env", "[commands.stand_weights]": "stand_weights"}.get(line.strip())
            continue
        if table is not None and (match := _CITATION.match(line)):
            tables[table][match["key"]] = (match["value"], int(match["line"]))
    return tables


def _toml_value(key: str, text: str) -> Any:
    return tomllib.loads(f"{key} = {text}")[key]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_each_weight_at_a_zero_command_is_the_stance_weight(model_id):
    """For every reward weight the stand block may swap: the stand weight if the block names it, else the walk's
    ``[env]`` value, equals the stance's, or the weight is a listed difference and the two really differ."""
    stance, walk, stand = _effective(model_id, "stance"), _effective(model_id, "locomotion"), _stand(model_id)
    listed = WEIGHT_DIFFERENCES[model_id]
    unknown = set(listed) - set(_defaults(model_id))
    assert not unknown, f"listed differences name no constructor weight: {sorted(unknown)}"
    for name in _defaults(model_id):
        if not _is_weight(model_id, name):
            continue
        at_stand = stand.get(name, walk[name])
        if name in listed:
            assert at_stand != stance[name], f"{name} is listed as a difference but a zero command pays the stance's"
        else:
            assert at_stand == stance[name], (
                f"{name}: a zero command pays {at_stand!r}, the stance {stance[name]!r}; restate it in "
                "[commands.stand_weights] or list the difference with its reason"
            )


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_the_stand_block_restates_no_weight_the_walk_already_pays(model_id):
    walk = _effective(model_id, "locomotion")
    assert _stand(model_id), "every walk declares a stand block"
    for name, weight in _stand(model_id).items():
        assert _is_weight(model_id, name), name
        assert weight != walk[name], f"{name} = {weight!r} is the walk's [env] value already"


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_each_stand_weight_and_setting_names_its_stance_line(model_id):
    """Every stand weight cites ``stance.toml:N``, as does every ``[env]`` setting the walk declares for a stand
    term, and line N of the model's stance.toml holds that key at that value (comments aside)."""
    cited = _citations(model_id)
    assert set(cited["stand_weights"]) == set(_stand(model_id)), "every stand weight cites its stance line"
    stance_lines = (_model_dir(model_id) / "stance.toml").read_text().splitlines()
    for table in ("stand_weights", "env"):
        for key, (value, number) in cited[table].items():
            source = stance_lines[number - 1]
            match = re.match(rf"^{key}\s*=\s*(?P<value>[^#]+?)\s*(#|$)", source)
            assert match, f"[{table}] {key} cites stance.toml:{number}, which reads {source!r}"
            assert _toml_value(key, match["value"]) == _toml_value(key, value), (table, key, number)


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_every_setting_is_the_stance_setting_or_a_listed_difference(model_id):
    """For every constructor parameter that is not a stand-block weight: the walk's value is the stance's (so a stand
    term runs the stance's settings), or the parameter is a listed difference and the two really differ.  Each
    setting the walk cites from the stance is a setting, not a weight, held off its constructor default."""
    stance, walk = _effective(model_id, "stance"), _effective(model_id, "locomotion")
    listed = SETTING_DIFFERENCES[model_id]
    assert not set(listed) - set(_defaults(model_id))
    for name in _defaults(model_id):
        if _is_weight(model_id, name):
            continue
        if name in listed:
            assert walk[name] != stance[name], f"{name} is listed as a difference but the walk holds the stance's"
        else:
            assert walk[name] == stance[name], (
                f"{name}: the walk runs {walk[name]!r}, the stance {stance[name]!r}; declare the stance's in the "
                "walk's [env] citing its line, or list the difference with its reason"
            )
    for name in _citations(model_id)["env"]:
        assert not _is_weight(model_id, name), f"[env] {name} is a weight: a stand weight goes in the stand block"
        assert walk[name] != _defaults(model_id)[name], f"[env] {name} cites the stance at the constructor default"


def _seed(model_id: ModelId, family: str) -> int:
    config = load_stage_config(model_id, "locomotion")["env_kwargs"]["command_config"]
    return next(seed for seed in range(3042, 3542) if reset_command_family(config, seed) == family)


def _rollout(model_id: ModelId, seed: int, **overrides: Any) -> list[tuple[float, bool, bool]]:
    """150 steps of a seeded random action (amplitude 0.3) from ``reset(seed=seed)``: (reward, flags) per step."""
    env = build_env(model_id, "locomotion", **overrides)
    try:
        env.reset(seed=seed)
        actions = np.random.default_rng([20261124, seed]).uniform(-0.3, 0.3, (150, env.action_space.shape[0]))
        steps = []
        for action in actions.astype(np.float32):
            _, reward, terminated, truncated, _ = env.step(action)
            steps.append((float(reward), bool(terminated), bool(truncated)))
            if terminated or truncated:
                break
        return steps
    finally:
        env.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_ids)
def test_the_stand_term_settings_move_no_cruise_reward_and_do_move_the_stand_reward(model_id):
    """The walk built with every cited ``[env]`` setting reverted to its constructor default: a cruise episode's
    rewards and flags are bit-identical (the settings' terms are at weight 0 there), and a stand episode's reward
    moves (the stand block runs those terms)."""
    settings = _citations(model_id)["env"]
    if not settings:
        pytest.skip(f"{model_id}'s stand terms run at their constructor settings, which are the stance's")
    reverted = {name: _defaults(model_id)[name] for name in settings}
    cruise, stand = _seed(model_id, CRUISE_FAMILY), _seed(model_id, STAND_FAMILY)
    assert _rollout(model_id, cruise) == _rollout(model_id, cruise, **reverted)
    committed, default = _rollout(model_id, stand), _rollout(model_id, stand, **reverted)
    assert committed != default, "the stand-term settings must reach the stand block's terms"
