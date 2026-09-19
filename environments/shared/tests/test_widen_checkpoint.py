"""Pins for ``scripts/widen_checkpoint.py`` (Phase C, WS-C2).

BEHAVIOR_RECIPES_PLAN §4.6 "Widening instead of retraining" / "Exact
transfer", invariant 7 (pins 1-4), decisions D-C8, D-C9, D-C10, D-C12 and
amendment A13.  The ``narrow_parent`` fixture synthesises the r -> r+1
crossing inside one checkout: a tiny PPO or SAC is trained on the REAL trex
stance environment seen through an observation wrapper that drops the
trailing command dims, stamped with the current plant identity mutated one
interface revision back, and saved exactly as a certified run directory
holds its handoff pair.  The ``widened`` fixture runs the tool ONCE per
algorithm over that parent — and once more over a PPO parent TWO revisions
back (``revision_gap=2``: the shape of the certified trex r11 stance
parents, D-C17) under ``max_revision_gap=2``, so every invariant-7 pin is
asserted on a bounded-gap widening too; all are module-cached and
read-only — every test that mutates a parent or a widened directory works
on a copy (``copy_parent`` / ``shutil.copytree``), and
``build_narrow_parent`` / ``narrow_identity`` / ``NarrowObservation`` stay
public for WS-C3.

Run with the SB3 interpreter (``JAX_PLATFORMS=cpu``, no ``MUJOCO_GL``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared import train_base  # noqa: E402
from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor  # noqa: E402
from environments.shared.command_frame import COMMAND_PROBE_VECTOR, COMMAND_WIDTH  # noqa: E402
from environments.shared.config import (  # noqa: E402
    LOAD_LINEAGE_KEYS,
    WIDEN_LINEAGE_KEYS,
    build_env,
    load_stage_config,
    record_stage_duration,
    save_stage_config,
)
from environments.shared.curriculum.checkpoints import load_vecnorm_stats, select_handoff_checkpoint  # noqa: E402
from environments.shared.curriculum.gate_schema import gate_config_view  # noqa: E402
from environments.shared.harnesses import freeze_recovery_gate  # noqa: E402
from environments.shared.plant_contract import (  # noqa: E402
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    PlantIdentity,
    attach_plant_identity,
    current_plant_identity,
)
from environments.shared.policy_loading import load_sb3_checkpoint  # noqa: E402
from environments.shared.result_bundle import GATE_VERDICT_FILENAME, sha256_file, write_gate_verdict  # noqa: E402
from environments.shared.scripts import widen_checkpoint as widen_module  # noqa: E402
from environments.shared.scripts.widen_checkpoint import (  # noqa: E402
    ACTION_DELTA_ATOL,
    DEFAULT_MAX_REVISION_GAP,
    FORBIDDEN_OUTPUT_FILES,
    MODEL_WIDEN_LINEAGE_ATTRIBUTE,
    VERIFICATION_ROLLOUT_SEED,
    VERIFICATION_ROLLOUT_STEPS,
    WIDEN_REPORT_FILENAME,
    WIDEN_REPORT_SCHEMA,
    WIDEN_TOOL_VERSION,
    WidenError,
    _check_max_revision_gap,
    identity_gate_errors,
    widen_checkpoint,
)
from environments.shared.scripts.widen_checkpoint import main as widen_main  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname, stage_label  # noqa: E402
from environments.shared.task_fingerprint import (  # noqa: E402
    MODEL_TASK_ATTRIBUTE,
    attach_task_fingerprint,
    derive_stage_task_fingerprint,
    read_checkpoint_attribute,
    read_checkpoint_task_fingerprint,
    validate_declared_parent,
)

SPECIES = "trex"
STAGE = "stance"
ALGORITHMS = ("ppo", "sac")
#: The ``widened`` fixture's parametrisation: one r-1 widening per algorithm
#: plus a PPO r-2 widening under ``max_revision_gap=2`` (D-C17).
WIDENED_CASES = ("ppo", "sac", "ppo_r2")
#: Every key the archive's ``mesozoic_widen_lineage`` attribute carries — the
#: same set for an identity-bearing parent and a legacy one (``revision_gap``
#: is then ``None``, never dropped).
WIDEN_LINEAGE_ATTRIBUTE_KEYS = frozenset(
    {
        "schema",
        "tool",
        "parent_checkpoint_sha256",
        "parent_normalization_sha256",
        "parent_task_sha256",
        "parent_plant_identity",
        "parent_path",
        "from_observation_dim",
        "to_observation_dim",
        "revision_gap",
        "padded_tensors",
        "inserted_at",
        "widened_at_commit",
    }
)
PARENT_RUN_NAME = "20260101_000000"
WIDENED_RUN_NAME = "20260102_000000"
PARENT_SEED = 7
PARENT_TIMESTEPS = 128
PARENT_DURATION_SECONDS = 12.5
FINGERPRINT_BACKEND = "stable-baselines3"

#: The first-layer tensors the tool must pad, per algorithm (SB3 2.9.0 names).
FIRST_LAYER_TENSORS = {
    "ppo": ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"),
    "sac": (
        "actor.latent_pi.0.weight",
        "critic.qf0.0.weight",
        "critic.qf1.0.weight",
        "critic_target.qf0.0.weight",
        "critic_target.qf1.0.weight",
    ),
}
#: The observation-action first layers (the SAC critics), whose zero block is INSERTED.
CRITIC_TENSORS = tuple(name for name in FIRST_LAYER_TENSORS["sac"] if name.startswith("critic"))
#: The optimizer members the tool pads and the state indices it must report
#: (the brief's SB3 2.9.0 parameter order: PPO policy_net.0 / value_net.0 at
#: policy.optimizer 1 / 5; SAC actor.latent_pi.0 at actor.optimizer 0 and
#: critic.qf0.0 / qf1.0 at critic.optimizer 0 / 6).
EXPECTED_OPTIMIZER_MEMBERS = {
    "ppo": {"policy.optimizer": [1, 5]},
    "sac": {"actor.optimizer": [0], "critic.optimizer": [0, 6]},
}


class NarrowObservation(gym.ObservationWrapper):
    """The pre-Phase-C interface: the real env minus its trailing command segment."""

    def __init__(self, env):
        super().__init__(env)
        width = int(env.observation_space.shape[0]) - COMMAND_WIDTH
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(width,), dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation[:-COMMAND_WIDTH], dtype=np.float32)


def narrow_identity(current: PlantIdentity, *, revision_gap: int = 1) -> PlantIdentity:
    """*current* *revision_gap* interface-only revisions back: r-k, three dims narrower, another interface digest.

    Every other field (physics digest, ``nq``/``nv``/``nu``, ``action_dim``)
    is the current one, so a gap above 1 models a chain of fingerprint-only
    bumps (the trex r11 -> r12 -> r13 history, D-C17).
    """
    seed = b"narrow parent policy interface" if revision_gap == 1 else f"narrow parent r-{revision_gap}".encode()
    digest = "sha256:" + hashlib.sha256(seed).hexdigest()
    return dataclasses.replace(
        current,
        policy_interface_revision=current.policy_interface_revision - revision_gap,
        observation_dim=current.observation_dim - COMMAND_WIDTH,
        policy_interface_sha256=digest,
    )


def legacy_closure_schedule(initial: float, final: float):
    """The closure ``train_base.linear_schedule`` returned before the loader change: pickled by value, bytecode and all."""

    def schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)

    return schedule


def build_narrow_parent(
    root: Path,
    algorithm: str,
    *,
    species: str = SPECIES,
    stage: str = STAGE,
    seed: int = PARENT_SEED,
    run_name: str = PARENT_RUN_NAME,
    handoff_name: str = "robust_best_model",
    revision_gap: int = 1,
    schedule_closures: bool = False,
) -> dict:
    """A certified-shaped parent run directory holding a narrow (r-*revision_gap*, default r-1) handoff pair.

    With *schedule_closures* the PPO parent trains under the CLOSURE schedules
    ``train_base.linear_schedule`` returned before the loader change (every
    Drive stage archive with a ``learning_rate_end`` holds them), so its
    archive's ``learning_rate`` / ``lr_schedule`` / ``clip_range`` members
    are cloudpickled bytecode.

    ``<root>/<run_name>/<stage_dirname>/models/<handoff_name>.zip`` +
    ``_vecnorm.pkl`` (both stamped with the narrow identity; the archive
    also carries the narrow task fingerprint) and a ``stage_config.json``
    run block with ``seed`` / ``n_envs`` / ``timesteps`` / ``duration_seconds``.
    Hidden widths stay away from the parent width and parent width +
    action_dim: the tool's tensor rule is width-based.
    """
    current = current_plant_identity(species)
    narrow = narrow_identity(current, revision_gap=revision_gap)
    stage_config = load_stage_config(species, stage)
    entry = load_stage_manifest(species).resolve(stage)
    stage_dir = root / run_name / stage_dirname(species, stage)
    models_dir = stage_dir / "models"
    models_dir.mkdir(parents=True)

    venv = VecNormalize(DummyVecEnv([lambda: NarrowObservation(build_env(species, stage))]))
    try:
        if algorithm == "ppo":
            schedules: dict = {}
            if schedule_closures:
                schedules = {
                    "learning_rate": legacy_closure_schedule(3e-4, 1e-5),
                    "clip_range": legacy_closure_schedule(0.2, 0.1),
                }
            model = PPO(
                "MlpPolicy",
                venv,
                n_steps=64,
                batch_size=32,
                policy_kwargs={"net_arch": [512, 256]},
                seed=seed,
                device="cpu",
                verbose=0,
                **schedules,
            )
        elif algorithm == "sac":
            model = SAC(
                "MlpPolicy",
                venv,
                buffer_size=256,
                learning_starts=64,
                batch_size=64,
                seed=seed,
                device="cpu",
                verbose=0,
            )
        else:
            raise ValueError(algorithm)
        model.learn(PARENT_TIMESTEPS)
        parent_task = derive_stage_task_fingerprint(
            species=species,
            stage=entry.reference,
            backend=FINGERPRINT_BACKEND,
            env_kwargs=stage_config["env_kwargs"],
            plant_identity=narrow.to_dict(),
        )
        attach_plant_identity(model, narrow)
        attach_task_fingerprint(model, parent_task)
        model.save(str(models_dir / f"{handoff_name}.zip"))
        attach_plant_identity(venv, narrow)
        venv.save(str(models_dir / f"{handoff_name}_vecnorm.pkl"))
    finally:
        venv.close()

    save_stage_config(
        stage_dir,
        entry.reference,
        stage_config,
        algorithm.upper(),
        extra={"seed": seed, "n_envs": 1, "timesteps": PARENT_TIMESTEPS},
        env_class=get_species_config(species).env_class,
        species=species,
        plant_identity=narrow,
        task_fingerprint=parent_task,
    )
    record_stage_duration(stage_dir, PARENT_DURATION_SECONDS)
    return {
        "algorithm": algorithm,
        "species": species,
        "stage": stage,
        "reference": entry.reference,
        "run_dir": stage_dir.parent,
        "stage_dir": stage_dir,
        "handoff_name": handoff_name,
        "model_zip": models_dir / f"{handoff_name}.zip",
        "vecnorm_pkl": models_dir / f"{handoff_name}_vecnorm.pkl",
        "current": current,
        "narrow": narrow,
        "task_fingerprint": parent_task,
        "seed": seed,
        "revision_gap": revision_gap,
    }


def copy_parent(parent: dict, root: Path) -> dict:
    """A private copy of *parent*'s run directory under *root*, for tests that mutate it."""
    run_dir = root / parent["run_dir"].name
    shutil.copytree(parent["run_dir"], run_dir)
    stage_dir = run_dir / parent["stage_dir"].name
    models_dir = stage_dir / "models"
    return {
        **parent,
        "run_dir": run_dir,
        "stage_dir": stage_dir,
        "model_zip": models_dir / f"{parent['handoff_name']}.zip",
        "vecnorm_pkl": models_dir / f"{parent['handoff_name']}_vecnorm.pkl",
    }


def rewrite_archive(zip_path: Path, mutate: Callable[[dict, dict], None]) -> None:
    """Rewrite an SB3 archive in place through its own serializer after *mutate(data, params)*."""
    data, params, pytorch_variables = load_from_zip_file(str(zip_path), device="cpu")
    mutate(data, params)
    save_to_zip_file(str(zip_path), data=data, params=params, pytorch_variables=pytorch_variables)


def restamp_identity(zip_path: Path, identity: "dict | None") -> None:
    """Replace (or, with None, remove) the plant identity an archive records."""

    def mutate(data, params):
        if identity is None:
            data.pop(MODEL_IDENTITY_ATTRIBUTE, None)
        else:
            data[MODEL_IDENTITY_ATTRIBUTE] = identity

    rewrite_archive(zip_path, mutate)


def rewrite_sidecar(pkl_path: Path, mutate: Callable[[Any], None]) -> None:
    """Rewrite a narrow VecNormalize sidecar in place after *mutate(wrapper)* (re-bound to the narrow env to save)."""
    with open(pkl_path, "rb") as handle:
        wrapper = pickle.load(handle)
    venv = DummyVecEnv([lambda: NarrowObservation(build_env(SPECIES, STAGE))])
    try:
        wrapper.set_venv(venv)
        mutate(wrapper)
        wrapper.save(str(pkl_path))
    finally:
        venv.close()


def widen_into(
    parent: dict,
    root: Path,
    *,
    run_name: str = WIDENED_RUN_NAME,
    label: str = "widen smoke",
    max_revision_gap: int = DEFAULT_MAX_REVISION_GAP,
) -> dict:
    """Run the tool over *parent* into ``<root>/<run_name>/<stage dir>`` (under *max_revision_gap*)."""
    target = root / run_name / stage_dirname(parent["species"], parent["stage"])
    result = widen_checkpoint(
        species=parent["species"],
        stage=parent["stage"],
        target_stage_dir=target,
        parent_stage_dir=parent["stage_dir"],
        label=label,
        max_revision_gap=max_revision_gap,
    )
    return {
        "parent": parent,
        "result": result,
        "target": target,
        "run_dir": target.parent,
        "algorithm": parent["algorithm"],
        "current": parent["current"],
        "parent_obs": parent["narrow"].observation_dim,
        "label": label,
        "revision_gap": parent["revision_gap"],
        "max_revision_gap": max_revision_gap,
    }


def stance_task_sha256(current: PlantIdentity) -> str:
    """The stance task digest the CURRENT config derives (what reuse rule 3 compares against)."""
    fingerprint = derive_stage_task_fingerprint(
        species=SPECIES,
        stage=load_stage_manifest(SPECIES).by_id(STAGE).reference,
        backend=FINGERPRINT_BACKEND,
        env_kwargs=load_stage_config(SPECIES, STAGE)["env_kwargs"],
        plant_identity=current.to_dict(),
    )
    return str(fingerprint["task_sha256"])


def padded_block(tensor: Any, parent_obs: int) -> Any:
    return tensor[:, parent_obs : parent_obs + COMMAND_WIDTH]


def assert_zero(block: Any, what: str) -> None:
    assert torch.equal(block, torch.zeros_like(block)), f"{what} is not exactly zero (max |x| = {block.abs().max()})"


def file_digests(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


# ── fixtures (module-cached, read-only) ───────────────────────────────────────


@pytest.fixture(scope="module")
def narrow_parent_ppo(tmp_path_factory):
    return build_narrow_parent(tmp_path_factory.mktemp("parent_ppo"), "ppo")


@pytest.fixture(scope="module")
def narrow_parent_sac(tmp_path_factory):
    return build_narrow_parent(tmp_path_factory.mktemp("parent_sac"), "sac")


@pytest.fixture(scope="module")
def narrow_parent_ppo_closures(tmp_path_factory):
    """A PPO parent whose archive stores its schedules as cloudpickled closures (the Drive parents' shape)."""
    return build_narrow_parent(tmp_path_factory.mktemp("parent_ppo_closures"), "ppo", schedule_closures=True)


@pytest.fixture(scope="module")
def narrow_parent_ppo_r2(tmp_path_factory):
    """A PPO parent TWO interface-only revisions back (the certified trex r11 parents' shape, D-C17)."""
    return build_narrow_parent(tmp_path_factory.mktemp("parent_ppo_r2"), "ppo", revision_gap=2)


@pytest.fixture(scope="module", params=ALGORITHMS, ids=ALGORITHMS)
def narrow_parent(request):
    """One narrow parent per algorithm, built once per module; treat it as read-only."""
    return request.getfixturevalue(f"narrow_parent_{request.param}")


@pytest.fixture(scope="module")
def widened_ppo(narrow_parent_ppo, tmp_path_factory):
    return widen_into(narrow_parent_ppo, tmp_path_factory.mktemp("widened_ppo"))


@pytest.fixture(scope="module")
def widened_sac(narrow_parent_sac, tmp_path_factory):
    return widen_into(narrow_parent_sac, tmp_path_factory.mktemp("widened_sac"))


@pytest.fixture(scope="module")
def widened_ppo_r2(narrow_parent_ppo_r2, tmp_path_factory):
    """The r-2 PPO parent widened under ``max_revision_gap=2`` (the default bound refuses it: ``test_refusals``)."""
    return widen_into(narrow_parent_ppo_r2, tmp_path_factory.mktemp("widened_ppo_r2"), max_revision_gap=2)


@pytest.fixture(scope="module", params=WIDENED_CASES, ids=WIDENED_CASES)
def widened(request):
    """The tool's output over each narrow parent (r-1 per algorithm, PPO r-2 under gap 2), run once; read-only."""
    return request.getfixturevalue(f"widened_{request.param}")


# ── the round trip: layout, report, sidecar, loadability ─────────────────────


def test_widen_round_trip(widened):
    """The widened pair is the parent under the current interface, laid out as a judge-ready root."""
    parent = widened["parent"]
    result = widened["result"]
    target = widened["target"]
    algorithm = widened["algorithm"]
    current = widened["current"]

    # Layout (D-C9): the parent's own handoff name plus byte-identical *_final copies, nothing judged.
    assert result.handoff_name == parent["handoff_name"]
    assert result.algorithm == algorithm
    assert result.model_zip == target / "models" / f"{parent['handoff_name']}.zip"
    assert result.vecnorm_pkl == target / "models" / f"{parent['handoff_name']}_vecnorm.pkl"
    label = stage_label(parent["reference"])
    assert result.final_zip == target / "models" / f"{label}_final.zip"
    assert result.final_vecnorm_pkl == target / "models" / f"{label}_final_vecnorm.pkl"
    assert result.final_zip.read_bytes() == result.model_zip.read_bytes()
    assert result.final_vecnorm_pkl.read_bytes() == result.vecnorm_pkl.read_bytes()
    assert sorted(path.name for path in (target / "models").iterdir()) == sorted(
        [
            f"{parent['handoff_name']}.zip",
            f"{parent['handoff_name']}_vecnorm.pkl",
            f"{label}_final.zip",
            f"{label}_final_vecnorm.pkl",
        ]
    )
    assert sorted(path.name for path in target.iterdir()) == sorted(
        ["models", "stage_config.json", "plant_identity.json", "task_fingerprint.json", WIDEN_REPORT_FILENAME]
    )
    assert not any((target / name).exists() for name in FORBIDDEN_OUTPUT_FILES)
    assert select_handoff_checkpoint(target / "models") == (
        parent["handoff_name"],
        str(target / "models" / parent["handoff_name"]),
        str(result.vecnorm_pkl),
    )

    # The run block: the parent's run facts, the label, the algorithm; identity re-stamped.
    record = json.loads((target / "stage_config.json").read_text())
    run = record["run"]
    assert (run["seed"], run["n_envs"], run["timesteps"]) == (parent["seed"], 1, PARENT_TIMESTEPS)
    assert run["duration_seconds"] == PARENT_DURATION_SECONDS
    assert run["label"] == widened["label"]
    assert record["algorithm"] == algorithm.upper()
    assert record["plant_identity"] == current.to_dict()
    assert json.loads((target / "plant_identity.json").read_text()) == current.to_dict()

    # The report (A19) and the result agree; hashes are the files' own.
    report = json.loads((target / WIDEN_REPORT_FILENAME).read_text())
    assert report == result.report
    assert report["schema"] == WIDEN_REPORT_SCHEMA
    assert report["tool"] == WIDEN_TOOL_VERSION
    assert (report["species"], report["stage"], report["stage_id"]) == (SPECIES, parent["reference"], STAGE)
    assert report["max_padded_column_abs"] == 0.0
    assert report["padded_columns_exactly_zero"] is True
    assert report["num_timesteps"] == PARENT_TIMESTEPS
    assert (report["from_observation_dim"], report["to_observation_dim"]) == (
        widened["parent_obs"],
        current.observation_dim,
    )
    assert report["command_width"] == COMMAND_WIDTH
    # D-C17: the gap crossed and the bound it was admitted under, on the report and the result.
    assert report["revision_gap"] == widened["revision_gap"] == result.revision_gap
    assert report["max_revision_gap"] == widened["max_revision_gap"]
    assert isinstance(report["revision_gap"], int) and isinstance(report["max_revision_gap"], int)
    assert 1 <= report["revision_gap"] <= report["max_revision_gap"]
    assert report["parent"]["policy_interface_revision"] == parent["narrow"].policy_interface_revision
    assert report["parent"]["policy_interface_revision"] == current.policy_interface_revision - report["revision_gap"]
    assert report["widened"]["checkpoint_sha256"] == sha256_file(result.model_zip)
    assert report["widened"]["normalization_sha256"] == sha256_file(result.vecnorm_pkl)
    assert report["widened"]["policy_interface_revision"] == current.policy_interface_revision
    assert report["parent"]["run_id"] == PARENT_RUN_NAME
    assert report["parent"]["legacy_plant"] is False
    assert report["rollout"] == {
        "seed": VERIFICATION_ROLLOUT_SEED,
        "steps": VERIFICATION_ROLLOUT_STEPS,
        "deterministic": True,
    }
    assert report["action_delta_atol"] == ACTION_DELTA_ATOL
    assert set(report["versions"]) == {"mujoco", "stable_baselines3", "torch", "gymnasium"}
    assert all(isinstance(version, str) for version in report["versions"].values())

    # The sidecar: padded statistics born reseeded, the identity re-stamped, loadable on the real env.
    alg_cls = PPO if algorithm == "ppo" else SAC
    with open(parent["vecnorm_pkl"], "rb") as handle:
        parent_norm = pickle.load(handle)
    venv = DummyVecEnv([lambda: build_env(SPECIES, STAGE)])
    try:
        normalizer = VecNormalize.load(str(result.vecnorm_pkl), venv)
        assert normalizer.obs_rms.mean.shape == (current.observation_dim,)
        assert np.array_equal(normalizer.obs_rms.mean[: widened["parent_obs"]], parent_norm.obs_rms.mean)
        assert np.array_equal(normalizer.obs_rms.var[: widened["parent_obs"]], parent_norm.obs_rms.var)
        assert np.array_equal(normalizer.obs_rms.mean[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH))
        assert np.array_equal(normalizer.obs_rms.var[-COMMAND_WIDTH:], np.ones(COMMAND_WIDTH))
        assert normalizer.obs_rms.count == parent_norm.obs_rms.count
        assert normalizer.training is False
        assert (normalizer.clip_obs, normalizer.epsilon) == (parent_norm.clip_obs, parent_norm.epsilon)
        assert np.array_equal(normalizer.ret_rms.mean, parent_norm.ret_rms.mean)
        assert getattr(normalizer, MODEL_IDENTITY_ATTRIBUTE) == current.to_dict()
        # The archive loads with an env (check_for_correct_spaces + exact_match over every member).
        model = alg_cls.load(str(result.model_zip), env=normalizer, device="cpu")
        assert model.num_timesteps == PARENT_TIMESTEPS
        assert model.observation_space == venv.observation_space
        assert model.observation_space.shape == (current.observation_dim,)
    finally:
        venv.close()


# ── invariant 7 pin 1: exact-zero padded columns ─────────────────────────────


def test_padded_columns_are_exactly_zero(widened):
    """Every first layer gained exactly COMMAND_WIDTH zero columns; every other tensor is untouched."""
    parent = widened["parent"]
    result = widened["result"]
    algorithm = widened["algorithm"]
    parent_obs = widened["parent_obs"]
    _, parent_params, _ = load_from_zip_file(str(parent["model_zip"]), device="cpu", load_data=False)
    _, widened_params, _ = load_from_zip_file(str(result.model_zip), device="cpu", load_data=False)

    assert set(result.report["padded_tensors"]) == set(FIRST_LAYER_TENSORS[algorithm])
    columns = list(range(parent_obs, parent_obs + COMMAND_WIDTH))
    for name in FIRST_LAYER_TENSORS[algorithm]:
        assert result.report["padded_tensors"][name] == columns
        before = parent_params["policy"][name]
        after = widened_params["policy"][name]
        assert after.shape == (before.shape[0], before.shape[1] + COMMAND_WIDTH)
        assert after.dtype == before.dtype
        assert_zero(padded_block(after, parent_obs), name)
        assert torch.equal(after[:, :parent_obs], before[:, :parent_obs])
        assert torch.equal(after[:, parent_obs + COMMAND_WIDTH :], before[:, parent_obs:])
    for name, tensor in widened_params["policy"].items():
        if name not in FIRST_LAYER_TENSORS[algorithm]:
            assert torch.equal(tensor, parent_params["policy"][name]), name
    assert set(widened_params["policy"]) == set(parent_params["policy"])
    # No hidden layer shares the parent width (the width-based tensor rule would refuse it).
    widths = {int(tensor.shape[1]) for tensor in parent_params["policy"].values() if tensor.ndim == 2}
    assert parent_obs in widths
    assert all(name.endswith(".0.weight") for name in FIRST_LAYER_TENSORS[algorithm])


def test_sac_critic_columns_are_inserted_before_the_action_block(widened_sac):
    """The critics read (obs, action): the zero block sits at [parent_obs, parent_obs+3) and the action columns shift intact."""
    parent = widened_sac["parent"]
    result = widened_sac["result"]
    parent_obs = widened_sac["parent_obs"]
    action_dim = widened_sac["current"].action_dim
    _, parent_params, _ = load_from_zip_file(str(parent["model_zip"]), device="cpu", load_data=False)
    _, widened_params, _ = load_from_zip_file(str(result.model_zip), device="cpu", load_data=False)

    assert set(result.report["inserted_at"]) == set(CRITIC_TENSORS)
    for name in CRITIC_TENSORS:
        assert result.report["inserted_at"][name] == parent_obs
        before = parent_params["policy"][name]
        after = widened_params["policy"][name]
        assert before.shape[1] == parent_obs + action_dim
        assert after.shape[1] == parent_obs + COMMAND_WIDTH + action_dim
        assert_zero(padded_block(after, parent_obs), name)
        # The action block is the parent's LAST action_dim columns, moved right by COMMAND_WIDTH.
        assert torch.equal(after[:, -action_dim:], before[:, -action_dim:])
        assert torch.equal(after[:, :parent_obs], before[:, :parent_obs])
        # Insertion, not an append: the appended slots hold the parent's action columns, never zeros.
        assert not torch.equal(after[:, -COMMAND_WIDTH:], torch.zeros_like(after[:, -COMMAND_WIDTH:]))
    # The actor reads the observation alone: its zero block is an append.
    actor = widened_params["policy"]["actor.latent_pi.0.weight"]
    assert actor.shape[1] == parent_obs + COMMAND_WIDTH
    assert "actor.latent_pi.0.weight" not in result.report["inserted_at"]
    # critic_target mirrors critic and is padded although no optimizer owns it.
    for tag in ("qf0", "qf1"):
        assert torch.equal(
            padded_block(widened_params["policy"][f"critic_target.{tag}.0.weight"], parent_obs),
            padded_block(widened_params["policy"][f"critic.{tag}.0.weight"], parent_obs),
        )


# ── invariant 7 pin 2: exact transfer ────────────────────────────────────────


def test_actions_equal_parent_on_zero_padded_and_live_command_observations(widened):
    """Over the seeded rollout the widened policy's actions equal the parent's, with zero AND probe commands (A13b)."""
    parent = widened["parent"]
    result = widened["result"]
    current = widened["current"]
    alg_cls = PPO if widened["algorithm"] == "ppo" else SAC

    parent_model = alg_cls.load(str(parent["model_zip"]), device="cpu")
    with open(parent["vecnorm_pkl"], "rb") as handle:
        parent_norm = pickle.load(handle)
    model, normalizer, _ = load_sb3_checkpoint(
        str(result.model_zip),
        str(result.vecnorm_pkl),
        env_factory=lambda: build_env(SPECIES, STAGE),
        guess_sidecar=False,
        plant_identity=current,
    )
    probe = np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float32)
    env = build_env(SPECIES, STAGE)
    delta_zero = 0.0
    delta_probe = 0.0
    steps = 0
    try:
        obs, _ = env.reset(seed=VERIFICATION_ROLLOUT_SEED)
        for _ in range(VERIFICATION_ROLLOUT_STEPS):
            observation = np.asarray(obs, dtype=np.float32)
            # command_mode = "none": the real env's command slice is zero, so the
            # zero-command comparison is the plain observation.
            assert np.array_equal(observation[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH, dtype=np.float32))
            parent_action, _ = parent_model.predict(
                parent_norm.normalize_obs(observation[:-COMMAND_WIDTH]), deterministic=True
            )
            widened_action, _ = model.predict(normalizer.normalize_obs(observation), deterministic=True)
            probed = observation.copy()
            probed[-COMMAND_WIDTH:] = probe
            normalized_probe = normalizer.normalize_obs(probed)
            # The probe reaches the network unclipped (the reseeded slice is mean 0 / var 1)...
            assert np.allclose(normalized_probe[-COMMAND_WIDTH:], probe, atol=1e-6, rtol=0)
            probed_action, _ = model.predict(normalized_probe, deterministic=True)
            # ...and only the zero columns keep it out of the action.
            assert widened_action.shape == parent_action.shape == (current.action_dim,)
            assert np.allclose(widened_action, parent_action, atol=ACTION_DELTA_ATOL, rtol=0)
            assert np.allclose(probed_action, parent_action, atol=ACTION_DELTA_ATOL, rtol=0)
            delta_zero = max(delta_zero, float(np.max(np.abs(widened_action - parent_action))))
            delta_probe = max(delta_probe, float(np.max(np.abs(probed_action - parent_action))))
            steps += 1
            obs, _, terminated, truncated, _ = env.step(parent_action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
        normalizer.close()
    assert steps == VERIFICATION_ROLLOUT_STEPS
    print(
        f"[{widened['algorithm']}] measured max action delta over {steps} steps: "
        f"zero command {delta_zero:.3g}, probe command {delta_probe:.3g} (atol {ACTION_DELTA_ATOL}); "
        f"tool recorded {result.max_action_delta_zero_command:.3g} / {result.max_action_delta_probe_command:.3g}"
    )
    assert result.max_action_delta_zero_command <= ACTION_DELTA_ATOL
    assert result.max_action_delta_probe_command <= ACTION_DELTA_ATOL
    assert result.report["max_action_delta_zero_command"] == result.max_action_delta_zero_command
    assert result.report["max_action_delta_probe_command"] == result.max_action_delta_probe_command


# ── D-C10: Adam moments ──────────────────────────────────────────────────────


def test_adam_moments_are_padded_and_a_stale_moment_is_refused(widened, tmp_path):
    """exp_avg / exp_avg_sq of every padded weight are padded identically; a missing member or a stale moment refuses."""
    parent = widened["parent"]
    result = widened["result"]
    algorithm = widened["algorithm"]
    parent_obs = widened["parent_obs"]
    _, parent_params, _ = load_from_zip_file(str(parent["model_zip"]), device="cpu", load_data=False)
    _, widened_params, _ = load_from_zip_file(str(result.model_zip), device="cpu", load_data=False)

    members = widen_module._OPTIMIZER_MEMBERS[algorithm]
    assert result.report["optimizer_members_padded"] == EXPECTED_OPTIMIZER_MEMBERS[algorithm]
    assert set(widened_params) == set(parent_params), "every archive member is kept"
    for member, prefix in members.items():
        names = [name for name in widened_params["policy"] if name.startswith(prefix)]
        widened_state = widened_params[member]["state"]
        parent_state = parent_params[member]["state"]
        assert set(widened_state) == set(parent_state)
        assert widened_params[member]["param_groups"] == parent_params[member]["param_groups"]
        for group in widened_params[member]["param_groups"]:
            assert group["weight_decay"] == 0
        padded_indices = result.report["optimizer_members_padded"][member]
        assert [names[index] for index in padded_indices] == [
            name for name in names if name in FIRST_LAYER_TENSORS[algorithm]
        ]
        for index in widened_state:
            name = names[int(index)]
            weight = widened_params["policy"][name]
            for key in ("exp_avg", "exp_avg_sq"):
                moment = widened_state[index][key]
                assert tuple(moment.shape) == tuple(weight.shape), f"{member} state[{index}].{key}"
                if int(index) in padded_indices:
                    assert_zero(padded_block(moment, parent_obs), f"{member} state[{index}].{key}")
                    assert torch.equal(moment[:, :parent_obs], parent_state[index][key][:, :parent_obs])
                    assert torch.equal(
                        moment[:, parent_obs + COMMAND_WIDTH :], parent_state[index][key][:, parent_obs:]
                    )
                else:
                    assert torch.equal(moment, parent_state[index][key])
            assert torch.equal(widened_state[index]["step"], parent_state[index]["step"])

    # A parent whose optimizer member is missing is refused, naming the member; nothing is left behind.
    first_member = next(iter(members))
    stripped = copy_parent(parent, tmp_path / "stripped")
    rewrite_archive(stripped["model_zip"], lambda data, params: params.pop(first_member))
    bound = widened["max_revision_gap"]
    target = tmp_path / "stripped_out" / stage_dirname(SPECIES, STAGE)
    with pytest.raises(WidenError, match=f"carries no {first_member!r} member"):
        widen_checkpoint(
            species=SPECIES,
            stage=STAGE,
            target_stage_dir=target,
            parent_stage_dir=stripped["stage_dir"],
            max_revision_gap=bound,
        )
    assert not target.exists()

    # A moment whose shape is not its weight's (an optimizer state from other weights) is refused as stale.
    stale = copy_parent(parent, tmp_path / "stale")
    stale_index = EXPECTED_OPTIMIZER_MEMBERS[algorithm][first_member][0]

    def corrupt(data, params):
        moment = params[first_member]["state"][stale_index]["exp_avg"]
        params[first_member]["state"][stale_index]["exp_avg"] = moment[:, :-1].clone()

    rewrite_archive(stale["model_zip"], corrupt)
    target = tmp_path / "stale_out" / stage_dirname(SPECIES, STAGE)
    with pytest.raises(WidenError, match="stale moment") as excinfo:
        widen_checkpoint(
            species=SPECIES,
            stage=STAGE,
            target_stage_dir=target,
            parent_stage_dir=stale["stage_dir"],
            max_revision_gap=bound,
        )
    assert f"{first_member} state[{stale_index}].exp_avg" in str(excinfo.value)
    assert not target.exists()

    # A moment the tool does not pad (an RMSprop ``square_avg``, an SGD ``momentum_buffer``) is refused,
    # never carried at the parent width: it would load and crash the first update (D-C10).
    foreign = copy_parent(parent, tmp_path / "foreign")

    def rmsprop_like(data, params):
        record = params[first_member]["state"][stale_index]
        record["square_avg"] = record.pop("exp_avg")

    rewrite_archive(foreign["model_zip"], rmsprop_like)
    target = tmp_path / "foreign_out" / stage_dirname(SPECIES, STAGE)
    with pytest.raises(WidenError, match="the tool does not pad") as excinfo:
        widen_checkpoint(
            species=SPECIES,
            stage=STAGE,
            target_stage_dir=target,
            parent_stage_dir=foreign["stage_dir"],
            max_revision_gap=bound,
        )
    assert f"{first_member} state[{stale_index}].square_avg" in str(excinfo.value)
    assert not target.exists()


def test_self_verification_asserts_every_widened_width_before_the_zero_block(widened, tmp_path):
    """``_verify`` refuses a weight or moment left at the parent width: an empty slice must never pass as zero."""
    parent = widened["parent"]
    result = widened["result"]
    report = result.report
    parent_obs = widened["parent_obs"]
    algorithm = widened["algorithm"]
    member, indices = next(iter(EXPECTED_OPTIMIZER_MEMBERS[algorithm].items()))
    prefix = widen_module._OPTIMIZER_MEMBERS[algorithm][member]
    _, params, _ = load_from_zip_file(str(result.model_zip), device="cpu", load_data=False)
    names = [name for name in params["policy"] if name.startswith(prefix)]
    name = names[indices[0]]
    widened_shapes = {key: tuple(params["policy"][key].shape) for key in report["padded_tensors"]}

    def verify(stage_dir: Path):
        return widen_module._verify(
            species=SPECIES,
            stage=parent["reference"],
            algorithm=algorithm,
            parent_zip=parent["model_zip"],
            parent_pkl=parent["vecnorm_pkl"],
            widened_zip=stage_dir / "models" / result.model_zip.name,
            widened_pkl=stage_dir / "models" / result.vecnorm_pkl.name,
            current=widened["current"],
            padded_tensors=report["padded_tensors"],
            optimizer_members_padded=report["optimizer_members_padded"],
            widened_shapes=widened_shapes,
        )

    intact = shutil.copytree(widened["target"], tmp_path / "intact")
    assert verify(intact)["padded_columns_exactly_zero"] is True

    narrow_moment = shutil.copytree(widened["target"], tmp_path / "narrow_moment")

    def unpad_moment(data, params):
        record = params[member]["state"][indices[0]]
        record["exp_avg"] = record["exp_avg"][:, :-COMMAND_WIDTH].clone()

    rewrite_archive(narrow_moment / "models" / result.model_zip.name, unpad_moment)
    with pytest.raises(WidenError, match="not its parameter's widened") as excinfo:
        verify(narrow_moment)
    assert f"{member} state[{indices[0]}].exp_avg" in str(excinfo.value)

    narrow_weight = shutil.copytree(widened["target"], tmp_path / "narrow_weight")

    def unpad_weight(data, params):
        tensor = params["policy"][name]
        params["policy"][name] = torch.cat([tensor[:, :parent_obs], tensor[:, parent_obs + COMMAND_WIDTH :]], dim=1)

    rewrite_archive(narrow_weight / "models" / result.model_zip.name, unpad_weight)
    with pytest.raises(WidenError, match=f"widened {name} has shape"):
        verify(narrow_weight)


# ── invariant 7 pin 3: one PPO update under initialize_next_stage ────────────


def test_one_ppo_update_completes_under_initialize_next_stage(widened_ppo):
    """The widened stem enters locomotion the way train_curriculum does, trains one update, and the columns stay zero."""
    current = widened_ppo["current"]
    result = widened_ppo["result"]
    parent_obs = widened_ppo["parent_obs"]
    manifest = load_stage_manifest(SPECIES)
    child = manifest.by_id("locomotion")
    parent_entry = manifest.by_id(STAGE)
    assert child.warm_start_from == parent_entry.id
    stem = str(result.model_zip)[: -len(".zip")]

    # The edge check train_curriculum applies before any directory exists.
    validate_declared_parent(
        read_checkpoint_task_fingerprint(result.model_zip),
        declared_parent=parent_entry.reference,
        species=SPECIES,
        child_stage=child.reference,
        artifact=stem,
    )

    species_cfg = get_species_config(SPECIES)
    locomotion = load_stage_config(SPECIES, child.reference)
    stage_configs = {child.reference: locomotion}
    gamma = locomotion["ppo_kwargs"]["gamma"]
    train_env = train_base.create_vec_env(
        species_cfg, stage_configs, child.reference, 1, 0, algorithm="ppo", gamma=gamma, plant_identity=current
    )
    eval_env = train_base.create_vec_env(
        species_cfg, stage_configs, child.reference, 1, 1000, algorithm="ppo", gamma=gamma, plant_identity=current
    )
    try:
        train_base._load_vecnorm_into_envs(
            stem,
            train_env,
            eval_env,
            plant_identity=current,
            task_load_mode="initialize_next_stage",
            command_mode=str(locomotion.get("env_kwargs", {}).get("command_mode", "none")),
        )
        assert train_env.obs_rms.mean.shape == (current.observation_dim,)
        assert np.array_equal(train_env.obs_rms.mean[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH))
        assert np.array_equal(eval_env.obs_rms.mean, train_env.obs_rms.mean)
        assert eval_env.training is False

        locomotion_task = derive_stage_task_fingerprint(
            species=SPECIES,
            stage=child.reference,
            backend=FINGERPRINT_BACKEND,
            env_kwargs=locomotion["env_kwargs"],
            plant_identity=current.to_dict(),
        )
        widened_task = read_checkpoint_task_fingerprint(result.model_zip)
        assert locomotion_task["task_sha256"] != widened_task["task_sha256"]
        model = train_base._create_or_load_model(
            train_base._ensure_sb3(),
            "ppo",
            {"n_steps": 64, "batch_size": 32, "device": "cpu"},
            train_env,
            stem,
            plant_identity=current,
            task_fingerprint=locomotion_task,
            task_load_mode="initialize_next_stage",
        )
        assert model.num_timesteps == PARENT_TIMESTEPS
        lineage = getattr(model, "mesozoic_task_lineage")
        assert lineage["mode"] == "initialize_next_stage"
        assert lineage["parent_task_sha256"] == widened_task["task_sha256"]
        assert lineage["child_task_sha256"] == locomotion_task["task_sha256"]
        assert lineage["parent_stage"] == parent_entry.reference
        assert getattr(model, MODEL_TASK_ATTRIBUTE) == locomotion_task
        assert getattr(model, MODEL_IDENTITY_ATTRIBUTE) == current.to_dict()
        # The premises of the pin: zero command dims, zero moments, no weight decay.
        for group in model.policy.optimizer.param_groups:
            assert group["weight_decay"] == 0
        state = model.policy.state_dict()
        before = {name: state[name].clone() for name in FIRST_LAYER_TENSORS["ppo"]}

        model.learn(total_timesteps=64, reset_num_timesteps=False)

        assert model.num_timesteps == PARENT_TIMESTEPS + 64
        state = model.policy.state_dict()
        parameters = dict(model.policy.named_parameters())
        for name in FIRST_LAYER_TENSORS["ppo"]:
            after = state[name]
            assert after.shape == (before[name].shape[0], current.observation_dim)
            assert_zero(padded_block(after, parent_obs), f"{name} after one update")
            # The update was real: the observation columns moved.
            assert not torch.equal(after[:, :parent_obs], before[name][:, :parent_obs])
            moments = model.policy.optimizer.state[parameters[name]]
            for key in ("exp_avg", "exp_avg_sq"):
                assert_zero(padded_block(moments[key], parent_obs), f"{name} {key} after one update")
    finally:
        train_env.close()
        eval_env.close()


# ── invariant 7 pin 4: lineage ───────────────────────────────────────────────


def test_lineage_records_the_parent_hashes_in_the_archive_and_the_run_block(widened):
    """The archive attribute and the run block both name the parent by digest; the task is re-stamped on both."""
    parent = widened["parent"]
    result = widened["result"]
    target = widened["target"]
    current = widened["current"]
    parent_obs = widened["parent_obs"]

    lineage = read_checkpoint_attribute(result.model_zip, MODEL_WIDEN_LINEAGE_ATTRIBUTE)
    assert lineage["schema"] == "mesozoic.widen-lineage/v1"
    assert lineage["tool"] == WIDEN_TOOL_VERSION
    assert lineage["parent_checkpoint_sha256"] == sha256_file(parent["model_zip"]) == result.parent_checkpoint_sha256
    assert lineage["parent_normalization_sha256"] == sha256_file(parent["vecnorm_pkl"])
    assert lineage["parent_normalization_sha256"] == result.parent_normalization_sha256
    assert lineage["parent_task_sha256"] == parent["task_fingerprint"]["task_sha256"]
    assert lineage["parent_plant_identity"] == parent["narrow"].to_dict()
    assert lineage["parent_path"] == str(parent["model_zip"])
    assert (lineage["from_observation_dim"], lineage["to_observation_dim"]) == (parent_obs, current.observation_dim)
    # D-C17: the archive records how many interface revisions the widening crossed.
    assert lineage["revision_gap"] == widened["revision_gap"] == result.report["revision_gap"]
    assert lineage["revision_gap"] == current.policy_interface_revision - parent["narrow"].policy_interface_revision
    assert set(lineage["padded_tensors"]) == set(FIRST_LAYER_TENSORS[widened["algorithm"]])
    assert lineage["inserted_at"] == result.report["inserted_at"]
    assert lineage["widened_at_commit"] == result.report["widened_at_commit"]
    assert set(lineage) == WIDEN_LINEAGE_ATTRIBUTE_KEYS
    # The parent recorded no task lineage (a root); none is invented.
    assert read_checkpoint_attribute(parent["model_zip"], "mesozoic_task_lineage") is None
    assert read_checkpoint_attribute(result.model_zip, "mesozoic_task_lineage") is None

    # Identity and task re-stamped on the archive; task_fingerprint.json == the archive's stamp.
    assert read_checkpoint_attribute(result.model_zip, MODEL_IDENTITY_ATTRIBUTE) == current.to_dict()
    archive_task = read_checkpoint_attribute(result.model_zip, MODEL_TASK_ATTRIBUTE)
    assert archive_task == json.loads((target / "task_fingerprint.json").read_text())
    record = json.loads((target / "stage_config.json").read_text())
    assert record["task_fingerprint"] == archive_task
    assert archive_task["task_sha256"] == stance_task_sha256(current) == result.report["widened"]["task_sha256"]
    assert archive_task["task_sha256"] != parent["task_fingerprint"]["task_sha256"]
    assert archive_task["plant"]["policy_interface_sha256"] == current.policy_interface_sha256
    assert parent["task_fingerprint"]["plant"]["policy_interface_sha256"] == parent["narrow"].policy_interface_sha256
    assert (archive_task["species"], archive_task["stage"]) == (SPECIES, parent["reference"])

    # The run block: every WIDEN_LINEAGE_KEYS value, none of the LOAD_LINEAGE_KEYS (D-C8).
    run = record["run"]
    assert set(WIDEN_LINEAGE_KEYS) <= set(run)
    assert all(run[key] is not None for key in WIDEN_LINEAGE_KEYS)
    assert not set(LOAD_LINEAGE_KEYS) & set(run)
    assert run["widened_from_path"] == str(parent["model_zip"])
    assert run["widened_from_checkpoint_sha256"] == lineage["parent_checkpoint_sha256"]
    assert run["widened_from_normalization_sha256"] == lineage["parent_normalization_sha256"]
    assert run["widened_from_task_sha256"] == parent["task_fingerprint"]["task_sha256"]
    assert run["widened_from_policy_interface_revision"] == parent["narrow"].policy_interface_revision
    assert run["widened_from_policy_interface_revision"] == current.policy_interface_revision - widened["revision_gap"]
    assert run["widened_from_policy_interface_sha256"] == parent["narrow"].policy_interface_sha256
    assert run["widened_from_run_id"] == PARENT_RUN_NAME
    assert run["widened_by"] == f"{WIDEN_TOOL_VERSION}@{lineage['widened_at_commit']}"
    assert set(run) == {"seed", "n_envs", "timesteps", "duration_seconds", "label", "hyperparameters_sha256"} | set(
        WIDEN_LINEAGE_KEYS
    )


# ── refusals ─────────────────────────────────────────────────────────────────


REFUSAL_CASES = (
    "two_revisions_behind",
    "three_revisions_behind_under_gap_2",
    "width_changing_hop_under_gap_2",
    "max_revision_gap_zero",
    "physics_sha256_differs",
    "action_dim_differs",
    "parent_verdict_failed",
    "occupied_target",
    "algorithm_mismatch",
    "missing_identity",
    "missing_sidecar",
    "unstamped_sidecar",
    "legacy_sidecar_foreign_species",
    "target_not_a_stage_dirname",
    "target_inside_parent_run",
    "parent_inside_target",
    "explicit_non_handoff_stem",
)


@pytest.mark.parametrize("case", REFUSAL_CASES)
def test_refusals(case, narrow_parent_ppo, tmp_path):
    """Every refusal names its reason and writes nothing (the target directory never appears)."""
    parent = copy_parent(narrow_parent_ppo, tmp_path / "parent")
    current = parent["current"]
    narrow = parent["narrow"]
    target = tmp_path / "widened" / stage_dirname(SPECIES, STAGE)
    kwargs: dict[str, Any] = dict(species=SPECIES, stage=STAGE, target_stage_dir=target)
    explicit: dict[str, Any] = dict(
        species=SPECIES,
        stage=STAGE,
        target_stage_dir=target,
        model_zip=parent["model_zip"],
        vecnorm_pkl=parent["vecnorm_pkl"],
        algorithm="ppo",
        seed=PARENT_SEED,
        n_envs=1,
        timesteps=PARENT_TIMESTEPS,
    )
    gate_prefix = f"is not the current {SPECIES} plant one interface-only revision behind"
    gap_2_prefix = f"is not the current {SPECIES} plant at most 2 interface-only revisions behind"
    fragments: tuple[str, ...]
    absent: tuple[str, ...] = ()

    if case == "two_revisions_behind":
        # The DEFAULT bound (D-C17): a parent two revisions back — the certified trex r11 parents' shape —
        # is refused with the gap, the bound and the opt-in flag named.
        restamp_identity(
            parent["model_zip"],
            dataclasses.replace(narrow, policy_interface_revision=narrow.policy_interface_revision - 1).to_dict(),
        )
        fragments = (
            gate_prefix,
            f"policy_interface_revision: parent={current.policy_interface_revision - 2}, "
            f"current={current.policy_interface_revision}",
            "gap 2 exceeds max_revision_gap=1",
            "--max-revision-gap 2 / max_revision_gap=2",
            "plant_versions.toml",
        )
    elif case == "three_revisions_behind_under_gap_2":
        # The bound is a bound: r-3 under max_revision_gap=2 is refused the same way.
        restamp_identity(
            parent["model_zip"],
            dataclasses.replace(narrow, policy_interface_revision=narrow.policy_interface_revision - 2).to_dict(),
        )
        kwargs["max_revision_gap"] = 2
        fragments = (
            gap_2_prefix,
            f"policy_interface_revision: parent={current.policy_interface_revision - 3}, "
            f"current={current.policy_interface_revision}",
            "gap 3 exceeds max_revision_gap=2",
            "--max-revision-gap 3 / max_revision_gap=3",
        )
    elif case == "width_changing_hop_under_gap_2":
        # A parent two revisions back whose intermediate hop changed the observation width: the revision
        # rule passes under max_revision_gap=2, the width condition still refuses (D-C17: only
        # fingerprint-only bumps can be crossed).
        restamp_identity(
            parent["model_zip"],
            dataclasses.replace(
                narrow,
                policy_interface_revision=narrow.policy_interface_revision - 1,
                observation_dim=current.observation_dim - 2 * COMMAND_WIDTH,
            ).to_dict(),
        )
        kwargs["max_revision_gap"] = 2
        fragments = (
            gap_2_prefix,
            f"observation_dim: parent={current.observation_dim - 2 * COMMAND_WIDTH} + {COMMAND_WIDTH} "
            f"!= current={current.observation_dim}",
        )
        absent = ("policy_interface_revision:",)
    elif case == "max_revision_gap_zero":
        # A bound below 1 is refused at the API before the parent is even resolved.
        kwargs["max_revision_gap"] = 0
        fragments = ("max_revision_gap must be an integer >= 1, not 0",)
    elif case == "physics_sha256_differs":
        other = "sha256:" + hashlib.sha256(b"another physics plant").hexdigest()
        restamp_identity(parent["model_zip"], dataclasses.replace(narrow, physics_sha256=other).to_dict())
        fragments = (gate_prefix, f"physics_sha256: parent={other!r}, current={current.physics_sha256!r}")
    elif case == "action_dim_differs":
        restamp_identity(parent["model_zip"], dataclasses.replace(narrow, action_dim=narrow.action_dim + 1).to_dict())
        fragments = (gate_prefix, f"action_dim: parent={current.action_dim + 1}, current={current.action_dim}")
    elif case == "parent_verdict_failed":
        curriculum = load_stage_config(SPECIES, STAGE)["curriculum_kwargs"]
        write_gate_verdict(
            parent["stage_dir"],
            species=SPECIES,
            stage=parent["reference"],
            stage_id=STAGE,
            gate_kind=curriculum["gate_kind"],
            gate_schema_version=curriculum["gate_schema_version"],
            passed=False,
            failures=["unsupported_duty_ucb 0.2153 > 0.0200"],
            task_sha256=parent["task_fingerprint"]["task_sha256"],
            judged_by="test_widen_checkpoint",
            checkpoint=parent["model_zip"],
            normalization=parent["vecnorm_pkl"],
            gate_config=gate_config_view(curriculum),
        )
        fragments = (f"{parent['stage_dir'] / GATE_VERDICT_FILENAME} records passed=False",)
    elif case == "occupied_target":
        target.mkdir(parents=True)
        (target / "stage_config.json").write_text('{"stage": "occupied"}\n', encoding="utf-8")
        fragments = (f"{target} already records a stage (stage_config.json present)",)
    elif case == "algorithm_mismatch":
        kwargs["algorithm"] = "sac"
        fragments = ("--algorithm sac disagrees with the PPO the parent's stage_config.json records",)
    elif case == "missing_identity":
        restamp_identity(parent["model_zip"], None)
        fragments = (f"{parent['model_zip']} has no plant identity", "--allow-legacy-plant")
    elif case == "missing_sidecar":
        parent["vecnorm_pkl"].unlink()
        fragments = ("no complete handoff pair",)
    elif case == "unstamped_sidecar":
        # A stamped archive beside an unstamped sidecar: the refusal names the sidecar (a raw-unpickled
        # VecNormalize has no class_attributes, so a plain getattr miss would recurse instead).
        rewrite_sidecar(parent["vecnorm_pkl"], lambda wrapper: wrapper.__dict__.pop(MODEL_IDENTITY_ATTRIBUTE))
        fragments = ("the sidecar does not record the parent archive's plant", str(parent["vecnorm_pkl"]))
        fragments += ("has no plant identity",)
    elif case == "legacy_sidecar_foreign_species":
        # Under the legacy allowance the sidecar's own identity still has to be the parent's species and width.
        restamp_identity(parent["model_zip"], None)
        foreign = current_plant_identity("velociraptor")
        rewrite_sidecar(parent["vecnorm_pkl"], lambda wrapper: attach_plant_identity(wrapper, foreign))
        kwargs["allow_legacy_plant"] = True
        fragments = (
            f"{parent['vecnorm_pkl']} records a velociraptor plant of observation_dim {foreign.observation_dim}, "
            f"not the {SPECIES} parent's {narrow.observation_dim}",
        )
    elif case == "target_not_a_stage_dirname":
        target = tmp_path / "widened" / "01_stanc"
        kwargs["target_stage_dir"] = target
        fragments = (f"{target} is not a {SPECIES} {STAGE} stage directory", "01_stance / stage1 / stance")
    elif case == "target_inside_parent_run":
        target = parent["stage_dir"] / "models" / stage_dirname(SPECIES, STAGE)
        kwargs["target_stage_dir"] = target
        fragments = (f"{target} lies inside the parent's directory {parent['run_dir']}",)
    elif case == "parent_inside_target":
        parent = copy_parent(narrow_parent_ppo, tmp_path / "stance")
        target = tmp_path / "stance"
        kwargs["target_stage_dir"] = target
        fragments = (f"{target} lies inside the parent's directory {parent['run_dir']} (or contains it)",)
    elif case == "explicit_non_handoff_stem":
        # The explicit-pair form never guesses a handoff name for another stem (D-C9: the parent's own).
        final_zip = parent["model_zip"].with_name("stage1_final.zip")
        final_pkl = parent["vecnorm_pkl"].with_name("stage1_final_vecnorm.pkl")
        shutil.copyfile(parent["model_zip"], final_zip)
        shutil.copyfile(parent["vecnorm_pkl"], final_pkl)
        kwargs = {**explicit, "model_zip": final_zip, "vecnorm_pkl": final_pkl}
        fragments = ("stage1_final.zip is not a handoff checkpoint (robust_best_model / best_model)",)
    else:  # pragma: no cover - the parametrisation is closed
        raise AssertionError(case)

    if "model_zip" not in kwargs:
        kwargs.setdefault("parent_stage_dir", parent["stage_dir"])
    parent_digests = file_digests(parent["run_dir"])
    with pytest.raises(WidenError) as excinfo:
        widen_checkpoint(**kwargs)
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, message
    for fragment in absent:
        assert fragment not in message, message
    assert file_digests(parent["run_dir"]) == parent_digests, "the parent run is never touched"

    if case == "occupied_target":
        # The occupant is untouched: the tool never removes what it did not write.
        assert sorted(path.name for path in target.iterdir()) == ["stage_config.json"]
        assert (target / "stage_config.json").read_text(encoding="utf-8") == '{"stage": "occupied"}\n'
        return
    if case in ("target_inside_parent_run", "parent_inside_target"):
        # The target lay inside the parent tree (or around it): the parent's files are all that exists there.
        assert not (parent["stage_dir"] / "models" / stage_dirname(SPECIES, STAGE)).exists()
        return
    assert not target.exists()
    assert not target.parent.exists()

    if case == "algorithm_mismatch":
        # The archive itself is sniffed in the explicit-pair form too.
        with pytest.raises(WidenError, match="is a PPO archive, not the SAC declared"):
            widen_checkpoint(**{**explicit, "algorithm": "sac"})
        assert not target.exists()
    elif case == "missing_sidecar":
        with pytest.raises(WidenError, match="sidecar not found"):
            widen_checkpoint(**explicit)
        assert not target.exists()
    elif case == "unstamped_sidecar":
        # A failure after the first write (the archive is written before the sidecar is read) removes
        # the target AND the run directory the tool created above it, but nothing that pre-existed.
        kept = tmp_path / "kept"
        kept.mkdir()
        (kept / "marker.txt").write_text("pre-existing\n", encoding="utf-8")
        kept_target = kept / WIDENED_RUN_NAME / stage_dirname(SPECIES, STAGE)
        with pytest.raises(WidenError, match="has no plant identity"):
            widen_checkpoint(**{**kwargs, "target_stage_dir": kept_target})
        assert not kept_target.parent.exists()
        assert sorted(path.name for path in kept.iterdir()) == ["marker.txt"]
        # Under the legacy allowance the unstamped sidecar is accepted (its width still checked) and
        # the archive's own identity keeps the gate.
        legacy = widen_checkpoint(**kwargs, allow_legacy_plant=True)
        assert legacy.report["parent"]["legacy_plant"] is False
        assert legacy.report["parent"]["policy_interface_revision"] == narrow.policy_interface_revision
        with open(legacy.vecnorm_pkl, "rb") as handle:
            assert pickle.load(handle).__dict__[MODEL_IDENTITY_ATTRIBUTE] == current.to_dict()
    elif case == "missing_identity":
        # The deliberate path: --allow-legacy-plant reads the width off the saved Box and
        # records a parent without an interface revision.
        legacy = widen_checkpoint(**kwargs, allow_legacy_plant=True)
        assert legacy.from_observation_dim == current.observation_dim - COMMAND_WIDTH
        assert legacy.report["parent"]["legacy_plant"] is True
        assert legacy.report["parent"]["policy_interface_revision"] is None
        # No parent revision, so no gap to measure: revision_gap is None (JSON null) in the report,
        # the result and the lineage attribute, while the bound it was admitted under is still recorded.
        assert legacy.revision_gap is None
        assert "revision_gap" in legacy.report and legacy.report["revision_gap"] is None
        assert legacy.report["max_revision_gap"] == DEFAULT_MAX_REVISION_GAP == 1
        lineage = read_checkpoint_attribute(legacy.model_zip, MODEL_WIDEN_LINEAGE_ATTRIBUTE)
        assert lineage["parent_plant_identity"] is None
        assert "revision_gap" in lineage and lineage["revision_gap"] is None
        assert set(lineage) == WIDEN_LINEAGE_ATTRIBUTE_KEYS
        run = json.loads((target / "stage_config.json").read_text())["run"]
        assert set(WIDEN_LINEAGE_KEYS) <= set(run)
        assert run["widened_from_policy_interface_revision"] is None
        assert run["widened_from_policy_interface_sha256"] is None
        assert run["widened_from_checkpoint_sha256"] == sha256_file(parent["model_zip"])
        assert read_checkpoint_attribute(legacy.model_zip, MODEL_IDENTITY_ATTRIBUTE) == current.to_dict()


def test_identity_gate_revision_gap_bound(narrow_parent_ppo_r2):
    """``identity_gate_errors`` admits ``1 <= current - parent <= max_revision_gap`` and nothing else (D-C17)."""
    current = narrow_parent_ppo_r2["current"]
    r2 = narrow_parent_ppo_r2["narrow"]
    assert current.policy_interface_revision - r2.policy_interface_revision == 2
    assert r2.observation_dim + COMMAND_WIDTH == current.observation_dim
    assert (r2.physics_sha256, r2.nq, r2.nv, r2.nu, r2.action_dim) == (
        current.physics_sha256,
        current.nq,
        current.nv,
        current.nu,
        current.action_dim,
    )

    def back(gap: int, **fields: Any) -> PlantIdentity:
        parent: PlantIdentity = dataclasses.replace(
            r2, policy_interface_revision=current.policy_interface_revision - gap, **fields
        )
        return parent

    assert identity_gate_errors(back(1), current) == []
    assert identity_gate_errors(back(1), current, max_revision_gap=2) == []
    assert identity_gate_errors(back(2), current, max_revision_gap=2) == []
    assert identity_gate_errors(back(2), current, max_revision_gap=3) == []
    assert identity_gate_errors(back(3), current, max_revision_gap=3) == []
    (problem,) = identity_gate_errors(back(2), current)
    assert problem.startswith(
        f"policy_interface_revision: parent={current.policy_interface_revision - 2}, "
        f"current={current.policy_interface_revision}"
    )
    assert "gap 2 exceeds max_revision_gap=1" in problem
    assert "--max-revision-gap 2 / max_revision_gap=2" in problem
    assert "plant_versions.toml" in problem
    (problem,) = identity_gate_errors(back(3), current, max_revision_gap=2)
    assert "gap 3 exceeds max_revision_gap=2" in problem
    # The gate widens forward only: the same or a newer revision is a gap below 1 under every bound.
    for gap in (0, -1):
        for bound in (1, 2):
            (problem,) = identity_gate_errors(back(gap), current, max_revision_gap=bound)
            assert f"(gap {gap}: the tool widens forward only" in problem
    # A wider bound never relaxes the other conditions: a width-changing hop stays refused.
    problems = identity_gate_errors(
        back(2, observation_dim=current.observation_dim - 2 * COMMAND_WIDTH), current, max_revision_gap=2
    )
    assert problems == [
        f"observation_dim: parent={current.observation_dim - 2 * COMMAND_WIDTH} + {COMMAND_WIDTH} "
        f"!= current={current.observation_dim}"
    ]
    other = "sha256:" + hashlib.sha256(b"another physics plant").hexdigest()
    problems = identity_gate_errors(back(2, physics_sha256=other), current, max_revision_gap=2)
    assert problems == [f"physics_sha256: parent={other!r}, current={current.physics_sha256!r}"]
    # The bound itself is validated: below 1 (or not an integer) is a refusal, never a pass.
    for bad in (0, -1, True, "2", 1.0, np.int64(0), np.float64(2.0)):
        with pytest.raises(WidenError, match="max_revision_gap must be an integer >= 1"):
            identity_gate_errors(back(1), current, max_revision_gap=bad)  # type: ignore[arg-type]
    # A numpy integer (a bound computed from manifest data) is an integer: admitted, and coerced to
    # a plain int so the messages and the JSON report carry `2`, not `np.int64(2)`.
    assert identity_gate_errors(back(2), current, max_revision_gap=np.int64(2)) == []  # type: ignore[arg-type]
    (problem,) = identity_gate_errors(back(3), current, max_revision_gap=np.int64(2))  # type: ignore[arg-type]
    assert "gap 3 exceeds max_revision_gap=2;" in problem and "np.int64" not in problem
    coerced = _check_max_revision_gap(np.int64(2))
    assert coerced == 2 and type(coerced) is int
    assert DEFAULT_MAX_REVISION_GAP == 1


# ── every repo loader ────────────────────────────────────────────────────────


def test_widened_pair_loads_through_every_repo_loader(widened):
    """policy_loading, the curriculum loader into create_vec_env envs, and both recovery-gate loaders accept the pair."""
    parent = widened["parent"]
    result = widened["result"]
    current = widened["current"]
    algorithm = widened["algorithm"]
    entry = load_stage_manifest(SPECIES).by_id(STAGE)

    # policy_loading.load_sb3_checkpoint with the current identity.
    model, normalizer, resolved = load_sb3_checkpoint(
        str(result.model_zip),
        str(result.vecnorm_pkl),
        env_factory=lambda: build_env(SPECIES, STAGE),
        guess_sidecar=False,
        plant_identity=current,
    )
    try:
        assert resolved == str(result.vecnorm_pkl)
        assert isinstance(model, PPO if algorithm == "ppo" else SAC)
        assert model.observation_space.shape == (current.observation_dim,)
        assert normalizer.obs_rms.mean.shape == (current.observation_dim,)
        assert normalizer.training is False and normalizer.norm_reward is False
        widened_mean = normalizer.obs_rms.mean.copy()
        widened_var = normalizer.obs_rms.var.copy()
    finally:
        normalizer.close()
    # The parent itself is refused by the same loader: the interface moved, and widening is what admits it.
    with pytest.raises(PlantCompatibilityError):
        load_sb3_checkpoint(
            str(parent["model_zip"]),
            str(parent["vecnorm_pkl"]),
            env_factory=lambda: build_env(SPECIES, STAGE),
            guess_sidecar=False,
            plant_identity=current,
        )

    # curriculum.load_vecnorm_stats into train_base.create_vec_env train / eval envs.
    species_cfg = get_species_config(SPECIES)
    stage_configs = {entry.reference: load_stage_config(SPECIES, entry.reference)}
    train_env = train_base.create_vec_env(
        species_cfg, stage_configs, entry.reference, 1, 0, algorithm=algorithm, plant_identity=current
    )
    eval_env = train_base.create_vec_env(
        species_cfg, stage_configs, entry.reference, 1, 1000, algorithm=algorithm, plant_identity=current
    )
    try:
        assert load_vecnorm_stats(str(result.vecnorm_pkl), train_env, eval_env, current_plant=current) is True
        assert np.array_equal(train_env.obs_rms.mean, widened_mean)
        assert np.array_equal(train_env.obs_rms.var, widened_var)
        assert np.array_equal(eval_env.obs_rms.mean, widened_mean)
        assert eval_env.training is False and eval_env.norm_reward is False
        assert train_env.training is True
        # ...and the parent's sidecar is refused there too — by SB3's own width check
        # (VecNormalize.load re-binds the wrapper before the plant is validated).
        with pytest.raises((AssertionError, PlantCompatibilityError), match="61|plant"):
            load_vecnorm_stats(str(parent["vecnorm_pkl"]), train_env, eval_env, current_plant=current)
    finally:
        train_env.close()
        eval_env.close()

    # freeze_recovery_gate: the source check (algorithm sniff, identity, re-stamped fingerprint) ...
    identity = freeze_recovery_gate._validate_checkpoint_source(result.model_zip, SPECIES, STAGE, algorithm)
    assert identity == current
    with pytest.raises(freeze_recovery_gate.GateResolutionError):
        freeze_recovery_gate._validate_checkpoint_source(
            result.model_zip, SPECIES, STAGE, "sac" if algorithm == "ppo" else "ppo"
        )
    with pytest.raises(PlantCompatibilityError):
        freeze_recovery_gate._validate_checkpoint_source(parent["model_zip"], SPECIES, STAGE, algorithm)
    # ... and the raw observation statistics against the current plant.
    stats = freeze_recovery_gate.load_vecnormalize_obs_stats(result.vecnorm_pkl, expected_plant=current)
    assert stats["mean"].shape == (current.observation_dim,)
    assert np.array_equal(stats["mean"], widened_mean)
    assert np.array_equal(stats["var"], widened_var)
    assert np.array_equal(stats["mean"][-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH))
    assert np.array_equal(stats["var"][-COMMAND_WIDTH:], np.ones(COMMAND_WIDTH))
    with pytest.raises(PlantCompatibilityError):
        freeze_recovery_gate.load_vecnormalize_obs_stats(parent["vecnorm_pkl"], expected_plant=current)


# ── the widened directory as a certified ROOT (ancestors rules 1-7) ──────────


def test_widened_stage_dir_is_reusable_as_a_root_after_a_passed_verdict(widened, tmp_path):
    """Once judged, the widened directory passes every reuse rule as a root; rules 3 and 4 still bite."""
    parent = widened["parent"]
    result = widened["result"]
    current = widened["current"]
    entry = load_stage_manifest(SPECIES).by_id(STAGE)
    curriculum = load_stage_config(SPECIES, STAGE)["curriculum_kwargs"]
    current_task = stance_task_sha256(current)
    directory_task = json.loads((widened["target"] / "task_fingerprint.json").read_text())["task_sha256"]
    assert directory_task == current_task
    handoff = parent["handoff_name"]

    def certified_copy(name: str, *, task_sha256: str, lineage: "dict[str, Any] | None" = None) -> tuple[Path, Path]:
        run_dir = tmp_path / name / widened["run_dir"].name
        shutil.copytree(widened["run_dir"], run_dir)
        stage_dir = run_dir / widened["target"].name
        if lineage:
            path = stage_dir / "stage_config.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["run"].update(lineage)
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        write_gate_verdict(
            stage_dir,
            species=SPECIES,
            stage=entry.reference,
            stage_id=entry.id,
            gate_kind=curriculum["gate_kind"],
            gate_schema_version=curriculum["gate_schema_version"],
            passed=True,
            failures=[],
            task_sha256=task_sha256,
            judged_by="test_widen_checkpoint",
            checkpoint=stage_dir / "models" / f"{handoff}.zip",
            normalization=stage_dir / "models" / f"{handoff}_vecnorm.pkl",
            gate_config=gate_config_view(curriculum),
        )
        return run_dir, stage_dir

    def find(run_dir: Path):
        return find_certified_ancestor(
            run_dir,
            species=SPECIES,
            entry=entry,
            current_task_sha256=current_task,
            plant_identity=current,
            current_gate_config=curriculum,
            parent_model_sha256=None,
            follow_records=True,
        )

    run_dir, stage_dir = certified_copy("certified", task_sha256=current_task)
    ancestor = find(run_dir)
    assert ancestor.stage_id == entry.id
    assert ancestor.run_id == WIDENED_RUN_NAME
    assert ancestor.source_run_dir == run_dir
    assert ancestor.stage_dir == stage_dir
    assert ancestor.handoff_name == handoff
    assert ancestor.model_stem == str(stage_dir / "models" / handoff)
    assert ancestor.model_zip == stage_dir / "models" / f"{handoff}.zip"
    assert ancestor.normalization_path == stage_dir / "models" / f"{handoff}_vecnorm.pkl"
    assert ancestor.model_sha256 == result.widened_checkpoint_sha256
    assert ancestor.normalization_sha256 == result.widened_normalization_sha256
    assert ancestor.task_sha256 == current_task
    assert ancestor.via == ()

    # Rule 4: a widened root that claims an initialize_next_stage entry is refused — which is why the
    # tool records the parent under WIDEN_LINEAGE_KEYS and never LOAD_LINEAGE_KEYS.
    chained_run, chained_stage = certified_copy(
        "chained",
        task_sha256=current_task,
        lineage={
            "load_path": str(parent["model_zip"]),
            "load_mode": "initialize_next_stage",
            "parent_checkpoint_sha256": result.parent_checkpoint_sha256,
        },
    )
    with pytest.raises(AncestorReuseError) as chained:
        find(chained_run)
    assert (
        f"{chained_stage} entered from a parent checkpoint ({result.parent_checkpoint_sha256}) under "
        f"initialize_next_stage, but {entry.id!r} is a root node in the current manifest"
    ) in str(chained.value)

    # Rule 3: a verdict that records the parent's OLD task hash is refused with the current task named.
    stale_task = parent["task_fingerprint"]["task_sha256"]
    assert stale_task != current_task
    stale_run, stale_stage = certified_copy("stale_task", task_sha256=stale_task)
    with pytest.raises(AncestorReuseError) as stale:
        find(stale_run)
    assert f"{stale_stage}/gate_verdict.json was judged under task {stale_task}" in str(stale.value)
    assert f"configured as task {current_task} now" in str(stale.value)


# ── rule 5: hashes stable after the self-verification ────────────────────────


def test_widened_pair_hashes_are_stable_after_self_verification(widened):
    """The digests taken before the tool's verification pass are the files' digests now, and stay so under a load."""
    result = widened["result"]
    current = widened["current"]
    report = result.report
    assert report["hashes_stable_after_verification"] is True
    before = (result.widened_checkpoint_sha256, result.widened_normalization_sha256)
    assert before == (report["widened"]["checkpoint_sha256"], report["widened"]["normalization_sha256"])
    assert (sha256_file(result.model_zip), sha256_file(result.vecnorm_pkl)) == before
    assert (sha256_file(result.final_zip), sha256_file(result.final_vecnorm_pkl)) == before

    # An independent load pass (what the judge and the trainer do) leaves every file unchanged.
    digests = file_digests(widened["target"])
    model, normalizer, _ = load_sb3_checkpoint(
        str(result.model_zip),
        str(result.vecnorm_pkl),
        env_factory=lambda: build_env(SPECIES, STAGE),
        guess_sidecar=False,
        plant_identity=current,
    )
    try:
        obs = np.zeros((1, current.observation_dim), dtype=np.float32)
        model.predict(normalizer.normalize_obs(obs), deterministic=True)
    finally:
        normalizer.close()
    assert file_digests(widened["target"]) == digests


# ── a parent whose schedules are bytecode (every pre-loader Drive archive) ───


def test_widening_a_closure_schedule_parent_yields_a_bytecode_free_archive(narrow_parent_ppo_closures, tmp_path):
    """The parent's cloudpickled schedule members are re-stated from its recorded hyperparameters, never re-pickled.

    The two dead widen sessions of 2026-09-19 re-pickled the r11 parent's Python 3.12 closures into a zip whose
    ``system_info.txt`` said 3.13, then died loading the parent bare (KNOWN_ISSUES, "SB3 archives are bound to the
    interpreter that saved them"). Widened archives now carry no bytecode at all, the report says what was
    re-stated, and the exact-transfer verification is unchanged.
    """
    from environments.shared.curriculum.schedules import LinearSchedule
    from environments.shared.policy_loading import inspect_sb3_archive, load_sb3_model

    parent = narrow_parent_ppo_closures
    parent_inspection = inspect_sb3_archive(parent["model_zip"])
    assert parent_inspection.bytecode_members == {"learning_rate", "lr_schedule", "clip_range"}
    widened = widen_into(parent, tmp_path / "widened_closures")
    result = widened["result"]
    assert inspect_sb3_archive(result.model_zip).bytecode_members == frozenset()
    assert inspect_sb3_archive(result.final_zip).bytecode_members == frozenset()
    # The recorded [ppo] block (the stage TOML's, saved by save_stage_config) is the source of the re-statement.
    block = load_stage_config(SPECIES, STAGE)["ppo_kwargs"]
    expected_lr = LinearSchedule(block["learning_rate"], block["learning_rate_end"])
    restated = result.report["schedule_members_restated"]
    assert set(restated) == {"learning_rate", "lr_schedule", "clip_range"}
    assert restated["learning_rate"] == restated["lr_schedule"] == repr(expected_lr)
    assert restated["clip_range"] == repr(float(block["clip_range"]))
    model = load_sb3_model(str(result.model_zip), algorithm="ppo", device="cpu")
    assert isinstance(model.learning_rate, LinearSchedule)
    assert (model.learning_rate.initial, model.learning_rate.final) == (expected_lr.initial, expected_lr.final)
    assert model.lr_schedule(1.0) == pytest.approx(expected_lr.initial)
    assert model.clip_range(1.0) == pytest.approx(float(block["clip_range"]))
    # Exact transfer held through the re-statement: the tool's own verification pinned it before writing the report.
    assert result.report["padded_columns_exactly_zero"] is True
    assert result.max_action_delta_zero_command <= ACTION_DELTA_ATOL
    assert result.max_action_delta_probe_command <= ACTION_DELTA_ATOL
    # A by-reference parent has nothing to re-state.
    plain = widen_into(narrow_parent_ppo_for_contrast(tmp_path), tmp_path / "widened_plain")
    assert plain["result"].report["schedule_members_restated"] == {}


def narrow_parent_ppo_for_contrast(root: Path) -> dict:
    """A by-reference (float-schedule) PPO parent built beside the closure one, for the empty-report contrast."""
    return build_narrow_parent(root / "contrast_parent", "ppo")


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_main_cli_round_trip(narrow_parent_ppo, tmp_path, capsys, caplog):
    """The argv form writes the layout and prints the report; a second run on the same target exits 1."""
    parent = narrow_parent_ppo
    current = parent["current"]
    target = tmp_path / "cli_run" / stage_dirname(SPECIES, STAGE)
    argv = [
        "--species",
        SPECIES,
        "--stage",
        STAGE,
        "--from-stage-dir",
        str(parent["stage_dir"]),
        "--to-stage-dir",
        str(target),
        "--label",
        "cli round trip",
    ]
    assert widen_main(argv) == 0
    printed = capsys.readouterr().out
    report = json.loads(printed)
    assert report == json.loads((target / WIDEN_REPORT_FILENAME).read_text())
    assert report["schema"] == WIDEN_REPORT_SCHEMA
    assert report["algorithm"] == "ppo"
    assert report["handoff_name"] == parent["handoff_name"]
    assert report["parent"]["stage_dir"] == str(parent["stage_dir"])
    label = stage_label(parent["reference"])
    assert sorted(path.name for path in (target / "models").iterdir()) == sorted(
        [
            f"{parent['handoff_name']}.zip",
            f"{parent['handoff_name']}_vecnorm.pkl",
            f"{label}_final.zip",
            f"{label}_final_vecnorm.pkl",
        ]
    )
    assert sorted(path.name for path in target.iterdir()) == sorted(
        ["models", "stage_config.json", "plant_identity.json", "task_fingerprint.json", WIDEN_REPORT_FILENAME]
    )
    run = json.loads((target / "stage_config.json").read_text())["run"]
    assert run["label"] == "cli round trip"
    assert run["widened_from_run_id"] == PARENT_RUN_NAME
    assert run["duration_seconds"] == PARENT_DURATION_SECONDS
    assert not set(LOAD_LINEAGE_KEYS) & set(run)
    assert json.loads((target / "plant_identity.json").read_text()) == current.to_dict()

    # The second invocation on the same target is refused as occupied; nothing changes, nothing is printed.
    digests = file_digests(target)
    with caplog.at_level(logging.ERROR, logger=widen_module.logger.name):
        assert widen_main(argv) == 1
    assert "Refusing to widen" in caplog.text
    assert f"{target} already records a stage (stage_config.json present)" in caplog.text
    assert capsys.readouterr().out == ""
    assert file_digests(target) == digests

    # The explicit-pair form: the run facts come from the flags, the run id from --parent-run-id,
    # and no duration is invented.
    explicit_target = tmp_path / "cli_explicit" / stage_dirname(SPECIES, STAGE)
    explicit = [
        "--species",
        SPECIES,
        "--stage",
        str(parent["reference"]),
        "--model",
        str(parent["model_zip"]),
        "--vecnorm",
        str(parent["vecnorm_pkl"]),
        "--algorithm",
        "ppo",
        "--seed",
        str(PARENT_SEED),
        "--n-envs",
        "1",
        "--timesteps",
        str(PARENT_TIMESTEPS),
        "--parent-run-id",
        PARENT_RUN_NAME,
        "--to-stage-dir",
        str(explicit_target),
    ]
    assert widen_main(explicit) == 0
    explicit_report = json.loads(capsys.readouterr().out)
    assert explicit_report["parent"]["stage_dir"] is None
    assert explicit_report["parent"]["run_id"] == PARENT_RUN_NAME
    # Both forms widen the same parent to the same weights (the archives themselves differ by
    # SB3's zip-entry timestamps, so the digests are not compared).
    for key in ("padded_tensors", "inserted_at", "optimizer_members_padded", "num_timesteps"):
        assert explicit_report[key] == report[key]
    assert (explicit_report["from_observation_dim"], explicit_report["to_observation_dim"]) == (
        report["from_observation_dim"],
        report["to_observation_dim"],
    )
    _, explicit_params, _ = load_from_zip_file(explicit_report["widened"]["model_path"], device="cpu", load_data=False)
    _, cli_params, _ = load_from_zip_file(report["widened"]["model_path"], device="cpu", load_data=False)
    assert set(explicit_params["policy"]) == set(cli_params["policy"])
    for name, tensor in cli_params["policy"].items():
        assert torch.equal(tensor, explicit_params["policy"][name]), name
    assert explicit_report["parent"]["checkpoint_sha256"] == report["parent"]["checkpoint_sha256"]
    explicit_run = json.loads((explicit_target / "stage_config.json").read_text())["run"]
    assert (explicit_run["seed"], explicit_run["n_envs"], explicit_run["timesteps"]) == (
        PARENT_SEED,
        1,
        PARENT_TIMESTEPS,
    )
    assert "duration_seconds" not in explicit_run
    assert "label" not in explicit_run
    assert explicit_run["widened_from_run_id"] == PARENT_RUN_NAME
    assert set(WIDEN_LINEAGE_KEYS) <= set(explicit_run)

    # ... and the declared algorithm is checked against the archive.
    mismatch_target = tmp_path / "cli_mismatch" / stage_dirname(SPECIES, STAGE)
    mismatch = [*explicit[:-1], str(mismatch_target)]
    mismatch[mismatch.index("--algorithm") + 1] = "sac"
    with caplog.at_level(logging.ERROR, logger=widen_module.logger.name):
        assert widen_main(mismatch) == 1
    assert "is a PPO archive, not the SAC declared" in caplog.text
    assert not mismatch_target.exists()

    # --max-revision-gap 0 is refused at the argparse boundary before widen_checkpoint() is entered
    # (exit 1, nothing written, nothing printed).  The exit code and the no-write guarantee are ALSO
    # given by the API's own _check_max_revision_gap (test_refusals[max_revision_gap_zero]); what this
    # pins beyond them is the boundary's wording and that the API refusal never had to fire.
    zero_target = tmp_path / "cli_gap_zero" / stage_dirname(SPECIES, STAGE)
    zero = [*argv[: argv.index("--to-stage-dir")], "--to-stage-dir", str(zero_target), "--max-revision-gap", "0"]
    with caplog.at_level(logging.ERROR, logger=widen_module.logger.name):
        assert widen_main(zero) == 1
    assert "Refusing to widen: --max-revision-gap must be >= 1, not 0" in caplog.text
    assert "max_revision_gap must be an integer >= 1" not in caplog.text
    assert capsys.readouterr().out == ""
    assert not zero_target.parent.exists()
    # ... and a non-integer value is argparse's own usage error.
    with pytest.raises(SystemExit):
        widen_main([*zero[:-1], "two"])
    assert not zero_target.parent.exists()


def test_main_cli_round_trip_with_max_revision_gap(narrow_parent_ppo_r2, tmp_path, capsys, caplog):
    """An r-2 parent: the default CLI refuses it naming the gap and the flag; ``--max-revision-gap 2`` widens it."""
    parent = narrow_parent_ppo_r2
    current = parent["current"]
    assert parent["revision_gap"] == 2
    target = tmp_path / "cli_r2" / stage_dirname(SPECIES, STAGE)
    argv = [
        "--species",
        SPECIES,
        "--stage",
        STAGE,
        "--from-stage-dir",
        str(parent["stage_dir"]),
        "--to-stage-dir",
        str(target),
        "--label",
        "cli r-2 round trip",
    ]
    with caplog.at_level(logging.ERROR, logger=widen_module.logger.name):
        assert widen_main(argv) == 1
    assert "Refusing to widen" in caplog.text
    assert f"is not the current {SPECIES} plant one interface-only revision behind" in caplog.text
    assert (
        f"policy_interface_revision: parent={current.policy_interface_revision - 2}, "
        f"current={current.policy_interface_revision} (gap 2 exceeds max_revision_gap=1"
    ) in caplog.text
    assert "--max-revision-gap 2 / max_revision_gap=2" in caplog.text
    assert "plant_versions.toml" in caplog.text
    assert capsys.readouterr().out == ""
    assert not target.parent.exists()

    assert widen_main([*argv, "--max-revision-gap", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads((target / WIDEN_REPORT_FILENAME).read_text())
    assert (report["revision_gap"], report["max_revision_gap"]) == (2, 2)
    assert report["parent"]["policy_interface_revision"] == current.policy_interface_revision - 2
    assert report["parent"]["policy_interface_sha256"] == parent["narrow"].policy_interface_sha256
    assert report["widened"]["policy_interface_revision"] == current.policy_interface_revision
    assert (report["from_observation_dim"], report["to_observation_dim"]) == (
        current.observation_dim - COMMAND_WIDTH,
        current.observation_dim,
    )
    assert report["padded_columns_exactly_zero"] is True
    assert report["max_action_delta_zero_command"] <= ACTION_DELTA_ATOL
    assert report["max_action_delta_probe_command"] <= ACTION_DELTA_ATOL
    run = json.loads((target / "stage_config.json").read_text())["run"]
    assert run["label"] == "cli r-2 round trip"
    assert run["widened_from_policy_interface_revision"] == current.policy_interface_revision - 2
    assert run["widened_from_run_id"] == PARENT_RUN_NAME
    assert not set(LOAD_LINEAGE_KEYS) & set(run)
    lineage = read_checkpoint_attribute(Path(report["widened"]["model_path"]), MODEL_WIDEN_LINEAGE_ATTRIBUTE)
    assert lineage["revision_gap"] == 2
    assert json.loads((target / "plant_identity.json").read_text()) == current.to_dict()
