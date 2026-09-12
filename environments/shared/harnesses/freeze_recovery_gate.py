"""Hand-run producer: freeze the recovery stage's gate resolution (plan P5).

STAGE1_SPLIT_PLAN §5 requires one atomic artifact — capability spec, null
manifest, decision procedure — frozen before a gate that compares against
measured nulls may advance anything.  ``curriculum/gate_resolver.py``
defines and ENFORCES that artifact; this module is the thing that MAKES
it: it rolls the null panels under the P3-calibrated judge at the stage's
real task fingerprint, writes ``gate_resolution.json``, and then reads it
back through ``require_gate_resolution`` so a record that cannot be
re-loaded at the current task is never left on disk claiming to be frozen.

Freezing a gate is explicit: the harness CLI or opt-in notebook runs it
before recovery training, and an existing resolution is replaced only
when ``--replace`` or ``replace=True`` says so.

**Where the numbers come from.**  Everything numeric here carries its
derivation record: the judge is the P3 calibration (first-runs record
§4.1/§4.3, defined once in ``recovery_evaluation``), the panel geometry
and the two gate thresholds are the P5 decisions taken on the §9 measured
results.  Nothing is re-derived at run time.

New species load a measured, species-specific recovery calibration profile.
Their frozen resolution also hashes ``evaluation_spec``: profile identity,
height reference, control timestep and safe set. A changed or missing judge
refuses evaluation. Historical T-Rex resolutions retain their original judge
and remain readable; its constants below are never a fallback for other species.

Examples (repo root):

    # The frozen record: statue null only — the default paired null
    python -m environments.shared.harnesses.freeze_recovery_gate \
        --stage-dir results/trex/recovery

    # Both nulls: the brace is the policy's post-settle mean, held
    python -m environments.shared.harnesses.freeze_recovery_gate \
        --stage-dir results/trex/recovery \
        --policy-zip <robust_best_model.zip> --vecnorm <..._vecnorm.pkl>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from environments.shared.config import build_env, load_stage_config
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.gate_resolver import (
    GateResolutionError,
    build_gate_resolution,
    require_gate_resolution,
    required_paired_nulls,
    write_gate_resolution,
)
from environments.shared.curriculum.recovery_gate import RecoveryGateThresholds
from environments.shared.recovery_calibration import RecoveryCalibration, load_recovery_calibration
from environments.shared.recovery_evaluation import (
    CALIBRATED_HEIGHT_REFERENCE_M,
    CALIBRATED_POSTURE_ONLY,
    RecoveryPanelEvidence,
    _safe_step,
    constant_action_controller,
    roll_recovery_panel,
    write_recovery_evidence,
    zero_action_controller,
)
from environments.shared.species_names import resolve_species_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The frozen capability spec.  These are the P5 values; the calibrated safe
# set and height reference they are paired with live in recovery_evaluation.
# ---------------------------------------------------------------------------

#: P5 DECISION (project owner), measured basis first-runs record §9.1/§9.5.
#: The attainable episode-success LCB95 at the training schedule is **0.361**
#: for the 3M checkpoint (20/40) and **0.338** for the 5M (19/40), against a
#: statue null whose one-sided UCB95 is 0.072.  0.30 certifies what the stage
#: can actually do today and stays clear of the null ceiling; the plan's
#: aspirational 0.725 needs a materially stronger policy, and the capability
#: ceiling between 165.5 N and 210 N (§9.2) is the recorded envelope this
#: threshold is honest about rather than hiding.
MIN_RECOVERY_SUCCESS_LCB = 0.30

#: P5 DECISION (project owner), measured basis first-runs record §9.1.  The
#: paired policy-minus-statue per-seed success LCB95 at the training
#: schedule is **+0.365** (3M) and **+0.340** (5M).  0.20 sits below both
#: and far above zero, so the authoritative criterion (split plan §6: when
#: a paired test exists it outranks any reward threshold) is a real
#: superiority claim, not a not-worse-than-nothing one.
MIN_PAIRED_SUCCESS_DELTA_LCB = 0.20

#: First-runs record §4.1: measured re-entry times have p90 = 84 steps, so
#: the 100-step window is validated by measurement rather than merely
#: retained from the provisional design.
T_RECOVER_STEPS = 100

#: The dwell that makes re-entry mean recovery rather than a touch-and-go
#: (recovery_gate.per_push_recovery); the window every §4.3 and §9 panel
#: was judged under.
DWELL_STEPS = 50

#: The registered panel: 40 episodes from seed 3042 (seeds 3042–3081), the
#: block every measured panel in §4.3 and §9 used.  ``min_eval_episodes``
#: is frozen at the panel size, so a short panel fails the gate instead of
#: quietly widening its own confidence interval.
MIN_EVAL_EPISODES = 40
#: The publication-protocol first seed (constants.PUBLICATION_SEED_START,
#: 3042), which the panel family starts from.
PANEL_SEED_START = PUBLICATION_SEED_START

#: The backend string every SB3 train path stamps into its stage
#: fingerprint (train_base), so the frozen ``task_sha256`` is the one a
#: gated run will present — a different backend string would freeze a
#: resolution no run can ever match, and the resolver would (correctly)
#: block forever.
FINGERPRINT_BACKEND = "stable-baselines3"

#: The tensors the NumPy forward pass needs out of an SB3 checkpoint.
#: ``mlp_extractor.policy_net.1``/``.3`` are the tanh activations and carry
#: no parameters; ``log_std`` is the exploration std, unused by a
#: deterministic prediction.
POLICY_HIDDEN_LAYERS = ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2")
POLICY_OUTPUT_LAYER = "action_net"


@dataclass(frozen=True)
class FrozenGateResolution:
    """One freeze: what was written, and the evidence behind it."""

    path: Path
    resolution: dict[str, Any]
    task_fingerprint: dict[str, Any]
    null_evidence: dict[str, RecoveryPanelEvidence]


# ---------------------------------------------------------------------------
# Task identity and environment
# ---------------------------------------------------------------------------


def stage_task_fingerprint(species: str, stage: "int | str") -> dict[str, Any]:
    """The stage's REAL task fingerprint, derived the way training derives it.

    Same call, same backend string, and the same current plant identity as
    ``train_base``, so the ``task_sha256`` frozen into the resolution is
    exactly the one a gated run will compute — the resolver's staleness
    check is only meaningful if both sides derive identically.
    """
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.task_fingerprint import derive_stage_task_fingerprint

    config = load_stage_config(species, stage)
    return derive_stage_task_fingerprint(
        species=species,
        stage=stage,
        backend=FINGERPRINT_BACKEND,
        env_kwargs=config.get("env_kwargs", {}),
        plant_identity=current_plant_identity(species).to_dict(),
    )


# ``build_env`` is the shared ``environments.shared.config.build_env``,
# re-exported above: species-generic through the registry (the
# perturbation engine is, by the 2026-08-15 standing constraint), even
# though P5 enables the recovery stage for T-Rex only.


# ---------------------------------------------------------------------------
# Checkpoint inference: SB3 when it works, NumPy when it does not
# ---------------------------------------------------------------------------


def load_vecnormalize_obs_stats(vecnorm_pkl: "str | Path", *, expected_plant: Any = None) -> dict[str, Any]:
    """Read the frozen observation statistics out of a VecNormalize pickle.

    Evaluation must normalize observations with the checkpoint's OWN stats;
    a policy fed raw observations is a different controller.  The pickle
    references Stable-Baselines3 classes, so unpickling needs SB3 importable
    even on the NumPy inference path — what §9 could not use was ``PPO.load``,
    not the package.
    """
    with open(vecnorm_pkl, "rb") as handle:
        vecnorm = pickle.load(handle)
    if expected_plant is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(vecnorm, expected_plant, artifact=str(vecnorm_pkl))
    return {
        "mean": np.asarray(vecnorm.obs_rms.mean, dtype=np.float64),
        "var": np.asarray(vecnorm.obs_rms.var, dtype=np.float64),
        "clip_obs": float(getattr(vecnorm, "clip_obs", 10.0)),
        "epsilon": float(getattr(vecnorm, "epsilon", 1e-8)),
    }


def normalize_observation(observation: Any, stats: Mapping[str, Any]) -> np.ndarray:
    """VecNormalize's observation transform: standardize, then clip."""
    normalized = (np.asarray(observation, dtype=np.float64) - stats["mean"]) / np.sqrt(stats["var"] + stats["epsilon"])
    return np.asarray(np.clip(normalized, -stats["clip_obs"], stats["clip_obs"]), dtype=np.float64)


def load_policy_weights(policy_zip: "str | Path") -> dict[str, np.ndarray]:
    """Extract the deterministic-forward tensors from an SB3 checkpoint ZIP.

    Reads ``policy.pth`` out of the archive with ``torch.load`` and keeps
    only arrays — the tensors, never the policy object — which is the part
    of the checkpoint that stayed loadable when ``PPO.load`` did not (§9's
    instrumentation note).  Missing tensors fail closed: a checkpoint with
    a different architecture must not be silently half-loaded.
    """
    import torch

    with zipfile.ZipFile(policy_zip) as archive:
        with archive.open("policy.pth") as member:
            payload = member.read()
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    weights = {str(name): np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64) for name, tensor in state.items()}
    required = [
        f"{layer}.{part}" for layer in (*POLICY_HIDDEN_LAYERS, POLICY_OUTPUT_LAYER) for part in ("weight", "bias")
    ]
    missing = [name for name in required if name not in weights]
    if missing:
        raise ValueError(
            f"{policy_zip} does not carry the expected MlpPolicy tensors (missing {missing}); "
            "the NumPy forward pass reproduces the two-tanh-layer default architecture only"
        )
    return weights


def numpy_deterministic_action(
    weights: Mapping[str, np.ndarray],
    observation: Any,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """SB3's ``predict(deterministic=True)`` for a Gaussian MlpPolicy, in NumPy.

    For PPO's default policy that prediction is exactly: two tanh layers,
    the linear ``action_net`` mean, and a clip to the action space
    (``squash_output`` is False, and the distribution's ``log_std`` is
    never consulted when deterministic).  Evaluation therefore needs none
    of torch's policy machinery — which is what made §9's panels possible
    when ``PPO.load`` segfaulted on the run image: this path was verified
    against ``predict(deterministic=True)`` to a maximum absolute
    difference of 6.5e-9 over 200 observations, so the panels are
    SB3-identical and only the inference path differs.
    """
    latent = np.asarray(observation, dtype=np.float64)
    for layer in POLICY_HIDDEN_LAYERS:
        latent = np.tanh(weights[f"{layer}.weight"] @ latent + weights[f"{layer}.bias"])
    action = weights[f"{POLICY_OUTPUT_LAYER}.weight"] @ latent + weights[f"{POLICY_OUTPUT_LAYER}.bias"]
    return np.asarray(np.clip(action, action_low, action_high), dtype=np.float64)


def policy_controller(
    policy_zip: "str | Path",
    vecnorm_pkl: "str | Path",
    *,
    action_space: Any,
    inference: str = "auto",
    algorithm: str = "ppo",
    expected_species: str | None = None,
    expected_stage: int | str | None = None,
) -> Callable[[Any], np.ndarray]:
    """A deterministic controller for a checkpoint, on whichever path works.

    ``algorithm`` selects PPO or SAC. ``inference`` selects the forward pass:
    ``sb3`` insists on that algorithm's loader, ``numpy`` supports PPO only,
    and ``auto`` tries SB3 first and falls back with a warning for PPO. The fallback catches an
    unusable ``PPO.load`` only when it RAISES — §9's failure was a segfault,
    which no ``except`` can see, so a run on such an image must pass
    ``--inference numpy`` rather than rely on ``auto``.
    """
    if inference not in ("auto", "sb3", "numpy"):
        raise ValueError(f"unknown inference mode {inference!r}; expected auto, sb3, or numpy")
    algorithm = algorithm.lower()
    if algorithm not in ("ppo", "sac"):
        raise ValueError(f"unknown recovery algorithm {algorithm!r}; expected ppo or sac")
    if algorithm == "sac" and inference == "numpy":
        raise ValueError("SAC recovery evaluation requires SB3 inference; the NumPy fallback supports PPO only")
    if (expected_species is None) != (expected_stage is None):
        raise ValueError("expected_species and expected_stage must be supplied together")
    if expected_species is not None and expected_stage is not None:
        plant = _validate_checkpoint_source(policy_zip, expected_species, expected_stage, algorithm)
        stats = load_vecnormalize_obs_stats(vecnorm_pkl, expected_plant=plant)
    else:
        stats = load_vecnormalize_obs_stats(vecnorm_pkl)

    if inference in ("auto", "sb3"):
        try:
            from stable_baselines3 import PPO, SAC

            model = (PPO if algorithm == "ppo" else SAC).load(str(policy_zip), device="cpu")
        except Exception as exc:  # noqa: BLE001 — any failure means "use the other path"
            if inference == "sb3" or algorithm == "sac":
                raise
            logger.warning("PPO.load failed (%s); falling back to the NumPy forward pass", exc)
        else:

            def predict_sb3(observation: Any) -> np.ndarray:
                action, _state = model.predict(
                    normalize_observation(observation, stats).astype(np.float32), deterministic=True
                )
                return np.asarray(action, dtype=np.float64)

            return predict_sb3

    weights = load_policy_weights(policy_zip)
    action_low = np.asarray(action_space.low, dtype=np.float64)
    action_high = np.asarray(action_space.high, dtype=np.float64)

    def predict_numpy(observation: Any) -> np.ndarray:
        return numpy_deterministic_action(
            weights,
            normalize_observation(observation, stats),
            action_low=action_low,
            action_high=action_high,
        )

    return predict_numpy


def _validate_checkpoint_source(policy_zip: str | Path, species: str, stage: int | str, algorithm: str) -> Any:
    """Validate source metadata even when inference uses the NumPy fallback."""
    from environments.shared.plant_contract import (
        MODEL_IDENTITY_ATTRIBUTE,
        current_plant_identity,
        validate_recorded_identity,
    )
    from environments.shared.stage_manifest import load_stage_manifest
    from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, validate_recorded_task

    try:
        with zipfile.ZipFile(policy_zip) as archive:
            metadata = json.loads(archive.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise GateResolutionError(f"cannot read recovery checkpoint metadata from {policy_zip}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise GateResolutionError(f"{policy_zip} contains invalid SB3 checkpoint metadata")
    is_ppo = "clip_range" in metadata and "n_epochs" in metadata
    is_sac = "target_entropy" in metadata and "replay_buffer_class" in metadata
    if is_ppo == is_sac or (algorithm == "ppo") != is_ppo:
        raise GateResolutionError(f"{policy_zip} does not identify the selected {algorithm.upper()} algorithm")
    identity = current_plant_identity(species)
    validate_recorded_identity(metadata.get(MODEL_IDENTITY_ATTRIBUTE), identity, artifact=str(policy_zip))
    task = metadata.get(MODEL_TASK_ATTRIBUTE)
    if task is not None and not isinstance(task, dict):
        raise GateResolutionError(f"{policy_zip} contains invalid task fingerprint metadata")
    reference = load_stage_manifest(species).resolve(stage).reference
    current_task = stage_task_fingerprint(species, reference)
    validate_recorded_task(task, current_task, mode="resume_same_stage", artifact=str(policy_zip))
    return identity


# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------


def _roll_null(
    env: Any,
    predict: Callable[[Any], np.ndarray],
    *,
    controller_id: str,
    episodes: int,
    seed: int,
    calibration: RecoveryCalibration | None = None,
) -> RecoveryPanelEvidence:
    """Roll one null panel under the calibrated judge — the only judge here."""
    thresholds = calibration.thresholds if calibration else None
    evidence = roll_recovery_panel(
        env,
        predict,
        controller_id=controller_id,
        episodes=episodes,
        seed=seed,
        t_recover_steps=thresholds.t_recover_steps if thresholds else T_RECOVER_STEPS,
        dwell_steps=thresholds.dwell_steps if thresholds else DWELL_STEPS,
        safe_set=calibration.safe_set if calibration else dict(CALIBRATED_POSTURE_ONLY),
        height_reference=calibration.height_reference_m if calibration else CALIBRATED_HEIGHT_REFERENCE_M,
    )
    _require_judged_pushes(evidence)
    return evidence


def _require_judged_pushes(evidence: RecoveryPanelEvidence) -> None:
    """A full-horizon episode with no judged disturbance proves nothing."""
    if any(episode.full_horizon and episode.n_pushes == 0 for episode in evidence.episodes):
        raise GateResolutionError(
            "recovery panel contains a full-horizon episode without a judged push; "
            "the horizon must include a delivered push and its recovery dwell"
        )


def _species_calibration(species: str, stage: int | str, *, env: Any = None) -> RecoveryCalibration | None:
    if resolve_species_id(species) == "trex":
        return None  # The historical judge remains explicit and T-Rex-only.
    return load_recovery_calibration(species, stage, env=env)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _quiet_brace_controller(
    species: str, predict: Callable[[Any], np.ndarray], *, calibration: RecoveryCalibration
) -> tuple[Callable[[Any], np.ndarray], dict[str, Any]]:
    """Derive a constant brace from quiet stance, before any pushed evaluation.

    The initial one-second settling interval and following one-second sample
    are physical durations, converted through this species' control timestep.
    A checkpoint that cannot survive that reference interval is not suitable
    for a stance warm start; partial trajectories never define its brace.
    """
    env = build_env(species, "stance")
    seeds = (6042, 6043, 6044, 6045)
    # An engineering compatibility check, not a learned recovery result: the
    # stance source must inhabit the same judge that recovery will be scored by.
    min_safe_fraction = 0.95
    try:
        if env.perturbation_capture_velocity_multiple != 0:
            raise GateResolutionError("brace derivation requires quiet stance with pushes disabled")
        settle_steps = max(1, round(1.0 / env.dt))
        sample_steps = max(1, round(1.0 / env.dt))
        actions: list[np.ndarray] = []
        quiet_safe_fractions: dict[str, float] = {}
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            safe_samples = 0
            for step in range(settle_steps + sample_steps):
                action = np.asarray(predict(obs), dtype=np.float64).ravel()
                if action.shape != env.action_space.shape or not np.all(np.isfinite(action)):
                    raise GateResolutionError("stance checkpoint produced an invalid action during brace derivation")
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    raise GateResolutionError(
                        f"stance checkpoint did not survive the complete quiet brace reference "
                        f"(seed {seed}, step {step + 1}/{settle_steps + sample_steps}); "
                        "use a qualified stance checkpoint before recovery training"
                    )
                if step >= settle_steps:
                    actions.append(action)
                    safe_samples += int(_safe_step(env, calibration.safe_set, calibration.height_reference_m))
            fraction = safe_samples / sample_steps
            quiet_safe_fractions[str(seed)] = fraction
            if fraction < min_safe_fraction:
                raise GateResolutionError(
                    f"stance checkpoint does not match the calibrated recovery safe set: seed {seed} has "
                    f"{fraction:.1%} quiet safe occupancy, below the {min_safe_fraction:.0%} engineering prerequisite; "
                    "recalibrate the judge from qualified stance evidence before recovery training"
                )
        held_action = np.mean(np.stack(actions), axis=0)
        return constant_action_controller(held_action), {
            "derivation": "quiet-stance-post-settle-mean/v1",
            "seeds": list(seeds),
            "control_dt_s": float(env.dt),
            "settle_steps": settle_steps,
            "sample_steps": sample_steps,
            "held_action": held_action.tolist(),
            "quiet_stance_min_safe_fraction": min_safe_fraction,
            "quiet_safe_fractions_by_seed": quiet_safe_fractions,
            "quiet_safe_set": calibration.safe_set,
            "quiet_height_reference_m": calibration.height_reference_m,
            "stance_env_kwargs": load_stage_config(species, "stance")["env_kwargs"],
        }
    finally:
        env.close()


def _validated_recovery_resolution(
    stage_dir: str | Path, species: str, stage: int | str
) -> tuple[dict[str, Any], RecoveryCalibration | None]:
    """Resolve the task, judge and pairing domain before training or evaluation."""
    fingerprint = stage_task_fingerprint(species, stage)
    resolution = require_gate_resolution(stage_dir, current_task_sha256=fingerprint["task_sha256"])
    spec = resolution["capability_spec"]
    calibration = _species_calibration(species, stage)
    active_safe_set = calibration.safe_set if calibration else dict(CALIBRATED_POSTURE_ONLY)
    if calibration is not None:
        if resolution.get("evaluation_spec") != calibration.evaluation_spec():
            raise GateResolutionError(
                "frozen recovery evaluation_spec is missing or differs from the current calibration; "
                "re-freeze under the current height reference, control clock and judge"
            )
        if spec != calibration.profile["capability_spec"]:
            raise GateResolutionError("frozen recovery capability_spec differs from the current calibration")

    # The stage directory's RECORDED task fingerprint is what the gate in
    # reporting/gates.py will judge against; when it exists and disagrees
    # with the fingerprint derived from the committed config, rolling a
    # panel would waste 40 episodes on a verdict the gate is going to
    # refuse anyway — refuse now, with the same staleness framing.
    recorded_fp = Path(stage_dir) / "task_fingerprint.json"
    if recorded_fp.exists():
        try:
            recorded_sha = json.loads(recorded_fp.read_text(encoding="utf-8")).get("task_sha256")
        except (OSError, ValueError):
            recorded_sha = None
        if recorded_sha is not None and recorded_sha != fingerprint["task_sha256"]:
            raise GateResolutionError(
                f"{recorded_fp} records task {recorded_sha}, but the committed config derives "
                f"{fingerprint['task_sha256']}: the stage was trained on a different task than "
                "the current config describes. Resolve the drift before rolling a panel."
            )

    # Enforce — not merely assert — that the judge about to roll IS the
    # judge the frozen nulls were measured under: the safe set lives in
    # recovery_evaluation.py, OUTSIDE task identity, so a future
    # recalibration would not trip the resolver's task-staleness check, and
    # pairing a new-judge policy panel against old-judge nulls would certify
    # on incommensurable evidence. Mirrors the freeze-time check.
    for controller_id, entry in sorted(resolution["null_manifest"].items()):
        if entry.get("safe_set") != active_safe_set:
            raise GateResolutionError(
                f"frozen null {controller_id!r} was measured under a different safe set than the "
                "current calibrated judge; the panel cannot be paired against it. Re-freeze the "
                "gate resolution under the current judge (a new calibration invalidates old nulls)."
            )

    # Pair against the panel the nulls actually rolled: their seed set is
    # the pairing domain, so the policy panel must match its size exactly.
    # A rehearsal freeze (--episodes below the frozen minimum) is refused
    # here, before any episode is rolled, rather than by the gate afterwards.
    null_episode_counts = {int(entry["n_episodes"]) for entry in resolution["null_manifest"].values()}
    if len(null_episode_counts) != 1:
        raise GateResolutionError(
            f"frozen nulls disagree on panel size ({sorted(null_episode_counts)}); the resolution "
            "cannot define a single pairing domain — re-freeze it."
        )
    (panel_episodes,) = null_episode_counts
    if panel_episodes < int(spec["min_eval_episodes"]):
        raise GateResolutionError(
            f"the frozen nulls hold {panel_episodes} episodes but the frozen spec demands at least "
            f"{spec['min_eval_episodes']}: this is a rehearsal freeze and can never certify. "
            "Re-freeze at the full panel size before rolling the policy."
        )
    for controller_id in required_paired_nulls(resolution):
        if controller_id not in resolution["null_manifest"]:
            raise GateResolutionError(
                f"required paired null {controller_id!r} is missing; re-freeze with the stance brace"
            )

    return resolution, calibration


def validate_recovery_resolution(
    stage_dir: str | Path,
    *,
    species: str = "trex",
    stage: int | str = "recovery",
    policy_zip: str | Path | None = None,
    vecnorm: str | Path | None = None,
    algorithm: str = "ppo",
) -> dict[str, Any]:
    """Refuse stale or rehearsal resolutions before spending a training budget."""
    if bool(policy_zip) != bool(vecnorm):
        raise ValueError("policy_zip and vecnorm must be supplied together")
    resolution, calibration = _validated_recovery_resolution(stage_dir, species, stage)
    if calibration and policy_zip is not None and vecnorm is not None:
        provenance = resolution.get("null_provenance", {}).get("brace", {})
        expected = {
            "checkpoint_sha256": _file_sha256(policy_zip),
            "vecnormalize_sha256": _file_sha256(vecnorm),
            "algorithm": algorithm.lower(),
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise GateResolutionError(
                "frozen brace source differs from the stance checkpoint, VecNormalize or algorithm selected "
                "for the warm start; re-freeze before recovery training"
            )
    return resolution


def roll_policy_panel(
    stage_dir: "str | Path",
    policy_zip: "str | Path",
    vecnorm_pkl: "str | Path",
    *,
    species: str = "trex",
    stage: "int | str" = "recovery",
    inference: str = "auto",
    algorithm: str = "ppo",
):
    """Roll the trained policy over the panel the frozen gate judges against.

    The counterpart of the freeze: reads the stage directory's
    ``gate_resolution.json`` through :func:`require_gate_resolution` (so a
    missing, tampered, or stale-task resolution refuses here, before any
    episode is rolled), and rolls the policy with EXACTLY the frozen decision
    procedure — the recorded panel seed, episode count, recovery window, and
    the calibrated judge the null manifest was measured under.  The returned
    evidence's ``successes_by_seed()`` is what
    ``generate_stage_artifacts(recovery_successes_by_seed=...)`` needs to
    make the frozen gate produce a verdict (gap review EE2).
    """
    resolution, calibration = _validated_recovery_resolution(stage_dir, species, stage)
    spec = resolution["capability_spec"]
    procedure = resolution["decision_procedure"]
    panel_episodes = int(next(iter(resolution["null_manifest"].values()))["n_episodes"])
    active_safe_set = calibration.safe_set if calibration else dict(CALIBRATED_POSTURE_ONLY)

    env = build_env(species, stage)
    try:
        predict = policy_controller(
            policy_zip,
            vecnorm_pkl,
            action_space=env.action_space,
            inference=inference,
            algorithm=algorithm,
            expected_species=species if calibration else None,
            expected_stage=stage if calibration else None,
        )
        evidence = roll_recovery_panel(
            env,
            predict,
            controller_id="policy",
            episodes=panel_episodes,
            seed=int(procedure["panel_seed_start"]),
            t_recover_steps=int(spec["recovery_t_recover_steps"]),
            dwell_steps=int(spec["recovery_dwell_steps"]),
            safe_set=active_safe_set,
            height_reference=calibration.height_reference_m if calibration else CALIBRATED_HEIGHT_REFERENCE_M,
        )
        _require_judged_pushes(evidence)
        return evidence
    finally:
        env.close()


def freeze_recovery_gate(
    stage_dir: "str | Path",
    *,
    species: str = "trex",
    stage: "int | str" = "recovery",
    episodes: int = MIN_EVAL_EPISODES,
    seed: int = PANEL_SEED_START,
    policy_zip: "str | Path | None" = None,
    vecnorm: "str | Path | None" = None,
    inference: str = "auto",
    algorithm: str = "ppo",
    replace: bool = False,
) -> FrozenGateResolution:
    """Roll the nulls, freeze the resolution, and verify it can be re-loaded.

    The statue (``zero_action``) null is always rolled: it is the null the
    resolver's paired criterion defaults to.  The brace null — the policy's
    post-settle mean action held constant, first-runs record §3.1, which
    dies *faster* than the statue (§9.4) — needs a checkpoint, so it is
    rolled only when both ``policy_zip`` and ``vecnorm`` are supplied.  One
    without the other is a mistake rather than a half-configuration — a
    policy read with the wrong observation statistics is a different
    controller — so it is refused.

    An existing ``gate_resolution.json`` is never overwritten implicitly:
    re-freezing is a decision, so it requires ``replace=True``.
    """
    if bool(policy_zip) != bool(vecnorm):
        raise ValueError("policy_zip and vecnorm must be supplied together (the brace null needs both)")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("recovery episodes must be a positive integer")
    stage_dir = Path(stage_dir)
    existing = stage_dir / "gate_resolution.json"
    if existing.is_file() and not replace:
        raise ValueError(
            f"{existing} already exists: a frozen gate resolution is replaced deliberately, never as a "
            "side effect. Re-run with replace=True (--replace) if re-freezing is what you mean."
        )

    task_fingerprint = stage_task_fingerprint(species, stage)
    env = build_env(species, stage)
    null_provenance: dict[str, Any] | None = None
    try:
        calibration = _species_calibration(species, stage, env=env)
        if (
            calibration
            and episodes >= calibration.thresholds.min_eval_episodes
            and "brace" in calibration.profile["required_paired_nulls"]
            and policy_zip is None
        ):
            raise GateResolutionError(
                "a full recovery freeze requires the stance checkpoint and VecNormalize for the brace null"
            )
        null_evidence: dict[str, RecoveryPanelEvidence] = {
            "zero_action": _roll_null(
                env,
                zero_action_controller(env.action_space.shape[0]),
                controller_id="zero_action",
                episodes=episodes,
                seed=seed,
                calibration=calibration,
            )
        }
        if policy_zip is not None and vecnorm is not None:
            # Lazy import: the historical brace helper imports this module's
            # constants. It holds a checkpoint's post-settle mean action.
            from environments.shared.harnesses.recovery_offdist_panel import brace_controller

            predict = policy_controller(
                policy_zip,
                vecnorm,
                action_space=env.action_space,
                inference=inference,
                algorithm=algorithm,
                expected_species=species if calibration else None,
                expected_stage="stance" if calibration else None,
            )
            if calibration:
                brace_predict, brace_metadata = _quiet_brace_controller(species, predict, calibration=calibration)
                brace_metadata.update(
                    algorithm=algorithm.lower(),
                    checkpoint_sha256=_file_sha256(policy_zip),
                    vecnormalize_sha256=_file_sha256(vecnorm),
                )
                null_provenance = {"brace": brace_metadata}
            else:
                brace_predict = brace_controller(env, predict)
            null_evidence["brace"] = _roll_null(
                env,
                brace_predict,
                controller_id="brace",
                episodes=episodes,
                seed=seed,
                calibration=calibration,
            )
    finally:
        env.close()

    thresholds = (
        calibration.thresholds
        if calibration
        else RecoveryGateThresholds(
            min_recovery_success_lcb=MIN_RECOVERY_SUCCESS_LCB,
            t_recover_steps=T_RECOVER_STEPS,
            dwell_steps=DWELL_STEPS,
            min_eval_episodes=MIN_EVAL_EPISODES,
            min_paired_success_delta_lcb=MIN_PAIRED_SUCCESS_DELTA_LCB,
        )
    )
    resolution = build_gate_resolution(
        task_fingerprint=task_fingerprint,
        thresholds=thresholds,
        null_evidence=null_evidence,
        panel_seed_start=seed,
        evaluation_spec=calibration.evaluation_spec() if calibration else None,
        null_provenance=null_provenance,
    )
    path = write_gate_resolution(stage_dir, resolution)

    # Verify by ENFORCEMENT, not by inspection: the same call the gate makes,
    # at the same task, on the file as written.  A resolution that cannot be
    # loaded back is not frozen — it is a trap left on disk.
    loaded = require_gate_resolution(stage_dir, current_task_sha256=task_fingerprint["task_sha256"])
    if loaded != resolution:
        raise RuntimeError(f"{path} did not read back as written; the frozen resolution is not trustworthy")
    active_safe_set = calibration.safe_set if calibration else dict(CALIBRATED_POSTURE_ONLY)
    for controller_id, entry in loaded["null_manifest"].items():
        if entry["safe_set"] != active_safe_set:
            raise RuntimeError(
                f"null panel {controller_id!r} was frozen under {entry['safe_set']}, not the calibrated "
                f"posture-only safe set {active_safe_set}"
            )
    for evidence in null_evidence.values():
        write_recovery_evidence(stage_dir, evidence)
    return FrozenGateResolution(
        path=path,
        resolution=resolution,
        task_fingerprint=task_fingerprint,
        null_evidence=null_evidence,
    )


def summarize(result: FrozenGateResolution) -> str:
    """The freeze report: what was frozen, and — loudly — what was not."""
    resolution = result.resolution
    spec = resolution["capability_spec"]
    manifest = resolution["null_manifest"]
    judge = resolution.get("evaluation_spec", {})
    safe_set = judge.get("safe_set", dict(CALIBRATED_POSTURE_ONLY))
    height_reference = judge.get("height_reference_m", CALIBRATED_HEIGHT_REFERENCE_M)
    lines = [
        f"wrote {result.path}",
        f"  resolution_sha256   {resolution['resolution_sha256']}",
        f"  task_sha256         {resolution['task_sha256']}",
        "  capability spec:",
        f"    gate_kind                     {spec['gate_kind']}",
        f"    min_recovery_success_lcb      {spec['min_recovery_success_lcb']}",
        f"    min_paired_success_delta_lcb  {spec['min_paired_success_delta_lcb']}",
        f"    recovery_t_recover_steps      {spec['recovery_t_recover_steps']}",
        f"    recovery_dwell_steps          {spec['recovery_dwell_steps']}",
        f"    min_eval_episodes             {spec['min_eval_episodes']}",
        f"  panel_seed_start      {resolution['decision_procedure']['panel_seed_start']}",
        f"  safe set              {safe_set}",
        f"  height reference      {height_reference} m (fixed)",
        f"  calibration           {judge.get('calibration_id', 'historical T-Rex P5')}",
        "  null manifest:",
    ]
    for controller_id, entry in sorted(manifest.items()):
        lines.append(
            f"    {controller_id:<12} {entry['n_successes']}/{entry['n_episodes']} succeeded, "
            f"UCB95 {entry['success_ucb95']:.4f}"
        )

    warnings: list[str] = []
    if "brace" not in manifest:
        if "brace" in judge.get("required_paired_nulls", []):
            warnings.append(
                "STATUE NULL ONLY: the required learned-stance brace was not frozen. This rehearsal cannot "
                "qualify recovery; re-freeze with the stance checkpoint and matched VecNormalize before training."
            )
        else:
            warnings.append(
                "STATUE NULL ONLY: no --policy-zip/--vecnorm was given, so the brace null (§3.1/§9.4) is "
                "NOT in this record. The paired criterion defaults to null_controller_id='zero_action', "
                "which IS frozen here, so the gate can run — but nothing can ever be paired against the "
                "brace until the resolution is re-frozen with a checkpoint."
            )
    short = {c: e["n_episodes"] for c, e in manifest.items() if e["n_episodes"] < spec["min_eval_episodes"]}
    if short:
        warnings.append(
            f"REHEARSAL, NOT THE FROZEN RECORD: null panels {short} are shorter than the spec's "
            f"min_eval_episodes {spec['min_eval_episodes']}. Re-run with the full panel before any "
            "gate consumes this file."
        )
    for warning in warnings:
        lines.append("")
        lines.extend(f"  !! {line}" for line in _wrap(warning))
    return "\n".join(lines)


def _wrap(text: str, width: int = 96) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


def main(argv: "list[str] | None" = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Freeze the recovery stage's gate resolution (plan P5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stage-dir", required=True, help="directory to write gate_resolution.json into")
    parser.add_argument("--species", default="trex")
    parser.add_argument("--stage", default="recovery")
    parser.add_argument("--episodes", type=int, default=MIN_EVAL_EPISODES, help="null panel size (frozen: 40)")
    parser.add_argument("--seed", type=int, default=PANEL_SEED_START, help="first panel seed (frozen: 3042)")
    parser.add_argument("--policy-zip", help="SB3 checkpoint (.zip); with --vecnorm, adds the brace null")
    parser.add_argument("--vecnorm", help="matching VecNormalize stats (.pkl)")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument(
        "--inference",
        choices=("auto", "sb3", "numpy"),
        default="auto",
        help="forward pass for the checkpoint; 'numpy' skips PPO.load entirely (see §9)",
    )
    parser.add_argument("--replace", action="store_true", help="overwrite an existing frozen resolution")
    args = parser.parse_args(argv)

    result = freeze_recovery_gate(
        args.stage_dir,
        species=args.species,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        policy_zip=args.policy_zip,
        vecnorm=args.vecnorm,
        inference=args.inference,
        algorithm=args.algorithm,
        replace=args.replace,
    )
    print(summarize(result))


if __name__ == "__main__":
    main()
