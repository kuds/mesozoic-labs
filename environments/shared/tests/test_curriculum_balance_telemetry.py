"""The balance telemetry's additions change no training: the in-suite inertness proof.

A short real T. rex stance run on CPU (768 steps, one env, an evaluation of
two episodes every 256 steps, the horizon 40 steps past the settle window,
seed 11) is trained twice: with every addition of this change, and with
each switched off -- the capture callback's ``report_extensions`` (the
eval-episode ledger, the in-training v2 screen, the balance and force
fields, the mirror-even leg DC, the reward-term split and the force
matrix), the action-jerk reward key, the PostureTrend watch, the
per-rollout episode and fall counts, the training-episode ledger, the
policy probe and the checkpoint archive rule.  The two runs train and evaluate exactly alike (the same
``evaluations.npz``, the same tensors in every checkpoint, the same
VecNormalize statistics), the plain run writes the key sets main writes
for the same run without this change, pinned below as they were derived on
``2e734e29`` (main's commits since, through ``c6292b1e``, change no
training, evaluation or configuration code), and the full run adds exactly
this change's keys, its ledgers and its probe file.

Both runs are instrumented alike, outside the code under test: a hash of
every ``RolloutBuffer.add`` (observation, action, reward, episode start,
value, log-probability), and before every rollout (so after every PPO
update) and at the end of ``learn()`` a hash of the policy's parameters,
its optimizer state and the global generators (``numpy.random``,
``random``, torch).  The two sequences must be identical: the training-side
telemetry, the probe's state bank included, draws no random number and
changes no update.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
EPISODES = 2
STEPS = 768

#: The keys main writes for this run without this change (at ``2e734e29``, unchanged through ``c6292b1e``),
#: termination causes aside (which causes a run meets is the run's; both runs here must meet the same ones).
#: Derived by running this module's ``_train`` in an export of ``2e734e29`` (2026-10-17).
BASE_GATE_PROGRESS_KEYS = frozenset(
    """
    action_ac_rms action_ac_rms_per_actuator action_dc_per_actuator action_dc_rms action_delta action_freq_hz
    action_jerk bilateral_support_duty duty_episodes full_horizon_fraction l_digit_share
    l_foot_cop_dominant_hz l_foot_cop_fore_aft l_foot_cop_fore_aft_gate l_foot_cop_std_m l_heel_margin_nm
    l_sole_cop_fore_aft l_sole_cop_fore_aft_gate l_sole_corner_lift_m l_sole_tilt_deg l_toe_d2_load_pos
    l_toe_d2_load_share l_toe_d3_load_pos l_toe_d3_load_share l_toe_d4_load_pos l_toe_d4_load_share
    l_toe_margin_nm leg_dc_l_minus_r_ankle_deg leg_dc_l_minus_r_hip_pitch_deg leg_dc_l_minus_r_knee_deg
    max_foot_cop_fore_aft_mean max_sole_cop_fore_aft_mean max_sole_cop_fore_aft_over_bar_fraction
    max_sole_corner_lift_m_mean max_sole_corner_lift_m_over_bar_fraction max_sole_tilt_deg_mean
    max_sole_tilt_deg_over_bar_fraction mean_reward r_digit_share r_foot_cop_dominant_hz r_foot_cop_fore_aft
    r_foot_cop_fore_aft_gate r_foot_cop_std_m r_heel_margin_nm r_sole_cop_fore_aft r_sole_cop_fore_aft_gate
    r_sole_corner_lift_m r_sole_tilt_deg r_toe_d2_load_pos r_toe_d2_load_share r_toe_d3_load_pos
    r_toe_d3_load_share r_toe_d4_load_pos r_toe_d4_load_share r_toe_margin_nm term_action_jerk
    term_action_saturation term_airborne_substeps term_alive term_approach term_backward
    term_bilateral_support term_bite term_drift term_energy term_floor_impact term_foot_flatness
    term_foot_load_balance term_forward term_gait term_head_clearance term_head_proximity term_heading
    term_height term_idle term_lateral term_leg_home_pose term_neck_posture term_nosedive term_posture
    term_smoothness term_speed term_spin term_stance_width term_tail term_tail_home_pose term_total
    timesteps unsupported_duty unsupported_duty_ucb
    """.split()
)
BASE_DIAGNOSTICS_KEYS = frozenset(
    """
    abs_speed action_abs_mean action_delta action_jerk action_saturation action_std approach_delta
    backward_vel bilateral_support_duty bite_success contact_asymmetry distance_traveled drift_distance
    food_reached foot_load_asymmetry foot_load_balance foot_load_imbalance forward_vel forward_z
    head_food_distance head_pelvis_rel_z head_tip_z heading_alignment heading_alignment_std height_error
    jaw_distance l_foot_contact l_foot_contact_duty l_foot_load_share lateral_vel leg_home_pose_error
    neck_posture_error pelvis_angular_vel pelvis_height pelvis_yaw_vel prey_distance r_foot_contact
    r_foot_contact_duty r_foot_load_share reward_action_saturation reward_airborne_substeps reward_alive
    reward_approach reward_backward reward_bilateral_support reward_bite reward_claw_proximity reward_drift
    reward_energy reward_floor_impact reward_food reward_foot_flatness reward_foot_load_balance reward_forward
    reward_gait reward_gait_symmetry reward_head_clearance reward_head_proximity reward_heading reward_height
    reward_home_pose reward_idle reward_lateral reward_leg_home_pose reward_neck_posture reward_nosedive
    reward_posture reward_proximity reward_smoothness reward_speed reward_spin reward_stance_width
    reward_strike reward_support_geom_coverage reward_tail reward_tail_home_pose reward_target reward_total
    rl_foot_contact rr_foot_contact single_support_duty spin_instability strike_success tail_instability
    target_success term_timesteps tilt_angle timesteps torso_height unsupported_duty
    """.split()
)
#: This change's ``gate_progress.npz`` keys on this run, the reward-term split and the screen's per-criterion
#: counts aside (derived from the base's ``term_`` keys and the stage's declared criteria below).
ADDED_GATE_PROGRESS_KEYS = frozenset(
    """
    actuator_force_frac_max actuator_force_frac_per_actuator actuator_force_limited_fraction com_fore_aft_mean_m
    fall_xcom_toe_lead_s hof_cop_term_mean_mps2 hof_counter_rotation_mean_mps2 hof_counter_rotation_rms_mps2
    l_ankle_moment_arm_m l_cone_use_max l_cone_use_mean l_shear_over_normal leg_dc_l_r_mean_ankle_deg
    leg_dc_l_r_mean_hip_pitch_deg leg_dc_l_r_mean_knee_deg leg_q_l_minus_r_ankle_deg leg_q_l_minus_r_hip_pitch_deg
    leg_q_l_minus_r_knee_deg leg_q_l_r_mean_ankle_deg leg_q_l_r_mean_hip_pitch_deg leg_q_l_r_mean_knee_deg
    r_ankle_moment_arm_m r_cone_use_max r_cone_use_mean r_shear_over_normal root_pitch_mean_deg
    screen_v2_clean_count screen_v2_clean_lcb screen_v2_episodes screen_v2_hop_or_fall_episodes
    screen_v2_yaw0_bars_met shear_over_normal time_to_toe_min_s xcom_fore_aft_max_m xcom_heel_margin_min_m
    xcom_past_toe_fraction xcom_tip_margin_min_m xcom_toe_margin_mean_m xcom_toe_margin_min_m
    """.split()
)


def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        digest.update(np.ascontiguousarray(np.asarray(array)).tobytes())
    return digest.hexdigest()


def _state_digest(model) -> str:
    """The policy's parameters, its optimizer's state and the global generators, as one hash."""
    import torch

    tensors: list = []

    def walk(value) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, (int, float)):
            tensors.append(np.float64(value))

    walk(model.policy.state_dict())
    walk(model.policy.optimizer.state_dict())
    numpy_state: Any = np.random.get_state()  # the legacy tuple: (name, keys, pos, has_gauss, cached_gaussian)
    return _digest(
        *tensors,
        numpy_state[1],
        np.array(numpy_state[2:4], dtype=np.int64),
        np.frombuffer(repr(random.getstate()).encode(), dtype=np.uint8),
        torch.get_rng_state(),
    )


def _instrument(monkeypatch, record: dict) -> None:
    """Hash every rollout-buffer add, and the state before every rollout and at the end of ``learn()``."""
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    add, collect, learn = RolloutBuffer.add, OnPolicyAlgorithm.collect_rollouts, OnPolicyAlgorithm.learn

    def hashed_add(self, *args, **kwargs):
        record["rollout_add"].append(_digest(*args, *kwargs.values()))
        return add(self, *args, **kwargs)

    def hashed_collect(self, *args, **kwargs):
        record["updates"].append(_state_digest(self))
        return collect(self, *args, **kwargs)

    def hashed_learn(self, *args, **kwargs):
        result = learn(self, *args, **kwargs)
        record["learn_end"] = _state_digest(self)
        return result

    monkeypatch.setattr(RolloutBuffer, "add", hashed_add)
    monkeypatch.setattr(OnPolicyAlgorithm, "collect_rollouts", hashed_collect)
    monkeypatch.setattr(OnPolicyAlgorithm, "learn", hashed_learn)


def _train(root: Path, monkeypatch, *, additions: bool) -> Path:
    """Train the T. rex stance for 768 steps into ``root/run/01_stance``; ``additions=False`` switches every
    addition of this change off.  The instrumentation's record goes to ``root/run/instrumentation.json``."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    import torch

    from environments.shared import curriculum, policy_probe, rollout_ledger, train_base
    from environments.shared.config import load_all_stages
    from environments.shared.diagnostics import DiagnosticsCallback
    from environments.shared.eval_diagnostics import StageAwareEvalCallback
    from environments.shared.species_registry import get_species_config

    monkeypatch.setattr(train_base, "_report_hpt_metrics", lambda *a, **k: None)
    monkeypatch.setattr(train_base, "_eval_episodes_for_stage", lambda config: EPISODES)
    if not additions:
        monkeypatch.setattr(StageAwareEvalCallback, "report_extensions", False)
        monkeypatch.setattr(
            DiagnosticsCallback,
            "REWARD_KEYS",
            [key for key in DiagnosticsCallback.REWARD_KEYS if key != "reward_action_jerk"],
        )
        monkeypatch.setattr(curriculum, "build_posture_trend_watch_callback", lambda *a, **k: None)
        monkeypatch.setattr(DiagnosticsCallback, "record_rollout_counts", False)
        monkeypatch.setattr(rollout_ledger, "build_training_ledger_callback", lambda *a, **k: None)
        monkeypatch.setattr(policy_probe, "build_policy_probe_callback", lambda *a, **k: None)
        monkeypatch.setattr(curriculum, "checkpoint_archive_every", lambda curriculum_kwargs: 0)
    record: dict[str, Any] = {"rollout_add": [], "updates": [], "learn_end": None}
    _instrument(monkeypatch, record)
    configs = load_all_stages(TYRANNOSAURUS_REX)
    stance = configs[1]
    stance["env_kwargs"]["max_episode_steps"] = int(stance["curriculum_kwargs"]["settle_steps"]) + 40
    stance["ppo_kwargs"].update(n_steps=256, batch_size=64, n_epochs=2)
    run = root / "run"
    run.mkdir(parents=True)
    (run / "provenance.json").write_text(json.dumps({"run_id": "balance-telemetry-proof"}), encoding="utf-8")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_base.train(
            get_species_config(TYRANNOSAURUS_REX),
            configs,
            1,
            total_timesteps=STEPS,
            n_envs=1,
            seed=11,
            eval_freq=256,
            save_freq=512,
            verbose=0,
            use_tensorboard=False,
            output_dir=str(run / "01_stance"),
        )
    finally:
        torch.set_num_threads(threads)
    (run / "instrumentation.json").write_text(json.dumps(record), encoding="utf-8")
    return run / "01_stance"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The run with this change's additions and the run without them, trained once for the module."""
    runs = []
    for additions, name in ((True, "full"), (False, "plain")):
        monkeypatch = pytest.MonkeyPatch()
        try:
            runs.append(_train(tmp_path_factory.mktemp(name), monkeypatch, additions=additions))
        finally:
            monkeypatch.undo()
    return tuple(runs)


def _keys(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _tensors(path: Path) -> dict[str, np.ndarray]:
    import torch

    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as archive:
        for member in ("policy.pth", "policy.optimizer.pth"):
            state = torch.load(io.BytesIO(archive.read(member)), map_location="cpu", weights_only=True)
            flat: list[tuple[str, np.ndarray]] = []

            def walk(prefix: str, value) -> None:
                if isinstance(value, torch.Tensor):
                    flat.append((prefix, value.numpy()))
                elif isinstance(value, dict):
                    for key in sorted(value, key=str):
                        walk(f"{prefix}/{key}", value[key])
                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        walk(f"{prefix}/{index}", item)

            walk(member, state)
            out.update(flat)
    return out


def test_the_additions_change_no_evaluation_no_checkpoint_and_no_normalization(trained):
    full, plain = trained
    first, second = _keys(full / "evaluations.npz"), _keys(plain / "evaluations.npz")
    assert sorted(first) == sorted(second)
    for key in first:
        assert np.array_equal(first[key], second[key]), key
    zips = sorted(path.name for path in (full / "models").glob("*.zip"))
    assert zips == sorted(path.name for path in (plain / "models").glob("*.zip")) and len(zips) >= 4
    for name in zips:
        tensors, reference = _tensors(full / "models" / name), _tensors(plain / "models" / name)
        assert sorted(tensors) == sorted(reference), name
        for key in tensors:
            assert np.array_equal(tensors[key], reference[key]), (name, key)
    for path in sorted((full / "models").glob("*vecnorm*.pkl")):
        with open(path, "rb") as handle:
            mine = pickle.load(handle)
        with open(plain / "models" / path.name, "rb") as handle:
            theirs = pickle.load(handle)
        for stats in ("obs_rms", "ret_rms"):
            for field in ("mean", "var", "count"):
                assert np.array_equal(getattr(getattr(mine, stats), field), getattr(getattr(theirs, stats), field))
        assert (mine.clip_obs, mine.clip_reward, mine.gamma) == (theirs.clip_obs, theirs.clip_reward, theirs.gamma)


def test_the_training_side_telemetry_draws_no_random_number_and_changes_no_update(trained):
    """Every rollout step, every PPO update and the generators at the end of ``learn()`` are the same."""
    full, plain = (json.loads((stage.parent / "instrumentation.json").read_text(encoding="utf-8")) for stage in trained)
    assert len(full["rollout_add"]) == STEPS and len(full["updates"]) == STEPS // 256 and full["learn_end"]
    assert full["rollout_add"] == plain["rollout_add"]
    assert full["updates"] == plain["updates"]
    assert full["learn_end"] == plain["learn_end"]


def test_the_plain_run_writes_the_bases_artifacts(trained):
    _full, plain = trained
    assert not (plain / "eval_episodes.csv").exists()
    assert not (plain / "train_episodes.csv").exists()
    assert not (plain / "policy_probe.npz").exists()
    gate = _keys(plain / "gate_progress.npz")
    diagnostics = _keys(plain / "diagnostics.npz")
    assert {key for key in gate if not key.startswith("termination_")} == BASE_GATE_PROGRESS_KEYS
    assert {key for key in diagnostics if not key.startswith("term_") or key == "term_timesteps"} == (
        BASE_DIAGNOSTICS_KEYS
    )


def test_the_full_run_adds_exactly_this_changes_keys_and_keeps_every_value_of_the_plain_runs(trained):
    from environments.shared.config import load_stage_config
    from environments.shared.eval_diagnostics import _screen_thresholds, screen_scalar_keys

    full, plain = trained
    for name in ("gate_progress.npz", "diagnostics.npz"):
        mine, theirs = _keys(full / name), _keys(plain / name)
        assert set(theirs) <= set(mine), name
        for key, values in theirs.items():
            assert np.array_equal(mine[key], values, equal_nan=True), (name, key)
    gate, plain_gate = _keys(full / "gate_progress.npz"), _keys(plain / "gate_progress.npz")
    terms = [key[len("term_") :] for key in plain_gate if key.startswith("term_")]
    thresholds = _screen_thresholds(load_stage_config(TYRANNOSAURUS_REX, "stance")["curriculum_kwargs"])
    expected = {
        *ADDED_GATE_PROGRESS_KEYS,
        *(f"{kind}_term_{term}" for kind in ("full_horizon", "fall") for term in terms),
        *(key for key in screen_scalar_keys(thresholds) if "_fail_" in key),
    }
    assert set(gate) - set(plain_gate) == expected
    plain_diagnostics = _keys(plain / "diagnostics.npz")
    causes = [key[len("term_") :] for key in plain_diagnostics if key.startswith("term_") and key != "term_timesteps"]
    assert set(_keys(full / "diagnostics.npz")) - set(plain_diagnostics) == {
        "reward_action_jerk",
        "rollout_episodes",
        "rollout_falls",
        *(f"rollout_term_count_{cause}" for cause in causes),
    }


def test_the_full_run_writes_the_ledger_and_the_progress_blocks(trained):
    full, _plain = trained
    with open(full / "eval_episodes.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    evaluations = _keys(full / "evaluations.npz")
    assert len(rows) == evaluations["results"].size == 3 * EPISODES
    assert [int(row["timesteps"]) for row in rows] == [int(step) for step in evaluations["timesteps"] for _ in range(2)]
    assert [float(row["return"]) for row in rows] == pytest.approx(evaluations["results"].ravel().tolist())
    record = json.loads((full / "progress.json").read_text(encoding="utf-8"))
    assert {"balance", "forces"} <= set(record) and "leg_q_deg" in record["posture"]
    assert "screen_v2" in record["evaluation"]
    with open(full / "train_episodes.csv", newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    # One env: each episode ends at the step count of the one before it plus its own length.
    steps = [int(row["timesteps"]) for row in train_rows]
    assert train_rows and steps == list(np.cumsum([int(row["length"]) for row in train_rows])) and steps[-1] <= STEPS
    assert all(int(row["length"]) <= 240 for row in train_rows)
    diagnostics = _keys(full / "diagnostics.npz")
    assert int(diagnostics["rollout_episodes"].sum()) == len(train_rows)
    assert int(diagnostics["rollout_falls"].sum()) == sum(row["cause"] != "truncated" for row in train_rows)
    assert record["training_rollouts"]["episodes_total"] == len(train_rows)
    assert record["policy"]["update_timesteps"] == STEPS
    plain_record = json.loads((_plain / "progress.json").read_text(encoding="utf-8"))
    assert not {"policy", "training_rollouts"} & set(plain_record)
    with np.load(full / "policy_probe.npz", allow_pickle=False) as probe:
        # One row per PPO update, and the whole bank although the stage's horizon is 240 steps.
        assert probe["update_timesteps"].tolist() == [256, 512, 768]
        assert len(probe["bank_labels"]) == 10 and int(probe["bank_horizon"]) == 601
        assert probe["bank_action_now"].shape == (3, 10, 15)
