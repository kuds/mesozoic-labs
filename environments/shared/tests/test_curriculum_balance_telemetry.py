"""The balance telemetry's additions change no training: the in-suite inertness proof.

A short real T. rex stance run on CPU (768 steps, one env, an evaluation of
two episodes every 256 steps, the horizon 40 steps past the settle window,
seed 11) is trained twice: with every addition of this change, and with
each switched off -- the capture callback's ``report_extensions`` (the
eval-episode ledger with its head posture and hip roll columns, its body
columns (commands, tail and body posture, energy, drift and slip, joint
stops, centre of mass and centroidal momentum, base wrenches, head rate) and
the recorder's site and body-dynamics captures behind them, the in-training
v2 screen, the
balance and force fields, the mirror-even leg DC, the reward-term split and
the force matrix), the action-jerk reward key, the PostureTrend watch, the
per-rollout episode and fall counts, the training-episode ledger, the
policy probe (its forward pass of every rollout buffer's observations for
the rollout clip shares included), the checkpoint archive rule and the
in-run K3 noise panel (``k3_panel``: enabled at test scale in both runs'
launch, every 128 steps on spawns 3042 and 3043, so three of its panels
fall inside a rollout, and switched off in the plain run by its builder;
its panels equal an offline run on the saved checkpoint pairs, and its
final panel runs after the final pair is saved).  The
two runs train and evaluate exactly alike (the same ``evaluations.npz``,
the same tensors in every checkpoint, the same VecNormalize statistics),
and so do a second pair trained with mirror-symmetric PPO (``[ppo]
symmetry = "mirror_kl"``, the stance5 LTSB arm's: ``MirrorKLPPO``'s
``log_std`` tie hook and symmetric observation statistics beside the
panel's CPU copy of the policy),
the plain run writes the key sets main writes for the same run without
this change, pinned below as they were derived on
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
changes no update.  Nothing records a TensorBoard key for the panel.
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
#: The K3 panel at test scale: every 128 steps (128, 384 and 640 inside a rollout of 256 steps; 512 on the checkpoint
#: pair's step), two spawns per test, the stance5 replication sidecar's mean sigma.
K3_EVERY = 128
K3_SEEDS = (3042, 3043)
K3_SIGMA_FILE = (
    Path(__file__).resolve().parents[3] / "docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json"
)
K3_SIGMA_KEY = "80"

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
    """Hash every rollout-buffer add, and the state before every rollout and at the end of ``learn()``; keep every
    logger key recorded."""
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    from environments.shared.k3_panel import K3PanelCallback

    add, collect, learn = RolloutBuffer.add, OnPolicyAlgorithm.collect_rollouts, OnPolicyAlgorithm.learn
    log = Logger.record
    panel = K3PanelCallback._panel

    def watched_panel(self, kind, **kwargs):
        models = Path(self.stage_dir) / "models"
        saved = (models / "stage1_final.zip").is_file() and (models / "stage1_final_vecnorm.pkl").is_file()
        record["k3_panels"].append([kind, int(self.num_timesteps), saved, record["learning"]])
        return panel(self, kind, **kwargs)

    def recorded(self, key, *args, **kwargs):
        if key not in record["logger_keys"]:
            record["logger_keys"].append(key)
        return log(self, key, *args, **kwargs)

    def hashed_add(self, *args, **kwargs):
        record["rollout_add"].append(_digest(*args, *kwargs.values()))
        return add(self, *args, **kwargs)

    def hashed_collect(self, *args, **kwargs):
        record["updates"].append(_state_digest(self))
        return collect(self, *args, **kwargs)

    def hashed_learn(self, *args, **kwargs):
        record["learning"] = True
        try:
            result = learn(self, *args, **kwargs)
        finally:
            record["learning"] = False
        record["learn_end"] = _state_digest(self)
        return result

    monkeypatch.setattr(RolloutBuffer, "add", hashed_add)
    monkeypatch.setattr(OnPolicyAlgorithm, "collect_rollouts", hashed_collect)
    monkeypatch.setattr(OnPolicyAlgorithm, "learn", hashed_learn)
    monkeypatch.setattr(Logger, "record", recorded)
    monkeypatch.setattr(K3PanelCallback, "_panel", watched_panel)


def _train(root: Path, monkeypatch, *, additions: bool, symmetry: bool = False) -> Path:
    """Train the T. rex stance for 768 steps into ``root/run/01_stance``; ``additions=False`` switches every
    addition of this change off, ``symmetry=True`` trains mirror-symmetric PPO (``[ppo] symmetry = "mirror_kl"``).
    The instrumentation's record goes to ``root/run/instrumentation.json``."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    import torch

    from environments.shared import curriculum, k3_panel, policy_probe, rollout_ledger, train_base
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
        monkeypatch.setattr(k3_panel, "build_k3_panel_callback", lambda *a, **k: None)
    record: dict[str, Any] = {
        "rollout_add": [],
        "updates": [],
        "learn_end": None,
        "logger_keys": [],
        "k3_panels": [],
        "learning": False,
    }
    _instrument(monkeypatch, record)
    configs = load_all_stages(TYRANNOSAURUS_REX)
    stance = configs[1]
    stance["env_kwargs"]["max_episode_steps"] = int(stance["curriculum_kwargs"]["settle_steps"]) + 40
    stance["ppo_kwargs"].update(n_steps=256, batch_size=64, n_epochs=2)
    if symmetry:
        stance["ppo_kwargs"]["symmetry"] = "mirror_kl"
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
            # The same launch option in both runs; the plain run's builder above returns no callback.
            k3_panel=k3_panel.K3PanelConfig(
                sigma_file=str(K3_SIGMA_FILE), sigma_key=K3_SIGMA_KEY, every=K3_EVERY, seeds=K3_SEEDS
            ),
        )
    finally:
        torch.set_num_threads(threads)
    (run / "instrumentation.json").write_text(json.dumps(record), encoding="utf-8")
    return run / "01_stance"


def _train_pair(tmp_path_factory, *, symmetry: bool) -> tuple[Path, Path]:
    runs = []
    for additions, name in ((True, "full"), (False, "plain")):
        monkeypatch = pytest.MonkeyPatch()
        try:
            folder = tmp_path_factory.mktemp(f"{name}_symmetric" if symmetry else name)
            runs.append(_train(folder, monkeypatch, additions=additions, symmetry=symmetry))
        finally:
            monkeypatch.undo()
    return runs[0], runs[1]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The run with this change's additions and the run without them, trained once for the module."""
    return _train_pair(tmp_path_factory, symmetry=False)


@pytest.fixture(scope="module")
def trained_symmetric(tmp_path_factory):
    """The same pair trained with mirror-symmetric PPO (the stance5 LTSB arm's ``[ppo] symmetry``)."""
    return _train_pair(tmp_path_factory, symmetry=True)


#: Both pairs: the stock PPO and the mirror-symmetric one.
PAIRS = ("trained", "trained_symmetric")


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


@pytest.mark.parametrize("pair", PAIRS)
def test_the_additions_change_no_evaluation_no_checkpoint_and_no_normalization(pair, request):
    full, plain = request.getfixturevalue(pair)
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


@pytest.mark.parametrize("pair", PAIRS)
def test_the_training_side_telemetry_draws_no_random_number_and_changes_no_update(pair, request):
    """Every rollout step, every PPO update and the generators at the end of ``learn()`` are the same."""
    full, plain = (
        json.loads((stage.parent / "instrumentation.json").read_text(encoding="utf-8"))
        for stage in request.getfixturevalue(pair)
    )
    assert len(full["rollout_add"]) == STEPS and len(full["updates"]) == STEPS // 256 and full["learn_end"]
    assert full["rollout_add"] == plain["rollout_add"]
    assert full["updates"] == plain["updates"]
    assert full["learn_end"] == plain["learn_end"]
    # The panels the full run rolled: on the grid inside learn() (three of them mid-rollout), and the final one
    # after learn() returned, with the final checkpoint pair on disk; the plain run rolled none.
    grid = [[kind, step, saved, inside] for kind, step, saved, inside in full["k3_panels"] if kind == "grid"]
    assert grid == [["grid", step, False, True] for step in range(K3_EVERY, STEPS + 1, K3_EVERY)]
    assert full["k3_panels"][-1] == ["final", STEPS, True, False] and len(full["k3_panels"]) == len(grid) + 1
    assert plain["k3_panels"] == []


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
    # The saturating-actuator columns are among the additions the plain run trains and evaluates exactly like.
    assert {"saturated_actuator", "saturated_actuator_fraction", "actuators_saturated"} <= set(rows[0])
    # So are the head posture and hip roll columns, before the body columns (last; the next test), and the
    # recorder's head-tip capture they read; they are ledger-only (the gate_progress.npz key set above is unchanged)
    # and reach no progress.json block.
    posture = [
        "head_tip_height_mean_m",
        "head_tip_height_min_m",
        "neck_pitch_mean_deg",
        "head_pitch_mean_deg",
        "r_hip_roll_mean_deg",
        "r_hip_roll_max_abs_deg",
        "l_hip_roll_mean_deg",
        "l_hip_roll_max_abs_deg",
    ]
    start = list(rows[0]).index(posture[0])
    assert list(rows[0])[start : start + len(posture)] == posture
    assert list(rows[0])[start + len(posture)] == "neck_pitch_act_command_mean"  # the body columns' first
    assert all(float(row["head_tip_height_mean_m"]) > 0.0 for row in rows if row["full_horizon"] == "1")
    assert not any(column in (full / "progress.json").read_text(encoding="utf-8") for column in posture)
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
        # The rollout clip shares: each update's whole buffer (256 steps of one env), per actuator.
        assert probe["rollout_rows"].tolist() == [256, 256, 256]
        for key in ("rollout_action_clip_high_share", "rollout_mu_beyond_low_share", "rollout_mu_q95"):
            assert probe[key].shape == (3, 15) and np.isfinite(probe[key]).all(), key
        counts = probe["rollout_action_clip_low_share"].astype(np.float64) * 256
        assert np.array_equal(counts, np.round(counts))
    assert "rollout_mu_beyond_share_max" in record["policy"]


def test_the_full_run_writes_the_body_columns_last_and_ledger_only(trained):
    """The body columns follow the posture columns; per evaluation, their command means and command jitter pool to
    the evaluation's own action_dc_per_actuator and action_delta; nothing of them reaches gate_progress.npz or
    progress.json, and the plain run (above) writes no ledger at all."""
    full, _plain = trained
    with open(full / "eval_episodes.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)
    first, last = header.index("neck_pitch_act_command_mean"), header.index("head_yaw_rate_rms_dps")
    body = header[first : last + 1]
    assert last == len(header) - 1 and len(body) == 118 and header[first - 1] == "l_hip_roll_max_abs_deg"
    gate = _keys(full / "gate_progress.npz")
    assert not set(body) & set(gate)

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert not set(body) & keys(json.loads((full / "progress.json").read_text(encoding="utf-8")))
    settle = 200
    names = [column[: -len("_command_mean")] for column in body if column.endswith("_command_mean")]
    assert len(names) == 15
    for index, timesteps in enumerate(gate["timesteps"].tolist()):
        episodes = [row for row in rows if int(row["timesteps"]) == int(timesteps)]
        windows = np.array([max(int(row["length"]) - settle, 0) for row in episodes], dtype=np.float64)
        means = np.array([[float(row[f"{name}_command_mean"]) for name in names] for row in episodes])
        means = np.where(windows[:, None] > 0, means, 0.0)
        pooled = (windows[:, None] * means).sum(axis=0) / windows.sum()
        np.testing.assert_allclose(pooled, gate["action_dc_per_actuator"][index], rtol=0, atol=1e-12)
        deltas = np.array([float(row["action_delta_mean"]) if w else 0.0 for row, w in zip(episodes, windows)])
        assert float((windows * deltas).sum() / windows.sum()) == pytest.approx(
            float(gate["action_delta"][index]), rel=0, abs=1e-12
        )
    for row in rows:
        if row["full_horizon"] == "1":
            assert float(row["tail_base_force_down_mean_n"]) > 100.0 and float(row["legs_work_abs_j"]) > 0.0
            assert (
                float(row["root_pitch_min_deg"])
                <= float(row["root_pitch_mean_deg"])
                <= float(row["root_pitch_max_deg"])
            )


def _k3_stage_factory():
    """The stance env the proof trained (its horizon 40 steps past the settle window), as the panel builds it."""
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    stance = load_all_stages(TYRANNOSAURUS_REX)[1]
    stance["env_kwargs"]["max_episode_steps"] = int(stance["curriculum_kwargs"]["settle_steps"]) + 40
    env_class, kwargs = get_species_config(TYRANNOSAURUS_REX).env_class, dict(stance["env_kwargs"])
    return (lambda: env_class(**kwargs)), int(kwargs["max_episode_steps"])


def test_the_k3_panel_rolls_the_saved_checkpoint_pairs_and_reaches_no_other_record(trained):
    """The full run rolls a panel at every 128 steps and at training end; the panel at 512 and the final one are
    what an offline run of the same arithmetic gives on the checkpoint pair saved at that step (lengths and returns
    bit for bit, twice); the plain run writes no panel file; no TensorBoard key, progress.json entry or
    gate_progress.npz key of it exists in either run."""
    import torch

    from environments.shared import k3_panel as K
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.policy_loading import load_sb3_checkpoint

    full, plain = trained
    assert not list(plain.glob("k3_panel*"))
    assert sorted(path.name for path in full.glob("k3_panel*")) == [K.K3_PANEL_CSV_FILENAME, K.K3_PANEL_JSON_FILENAME]
    document = json.loads((full / K.K3_PANEL_JSON_FILENAME).read_text(encoding="utf-8"))
    assert document["schema"] == K.K3_PANEL_SCHEMA and document["session_problem"] is None
    panels = document["panels"]
    assert [(panel["kind"], panel["timesteps"]) for panel in panels] == [
        *(("grid", step) for step in range(K3_EVERY, STEPS + 1, K3_EVERY)),
        ("final", STEPS),
    ]
    sidecar = json.loads(K3_SIGMA_FILE.read_text(encoding="utf-8"))
    sigma = np.asarray(sidecar["control_sigma"][K3_SIGMA_KEY]["vector"], dtype=np.float64)
    assert document["spec"]["joint_sigma"]["path_in_file"] == f"control_sigma.{K3_SIGMA_KEY}"
    assert document["spec"]["joint_sigma"]["sha256"] == hashlib.sha256(sigma.astype("<f8").tobytes()).hexdigest()
    for panel in panels:
        assert panel["failure"] is None and panel["global_rng_moved"] is False
        assert panel["own_episodes"] == panel["joint_episodes"] == len(K3_SEEDS)
    with open(full / K.K3_PANEL_CSV_FILENAME, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(K3_SEEDS) * len(panels) and list(rows[0]) == list(K.CSV_COLUMNS)

    factory, horizon = _k3_stage_factory()
    for panel, (policy_zip, sidecar_pkl) in (
        (panels[512 // K3_EVERY - 1], ("stage1_512_steps.zip", "stage1_vecnormalize_512_steps.pkl")),
        (panels[-1], ("stage1_final.zip", "stage1_final_vecnorm.pkl")),
    ):
        model, normalizer, _ = load_sb3_checkpoint(
            str(full / "models" / policy_zip),
            str(full / "models" / sidecar_pkl),
            factory,
            guess_sidecar=False,
            plant_identity=current_plant_identity(TYRANNOSAURUS_REX),
        )
        # The weights and statistics the panel copied are the pair's.
        assert K.state_dict_sha256(model.policy.state_dict()) == panel["policy_state_sha256"], policy_zip
        statistics = K.copy_normalization(normalizer)
        assert statistics is not None
        assert hashlib.sha256(statistics.mean.astype("<f8").tobytes()).hexdigest() == panel["obs_mean_sha256"]
        assert hashlib.sha256(statistics.var.astype("<f8").tobytes()).hexdigest() == panel["obs_var_sha256"]
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            offline = [
                K.run_k3_panel(model.policy, statistics, factory, joint_sigma=sigma, seeds=K3_SEEDS, horizon=horizon)
                for _ in range(2)
            ]
        finally:
            torch.set_num_threads(threads)
        assert offline[0] == offline[1]
        recorded = [row for row in rows if int(row["panel"]) == panel["panel"]]
        assert [
            (row["test"], int(row["seed"]), int(row["length"]), float(row["return"]), row["termination_reason"])
            for row in recorded
        ] == [(row["test"], row["seed"], row["length"], row["return"], row["termination_reason"]) for row in offline[0]]

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    for stage in (full, plain):
        instrumentation = json.loads((stage.parent / "instrumentation.json").read_text(encoding="utf-8"))
        assert instrumentation["logger_keys"] and not any("k3" in key for key in instrumentation["logger_keys"])
        assert not any("k3" in key for key in keys(json.loads((stage / "progress.json").read_text(encoding="utf-8"))))
        assert not any("k3" in key for key in _keys(stage / "gate_progress.npz"))


def test_the_k3_panel_runs_beside_mirror_symmetric_ppo(trained_symmetric):
    """The symmetric pair trained MirrorKLPPO (recorded in the stage config and in every checkpoint), and the full
    run's panels ran in full beside its ``log_std`` tie hook and symmetric statistics, moving no global generator."""
    from environments.shared import k3_panel as K
    from environments.shared.symmetry import CHECKPOINT_ATTRIBUTE
    from environments.shared.task_fingerprint import read_checkpoint_attribute

    full, plain = trained_symmetric
    for stage in (full, plain):
        assert '"mirror_kl"' in (stage / "stage_config.json").read_text(encoding="utf-8")
        assert read_checkpoint_attribute(stage / "models" / "stage1_final.zip", CHECKPOINT_ATTRIBUTE)
    assert not list(plain.glob("k3_panel*"))
    panels = json.loads((full / K.K3_PANEL_JSON_FILENAME).read_text(encoding="utf-8"))["panels"]
    assert [panel["kind"] for panel in panels] == ["grid"] * (STEPS // K3_EVERY) + ["final"]
    for panel in panels:
        assert panel["failure"] is None and panel["global_rng_moved"] is False
        assert panel["own_episodes"] == panel["joint_episodes"] == len(K3_SEEDS)


@pytest.mark.parametrize("pair", PAIRS)
def test_the_reports_stub_unpickler_reads_the_runs_own_pickles_as_sb3_does(pair, request):
    """The training-history report's normalization page reads the run's VecNormalize pickles without SB3's classes
    (``training_history.read_vecnormalize_stats``): the statistics equal ``pickle.load``'s, and at 512 (a rollout
    end) the pickle cast to float32 equals the policy probe's row of that update."""
    from environments.shared.reporting import training_history as th

    full, _plain = request.getfixturevalue(pair)
    for path in sorted((full / "models").glob("stage1_*vec*.pkl")):
        stats = th.read_vecnormalize_stats(path)
        with open(path, "rb") as handle:
            reference = pickle.load(handle)
        assert np.array_equal(stats["mean"], reference.obs_rms.mean) and np.array_equal(
            stats["var"], reference.obs_rms.var
        )
        assert (stats["epsilon"], stats["clip_obs"]) == (reference.epsilon, reference.clip_obs)
    run = th.load_run_history(full, read_std=False)
    probe = run.obs_norm["probe"]
    assert [check["timesteps"] for check in probe["pickle_checks"]] == [512, STEPS]
    assert probe["pickle_checks_equal"] == 2 and run.obs_norm["errors"] == []
