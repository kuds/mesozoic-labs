"""Fresh gait evidence, exact checkpoint bindings, and CLI error boundaries.

Panels are intentionally short: these tests validate the producer and its
refusals, not locomotion quality or biologically calibrated acceptance limits.
A real tiny PPO/VecNormalize pair exercises saved-policy inference.
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND, provisional_gait_criteria
from environments.shared.gait import report as producer
from environments.shared.gait.identity import measurement_protocol, protocol_sha256
from environments.shared.gait.types import GaitProtocol
from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
from environments.shared.policy_loading import PolicyLoadError
from environments.shared.result_bundle.errors import ResultBundleError
from environments.shared.result_bundle.hashing import sha256_file
from environments.shared.scripts import gait_report as cli
from environments.shared.task_fingerprint import TaskFingerprintError, attach_task_fingerprint, stage_task_fingerprint
from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def species_config():
    return SimpleNamespace(species="trex", env_class=TRexEnv)


@pytest.fixture
def stage_config():
    return {
        "_gait_stage": "locomotion",
        "env_kwargs": {"max_episode_steps": 3, "reset_noise_scale": 0.01},
        "curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_forward_vel": 0.1},
    }


@pytest.fixture
def fresh_plant_identity_cache():
    """Preflight must validate real code, independent of earlier test doubles."""
    from environments.shared.plant_contract import clear_plant_identity_cache

    clear_plant_identity_cache()
    try:
        yield
    finally:
        clear_plant_identity_cache()


def _prior_certificate(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "gait_report.json"
    path.write_text(json.dumps({"status": "complete", "certified": True, "certification_eligible": True}))
    return path


def _assert_incomplete(output, *, error=None):
    report = json.loads((output / "gait_report.json").read_text())
    assert report["status"] == "incomplete"
    assert report["certification_eligible"] is False
    assert not report.get("certified", False)
    if error is not None:
        assert error in report["error"]
    return report


def _strict_config(config, **updates):
    value = copy.deepcopy(config)
    env = TRexEnv(**value["env_kwargs"])
    try:
        protocol = measurement_protocol(
            "trex",
            GaitProtocol(),
            settle_s=0.0,
            direction_xy=(1.0, 0.0),
            horizon=value["env_kwargs"]["max_episode_steps"],
            physics_dt_s=float(env.model.opt.timestep),
            control_dt_s=float(env.dt),
            episodes=2,
            seed_start=PUBLICATION_SEED_START,
        )
    finally:
        env.close()
    value["curriculum_kwargs"] = {
        **provisional_gait_criteria("biped_walk"),
        "gate_kind": GAIT_GATE_KIND,
        "gate_schema_version": 1,
        "gait_profile": "biped_walk",
        "measurement_protocol_sha256": protocol_sha256(protocol),
        "min_eval_episodes": 2,
        "gait_panel_seed_start": PUBLICATION_SEED_START,
        "min_gait_success_lcb": 0.8,
        "min_episode_forward_vel": 0.1,
        "min_episode_duration_s": 0.03,
        **updates,
    }
    return value


def test_real_zero_action_panel_writes_replayable_traces_and_matching_csv(tmp_path, species_config, stage_config):
    original = copy.deepcopy(stage_config)
    report = producer.write_gait_report(
        species_config, stage_config, None, None, tmp_path, episodes=2, seed=17, settle_s=0.0
    )
    assert stage_config == original
    assert report["status"] == "complete"
    assert report["controller"] == "zero_action_reference"
    assert report["report_only"] is True
    assert report["certified"] is report["certification_eligible"] is False
    assert report["task_validated"] is False
    assert report["threshold_status"] == "provisional development criteria"
    assert report["statistics"]["n_episodes"] == 2
    assert report["statistics"]["success_count"] == 0
    assert report["checkpoint_sha256"] is report["normalization_sha256"] is None
    assert [row["seed"] for row in report["episodes"]] == [17, 18]
    assert report["panel"]["unique_reset_states"] == 2
    # The report records the seeds it actually rolled, in its role.
    assert report["panel"]["seed_role"] == "development"
    assert report["seed_start"] == report["thresholds"]["gait_panel_seed_start"] == 17
    # Runtime versions are informational, outside the hashed protocol.
    assert set(report["runtime"]) == {"mujoco", "numpy", "python"}
    assert "mujoco_version" not in json.dumps(report["measurement_protocol"])
    assert report["measurement_protocol_sha256"] == protocol_sha256(report["measurement_protocol"])
    assert json.loads((tmp_path / "gait_report.json").read_text()) == report
    assert sha256_file(tmp_path / "gait_panel.csv") == report["panel_csv"]["sha256"]
    with (tmp_path / "gait_panel.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for index, (row, episode, trace_binding) in enumerate(zip(rows, report["episodes"], report["traces"], strict=True)):
        assert json.loads(row["metrics_json"]) == episode
        assert row["measurement_protocol_sha256"] == report["measurement_protocol_sha256"]
        assert episode["telemetry_valid"] is True
        assert episode["length"] == 3
        assert all(foot["complete_cycles"] == 0 for foot in episode["per_foot"].values())
        path = tmp_path / trace_binding["path"]
        assert sha256_file(path) == trace_binding["sha256"]
        with zipfile.ZipFile(path) as archive:
            assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_DEFLATED}
        with np.load(path, allow_pickle=False) as trace:
            assert bool(trace["physics_diverged"]) is False
            assert episode["physics_diverged"] is False
            assert episode["reset_state_sha256"] == producer.reset_state_digest(
                trace["reset_qpos"], trace["reset_qvel"], trace["reset_mocap_pos"]
            )
            samples = len(trace["time_s"])
            assert trace["floor_force_n"].shape == (samples, 2)
            assert trace["foot_position_m"].shape == (samples, 2, 3)
            assert trace["time_s"][0] == 0.0
            assert np.all(np.diff(trace["time_s"]) > 0)
            assert np.isfinite(trace["foot_clearance_m"]).all()
            replay = TRexEnv(**stage_config["env_kwargs"])
            try:
                assert samples == 1 + 3 * replay.frame_skip
                np.testing.assert_allclose(np.diff(trace["time_s"]), replay.model.opt.timestep, rtol=1e-13)
                replay.reset(seed=17 + index)
                initial_position = replay.data.qpos[:3].copy()
                for _ in range(3):
                    replay.step(np.zeros(replay.action_space.shape))
                np.testing.assert_array_equal(trace["root_position_m"][0], initial_position)
                np.testing.assert_array_equal(trace["root_position_m"][-1], replay.data.qpos[:3])
            finally:
                replay.close()


@pytest.mark.parametrize(
    "options,match",
    [
        ({"episodes": 0}, "episodes"),
        ({"episodes": True}, "episodes"),
        ({"episodes": 1.5}, "episodes"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"settle_s": -0.1}, "settle_s"),
        ({"settle_s": float("nan")}, "settle_s"),
        ({"direction_xy": (0.0, 0.0)}, "direction"),
        ({"direction_xy": (1.0,)}, "direction"),
        ({"direction_xy": (float("inf"), 0.0)}, "direction"),
        ({"settle_s": "1"}, "settle_s"),
        # A report-only panel never rolls the reserved certification block 3042-3081.
        ({"seed": PUBLICATION_SEED_START}, "certification block"),
        ({"seed": PUBLICATION_SEED_START - 5, "episodes": 6}, "certification block"),
        ({"seed": PUBLICATION_SEED_START + 39}, "certification block"),
    ],
)
def test_invalid_arguments_are_refused_before_touching_a_prior_report(
    tmp_path, species_config, stage_config, options, match
):
    prior = _prior_certificate(tmp_path)
    before = prior.read_bytes()
    kwargs = {"episodes": 1, "seed": 0, "settle_s": 0.0, **options}
    with pytest.raises(ValueError, match=match):
        producer.write_gait_report(species_config, stage_config, None, None, tmp_path, **kwargs)
    # A typo invalidates nothing: the earlier report stays as it was.
    assert prior.read_bytes() == before
    assert not (tmp_path / "gait_panel.csv").exists()


def test_report_only_panel_beside_the_certification_block_is_allowed(tmp_path, species_config, stage_config):
    report = producer.write_gait_report(
        species_config, stage_config, None, None, tmp_path, episodes=1, seed=PUBLICATION_SEED_START + 40, settle_s=0.0
    )
    assert report["seed_start"] == PUBLICATION_SEED_START + 40


def test_default_report_only_panel_is_short_and_on_the_development_block(tmp_path, species_config, stage_config):
    from environments.shared.constants import DEVELOPMENT_GAIT_SEED_START

    report = producer.write_gait_report(species_config, stage_config, None, None, tmp_path, settle_s=0.0)
    assert report["panel"]["episodes"] == producer.DEFAULT_DEVELOPMENT_EPISODES == 10
    assert report["seed_start"] == DEVELOPMENT_GAIT_SEED_START
    assert [row["seed"] for row in report["episodes"]] == list(range(9000, 9010))


def test_rewrite_clears_every_earlier_panel_file(tmp_path, species_config, stage_config):
    """A stale trace from an earlier, larger panel never survives into a new report."""
    producer.write_gait_report(species_config, stage_config, None, None, tmp_path, episodes=3, seed=9000, settle_s=0.0)
    (tmp_path / "gait_traces" / "episode_0007.npz").write_bytes(b"stray")
    report = producer.write_gait_report(
        species_config, stage_config, None, None, tmp_path, episodes=2, seed=9100, settle_s=0.0
    )
    assert sorted(path.name for path in (tmp_path / "gait_traces").iterdir()) == [
        "episode_0000.npz",
        "episode_0001.npz",
    ]
    assert [entry["path"] for entry in report["traces"]] == [
        "gait_traces/episode_0000.npz",
        "gait_traces/episode_0001.npz",
    ]
    with (tmp_path / "gait_panel.csv").open(newline="") as handle:
        assert [row["seed"] for row in csv.DictReader(handle)] == ["9100", "9101"]


def test_failed_rewrite_removes_earlier_traces_with_the_report(tmp_path, species_config, stage_config):
    producer.write_gait_report(species_config, stage_config, None, None, tmp_path, episodes=2, seed=9000, settle_s=0.0)
    with pytest.raises(ValueError, match="positive analysis window"):
        producer.write_gait_report(
            species_config, stage_config, None, None, tmp_path, episodes=2, seed=9000, settle_s=5.0
        )
    _assert_incomplete(tmp_path, error="positive analysis window")
    assert not (tmp_path / "gait_traces").exists() and not (tmp_path / "gait_panel.csv").exists()


def test_settling_that_removes_entire_panel_window_is_refused(tmp_path, species_config, stage_config):
    _prior_certificate(tmp_path)
    with pytest.raises(ValueError, match="positive analysis window"):
        producer.write_gait_report(species_config, stage_config, None, None, tmp_path, episodes=1, seed=0, settle_s=1.0)
    _assert_incomplete(tmp_path, error="positive analysis window")


@pytest.mark.parametrize("location", ["output", "parent", "ancestor", "symlink_ancestor"])
def test_complete_result_bundle_is_refused_before_any_write(tmp_path, species_config, stage_config, location):
    output = tmp_path / "panel"
    guarded = output if location == "output" else tmp_path
    if location in ("ancestor", "symlink_ancestor"):
        guarded = tmp_path / "completed_run"
        output = guarded / "diagnostics" / "panel"
    guarded.mkdir(parents=True, exist_ok=True)
    marker = guarded / "artifact_manifest.json"
    marker.write_text('{"status":"complete"}')
    if location == "symlink_ancestor":
        (guarded / "diagnostics").mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(guarded / "diagnostics", target_is_directory=True)
        output = alias / "panel"
    if location == "output":
        old_report = _prior_certificate(output)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ResultBundleError, match="immutable"):
        producer.write_gait_report(species_config, stage_config, None, None, output, episodes=1, seed=0, settle_s=0.0)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    if location != "output":
        assert not output.exists()
    else:
        assert json.loads(old_report.read_text())["certified"] is True


@pytest.fixture(scope="module")
def saved_pair_source(tmp_path_factory):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    directory = tmp_path_factory.mktemp("gait-saved-policy")
    env_kwargs = {"max_episode_steps": 3, "reset_noise_scale": 0.01}
    plant = current_plant_identity("trex")
    task = stage_task_fingerprint("trex", "locomotion", env_kwargs=env_kwargs, plant_identity=plant)
    normalizer = VecNormalize(DummyVecEnv([lambda: TRexEnv(**env_kwargs)]), norm_obs=True, norm_reward=True)
    try:
        # Distinct, nonidentity statistics make forgotten normalization
        # observable. No optimization or convergence is claimed by this test.
        normalizer.obs_rms.mean[:] = np.linspace(-0.7, 0.7, normalizer.obs_rms.mean.size)
        normalizer.obs_rms.var[:] = np.linspace(0.5, 2.0, normalizer.obs_rms.var.size)
        normalizer.obs_rms.count = 123.0
        model = sb3.PPO(
            "MlpPolicy",
            normalizer,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            seed=77,
            device="cpu",
        )
        attach_plant_identity(model, plant)
        attach_plant_identity(normalizer, plant)
        attach_task_fingerprint(model, task)
        model.save(directory / "policy.zip")
        normalizer.save(directory / "policy_vecnorm.pkl")
        (directory / "stage_config.json").write_text(
            json.dumps({"run": {"seed": 77, "n_envs": 1}, "env_kwargs": env_kwargs, "task_fingerprint": task})
        )
    finally:
        normalizer.close()
    return directory


@pytest.fixture
def saved_pair(tmp_path, saved_pair_source):
    model, norm = tmp_path / "policy.zip", tmp_path / "policy_vecnorm.pkl"
    shutil.copyfile(saved_pair_source / model.name, model)
    shutil.copyfile(saved_pair_source / norm.name, norm)
    shutil.copyfile(saved_pair_source / "stage_config.json", tmp_path / "stage_config.json")
    return model, norm


def test_real_saved_pair_applies_frozen_normalization_then_deterministic_prediction(
    tmp_path, species_config, stage_config, saved_pair, monkeypatch
):
    original_loader = producer.load_sb3_checkpoint
    calls, loaded = [], {}

    def inspected_loader(*args, **kwargs):
        assert kwargs["guess_sidecar"] is False
        assert kwargs["allow_unnormalized"] is False
        model, normalizer, path = original_loader(*args, **kwargs)
        assert normalizer.training is False
        assert normalizer.norm_reward is False
        loaded.update(normalizer=normalizer, before=copy.deepcopy(normalizer.obs_rms.__dict__))
        normalize, predict = normalizer.normalize_obs, model.predict

        def normalized(obs):
            actual = normalize(obs)
            expected = np.clip(
                (obs - normalizer.obs_rms.mean) / np.sqrt(normalizer.obs_rms.var + normalizer.epsilon),
                -normalizer.clip_obs,
                normalizer.clip_obs,
            ).astype(np.float32)
            np.testing.assert_array_equal(actual, expected)
            calls.append((obs.copy(), actual.copy()))
            return actual

        def predicted(obs, *, deterministic):
            assert deterministic is True
            np.testing.assert_array_equal(obs, calls[-1][1])
            return predict(obs, deterministic=deterministic)

        monkeypatch.setattr(normalizer, "normalize_obs", normalized)
        monkeypatch.setattr(model, "predict", predicted)
        return model, normalizer, path

    monkeypatch.setattr(producer, "load_sb3_checkpoint", inspected_loader)
    model, norm = saved_pair
    before = sha256_file(model), sha256_file(norm)
    report = producer.write_gait_report(
        species_config, stage_config, model.with_suffix(""), norm, tmp_path / "panel", episodes=2, seed=19, settle_s=0.0
    )
    assert len(calls) == 6
    assert any(not np.array_equal(raw, normalized) for raw, normalized in calls)
    assert report["controller"] == "selected_checkpoint"
    assert report["task_validated"] is True
    assert report["certification_eligible"] is True
    assert report["report_only"] is True and report["certified"] is False
    assert (report["checkpoint_sha256"], report["normalization_sha256"]) == before
    assert (sha256_file(model), sha256_file(norm)) == before
    for name in ("mean", "var", "count"):
        np.testing.assert_array_equal(getattr(loaded["normalizer"].obs_rms, name), loaded["before"][name])


def test_explicit_normalization_is_required_even_when_matching_sidecar_exists(
    tmp_path, species_config, stage_config, saved_pair
):
    model, _ = saved_pair
    output = tmp_path / "panel"
    _prior_certificate(output)
    with pytest.raises(ValueError, match="explicit matched VecNormalize"):
        producer.write_gait_report(species_config, stage_config, model, None, output, episodes=1, settle_s=0.0)
    _assert_incomplete(output, error="explicit matched VecNormalize")


def test_corrupt_normalization_is_fatal_and_clears_prior_result(tmp_path, species_config, stage_config, saved_pair):
    model, norm = saved_pair
    norm.write_bytes(b"broken saved statistics")
    output = tmp_path / "panel"
    _prior_certificate(output)
    with pytest.raises(PolicyLoadError, match="cannot read VecNormalize"):
        producer.write_gait_report(species_config, stage_config, model, norm, output, episodes=1, settle_s=0.0)
    _assert_incomplete(output, error="cannot read VecNormalize")


def test_checkpoint_task_mismatch_is_refused_before_model_inference(
    tmp_path, species_config, stage_config, saved_pair, monkeypatch
):
    config = copy.deepcopy(stage_config)
    config["env_kwargs"]["alive_bonus"] = 123.0
    output = tmp_path / "panel"
    _prior_certificate(output)

    def forbidden_loader(*args, **kwargs):
        pytest.fail("task-mismatched checkpoint reached the inference loader")

    monkeypatch.setattr(producer, "load_sb3_checkpoint", forbidden_loader)
    with pytest.raises(TaskFingerprintError, match="task"):
        producer.write_gait_report(species_config, config, *saved_pair, output, episodes=1, settle_s=0.0)
    _assert_incomplete(output, error="TaskFingerprintError")


@pytest.mark.parametrize("replacement", ["checkpoint", "normalization"])
def test_replaced_policy_files_cannot_be_attributed_to_completed_evaluation(
    tmp_path, species_config, stage_config, saved_pair, monkeypatch, replacement
):
    roll = producer.roll_gait_panel

    def replace_after_panel(*args, **kwargs):
        result = roll(*args, **kwargs)
        saved_pair[0 if replacement == "checkpoint" else 1].write_bytes(b"different bytes after rollout")
        return result

    monkeypatch.setattr(producer, "roll_gait_panel", replace_after_panel)
    output = tmp_path / "panel"
    _prior_certificate(output)
    with pytest.raises(ValueError, match="changed during gait evaluation"):
        producer.write_gait_report(species_config, stage_config, *saved_pair, output, episodes=1, settle_s=0.0)
    _assert_incomplete(output, error="changed during gait evaluation")
    assert not (output / "gait_panel.csv").exists()


@pytest.mark.parametrize("episodes,seed", [(1, PUBLICATION_SEED_START), (2, PUBLICATION_SEED_START + 1)])
def test_strict_panel_requires_declared_episode_count_and_seed_start(
    tmp_path, species_config, stage_config, episodes, seed
):
    config = _strict_config(stage_config)
    prior = _prior_certificate(tmp_path)
    before = prior.read_bytes()
    with pytest.raises(ValueError, match="fixed episode count and seed start"):
        producer.write_gait_report(
            species_config, config, None, None, tmp_path, episodes=episodes, seed=seed, settle_s=0.0
        )
    assert prior.read_bytes() == before


def test_zero_action_reference_cannot_enter_certification(tmp_path, species_config, stage_config):
    config = _strict_config(stage_config)
    _prior_certificate(tmp_path)
    with pytest.raises(ValueError, match="zero-action reference cannot be certified"):
        producer.write_gait_report(
            species_config, config, None, None, tmp_path, episodes=2, seed=PUBLICATION_SEED_START, settle_s=0.0
        )
    _assert_incomplete(tmp_path, error="zero-action reference cannot be certified")


def test_strict_saved_pair_refuses_unpinned_measurement_protocol(tmp_path, species_config, stage_config, saved_pair):
    config = _strict_config(stage_config, measurement_protocol_sha256="sha256:" + "0" * 64)
    output = tmp_path / "panel"
    with pytest.raises(ValueError, match="declared gait protocol hash differs"):
        producer.write_gait_report(
            species_config, config, *saved_pair, output, episodes=2, seed=PUBLICATION_SEED_START, settle_s=0.0
        )
    _assert_incomplete(output, error="declared gait protocol hash differs")


def test_strict_saved_pair_with_current_protocol_completes_evidence_but_fails_gait(
    tmp_path, species_config, stage_config, saved_pair
):
    config = _strict_config(stage_config)
    report = producer.write_gait_report(
        species_config, config, *saved_pair, tmp_path / "panel", episodes=2, seed=PUBLICATION_SEED_START, settle_s=0.0
    )
    assert report["status"] == "complete"
    assert report["report_only"] is False
    assert report["certification_eligible"] is True
    assert report["certified"] is False
    assert report["statistics"]["success_count"] == 0
    assert report["threshold_status"] == "explicit declared gate criteria"
    assert report["episode_failure_counts"]
    # Failures aggregate on stable rail ids, never on per-episode value text.
    assert all("/" in rail and ":" not in rail for rail in report["episode_failure_counts"])
    assert len(report["episode_labels"]) == 2


def _cli_summary():
    return {
        "report_only": True,
        "certified": False,
        "statistics": {"n_episodes": 2, "success_count": 0},
        "measurement_protocol_sha256": "sha256:" + "a" * 64,
        "episode_failure_counts": {"participation/complete_cycles_min": 2},
    }


def test_cli_loads_raw_task_json_and_protocol_options_and_forwards_explicit_arguments(tmp_path, monkeypatch, capsys):
    env_json, protocol_json = tmp_path / "task.json", tmp_path / "protocol.json"
    task = {"max_episode_steps": 3, "reset_noise_scale": 0.01, "alive_bonus": 2.0}
    env_json.write_text(json.dumps(task))
    protocol_json.write_text(json.dumps({"contact_force_bw_per_limb": 0.03, "chatter_fill_s": 0.012}))
    recorded = {}

    def writer(species, config, model, norm, output, **options):
        recorded.update(species=species.species, config=config, model=model, norm=norm, output=output, **options)
        return _cli_summary()

    monkeypatch.setattr(cli, "write_gait_report", writer)
    exit_code = cli.main(
        [
            "trex",
            "--stage",
            "locomotion",
            "--model",
            "selected.zip",
            "--vecnorm",
            "matched.pkl",
            "--out-dir",
            str(tmp_path / "panel"),
            "--env-json",
            str(env_json),
            "--protocol-json",
            str(protocol_json),
            "--episodes",
            "2",
            "--seed",
            "17",
            "--settle-s",
            "0.25",
            "--direction",
            "3",
            "4",
            "--allow-legacy-plant",
        ]
    )
    assert exit_code == 0
    assert recorded["species"] == "trex"
    assert recorded["config"]["env_kwargs"] == task  # Exact frozen replacement, not an accidental merge.
    assert recorded["config"]["_gait_stage"] == 2
    assert recorded["model"] == "selected.zip" and recorded["norm"] == "matched.pkl"
    assert recorded["episodes"] == 2 and recorded["seed"] == 17
    assert recorded["settle_s"] == 0.25 and recorded["direction_xy"] == (3.0, 4.0)
    assert recorded["allow_legacy_plant"] is True
    assert recorded["protocol"] == GaitProtocol(contact_force_bw_per_limb=0.03, chatter_fill_s=0.012)
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == {
        **{key: value for key, value in _cli_summary().items() if key != "episode_failure_counts"},
        "failure_counts": _cli_summary()["episode_failure_counts"],
    }


def test_cli_committed_stage_defaults_use_development_panel_and_zero_controller(tmp_path, monkeypatch):
    from environments.shared.constants import DEVELOPMENT_GAIT_SEED_START

    recorded = {}

    def writer(species, config, model, norm, output, **options):
        recorded.update(model=model, norm=norm, **options)
        return _cli_summary()

    monkeypatch.setattr(cli, "write_gait_report", writer)
    assert cli.main(["trex", "--zero-action", "--out-dir", str(tmp_path)]) == 0
    assert recorded["model"] is recorded["norm"] is None
    # A reward-gated stage gets a short development panel, never the certification block.
    assert recorded["episodes"] == 10
    assert recorded["seed"] == DEVELOPMENT_GAIT_SEED_START
    assert recorded["protocol"] == GaitProtocol()


@pytest.mark.parametrize(
    "options",
    [
        ["--episodes", "0"],
        ["--episodes", "-3"],
        ["--seed", "-1"],
        ["--settle-s", "nan"],
        ["--settle-s", "-0.5"],
        ["--direction", "0", "0"],
        ["--direction", "inf", "0"],
        ["--seed", "3042"],  # a report-only panel on the certification block
        ["--seed", "3000", "--episodes", "50"],
    ],
)
def test_cli_invalid_numbers_are_usage_errors_that_leave_an_existing_report(tmp_path, monkeypatch, capsys, options):
    monkeypatch.setattr(
        cli, "write_gait_report", lambda *args, **kwargs: pytest.fail("invalid usage reached the writer")
    )
    prior = _prior_certificate(tmp_path)
    before = prior.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main(["trex", "--zero-action", "--out-dir", str(tmp_path), *options])
    assert error.value.code == 2
    assert "usage:" in capsys.readouterr().err
    assert prior.read_bytes() == before


@pytest.mark.parametrize("content", ['{"no_such_option": 1}', '{"chatter_fill_s": -1}', "[]"])
def test_cli_invalid_protocol_options_are_usage_errors(tmp_path, monkeypatch, capsys, content):
    monkeypatch.setattr(
        cli, "write_gait_report", lambda *args, **kwargs: pytest.fail("invalid usage reached the writer")
    )
    options = tmp_path / "protocol.json"
    options.write_text(content)
    with pytest.raises(SystemExit) as error:
        cli.main(["trex", "--zero-action", "--out-dir", str(tmp_path / "panel"), "--protocol-json", str(options)])
    assert error.value.code == 2
    assert "--protocol-json" in capsys.readouterr().err
    assert not (tmp_path / "panel").exists()


def test_cli_writer_refusal_returns_three_and_prints_a_clear_error(tmp_path, monkeypatch, capsys):
    def refused(*args, **kwargs):
        raise ValueError("explicit matched normalization is required")

    monkeypatch.setattr(cli, "write_gait_report", refused)
    assert cli.main(["trex", "--model", "selected.zip", "--out-dir", str(tmp_path)]) == 3
    output = capsys.readouterr()
    assert not output.out
    assert "Gait evaluation refused: ValueError: explicit matched normalization is required" in output.err


@pytest.mark.parametrize(
    "extra_args",
    [[], ["--model", "policy.zip", "--zero-action"], ["--zero-action", "--direction", "one", "two"]],
)
def test_cli_invalid_usage_returns_argparse_status_two(tmp_path, capsys, extra_args):
    with pytest.raises(SystemExit) as error:
        cli.main(["trex", "--out-dir", str(tmp_path), *extra_args])
    assert error.value.code == 2
    assert "usage:" in capsys.readouterr().err
    assert not (tmp_path / "gait_report.json").exists()


@pytest.mark.parametrize("bad_json", ["[]", "null", "not json"])
def test_cli_bad_frozen_task_json_is_a_clear_refusal(tmp_path, capsys, bad_json):
    path = tmp_path / "task.json"
    path.write_text(bad_json)
    assert cli.main(["trex", "--zero-action", "--out-dir", str(tmp_path / "panel"), "--env-json", str(path)]) == 3
    assert "Gait evaluation refused:" in capsys.readouterr().err
    assert not (tmp_path / "panel").exists()


def test_cli_real_zero_action_smoke_uses_raw_json_without_touching_frozen_task(tmp_path, capsys):
    path = tmp_path / "task.json"
    path.write_text('{"max_episode_steps":3,"reset_noise_scale":0.01}')
    before = path.read_bytes()
    output = tmp_path / "panel"
    assert (
        cli.main(
            [
                "trex",
                "--zero-action",
                "--out-dir",
                str(output),
                "--env-json",
                str(path),
                "--episodes",
                "1",
                "--seed",
                "0",
                "--settle-s",
                "0",
            ]
        )
        == 0
    )
    assert path.read_bytes() == before
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["report_only"] is True and stdout["certified"] is False
    report = json.loads((output / "gait_report.json").read_text())
    assert report["episodes"][0]["length"] == 3
    assert len(report["traces"]) == 1


@pytest.mark.parametrize("provide_output", [False, True])
def test_protocol_only_cli_prints_exact_identity_without_rolling_or_writing(
    tmp_path, monkeypatch, capsys, provide_output, fresh_plant_identity_cache
):
    from environments.shared import plant_contract

    config = cli.load_stage_config("trex", 2)
    env = TRexEnv(**config["env_kwargs"])
    try:
        expected = measurement_protocol(
            "trex",
            GaitProtocol(),
            settle_s=1.0,
            direction_xy=(1.0, 0.0),
            horizon=config["env_kwargs"]["max_episode_steps"],
            physics_dt_s=float(env.model.opt.timestep),
            control_dt_s=float(env.dt),
            episodes=40,
            seed_start=3042,
        )
    finally:
        env.close()

    def forbidden(*args, **kwargs):
        pytest.fail("protocol preflight must not reset, step, load a policy, or write a panel")

    validate = plant_contract.validate_environment_plant

    def validate_then_guard(env, identity, **kwargs):
        # Plant identity hashes reset code, so retain its real validation
        # before instrumenting this one live instance for episode use.
        validate(env, identity, **kwargs)
        monkeypatch.setattr(env, "reset", forbidden)
        monkeypatch.setattr(env, "step", forbidden)

    monkeypatch.setattr(plant_contract, "validate_environment_plant", validate_then_guard)
    monkeypatch.setattr(cli, "write_gait_report", forbidden)
    monkeypatch.setattr(producer, "load_sb3_checkpoint", forbidden)
    monkeypatch.setattr(producer, "roll_gait_panel", forbidden)
    arguments = ["trex", "--protocol-only", "--episodes", "40", "--seed", "3042"]
    if provide_output:
        arguments.extend(["--out-dir", str(tmp_path / "unused_panel")])
    assert cli.main(arguments) == 0
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == {
        "measurement_protocol_sha256": protocol_sha256(expected),
        "measurement_protocol": expected,
    }
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options,match",
    [
        (["--episodes", "0"], "--episodes"),
        (["--episodes", "-1"], "--episodes"),
        (["--seed", "-1"], "--seed"),
        (["--settle-s", "-0.1"], "--settle-s"),
        (["--settle-s", "nan"], "--settle-s"),
        (["--direction", "0", "0"], "--direction"),
        (["--direction", "nan", "1"], "--direction"),
        (["--direction", "inf", "1"], "--direction"),
    ],
)
def test_protocol_only_cli_invalid_numbers_are_usage_errors_without_output(tmp_path, capsys, options, match):
    with pytest.raises(SystemExit) as error:
        cli.main(["trex", "--protocol-only", "--out-dir", str(tmp_path / "unused_panel"), *options])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert "usage:" in output.err and match in output.err
    assert not list(tmp_path.iterdir())


def test_protocol_only_cli_refuses_a_window_the_plant_cannot_hold(tmp_path, capsys):
    assert cli.main(["trex", "--protocol-only", "--out-dir", str(tmp_path / "unused_panel"), "--settle-s", "1000"]) == 3
    output = capsys.readouterr()
    assert not output.out
    assert "Gait evaluation refused:" in output.err and "positive analysis window" in output.err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("horizon", [0, -1])
def test_protocol_only_cli_invalid_horizon_is_refused_without_a_panel(tmp_path, capsys, horizon):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"max_episode_steps": horizon, "reset_noise_scale": 0.01}))
    assert (
        cli.main(
            [
                "trex",
                "--protocol-only",
                "--env-json",
                str(task),
                "--settle-s",
                "0",
                "--out-dir",
                str(tmp_path / "panel"),
            ]
        )
        == 3
    )
    output = capsys.readouterr()
    assert not output.out
    assert "Gait evaluation refused:" in output.err
    assert not (tmp_path / "panel").exists()


class _DivergingRecorder(producer.SubstepContactRecorder):
    """Blows the second episode's velocities up after two substeps, as an unstable policy would.

    MuJoCo answers with its automatic reset inside the next ``mj_step``.
    """

    created = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).created += 1
        self._inject = type(self).created == 2

    def _record_substep(self):
        super()._record_substep()
        if self._inject and len(self._rows) == 3:
            self.env.data.qvel[:] = 1e12
            self._inject = False


def test_divergence_fails_one_episode_and_the_panel_completes(tmp_path, species_config, stage_config, monkeypatch):
    _DivergingRecorder.created = 0
    monkeypatch.setattr(producer, "SubstepContactRecorder", _DivergingRecorder)
    report = producer.write_gait_report(
        species_config, stage_config, None, None, tmp_path, episodes=3, seed=9000, settle_s=0.0
    )
    assert report["status"] == "complete"
    first, diverged, third = report["episodes"]
    assert first["telemetry_valid"] and third["telemetry_valid"]
    assert diverged["telemetry_valid"] is False and diverged["physics_diverged"] is True
    assert diverged["telemetry_errors"] == [
        "physics_diverged: MuJoCo reset the simulation after a numerical instability"
    ]
    assert diverged["termination_reason"] == "physics_diverged" and diverged["completed_horizon"] is False
    with np.load(tmp_path / report["traces"][1]["path"]) as trace:
        assert bool(trace["physics_diverged"]) and len(trace["time_s"]) == 3


def _certifiable_stage_dir(root, saved_pair_source):
    models = root / "models"
    models.mkdir(parents=True)
    shutil.copyfile(saved_pair_source / "policy.zip", models / "robust_best_model.zip")
    shutil.copyfile(saved_pair_source / "policy_vecnorm.pkl", models / "robust_best_model_vecnorm.pkl")
    shutil.copyfile(saved_pair_source / "stage_config.json", root / "stage_config.json")
    return models / "robust_best_model.zip", models / "robust_best_model_vecnorm.pkl"


def test_producer_to_reader_accepts_a_genuine_panel_in_a_reused_directory(
    tmp_path, species_config, stage_config, saved_pair_source, monkeypatch
):
    """The real writer's evidence is readable by the real reader: seed-tied traces, replayed metrics.

    The directory first holds a larger report-only panel (stale traces), and
    one certification episode diverges; neither makes the evidence unreadable.
    """
    from environments.shared.reporting.gates import gait_statistics

    root = tmp_path / "stage"
    model, norm = _certifiable_stage_dir(root, saved_pair_source)
    producer.write_gait_report(species_config, stage_config, model, norm, root, episodes=4, seed=9000, settle_s=0.0)
    assert len(list((root / "gait_traces").iterdir())) == 4
    config = _strict_config(stage_config)
    _DivergingRecorder.created = 0
    monkeypatch.setattr(producer, "SubstepContactRecorder", _DivergingRecorder)
    report = producer.write_gait_report(
        species_config, config, model, norm, root, episodes=2, seed=PUBLICATION_SEED_START, settle_s=0.0
    )
    assert report["certification_eligible"] is True and report["episodes"][1]["physics_diverged"] is True
    stats, failures = gait_statistics(root, config["curriculum_kwargs"])
    assert not failures and stats is not None
    assert stats["selected_gait_n_episodes"] == 2 and stats["passed"] is False
    assert stats["episode_failures"][1] == ["episode/telemetry_valid: telemetry is invalid or missing"]


def _recorded_stage_dir(root, species):
    """A stage directory recorded the way ``train_curriculum`` records one, for *species*' locomotion.

    ``save_stage_config`` writes ``stage_config.json`` (every constructor
    default out, under ``reward_weights``) with the task fingerprint the
    trainer derives from the stage's own ``[env]``; a tiny PPO pair carries
    the same fingerprint.  Returns the strict stage config and the pair.
    """
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.config import load_all_stages, save_stage_config
    from environments.shared.gait.identity import stage_measurement_protocol
    from environments.shared.species_registry import get_species_config

    species_cfg = get_species_config(species)
    config = copy.deepcopy(load_all_stages(species)[2])
    config["env_kwargs"]["max_episode_steps"] = 3
    config["curriculum_kwargs"] = {
        **provisional_gait_criteria("biped_walk"),
        "gate_kind": GAIT_GATE_KIND,
        "gate_schema_version": 1,
        "gait_profile": "biped_walk",
        "min_eval_episodes": 2,
        "gait_panel_seed_start": PUBLICATION_SEED_START,
        "min_gait_success_lcb": 0.8,
        "min_episode_forward_vel": 0.1,
        "min_episode_duration_s": 0.03,
    }
    config["curriculum_kwargs"]["measurement_protocol_sha256"] = protocol_sha256(
        stage_measurement_protocol(species_cfg, config, episodes=2, seed_start=PUBLICATION_SEED_START, settle_s=0.0)
    )
    plant = current_plant_identity(species)
    task = stage_task_fingerprint(species, 2, stage_config=config, plant_identity=plant)
    models = root / "models"
    models.mkdir(parents=True)
    normalizer = VecNormalize(
        DummyVecEnv([lambda: species_cfg.env_class(**config["env_kwargs"])]), norm_obs=True, norm_reward=True
    )
    try:
        model = sb3.PPO(
            "MlpPolicy",
            normalizer,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            seed=77,
            device="cpu",
        )
        attach_plant_identity(model, plant)
        attach_plant_identity(normalizer, plant)
        attach_task_fingerprint(model, task)
        model.save(models / "robust_best_model.zip")
        normalizer.save(models / "robust_best_model_vecnorm.pkl")
    finally:
        normalizer.close()
    save_stage_config(
        root,
        2,
        config,
        "PPO",
        extra={"seed": 77, "n_envs": 1},
        env_class=species_cfg.env_class,
        species=species,
        plant_identity=plant,
        task_fingerprint=task,
    )
    return species_cfg, config, models / "robust_best_model.zip", models / "robust_best_model_vecnorm.pkl"


@pytest.mark.parametrize("species", ["compsognathus", "trex"])
def test_producer_to_reader_reads_the_stage_config_the_trainer_records(tmp_path, species):
    """PL-1: the reader accepts a genuine panel beside the ``stage_config.json`` ``save_stage_config`` writes.

    Compsognathus' quiet-push carve-out depends on which push keys the stage
    set, which ``reward_weights`` (every default written out) cannot say, so
    a task re-derived from it kept the five push keys and the reader refused
    every compsognathus panel after the whole training budget.
    """
    from environments.shared.reporting.gates import gait_statistics

    root = tmp_path / "stage"
    species_cfg, config, model, norm = _recorded_stage_dir(root, species)
    saved = json.loads((root / "stage_config.json").read_text())
    if species == "compsognathus":
        recorded_env = saved["task_fingerprint"]["env"]
        assert "perturbation_interval" in saved["reward_weights"] and "perturbation_interval" not in recorded_env
    report = producer.write_gait_report(
        species_cfg,
        dict(config, _gait_stage=2),
        model,
        norm,
        root,
        episodes=2,
        seed=PUBLICATION_SEED_START,
        settle_s=0.0,
    )
    assert report["status"] == "complete" and report["certification_eligible"] is True
    stats, failures = gait_statistics(root, config["curriculum_kwargs"])
    assert not failures and stats is not None and stats["selected_gait_n_episodes"] == 2


def test_reader_refuses_a_recorded_constructor_that_builds_another_task(tmp_path):
    """A constructor value the recorded task names, changed in ``stage_config.json``, is named and refused."""
    from environments.shared.reporting.gates import gait_statistics

    root = tmp_path / "stage"
    species_cfg, config, model, norm = _recorded_stage_dir(root, "compsognathus")
    producer.write_gait_report(
        species_cfg,
        dict(config, _gait_stage=2),
        model,
        norm,
        root,
        episodes=2,
        seed=PUBLICATION_SEED_START,
        settle_s=0.0,
    )
    path = root / "stage_config.json"
    saved = json.loads(path.read_text())
    saved["reward_weights"]["forward_vel_weight"] = saved["reward_weights"]["forward_vel_weight"] + 1.0
    path.write_text(json.dumps(saved))
    stats, failures = gait_statistics(root, config["curriculum_kwargs"])
    assert stats is None
    assert failures == [
        "gait evidence could not be read: ValueError: the recorded environment constructor does not build "
        "the gait report's task: forward_vel_weight"
    ]


def test_constructor_task_differences_ignores_carved_out_keys_and_names_changed_ones():
    """Every key the task names is compared as the fingerprint hashes it; carved-out keys are not named."""
    from environments.shared.task_fingerprint import constructor_task_differences

    task = stage_task_fingerprint("compsognathus", 2, env_kwargs={})
    assert "perturbation_interval" not in task["env"]
    explicit = {**{key: value for key, value in task["env"].items()}, "perturbation_interval": 2.0}
    assert constructor_task_differences("compsognathus", explicit, task) == []
    assert constructor_task_differences("compsognathus", {}, task) == []
    healthy = task["env"]["max_episode_steps"]
    assert constructor_task_differences("compsognathus", {"max_episode_steps": float(healthy)}, task) == [
        "max_episode_steps"
    ]
    assert constructor_task_differences("compsognathus", {}, {"task_sha256": "x"}) == ["env"]


@pytest.mark.parametrize("missing,flag", [("min_eval_episodes", "--episodes"), ("gait_panel_seed_start", "--seed")])
def test_cli_gait_block_without_a_panel_key_is_a_usage_error_or_a_refusal(tmp_path, monkeypatch, capsys, missing, flag):
    """PL-4: a ``locomotion_gait/v2`` block missing a panel key never ends in a traceback (exit 1).

    Without the explicit option it is a usage error (2); given explicitly,
    the protocol is planned, and a panel is refused by the writer's
    validation of the block (3) before anything is written.
    """
    gated = _strict_config({"env_kwargs": {"max_episode_steps": 3, "reset_noise_scale": 0.01}})
    del gated["curriculum_kwargs"][missing]
    monkeypatch.setattr(cli, "load_stage_config", lambda species, stage, config_path=None: copy.deepcopy(gated))
    with pytest.raises(SystemExit) as error:
        cli.main(["trex", "--protocol-only"])
    assert error.value.code == 2 and flag in capsys.readouterr().err
    explicit = ["--episodes", "2", "--seed", str(PUBLICATION_SEED_START), "--settle-s", "0"]
    assert cli.main(["trex", "--protocol-only", *explicit]) == 0
    assert json.loads(capsys.readouterr().out)["measurement_protocol_sha256"].startswith("sha256:")
    output = tmp_path / "panel"
    assert cli.main(["trex", "--zero-action", "--out-dir", str(output), *explicit]) == 3
    assert missing in capsys.readouterr().err
    assert not output.exists()
