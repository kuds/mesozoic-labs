"""The report-only heading probe (decision D-D27): spawns turned about vertical.

Both physics-r8 trex stances fell when spawned 90 degrees off where the
statue stands; the stance stage always spawns facing +x, so no gate input
can see it.  The probe turns the spawn after ``reset`` -- the animal alone,
or with the prey and the episode's reference direction -- and these tests
pin that the turn is exactly a rotation about vertical through the root and
that it changes nothing else.
"""

from __future__ import annotations

import json
import math
import tomllib

import numpy as np
import pytest

from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.curriculum.gate_schema import _DIAGNOSTIC_KEYS
from environments.shared.reporting.stance_report import (
    HEADING_PROBE_JSON,
    SpawnYaw,
    _apply_spawn_yaw,
    _root_yaw,
    build_stance_gate_report,
    declares_heading_probe,
    heading_probe_warnings,
    probe_stem,
    render_heading_probe,
    spawn_yaw_variants,
    stage_heading_probe_warnings,
    write_heading_probe,
)
from environments.shared.species_catalog import DEFAULT_MANIFEST_PATH
from environments.trex.envs.trex_env import TRexEnv

#: Each species' gait, from the species manifest: every biped's stance declares the heading probe, and no
#: other species' stance does, so a new species lands in one of the two tests below.
_GAITS = {
    entry["id"]: entry["gait"] for entry in tomllib.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))["species"]
}


@pytest.fixture
def env():
    e = TRexEnv(prey_distance_range=(10.0, 15.0), reset_noise_scale=0.05)
    yield e
    e.close()


def _obs_slices(env) -> dict[str, slice]:
    """Where TRexEnv._get_obs (bipedal-target-heading-free/v1, policy interface r14) puts its root and prey channels.

    After the joint positions and velocities: projected gravity, the gyro, the heading-frame velocity, the
    accelerometer, the two feet, then the pelvis-frame prey direction.
    """
    start = (env.model.nq - 7) + (env.model.nv - 6)
    return {
        "pelvis_gravity": slice(start, start + 3),
        "pelvis_heading_velocity": slice(start + 6, start + 9),
        "prey_direction_body": slice(start + 14, start + 17),
    }


def _turned_xy(yaw_deg: float, vector: np.ndarray) -> np.ndarray:
    theta = math.radians(yaw_deg)
    x, y, z = (float(value) for value in vector)
    return np.array([math.cos(theta) * x - math.sin(theta) * y, math.sin(theta) * x + math.cos(theta) * y, z])


class TestTurn:
    @pytest.mark.parametrize("yaw_deg", [-90.0, 45.0])
    def test_the_animal_turns_about_vertical_and_nothing_else_moves(self, env, yaw_deg):
        env.reset(seed=3042)
        joints = env.data.qpos[7:].copy()
        root = env.data.qpos[0:3].copy()
        heights = env.data.xpos[:, 2].copy()
        prey = env.data.mocap_pos[0].copy()
        yaw_before = _root_yaw(env)
        slices = _obs_slices(env)
        obs_before = env._get_obs()
        pelvis_before = env.data.xmat[env.pelvis_id].reshape(3, 3).copy()
        prey_offset = env.data.mocap_pos[0] - env.data.xpos[env.pelvis_id]
        prey_direction_before = prey_offset / (np.linalg.norm(prey_offset) + 1e-8)
        obs = _apply_spawn_yaw(env, SpawnYaw(yaw_deg))
        assert math.degrees(math.remainder(_root_yaw(env) - yaw_before, 2 * math.pi)) == pytest.approx(yaw_deg)
        np.testing.assert_array_equal(env.data.qpos[7:], joints)
        np.testing.assert_array_equal(env.data.qpos[0:3], root)
        # A rotation about vertical moves no height: the spawn stays on the floor exactly as it was.
        np.testing.assert_allclose(env.data.xpos[:, 2], heights, atol=1e-12)
        np.testing.assert_array_equal(env.data.mocap_pos[0], prey)
        np.testing.assert_array_equal(obs, env._get_obs())
        # The observation reads the TURNED state (the derived sensors recomputed, not the reset's).  It
        # carries no world heading since policy interface r14, so the turn alone moves nothing in it but the
        # prey's bearing: the prey, left behind, now lies where the reset's prey direction turned by -yaw
        # about vertical lies, seen from the reset's pelvis frame.
        prey_bearing = slices["prey_direction_body"]
        others = np.ones(obs.shape, dtype=bool)
        others[prey_bearing] = False
        np.testing.assert_allclose(obs[others], obs_before[others], atol=1e-6)
        expected_bearing = pelvis_before.T @ _turned_xy(-yaw_deg, prey_direction_before)
        np.testing.assert_allclose(obs[prey_bearing], expected_bearing, atol=1e-6)
        assert np.max(np.abs(obs[prey_bearing] - obs_before[prey_bearing])) > 0.5
        # The turned state was posed out of band, so the step-tagged substep aggregates are dropped.
        assert env._substep_contact_step == -1

    def test_the_whole_scene_turns_the_prey_and_the_reference_direction_with_it(self, env):
        env.reset(seed=3042)
        root_xy = env.data.qpos[0:2].copy()
        prey_rel = env.data.mocap_pos[0, :2] - root_xy
        reference = env._initial_prey_dir_2d.copy()
        obs_before = env._get_obs()
        obs = _apply_spawn_yaw(env, SpawnYaw(90.0, whole_scene=True))
        np.testing.assert_allclose(env.data.mocap_pos[0, :2] - root_xy, [-prey_rel[1], prey_rel[0]], atol=1e-12)
        np.testing.assert_allclose(env._initial_prey_dir_2d, [-reference[1], reference[0]], atol=1e-12)
        # Since policy interface r14 the observation carries no world heading: the turned scene reads the same.
        np.testing.assert_allclose(obs, obs_before, atol=1e-6)

    def test_the_statue_scores_the_same_episode_from_a_turned_scene(self):
        """Physics and reward are rotation-invariant for a constant command when the prey turns too."""
        stage_config = load_stage_config("trex", "stance")

        def roll(spawn_yaw):
            report = build_stance_gate_report(
                "trex", 1, stage_config=stage_config, zero_action=True, episodes=1, seed=3042, spawn_yaw=spawn_yaw
            )
            return report

        straight = roll(SpawnYaw(0.0))
        turned = roll(SpawnYaw(90.0, whole_scene=True))
        assert turned["metrics"]["episode_length_mean"] == straight["metrics"]["episode_length_mean"] == 1000
        assert turned["metrics"]["reward_mean"] == pytest.approx(straight["metrics"]["reward_mean"], rel=1e-6)
        assert turned["episode_yaw_change_deg"][0] == pytest.approx(straight["episode_yaw_change_deg"][0], abs=1e-3)
        assert probe_stem(turned) == "stance_gate_probe_heading"
        assert turned["spawn_yaw"] == {"kind": "spawn_yaw/v1", "yaw_deg": 90.0, "whole_scene": True}


class TestYawChange:
    def test_a_turn_past_180_degrees_is_accumulated_not_folded_back(self):
        """Three 100-degree steps are a 300-degree turn; read off the wrapped yaw they would be 60."""
        from environments.shared.reporting.stance_report import _roll_episodes

        class _Spinning:
            """Turns its root 100 degrees about vertical on every step, for three steps."""

            def __init__(self) -> None:
                self.unwrapped = self
                self.data = type("Data", (), {})()
                self.yaw_deg = 0.0

            def _pose(self) -> None:
                half = math.radians(self.yaw_deg) / 2.0
                self.data.qpos = np.array([0.0, 0.0, 1.0, math.cos(half), 0.0, 0.0, math.sin(half)])

            def reset(self, seed=None):
                self.yaw_deg = 0.0
                self._pose()
                return np.zeros(1), {}

            def step(self, action):
                self.yaw_deg += 100.0
                self._pose()
                return np.zeros(1), 1.0, False, self.yaw_deg >= 300.0, {}

        yaw_changes: list[float] = []
        _roll_episodes(
            _Spinning(),
            predict=lambda obs: np.zeros(1),
            episodes=1,
            seed=0,
            settle_steps=10,
            lengths=[],
            rewards=[],
            duties=[],
            bilateral_duties=[],
            single_duties=[],
            terminations={},
            components={},
            action_stats={},
            yaw_changes=yaw_changes,
        )
        assert yaw_changes == [pytest.approx(300.0)]


class TestProbeTable:
    def test_the_rows_are_the_control_then_each_offset_both_ways(self):
        variants = spawn_yaw_variants([90.0, -90.0, 0.0, 45.0])
        assert variants[0] == SpawnYaw(0.0)
        assert [(v.yaw_deg, v.whole_scene) for v in variants[1:]] == [
            (-90.0, False),
            (-90.0, True),
            (45.0, False),
            (45.0, True),
            (90.0, False),
            (90.0, True),
        ]

    def test_render_and_write(self, tmp_path):
        def report(yaw_deg, whole_scene, length, reward):
            return {
                "spawn_yaw": SpawnYaw(yaw_deg, whole_scene).as_dict(),
                "metrics": {
                    "full_horizon_fraction": length / 1000,
                    "reward_mean": reward,
                    "episode_length_mean": length,
                },
                "episode_yaw_change_deg": [3.0, 5.0],
                "terminations": {},
                "horizon": 1000,
                "policy": "robust_best_model.zip",
                "seed": 3042,
            }

        policy = [report(0.0, False, 1000, 3770.0), report(90.0, False, 200, 400.0), report(90.0, True, 200, 410.0)]
        statue = [report(0.0, False, 1000, 3765.0), report(90.0, False, 1000, 3680.0), report(90.0, True, 1000, 3765.0)]
        text, payload = render_heading_probe(policy, statue, probe_episodes=2)
        assert "REPORT ONLY" in payload["note"] and "not a gate result" in text
        assert [row["whole_scene"] for row in payload["rows"]] == [False, False, True]
        assert payload["rows"][1]["policy"]["full_horizon_fraction"] == 0.2
        written = write_heading_probe(tmp_path, policy, statue, probe_episodes=2)
        assert written["heading_probe_txt"].name == "stance_heading_probe.txt"
        assert written["heading_probe_json"].exists()

    @pytest.mark.parametrize("species", sorted(s for s, gait in _GAITS.items() if gait == "Bipedal"))
    def test_the_probe_key_is_a_diagnostic_declared_by_every_biped_stance(self, species):
        # D-D27 for the trex; the maintainer's decision of 2026-10-11 for the velociraptor and both compsognathus.
        assert "stance_probe_spawn_yaw_deg" in _DIAGNOSTIC_KEYS
        curriculum = load_stage_config(species, "stance")["curriculum_kwargs"]
        assert curriculum["stance_probe_spawn_yaw_deg"] == [-90.0, -45.0, 45.0, 90.0]
        assert declares_heading_probe(curriculum)

    @pytest.mark.parametrize("species", sorted(s for s, gait in _GAITS.items() if gait != "Bipedal"))
    def test_no_quadruped_stance_declares_the_probe(self, species):
        # Their reward_and_length gate rolls no stance report, and the probe runs only after one.
        curriculum = load_stage_config(species, "stance")["curriculum_kwargs"]
        assert curriculum["gate_kind"] == "reward_and_length/v1"
        assert "stance_probe_spawn_yaw_deg" not in curriculum
        assert not declares_heading_probe(curriculum)


#: Seed 52's probe (``20261009_155723``; docs/HEADING_INVARIANCE_DESIGN_2026_10.md, "Later runs"): the policy's
#: full-horizon fraction per (yaw, whole scene).  It passed stance_quality/v2 and falls when turned -90 or -45.
SEED_52 = {
    (0.0, False): 1.0,
    (-90.0, False): 0.0,
    (-90.0, True): 0.0,
    (-45.0, False): 0.0,
    (-45.0, True): 0.0,
    (45.0, False): 0.875,
    (45.0, True): 1.0,
    (90.0, False): 0.125,
    (90.0, True): 1.0,
}
STANDS = dict.fromkeys(SEED_52, 1.0)


def _sweeps(policy=SEED_52, statue=STANDS):
    """The policy and statue panels of a four-offset probe, each row at the given full-horizon fraction."""

    def report(variant, fraction):
        return {
            "spawn_yaw": variant.as_dict(),
            "metrics": {"full_horizon_fraction": fraction, "reward_mean": 3700.0, "episode_length_mean": 1000.0},
            "episode_yaw_change_deg": [3.0, 5.0],
            "terminations": {},
            "horizon": 1000,
            "policy": "robust_best_model.zip",
            "seed": 3042,
        }

    variants = spawn_yaw_variants([-90.0, -45.0, 45.0, 90.0])
    return (
        [report(v, policy[(v.yaw_deg, v.whole_scene)]) for v in variants],
        [report(v, statue[(v.yaw_deg, v.whole_scene)]) for v in variants],
    )


def _payload(policy=SEED_52, statue=STANDS) -> dict:
    """The probe's JSON as written (non-finite values as null)."""
    _, payload = render_heading_probe(*_sweeps(policy, statue), probe_episodes=8)
    written: dict = json.loads(json.dumps(payload))
    return written


class TestHandoffWarning:
    """The maintainer's decision of 2026-10-11: handoff tooling warns, without blocking, when a whole-scene
    row's full-horizon fraction is below 0.95, v2's per-panel rail; the probe otherwise stays report only."""

    def test_a_whole_scene_row_below_the_rail_warns_and_names_it(self):
        (message,) = heading_probe_warnings(_payload())
        assert "below 0.95, the v2 per-panel rail, on 2 of 4 whole-scene rows" in message
        assert "(-90 deg 0.00 (statue 1.00); -45 deg 0.00 (statue 1.00); 8 episodes per row)" in message
        assert "+45 deg" not in message and "+90 deg" not in message
        assert "no verdict changes and nothing is blocked" in message

    def test_rows_at_or_above_the_rail_do_not_warn(self):
        assert heading_probe_warnings(_payload(STANDS)) == []
        assert heading_probe_warnings(_payload(dict.fromkeys(SEED_52, 0.95))) == []

    def test_one_fall_in_eight_warns(self):
        (message,) = heading_probe_warnings(_payload({**STANDS, (90.0, True): 7 / 8}))
        assert "on 1 of 4 whole-scene rows (+90 deg 0.88 (statue 1.00)" in message

    def test_the_unturned_control_and_animal_only_rows_never_warn(self):
        falls_unless_scene = {key: 1.0 if key[1] else 0.0 for key in SEED_52}
        assert heading_probe_warnings(_payload(falls_unless_scene)) == []

    def test_a_statue_row_below_the_rail_is_reported_as_a_failed_control(self):
        statue = {**STANDS, (45.0, True): 0.75, (45.0, False): 0.5}
        (message,) = heading_probe_warnings(_payload(STANDS, statue))
        assert message.startswith("the zero-action statue is itself below 0.95 on whole-scene rows +45 deg 0.75:")
        assert "the control did not stand" in message
        # Both are reported, each on its own line: the policy falls where the statue stands.
        policy_warning, statue_warning = heading_probe_warnings(_payload(SEED_52, statue))
        assert "on 2 of 4 whole-scene rows" in policy_warning and "+45 deg 0.75" in statue_warning

    @pytest.mark.parametrize("fraction", [None, math.nan, "1.0", True, -0.1, 1.5, 10**400], ids=repr)
    def test_unmeasured_rows_warn_as_unmeasured(self, fraction):
        payload = _payload(STANDS)
        (row,) = [row for row in payload["rows"] if row["whole_scene"] and row["yaw_deg"] == -45.0]
        row["policy"]["full_horizon_fraction"] = fraction
        (message,) = heading_probe_warnings(payload)
        assert "whole-scene rows -45 deg; those headings are unmeasured" in message
        # The statue control of the same row, unmeasured, is reported apart.
        row["policy"]["full_horizon_fraction"] = 1.0
        row["statue"]["full_horizon_fraction"] = fraction
        (message,) = heading_probe_warnings(payload)
        assert message.startswith("the zero-action statue's full-horizon fraction is not a share in [0, 1] on")
        assert "whole-scene rows -45 deg; those controls are unmeasured" in message

    def test_a_row_whose_heading_is_not_a_finite_float_is_named_by_its_index(self):
        payload = _payload({**STANDS, (-45.0, True): 0.0})
        (index,) = [i for i, row in enumerate(payload["rows"]) if row["whole_scene"] and row["yaw_deg"] == -45.0]
        for yaw in (None, math.inf, 10**400):
            payload["rows"][index]["yaw_deg"] = yaw
            (message,) = heading_probe_warnings(payload)
            assert f"(row {index} 0.00 (statue 1.00); 8 episodes per row)" in message

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda payload: payload.update(schema="mesozoic.stance-heading-probe/v0"),
            lambda payload: payload.pop("schema"),
            lambda payload: payload.update(probe_episodes=0),
            lambda payload: payload.update(probe_episodes=8.0),
            lambda payload: payload.pop("probe_episodes"),
            lambda payload: payload.update(rows="rows"),
            lambda payload: payload.update(rows=[row for row in payload["rows"] if not row["whole_scene"]]),
            lambda payload: payload.update(rows=[None, 3, *payload["rows"][:1]]),
        ],
        ids=["schema", "no-schema", "no-episodes", "float-episodes", "episodes-missing", "rows", "no-scene", "junk"],
    )
    def test_a_malformed_probe_warns_instead_of_raising(self, mutate):
        payload = _payload(STANDS)
        mutate(payload)
        (message,) = heading_probe_warnings(payload)
        assert message.endswith("heading robustness is unmeasured")

    @pytest.mark.parametrize("payload", [None, [], "probe", 3.0], ids=repr)
    def test_a_record_that_is_not_an_object_warns_instead_of_raising(self, payload):
        (message,) = heading_probe_warnings(payload)
        assert "is not a mesozoic.stance-heading-probe/v1 record" in message

    def test_the_probe_text_carries_the_warning_and_the_json_is_unchanged(self):
        text, payload = render_heading_probe(*_sweeps(), probe_episodes=8)
        assert "  HANDOFF WARNING (the maintainer's decision of 2026-10-11" in text
        # The warning is wrapped under its heading, before the closing notes, and the text is still a probe.
        block = text.split("HANDOFF WARNING", 1)[1].split("The unturned row is the harness check", 1)[0]
        assert "on 2 of 4 whole-scene rows" in " ".join(block.split())
        assert all(len(line) <= 120 for line in text.splitlines())
        assert "not a gate result" in text
        # Report only: no key, no verdict, nothing a reader could gate on is added to the record.
        assert set(payload) == {"schema", "policy", "probe_episodes", "seed", "horizon", "note", "rows"}
        assert payload["schema"] == "mesozoic.stance-heading-probe/v1"
        quiet, quiet_payload = render_heading_probe(*_sweeps(STANDS), probe_episodes=8)
        assert "HANDOFF WARNING" not in quiet and set(quiet_payload) == set(payload)

    def test_the_stage_reader_reads_what_the_writer_wrote(self, tmp_path):
        written = write_heading_probe(tmp_path, *_sweeps(), probe_episodes=8)
        assert written["heading_probe_json"] == tmp_path / HEADING_PROBE_JSON
        assert stage_heading_probe_warnings(tmp_path) == heading_probe_warnings(_payload())
        write_heading_probe(tmp_path, *_sweeps(STANDS), probe_episodes=8)
        assert stage_heading_probe_warnings(tmp_path) == []
        (tmp_path / HEADING_PROBE_JSON).write_text("{truncated", encoding="utf-8")
        (message,) = stage_heading_probe_warnings(tmp_path)
        assert "is not a readable JSON object; heading robustness is unmeasured" in message

    def test_an_absent_probe_warns_only_when_the_stage_declares_it(self, tmp_path):
        assert stage_heading_probe_warnings(tmp_path) == []
        assert stage_heading_probe_warnings(tmp_path, declared=False) == []
        (message,) = stage_heading_probe_warnings(tmp_path, declared=True)
        assert "is missing although the stage declares the heading probe; heading robustness is unmeasured" in message
        # The remedy rolls it into a scratch directory: --out-dir also writes a gate report, never into the stage.
        assert "--out-dir <a scratch directory>" in message and f"--out-dir {tmp_path}" not in message
        # Undeclared by the caller, the stage's recorded config decides: the trex and velociraptor stances
        # declare the probe, so a stance the command-line curriculum trained (it rolls none) warns; the
        # brachiosaurus stance does not, and nor does a velociraptor stance recorded before its config declared
        # the probe (2026-10-11), whatever the checkout's config now says.
        trex, raptor, brachio = tmp_path / "trex", tmp_path / "raptor", tmp_path / "brachio"
        save_stage_config(trex, 1, load_stage_config("trex", "stance"), "PPO", species="trex")
        save_stage_config(raptor, 1, load_stage_config("velociraptor", "stance"), "PPO", species="velociraptor")
        save_stage_config(brachio, 1, load_stage_config("brachiosaurus", "stance"), "PPO", species="brachiosaurus")
        old_raptor, before = tmp_path / "old_raptor", load_stage_config("velociraptor", "stance")
        before["curriculum_kwargs"].pop("stance_probe_spawn_yaw_deg")
        save_stage_config(old_raptor, 1, before, "PPO", species="velociraptor")
        assert len(stage_heading_probe_warnings(trex)) == 1
        assert len(stage_heading_probe_warnings(raptor)) == 1
        assert stage_heading_probe_warnings(brachio) == []
        assert stage_heading_probe_warnings(old_raptor) == []
        assert stage_heading_probe_warnings(trex, declared=False) == []

    def test_declares_heading_probe(self):
        assert declares_heading_probe(load_stage_config("trex", "stance")["curriculum_kwargs"])
        assert declares_heading_probe(load_stage_config("velociraptor", "stance")["curriculum_kwargs"])
        assert not declares_heading_probe(load_stage_config("brachiosaurus", "stance")["curriculum_kwargs"])
        assert not declares_heading_probe(load_stage_config("trex", "recovery")["curriculum_kwargs"])
        stance = {"gate_kind": "stance_quality/v2"}
        assert declares_heading_probe({**stance, "stance_probe_spawn_yaw_deg": [0.0, "north", 90.0]})
        assert declares_heading_probe({**stance, "stance_probe_spawn_yaw_deg": -45.0})
        for raw in (None, [], [0.0], ["north"], [math.nan], [10**400]):
            assert not declares_heading_probe({**stance, "stance_probe_spawn_yaw_deg": raw})
        assert not declares_heading_probe({"gate_kind": "reward_and_length/v1", "stance_probe_spawn_yaw_deg": [90.0]})
        assert not declares_heading_probe(None)
