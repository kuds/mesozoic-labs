"""Tests for reporting-only T. rex stance diagnostics."""

import csv

import numpy as np
import pytest

from environments.shared.stance_diagnostics import (
    capture_trex_stance_snapshot,
    derive_stance_info,
    write_stance_diagnostics_csv,
)


def test_derive_stance_info_reports_duty_load_share_and_asymmetry():
    result = derive_stance_info({"r_foot_contact": 75.0, "l_foot_contact": 25.0})

    assert result["r_foot_contact_duty"] == 1.0
    assert result["l_foot_contact_duty"] == 1.0
    assert result["bilateral_support_duty"] == 1.0
    assert result["single_support_duty"] == 0.0
    assert result["r_foot_load_share"] == pytest.approx(0.75)
    assert result["l_foot_load_share"] == pytest.approx(0.25)
    assert result["foot_load_imbalance"] == pytest.approx(0.5)
    assert result["foot_load_asymmetry"] == pytest.approx(0.5)
    assert result["foot_load_balance"] == pytest.approx(0.5)


def test_derive_stance_info_distinguishes_single_and_unsupported_contact():
    single = derive_stance_info({"r_foot_contact": 1.0, "l_foot_contact": 0.0})
    unsupported = derive_stance_info({"r_foot_contact": 0.0, "l_foot_contact": 0.0})

    assert single["single_support_duty"] == 1.0
    assert single["bilateral_support_duty"] == 0.0
    assert unsupported["unsupported_duty"] == 1.0
    assert unsupported["foot_load_balance"] == 0.0


def test_snapshot_gracefully_records_info_without_mujoco_environment():
    row = capture_trex_stance_snapshot(
        object(),
        {
            "r_foot_contact": 60.0,
            "l_foot_contact": 40.0,
            "drift_distance": 0.2,
            "pelvis_yaw_vel": -0.1,
        },
        step=7,
    )

    assert row["step"] == 7.0
    assert row["r_foot_load_share"] == pytest.approx(0.6)
    assert row["drift_distance"] == pytest.approx(0.2)
    assert row["pelvis_yaw_vel"] == pytest.approx(-0.1)


def test_snapshot_records_geometry_without_mutating_simulation_state():
    from environments.trex.envs.trex_env import TRexEnv

    env = TRexEnv(reset_noise_scale=0.0)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        before = {
            "qpos": env.data.qpos.copy(),
            "qvel": env.data.qvel.copy(),
            "act": env.data.act.copy(),
            "ctrl": env.data.ctrl.copy(),
            "time": float(env.data.time),
        }

        row = capture_trex_stance_snapshot(env, info, step=1)

        right_foot = np.asarray(env.data.site_xpos[env.r_foot_site_id], dtype=float)
        left_foot = np.asarray(env.data.site_xpos[env.l_foot_site_id], dtype=float)
        pelvis = np.asarray(env.data.xpos[env.pelvis_id], dtype=float)
        support_midpoint = (right_foot + left_foot) / 2.0
        pelvis_support_offset = pelvis - support_midpoint

        assert [row[f"r_foot_{axis}"] for axis in "xyz"] == pytest.approx(right_foot)
        assert [row[f"l_foot_{axis}"] for axis in "xyz"] == pytest.approx(left_foot)
        assert row["stance_width"] == pytest.approx(abs(right_foot[1] - left_foot[1]))
        assert row["foot_fore_aft_offset"] == pytest.approx(right_foot[0] - left_foot[0])
        assert row["foot_fore_aft_separation"] == pytest.approx(abs(right_foot[0] - left_foot[0]))
        assert [row[f"support_midpoint_{axis}"] for axis in "xyz"] == pytest.approx(support_midpoint)
        assert [row[f"pelvis_support_offset_{axis}"] for axis in "xyz"] == pytest.approx(pelvis_support_offset)
        assert row["pelvis_support_offset_xy"] == pytest.approx(np.linalg.norm(pelvis_support_offset[:2]))
        assert row["hip_roll_home_error_rad"] == pytest.approx(
            np.mean(
                [
                    row["r_hip_roll_home_error_rad"],
                    row["l_hip_roll_home_error_rad"],
                ]
            )
        )

        np.testing.assert_array_equal(env.data.qpos, before["qpos"])
        np.testing.assert_array_equal(env.data.qvel, before["qvel"])
        np.testing.assert_array_equal(env.data.act, before["act"])
        np.testing.assert_array_equal(env.data.ctrl, before["ctrl"])
        assert env.data.time == before["time"]
    finally:
        env.close()


def test_write_stance_csv_unions_columns(tmp_path):
    path = tmp_path / "stance.csv"
    result = write_stance_diagnostics_csv(path, [{"step": 1, "a": 2.0}, {"step": 2, "b": 3.0}])

    assert result == path
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["a"] == "2.0"
    assert rows[1]["b"] == "3.0"


def test_write_stance_csv_skips_empty_rows(tmp_path):
    path = tmp_path / "stance.csv"

    assert write_stance_diagnostics_csv(path, []) is None
    assert not path.exists()
