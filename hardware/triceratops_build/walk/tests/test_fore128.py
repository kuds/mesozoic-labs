"""Tests for walk/fore128_walk60.py: at its defaults, on the walk's own model, it reproduces the stored judged and
sweep episodes exactly, so the parameters it adds (model, delay points, push axis, seed start, peak drive speed)
leave every existing result unchanged. Run by hand:
    python -m pytest hardware/triceratops_build/walk/tests
"""

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

WALK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WALK))

import fore128_walk60 as F  # noqa: E402
import judge_walk60 as JW  # noqa: E402
import sweep_walk60 as SW  # noqa: E402

S = F.S
ROLL_ANKLE = Path(F.walk60.plant_config().model_path).name


@pytest.fixture(autouse=True)
def _restore_sweep_module():
    """The driver's workers patch two names in rl/robustness/sweep.py (in their own processes); put them back after
    each test so other tests in this process see the sweep as it is."""
    yield
    S.episode_record, S.lateral_axis = F._ORIG_RECORD, F._ORIG_LATERAL


def _stored_sweep():
    return json.loads(gzip.decompress((WALK / "results" / "sweep_walk60" / "sweep_walk60.json.gz").read_bytes()))


def test_defaults_are_the_existing_tools():
    """No model given: the sweep driver's own env config. No delay points or push axis given: the sweep's own
    families (terrain aside) and points, with the same keys. The judged conditions are the judge's."""
    assert F.env_config(None) == SW.env_config()
    assert F.env_config(ROLL_ANKLE) == SW.env_config()
    mine = F.families()
    theirs = [f for f in S.FAMILIES if f[0] != "terrain"]
    assert mine == theirs
    _, p_mine = S.plan(families=mine)
    _, p_all = S.plan()
    assert set(p_mine) <= set(p_all)
    conds = F.judged_conditions()
    assert set(conds) == set(JW.JUDGED) | set(JW.REPORTED)
    for n, (lab, plant, imp) in JW.JUDGED.items():
        assert conds[n] == (lab, plant, imp, JW.COMMAND, 10)


def test_added_delay_points_keep_the_existing_ones():
    """--delays adds points; every point the sweep defines keeps its key, so its episodes are the same."""
    fams = F.families(delays=[0, 10, 16, 17, 18, 19, 20, 30, 40, 60])
    d = [f for f in fams if f[0] == "delay"]
    assert len(d) == 2 and all(f[3] == (0, 10, 16, 17, 18, 19, 20, 30, 40, 60) for f in d)
    fams = F.families(delays=[0, 2, 5, 10, 15, 16, 17, 18, 19, 20, 30, 40, 60])
    _, p_new = S.plan(families=fams)
    _, p_old = S.plan(families=F.families())
    assert set(p_old) <= set(p_new)


def test_forward_axis():
    yaw = math.radians(30.0)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(F.forward_axis(R.ravel()), [math.cos(yaw), math.sin(yaw), 0.0])
    assert np.allclose(F._ORIG_LATERAL(R.ravel()), [-math.sin(yaw), math.cos(yaw), 0.0])
    fams = F.families(push_axis="fore_aft")
    assert [f[0] for f in fams] == ["push"]
    assert fams[0][3] == [f for f in S.FAMILIES if f[0] == "push"][0][3]


def test_sweep_episode_reproduces_the_stored_one():
    """The driver's sweep path (sideways, the walk's own model) reproduces the stored +1.25 m/s push episode, seed
    200000, exactly, and adds the peak drive speed; the sideways worker leaves the sweep's lateral axis in place."""
    res = _stored_sweep()
    key = next(k for k, v in res["points"].items() if v["point"]["push_dv"] == 1.25)
    stored = next(e for e in res["points"][key]["episodes"] if e["seed"] == 200000)
    F._init_worker(F.env_config(None), "sideways")
    assert S.lateral_axis is F._ORIG_LATERAL
    _, rec = SW._task((key, S.make_point(push_dv=1.25), 200000))
    for k in ("fell", "forward_speed_mps", "sideways_drift_m", "heading_change_deg", "heat", "at_nominal_limit"):
        assert rec[k] == stored[k], k
    assert rec["push"]["lateral_peak_m"] == stored["push"]["lateral_peak_m"]
    assert len(rec["peak_rpm"]) == 12 and all(p >= q >= 0 for p, q in zip(rec["peak_rpm"], rec["p99_rpm"]))


def test_fore_aft_push_goes_along_the_body():
    """With push_axis fore_aft the same impulse goes along the chassis's forward axis: the push's direction at onset
    is the body's forward axis, and the robot's centre of mass speeds up forward against the same seed's unpushed
    episode."""
    F._init_worker(F.env_config(None), "fore_aft")
    try:
        env, ctrl = SW.env_and_controller(None)
        seen = {}
        orig = S.Pusher

        class Spy(orig):
            def after_step(self):
                super().after_step()
                if self.direction is not None and "dir" not in seen:
                    seen["dir"] = self.direction.copy()
                    seen["x"] = F.forward_axis(self.d.xmat[self.body])
                    seen["force"] = self.force.copy()
                    seen["mass"] = self.mass

        S.Pusher = Spy
        try:
            rec = S.run_episode(env, None, None, 200000, S.make_point(push_dv=0.5), controller=ctrl.act)
        finally:
            S.Pusher = orig
        assert np.allclose(seen["dir"], seen["x"])
        assert np.allclose(seen["force"], seen["mass"] * 0.5 / S.PUSH_S * seen["x"])
        assert not rec["fell"]
        base = S.run_episode(env, None, None, 200000, S.make_point(), controller=ctrl.act)
        moved = np.asarray(rec["push"]["_lat_left"]) - np.asarray(base["push"]["_lat_left"])
        assert moved[:30].min() > -1e-3 and moved[:30].max() > 0.005  # first 0.3 s: further along the push, m
        com_dv = rec["push"]["_com_v_left_after"] - base["push"]["_com_v_left_after"]
        assert 0.05 < com_dv < 0.5  # the centre of mass gains part of the 0.5 m/s free-body change, forward
    finally:
        F._init_worker(F.env_config(None), "sideways")
    assert S.lateral_axis is F._ORIG_LATERAL


def test_judged_episode_reproduces_the_stored_one():
    """The driver's judged path on the walk's own model reproduces the judged run's J1 episode, seed 101, exactly
    (results/walk60_2026-11-19/results.json); the contact-slip rerun has the same trajectory."""
    res = json.loads((WALK / "results" / "walk60_2026-11-19" / "results.json").read_text())
    stored = next(e for e in res["judged"]["J1"]["episodes"] if e["seed"] == 101)
    lab, plant, imp = JW.JUDGED["J1"]
    _, contact, r = F.judged_episode(("J1", 101, plant, imp, JW.COMMAND, 10, False))
    assert contact is False
    assert r["speed_mps"] == stored["speed_mps"]
    for k in ("forward_speed_mps", "sideways_drift_m", "heading_change_deg", "slip_ratio", "power_w"):
        assert r["measures"][k] == stored["measures"][k], k
    for nm, d in stored["measures"]["drives"].items():
        assert r["measures"]["drives"][nm] == d, nm
    assert len(r["drive_speed"]["peak_rpm"]) == 12
    _, contact, rc = F.judged_episode(("J1", 101, plant, imp, JW.COMMAND, 10, True))
    assert contact is True
    assert rc["measures"]["forward_speed_mps"] == r["measures"]["forward_speed_mps"]
    assert rc["drive_speed"] == r["drive_speed"]


def test_slow_gait_windows():
    """fore128_slow_gait.windows: forward speed along the start heading, the larger hind knee's RMS / 6.75 N m and
    the mean pitch over 2-8 s and 10-20 s, on a synthetic 1 ms log."""
    import fore128_slow_gait as SG

    t = np.arange(0.0, 20.0, 0.001)
    yaw0 = math.radians(20.0)
    v = np.where(t < 9.0, 0.25, 0.19)
    s = np.cumsum(v) * 0.001
    pos = np.stack([s * math.cos(yaw0), s * math.sin(yaw0), np.zeros_like(t)], axis=1)
    names = [f"{c}_{a}" for c in ("left_fore", "right_fore", "left_hind", "right_hind")
             for a in ("roll", "pitch", "flexion")]
    tau = np.zeros((len(t), 12))
    tau[:, names.index("left_hind_flexion")] = np.where(t < 9.0, 0.5, 1.05) * 6.75
    rp = np.stack([np.zeros_like(t), np.where(t < 9.0, 0.0, -1.3)], axis=1)
    out = SG.windows({"t": t, "pos": pos, "heading0": yaw0, "names": names, "tau": tau, "rp": rp})
    assert abs(out["speed_2_8_mps"] - 0.25) < 1e-3 and abs(out["speed_10_20_mps"] - 0.19) < 1e-3
    assert abs(out["hind_knee_rms_over_rating_2_8"] - 0.5) < 1e-3
    assert abs(out["hind_knee_rms_over_rating_10_20"] - 1.05) < 1e-3
    assert abs(out["pitch_10_20_deg"] + 1.3) < 1e-6 and out["pitch_2_8_deg"] == 0.0


RESULTS = WALK / "results" / "walk60_fore128_2026-11-27"


def test_fore_aft_push_rows_carry_no_recovery():
    """The sweep's recovery rule (speed along the push within 0.05 m/s) cannot be met walking forward, so fore-aft
    push rows set it to None with a note; sideways rows keep the sweep's own counts. The extra travel along the push
    is the sweep's lateral_extra_peak_m under an axis-neutral name."""
    for axis in ("fore_aft", "sideways"):
        res = F.read_gz(RESULTS / f"{axis}_200000" / f"sweep_walk60_fore128_{axis}_200000.json.gz")
        fam = next(f for f in res["families"] if f["sweep"] == "push")
        for row in fam["rows"]:
            ps = F.point_summary(res, row["key"])["push"]
            q = res["points"][row["key"]]["summary"]["push"]
            assert ps["extra_travel_along_push_m_mean"] == q["lateral_extra_peak_m_mean"]
            assert not any(k.startswith("lateral_") for k in ps)
            if axis == "fore_aft":
                assert ps["recovered"] is None and ps["recovery_s_median"] is None
                assert "0.05 m/s" in ps["recovery_note"]
            else:
                assert ps["recovered"] == q["recovered"] and "recovery_note" not in ps


_G = {"episodes": 10, "falls": 0, "slow_after": 3, "slow_hot_after": 2, "speed_2_8_range_mps": [0.25, 0.25],
      "speed_10_20_range_mps": [0.19, 0.25], "hind_knee_10_20_range": [0.6, 1.05], "pitch_10_20_mean_deg_slow": -1.3}


def _sweep_eps(res, keys):
    """Slow-gait-style episode rows (dv, seed, fell, the sweep's 2-20 s speed) for the sweep points ``keys``."""
    return [{"dv": res["points"][k]["point"]["push_dv"], "seed": e["seed"], "fell": e["fell"],
             "forward_speed_mps": e["forward_speed_mps"]} for k in keys for e in res["points"][k]["episodes"]]


def test_windows_attach_only_to_the_same_episodes():
    """A slow-gait record is put beside the sweep's push rows only for the same model, axis and seeds, and says
    whether its episodes are the sweep's: the same falls and the same 2-20 s speeds, seed by seed. A fall flipped or a
    speed changed on one seed, or a record without the sweep's speeds, reads as not the same episodes."""
    res = F.read_gz(RESULTS / "fore_aft_200000" / "sweep_walk60_fore128_fore_aft_200000.json.gz")
    fam = next(f for f in res["families"] if f["sweep"] == "push")
    eps = _sweep_eps(res, [r["key"] for r in fam["rows"]])
    sg = {"mode": "sweep", "model": res["env"]["model"], "push_axis": "fore_aft", "point": {}, "seeds": [200000, 200009],
          "summary": [{"dv": -0.3, **_G}], "episodes": eps}

    def attached(sg_):
        ss = F.sweep_summary(res)
        F.attach_sweep_windows(ss, res, [("x.json", sg_)])
        return {r["x"]: r for r in ss["families"][0]["rows"]}

    rows = attached(sg)
    w = rows[-0.3]["windows_10_20s"]
    assert w["same_episodes_as_sweep"] is True and w["slow_episodes"] == 3 and w["slow_hot_episodes"] == 2
    assert "windows_10_20s" not in rows[0.3]
    i = next(j for j, e in enumerate(eps) if e["dv"] == -0.3)
    for change in ({"fell": not eps[i]["fell"]}, {"forward_speed_mps": eps[i]["forward_speed_mps"] + 1e-9},
                   {"forward_speed_mps": None}):
        bad = [dict(e, **change) if j == i else e for j, e in enumerate(eps)]
        rows = attached({**sg, "episodes": bad})
        assert rows[-0.3]["windows_10_20s"]["same_episodes_as_sweep"] is False, change
    for other in ({"push_axis": "sideways"}, {"model": "walk60"}, {"seeds": [300000, 300009]}, {"mode": "judged"}):
        rows = attached({**sg, **other})
        assert not any("windows_10_20s" in r for r in rows.values()), other


def test_point_windows_attach_to_their_own_row():
    """A slow-gait record with a further point setting (kp x0.8, no push) goes beside the kp family's 0.8 row only,
    not beside the push family's unpushed row or any other row."""
    res = F.read_gz(RESULTS / "sideways_200000" / "sweep_walk60_fore128_sideways_200000.json.gz")
    key = next(k for k, v in res["points"].items() if v["point"]["kp_scale"] == 0.8)
    sg = {"mode": "sweep", "model": res["env"]["model"], "push_axis": "sideways", "point": {"kp_scale": 0.8},
          "seeds": [200000, 200009], "summary": [{"dv": 0.0, **_G}], "episodes": _sweep_eps(res, [key])}
    ss = F.sweep_summary(res)
    F.attach_sweep_windows(ss, res, [("kp.json", sg)])
    found = [(f["sweep"], f["x_key"], r["x"], r["windows_10_20s"]) for f in ss["families"] for r in f["rows"]
             if "windows_10_20s" in r]
    assert [(a, b, c) for a, b, c, _ in found] == [("drive", "kp_scale", 0.8)]
    assert found[0][3]["same_episodes_as_sweep"] is True and found[0][3]["source"] == "kp.json"
    nominal = next(k for k, v in res["points"].items() if v["point"]["kp_scale"] == 1.0
                   and v["point"]["push_dv"] == 0.0 and v["point"]["delay_ms"] == 10 and v["point"]["friction"] is None
                   and v["point"]["mass_scale"] == 1.0 and v["point"]["payload_kg"] == 0.0
                   and v["point"]["force_scale"] == 1.0 and v["point"]["command"] == 0.25)
    sg0 = {**sg, "point": {}, "episodes": _sweep_eps(res, [nominal])}
    ss0 = F.sweep_summary(res)
    F.attach_sweep_windows(ss0, res, [("push.json", sg0)])
    found0 = [(f["sweep"], r["x"]) for f in ss0["families"] for r in f["rows"] if "windows_10_20s" in r]
    assert found0 == [("push", 0.0)]


def test_judged_windows_check_both_speeds():
    """Judged windows are matched to the stored episodes on the whole-episode speed and the 2-20 s speed; either one
    off reads as not the same episodes."""
    raw = F.read_gz(RESULTS / "judged_101-110.json.gz")
    stored = raw["conditions"]["J1"]["episodes"]
    eps = [{"condition": "J1", "seed": e["seed"], "fell": e["fell"], "speed_mps": e["speed_mps"],
            "forward_speed_mps": e["measures"]["forward_speed_mps"]} for e in stored]
    sg = {"mode": "judged", "model": raw["model"], "seeds": [101, 110], "summary": [{"condition": "J1", **_G}],
          "episodes": eps}

    def same(sg_):
        js = {"conditions": {"J1": {}}}
        F.attach_judged_windows(js, raw, [("j.json", sg_)])
        return js["conditions"]["J1"]["windows_10_20s"]["same_episodes"]

    assert same(sg) is True
    for k in ("speed_mps", "forward_speed_mps"):
        assert same({**sg, "episodes": [dict(eps[0], **{k: eps[0][k] + 1e-9})] + eps[1:]}) is False, k


def test_rule_cells_name_each_rule():
    s = {"prereg_188mm_thresholds": {"W1_no_fall": True, "W2_mean_at_least_0.225_none_under_0.20": False,
                                     "W3_every_episode_under_1.0": False, "W4_slip_at_origin_at_most_0.25": True},
         "falls": 0, "speed_mean_mps": 0.2211, "speed_min_mps": 0.1932, "heat_worst_episode": {"rms_over_continuous": 1.104},
         "slip_ratio_mean_by_foot_at_origin": {"a": 0.13, "b": None}}
    cell = F._rules_cell("J3", s)
    assert cell == ("W1 met (0 falls); W2 **not met** (0.221, min 0.193); W3 **not met** (1.10); W4 met (0.13)")
    assert F._rules_cell("J1d17", s).startswith("orientation only: W1 met")
    assert F._rules_cell("R3", {}).startswith("none")


def test_committed_summaries_regenerate_exactly(tmp_path):
    """summary.json and RESULTS.md, here and in capsule188_check/, are what summarise writes from the committed raw
    files (and slow_gait/ for the main one)."""
    F.summarise(RESULTS, tmp_path / "main", RESULTS / "slow_gait")
    for n in ("summary.json", "RESULTS.md"):
        assert (tmp_path / "main" / n).read_bytes() == (RESULTS / n).read_bytes(), n
    c = RESULTS / "capsule188_check"
    F.summarise(c, tmp_path / "c188", RESULTS / "slow_gait")
    for n in ("summary.json", "RESULTS.md"):
        assert (tmp_path / "c188" / n).read_bytes() == (c / n).read_bytes(), n


def test_slow_gait_push_points_and_groups():
    import fore128_slow_gait as SG

    assert SG.PUSHES == list(next(f for f in S.FAMILIES if f[0] == "push")[3])
    rows = [{"fell": False, "speed_2_8_mps": 0.25, "speed_10_20_mps": v, "hind_knee_rms_over_rating_2_8": 0.6,
             "hind_knee_rms_over_rating_10_20": h, "pitch_2_8_deg": 0.0, "pitch_10_20_deg": p}
            for v, h, p in ((0.19, 1.05, -1.3), (0.25, 0.65, 0.1), (0.20, 1.03, -1.2), (0.215, 0.75, 0.8),
                            (0.24, 1.10, -1.0))]
    rows.append({"fell": True, "speed_10_20_mps": None})
    g = SG.group_summary(rows)
    # slow: under 0.225 m/s (3); slow and hot: also the hind knee at or over 1.0 (2); fast but hot is neither
    assert g["episodes"] == 6 and g["falls"] == 1 and g["slow_after"] == 3 and g["slow_hot_after"] == 2
    assert g["speed_10_20_range_mps_slow"] == [0.19, 0.215] and g["hind_knee_10_20_range_slow"] == [0.75, 1.05]
    assert abs(g["pitch_10_20_mean_deg_slow"] - (-1.3 - 1.2 + 0.8) / 3) < 1e-9
    assert g["pitch_10_20_range_deg_slow"] == [-1.3, 0.8]
    assert g["speed_10_20_range_mps_slow_hot"] == [0.19, 0.20] and g["hind_knee_10_20_range_slow_hot"] == [1.03, 1.05]
    assert abs(g["pitch_10_20_mean_deg_slow_hot"] + 1.25) < 1e-9


def test_slow_gait_episodes_are_the_stored_ones():
    """fore128_slow_gait.py's episodes are the stored ones: a sweep-mode episode (fore-aft push -0.3 m/s, seed 200000)
    has the sweep's fall and 2-20 s speed, and a judged-mode episode (J1, seed 101) has the judged run's whole-episode
    and 2-20 s speeds, all on the adopted model."""
    import fore128_slow_gait as SG

    res = F.read_gz(RESULTS / "fore_aft_200000" / "sweep_walk60_fore128_fore_aft_200000.json.gz")
    key = next(k for k, v in res["points"].items() if v["point"]["push_dv"] == -0.3)
    stored = next(e for e in res["points"][key]["episodes"] if e["seed"] == 200000)
    SG._init(F.MODEL, "fore_aft")
    try:
        r = SG._job((-0.3, 200000, {}))
    finally:
        F._init_worker(F.env_config(None), "sideways")
    assert r["fell"] == stored["fell"] and r["forward_speed_mps"] == stored["forward_speed_mps"]
    assert r["speed_10_20_mps"] is not None
    raw = F.read_gz(RESULTS / "judged_101-110.json.gz")
    st = next(e for e in raw["conditions"]["J1"]["episodes"] if e["seed"] == 101)
    lab, plant, imp = JW.JUDGED["J1"]
    rj = SG._judged_job(("J1", 101, {**plant, "model_path": str(F.GEN / F.MODEL)}, imp, JW.COMMAND, 10))
    assert rj["speed_mps"] == st["speed_mps"] and rj["forward_speed_mps"] == st["measures"]["forward_speed_mps"]
