"""Tests of pilot4_mech.py (pytest; walking pilot 4 scratch, never merged).

Pure functions on synthetic gaits and ledgers (the symmetry index, the boxcar and counter-turn, t_start, both
debounces, the per-touchdown block, the per-stride range, the ledger's leg-work ratio, the leg power), the refusals,
and pins of the constants against the gait-report lanes' recorded outputs when those files are on disk; the frozen
probe's debounce and _events are loaded from the pilot-4 tree (stdlib + numpy only at import) and compared with the
copies here. Nothing here rolls an env, imports the repository or trains.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot4_mech as mm  # noqa: E402

SP = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad")
G = SP / "work/gait_report"
V3 = SP / "work/pilot4/postfix/fix/criteria_pilot4_draft_v3.txt"
PROBE = TOOLS / "tree" / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"


def load_probe():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("gait_probe_frozen_test", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- small helpers


def test_si_med_and_boxcar():
    assert mm.si(1.6, 3.1) == pytest.approx((1.6 - 3.1) / 2.35)
    assert math.isnan(mm.si(0.0, 0.0)) and math.isnan(mm.si(float("nan"), 1.0))
    assert mm.med([1.0, float("nan"), 3.0]) == 2.0 and math.isnan(mm.med([]))
    assert np.allclose(mm.moving_avg(np.arange(5.0), 2), [0.5, 1.5, 2.5, 3.5])


def test_counter_turn_is_zero_for_a_monotonic_veer_and_reads_a_reversal():
    assert mm.counter_turn(-2.5 * np.arange(500) * 0.01) == 0.0
    # right 10 deg, back left 6, right again 20: the swing against the net (right) turn is 6
    psi = np.concatenate([np.linspace(0, -10, 100), np.linspace(-10, -4, 100), np.linspace(-4, -24, 100)])
    assert mm.counter_turn(psi) == pytest.approx(6.0)
    assert math.isnan(mm.counter_turn([1.0]))


def test_t_start_on_a_constant_and_a_ramped_walk():
    dt = 0.01
    t = np.arange(1001) * dt
    const = np.stack([2.0 * t, 0 * t, 0 * t], axis=1)
    assert mm.t_start(const, dt) == pytest.approx(1.0)
    ramp_v = np.minimum(t / 3.0, 1.0) * 2.0  # reaches full speed at 3 s
    ramp = np.stack([np.concatenate([[0.0], np.cumsum(ramp_v[:-1] * dt)]), 0 * t, 0 * t], axis=1)
    ts = mm.t_start(ramp, dt)
    assert 2.4 < ts < 3.0
    assert math.isnan(mm.t_start(np.zeros((150, 3)), dt))  # never two seconds left to compare


def test_the_two_debounces_differ_where_the_lanes_differ():
    x = np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)  # a 1-step gap followed by a 1-step blip
    gaps_first = mm.debounce_gaps_first(x)
    blips_first = mm.debounce_blips_first(x)
    assert gaps_first.tolist() == [1, 1, 1, 1, 0, 0, 1, 1]
    assert blips_first.tolist() == [1, 1, 0, 0, 0, 0, 1, 1]


@pytest.mark.skipif(not PROBE.is_file(), reason="no pilot-4 tree")
def test_the_copies_equal_the_frozen_probes_debounce_and_events():
    gp = load_probe()
    rng = np.random.default_rng(3)
    for _ in range(200):
        x = rng.random(80) < 0.5
        assert np.array_equal(gp.debounce(x), mm.debounce_gaps_first(x))
        d = gp.debounce(x)
        a, b = gp._events(d)
        c, e = mm.events(d)
        assert np.array_equal(a, c) and np.array_equal(b, e)


# ---------------------------------------------------------------- a synthetic two-footed gait


def synthetic_episode(T=1000, period=40, scuff_every=0, dt=0.01, bw=840.0):
    """Feet r (index 0) and l (index 1) alternate: each down for 20 of every 40 steps, l half a period later. The
    right foot lands at 3.0 BW, the left at 1.5 BW (on the touchdown step only); the pelvis moves +x at 2 m/s; the
    touching foot is 0.7 m (r) or 0.35 m (l) ahead of the other. Optionally a 1-step right-foot touch mid-swing."""
    frac = np.zeros((T, 2))
    fmax = np.zeros((T, 2))
    fmean = np.zeros((T, 2))
    for k in range(T):
        for f, off in ((0, 0), (1, period // 2)):
            ph = (k - off) % period
            if ph < period // 2:
                frac[k, f] = 1.0
                fmean[k, f] = 0.5 * bw
                if ph == 0:
                    fmax[k, f] = (3.0 if f == 0 else 1.5) * bw
                else:
                    fmax[k, f] = 0.6 * bw
        if scuff_every and k % period == 30:
            frac[k, 0] = 0.2
            fmax[k, 0] = 0.8 * bw
    x = 2.0 * np.arange(T + 1) * dt
    root = np.stack([x, np.zeros(T + 1), np.full(T + 1, 0.9)], axis=1)
    feet = np.zeros((T + 1, 2, 3))
    for k in range(1, T + 1):
        s = k - 1  # the state after step s; right step = 0.35 + 0.35, left step = 0.175 + 0.175
        feet[k, 0, 0] = x[k] + (0.35 if (s % period) < period // 2 else -0.175)
        feet[k, 1, 0] = x[k] + (0.175 if ((s - period // 2) % period) < period // 2 else -0.35)
    feet[0] = feet[1]
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (T + 1, 1))
    return {"length": T, "root_pos": root, "quat": quat, "foot_pos": feet, "foot_h": np.zeros((T + 1, 2)),
            "floor_contact_substep_frac": frac, "ffoot_max": fmax, "ffoot_mean": fmean}


def test_per_touchdown_metrics_on_a_synthetic_gait():
    E = synthetic_episode()
    r = mm.per_touchdown_metrics(E, iL=1, iR=0, dt=0.01, bw=840.0)
    assert r["land_peak_bw_R"] == pytest.approx(3.0) and r["land_peak_bw_L"] == pytest.approx(1.5)
    assert r["land_peak_bw_SI"] == pytest.approx((1.5 - 3.0) / 2.25)
    assert r["stance_s_L"] == pytest.approx(0.20) and r["stance_s_R"] == pytest.approx(0.20)
    assert r["step_len_R"] == pytest.approx(0.70, abs=1e-6) and r["step_len_L"] == pytest.approx(0.35, abs=1e-6)
    assert r["scuff_share_R"] == 0.0 and r["scuff_touched_R"] == 0 and r["scuff_swings_R"] > 20


def test_a_mid_swing_touch_is_a_scuff_and_an_edge_touch_is_not():
    E = synthetic_episode(scuff_every=1)
    r = mm.per_touchdown_metrics(E, iL=1, iR=0, dt=0.01, bw=840.0)
    assert r["scuff_share_R"] == 1.0 and r["scuff_share_L"] == 0.0
    E2 = synthetic_episode()
    frac = E2["floor_contact_substep_frac"]
    for k in range(len(frac)):
        if k % 40 == 21:  # one step after the right lift-off: inside the 2-step edge, not a scuff
            frac[k, 0] = 0.2
    assert mm.per_touchdown_metrics(E2, iL=1, iR=0, dt=0.01, bw=840.0)["scuff_share_R"] == 0.0


def test_stride_rom_reads_each_sides_range_per_own_stride():
    T, period = 1000, 40
    frac = synthetic_episode(T=T, period=period)["floor_contact_substep_frac"]
    k = np.arange(T)
    q = {"l_knee": 6.0 * np.sin(2 * np.pi * k / period), "r_knee": 14.0 * np.sin(2 * np.pi * k / period),
         "l_ankle": np.zeros(T), "r_ankle": np.zeros(T), "l_hip_pitch": 5.0 * np.cos(2 * np.pi * k / period),
         "r_hip_pitch": 5.0 * np.cos(2 * np.pi * k / period)}
    r = mm.stride_rom(frac, q, iL=1, iR=0)
    assert r["knee_rom_L_deg"] == pytest.approx(12.0, abs=0.1) and r["knee_rom_R_deg"] == pytest.approx(28.0, abs=0.2)
    assert r["knee_rom_ratio"] == pytest.approx(12.0 / 28.0, abs=0.01)
    assert math.isnan(r["ankle_rom_ratio"]) and r["hip_pitch_rom_ratio"] == pytest.approx(1.0)


def test_stride_rom_reads_only_strides_from_touchdowns_at_step_100_on():
    T, period = 200, 40
    frac = synthetic_episode(T=T, period=period)["floor_contact_substep_frac"]
    k = np.arange(T)
    amp = np.where(k < 100, 10.0, 6.0)  # the start-up strides move the knee further
    q = {n: amp * np.sin(2 * np.pi * k / period) for n in ("l_knee", "r_knee", "l_ankle", "r_ankle", "l_hip_pitch",
                                                           "r_hip_pitch")}
    r = mm.stride_rom(frac, q, iL=1, iR=0)
    assert r["knee_rom_R_deg"] == pytest.approx(12.0, abs=0.1)  # right touchdowns at 120, 160: one stride


def test_weave_reads_no_t_start_as_not_run():
    out = mm.weave(np.zeros((150, 3)), np.tile([1.0, 0, 0, 0], (150, 1)), 0.01)
    assert out["t_start_s"] is None and math.isnan(out["counter_turn_deg"]) and "t_start" in out["why_not"]


# ---------------------------------------------------------------- the ledger's leg-work ratio


def write_ledger(path: Path, rows: list[dict], header=None):
    header = header or ["evaluation", "timesteps", "episode", "full_horizon", "cause", "r_leg_work_abs_j",
                        "l_leg_work_abs_j"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def ledger_rows(steps, ratio_of, fh=lambda s, e: 1):
    rows = []
    for i, s in enumerate(steps):
        for e in range(10):
            rows.append({"evaluation": i, "timesteps": s, "episode": e, "full_horizon": fh(s, e), "cause": "",
                         "r_leg_work_abs_j": 1000.0 * ratio_of(s, e), "l_leg_work_abs_j": 1000.0})
    return rows


def test_ledger_window_is_the_last_six_at_or_before_the_checkpoint(tmp_path):
    steps = [65536 * k for k in range(1, 31)]
    p = tmp_path / "eval_episodes.csv"
    write_ledger(p, ledger_rows(steps, lambda s, e: 2.0 if s <= 458752 else 0.5))
    h, rows = mm.read_ledger(p)
    at_500k = mm.ledger_asymmetry(h, rows, through=500_000)
    assert at_500k["evaluations"] == 6 and at_500k["from_timesteps"] == 131072 and at_500k["to_timesteps"] == 458752
    assert at_500k["r_over_l_median"] == 2.0 and at_500k["max_over_min"] == 2.0 and at_500k["larger_side"] == "right"
    at_2m = mm.ledger_asymmetry(h, rows, through=2_000_000)
    assert at_2m["to_timesteps"] == 1966080 and at_2m["r_over_l_median"] == 0.5 and at_2m["max_over_min"] == 2.0
    assert at_2m["larger_side"] == "left" and at_2m["left_larger"] == 60


def test_ledger_reads_full_horizon_then_cause_and_skips_bad_rows(tmp_path):
    steps = [65536, 131072]
    rows = ledger_rows(steps, lambda s, e: 3.0 if e < 5 else 1.5, fh=lambda s, e: 1 if e < 5 else 0)
    p = tmp_path / "a.csv"
    write_ledger(p, rows)
    h, r = mm.read_ledger(p)
    assert mm.ledger_asymmetry(h, r, through=10 ** 9)["r_over_l_median"] == 3.0  # only full-horizon episodes
    for row in rows:
        row["full_horizon"] = ""
        row["cause"] = "truncated" if row["episode"] < 5 else "terminated"
    write_ledger(p, rows)
    h, r = mm.read_ledger(p)
    assert mm.ledger_asymmetry(h, r, through=10 ** 9)["r_over_l_median"] == 3.0  # cause as the fallback
    rows[0]["l_leg_work_abs_j"] = 0.0  # left 0: the ratio is NaN and left out
    write_ledger(p, rows)
    h, r = mm.read_ledger(p)
    assert mm.ledger_asymmetry(h, r, through=10 ** 9)["episodes"] == 9


def test_ledger_not_run_reasons(tmp_path):
    p = tmp_path / "b.csv"
    write_ledger(p, [{"timesteps": 65536}], header=["timesteps", "r_leg_work_abs_j"])
    h, r = mm.read_ledger(p)
    assert "lacks" in mm.ledger_asymmetry(h, r, through=10 ** 9)["why_not"]
    write_ledger(p, ledger_rows([65536], lambda s, e: 1.0))
    h, r = mm.read_ledger(p)
    assert "no evaluation at or before" in mm.ledger_asymmetry(h, r, through=1000)["why_not"]


def test_leg_power_sums_each_sides_medians():
    names = [f"{s}_{j}_act" for s in "rl" for j in mm.LEG_JOINTS] + ["neck_pitch_act"]
    eps = [{n: (10.0 if n.startswith("r_") else 4.0) for n in names}, {n: (12.0 if n.startswith("r_") else 6.0)
                                                                       for n in names}]
    lp = mm.leg_power(eps, names)
    assert lp["R_W"] == pytest.approx(44.0) and lp["L_W"] == pytest.approx(20.0)
    assert lp["max_over_min"] == pytest.approx(2.2) and lp["larger_side"] == "right"
    assert mm.leg_power([], names)["read"] is False


# ---------------------------------------------------------------- refusals


def test_output_cell_and_parent_refusals(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    run = tmp_path / "run"
    (run / "03_locomotion").mkdir(parents=True)
    (run / "provenance.json").write_text("{}")
    led = run / "03_locomotion" / "eval_episodes.csv"
    assert mm.output_problems(tmp_path / "o.json", tree=tree, inputs=[led], run_copy=str(run)) == []
    assert any("run directory" in m for m in mm.output_problems(run / "o.json", tree=None, inputs=[led],
                                                                 run_copy=None))
    assert any("inside the tree" in m for m in mm.output_problems(tree / "o.json", tree=tree, inputs=[],
                                                                   run_copy=None))
    b = {"schema": mm.PILOT3_CELL_SCHEMA, "head": "h"}
    assert any("pilot 3's" in m for m in mm.cell_problems(b, head="h", dirty="", validation=False))
    assert mm.cell_problems(b, head="h", dirty="", validation=True) == []
    z, p = mm.parent_paths(tmp_path, 60)
    assert all("missing" in m for m in mm.parent_problems(60, z, p))


def test_existing_json_is_refused_before_the_tree_is_imported(tmp_path, monkeypatch):
    out = tmp_path / "m.json"
    out.write_text("{}")
    monkeypatch.chdir(tmp_path)
    before = "environments" in sys.modules
    assert mm.main(["--parent", "42", "--json", str(out)]) == 2
    assert mm.main(["--parent", "42", "--json", str(tmp_path / "n.json"), "--episodes", "41"]) == 2
    assert ("environments" in sys.modules) == before and not (tmp_path / "n.json").exists()


# ---------------------------------------------------------------- pins against the lanes' records


@pytest.mark.skipif(not (G / "fix/out/stride_check.txt").is_file(), reason="gait-report lanes not on disk")
def test_the_references_are_the_lanes_records():
    st = (G / "fix/out/stride_check.txt").read_text()
    for seed in (42, 52, 60):
        block = st.split(f"== pol_s{seed}_8M")[1].split("==")[0]
        L, R = (float(x) for x in block.split("knee per-stride ROM deg: L ")[1].split("\n")[0].split("  R "))
        assert (L, R) == mm.PILOT3_REFERENCE["knee_rom_L_R_deg"][seed]
        ratio = float(block.split("rom_knee")[1].split("med")[1].split()[0])
        assert round(ratio, 3) == mm.PILOT3_REFERENCE["knee_rom_ratio"][seed]
    led = (G / "phase1/fix/pilot3_final/ledger_summary.txt").read_text()
    for seed in (42, 52, 60):
        assert f"value {mm.PILOT3_REFERENCE['ledger_leg_work_max_over_min'][seed]}" in led
    sd = (G / "survey_metrics/out/scuff_detail.txt").read_text()
    s52 = json.loads(sd.split("pol_s52_8M ")[1].split("\n")[0])
    assert (s52["R"]["touched"], s52["R"]["swings"]) == mm.PILOT3_REFERENCE["scuff_R_pooled_touched_of_swings"][52]
    lp = json.loads((G / "plan/out/leg_power.json").read_text())
    for seed in (42, 52, 60):
        assert lp[f"pol_s{seed}_8M"]["max_over_min"] == mm.PILOT3_REFERENCE["leg_power_max_over_min"][seed]


@pytest.mark.skipif(not V3.is_file(), reason="v3 not on disk")
def test_the_parent_table_is_v3s():
    text = V3.read_text()
    for seed, (z, p) in mm.PARENTS.items():
        assert z.split(":", 1)[1] in text and p.split(":", 1)[1] in text, seed


def test_constants_are_the_lanes():
    assert (mm.SETTLE, mm.LANDING_STEPS, mm.SCUFF_EDGE_STEPS, mm.TRAVEL_HALF_STEPS) == (100, 5, 2, 25)
    assert mm.MOVAVG_SAMPLES == 100 and mm.LEDGER_EVALUATIONS == 6 and mm.PANEL_EPISODES == 40
    assert mm.LEG_JOINTS == ("hip_pitch", "hip_roll", "knee", "ankle")
