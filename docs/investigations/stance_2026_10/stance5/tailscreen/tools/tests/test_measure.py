"""Synthetic tests for measure_tailscreen.py's reductions (build tier: numpy only; no trace run, no training).

A fake trace folder (episode_trace.json + 20 per-step CSVs in the trace tool's layout) exercises the per-episode flags
(TAIL-UP, OTHER by saturation with actuator and sign, OTHER by short episode, falls, the 0.10 bar being strict, an
episode that ends before the window, OTHER on a tail-up episode, OTHER by tail_3's lower side), the episode-level
independence of OTHER from TAIL-UP, the panel aggregates, the panels.csv lock/append (with the under-lock attempt
check), the crash rule's retirement and the gzip step.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("measure_tailscreen")
NAMES = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
         "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
         "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]
T3, KNEE = 14, 5
SETTLE = 200


def _episode(seed, length, cmd, terminated):
    raw = cmd.copy()
    jp = 0.1 * cmd
    w = slice(SETTLE, length)
    wc = np.abs(cmd[w])
    acts = {}
    sat = (wc >= 0.99).mean(axis=0) if wc.shape[0] else None
    for i, n in enumerate(NAMES):
        if wc.shape[0]:
            acts[n] = {"index": i, "command_window_mean": float(cmd[w, i].mean()),
                       "raw_mean_window_mean": float(raw[w, i].mean()), "saturated_fraction": float(sat[i]),
                       "near_saturated_fraction": float((wc[:, i] >= 0.9).mean()),
                       "raw_mean_max": float(raw[w, i].max()), "joint_pos_window_mean": float(jp[w, i].mean())}
        else:
            acts[n] = {k: None for k in ("command_window_mean", "raw_mean_window_mean", "saturated_fraction",
                                         "near_saturated_fraction", "raw_mean_max", "joint_pos_window_mean")}
            acts[n]["index"] = i
    ep = {"seed": seed, "length": length, "return": float(length), "terminated": terminated,
          "truncated": not terminated, "max_actuator_saturation_fraction": float(sat.max()) if sat is not None else None,
          "actuators": acts}
    rows = []
    for s in range(length):
        r = {"step": s, "time_s": (s + 1) * 0.01, "in_window": int(s >= SETTLE), "reward": 1.0}
        for i, n in enumerate(NAMES):
            r[f"{n}_raw_mean"] = float(raw[s, i])
            r[f"{n}_command"] = float(cmd[s, i])
            r[f"{n}_joint_pos"] = float(jp[s, i])
        rows.append(r)
    return ep, rows


def make_trace(d: Path, edit=None):
    d.mkdir(parents=True)
    eps = []
    for k, seed in enumerate(range(3042, 3062)):
        L = 1000
        cmd = np.zeros((L, 15))
        cmd[:, T3] = 0.5
        term = False
        if k == 0:  # tail-up: tail_3 at +1 over the whole window
            cmd[:, T3] = 1.0
        elif k == 1:  # saturation elsewhere: r_knee at -1 on half the window
            cmd[SETTLE:600, KNEE] = -1.0
        elif k == 2:  # a fall at 500
            L, term = 500, True
        elif k == 3:  # ends before the window
            L, term = 150, True
        elif k == 4:  # exactly 10% of window steps at +0.995: not above the bar
            cmd[SETTLE:SETTLE + 80, T3] = 0.995
        elif k == 5:  # tail-up AND short: a TAIL-UP flag and an OTHER flag (short)
            cmd[:, T3] = 1.0
            L, term = 900, True
        elif k == 6:  # tail-up AND r_knee at -1 on half the window: TAIL-UP and OTHER (saturation)
            cmd[:, T3] = 1.0
            cmd[SETTLE:600, KNEE] = -1.0
        elif k == 7:  # tail_3 pinned DOWN: not tail-up; OTHER by tail_3's lower side
            cmd[:, T3] = -1.0
        if edit is not None:
            edit(k, cmd)
        ep, rows = _episode(seed, L, cmd[:L], term)
        eps.append(ep)
        with open(d / f"episode_trace_seed{seed}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    report = {"schema": "mesozoic.episode-trace/v1", "actuator_names": NAMES, "traced_actuators": NAMES,
              "deterministic": True, "stochastic_seed": None, "holds": [], "checkpoint_plant_validated": True,
              "seeds": list(range(3042, 3062)), "settle_steps": SETTLE, "horizon": 1000, "saturation_abs": 0.99,
              "near_saturation_abs": 0.9, "episodes": eps}
    (d / "episode_trace.json").write_text(json.dumps(report))
    return d


def test_flags_and_aggregates(tmp_path):
    d = make_trace(tmp_path / "panel")
    agg, flags, problems = M.reduce_panel(d)
    assert problems == []
    by = {f["seed"]: f for f in flags}
    assert by[3042]["tailup_flag"] == 1 and by[3042]["other_flag"] == 0
    assert by[3042]["gate_actuator"] == "tail_3_pitch_act" and by[3042]["gate_sign"] == "+"
    assert by[3043]["tailup_flag"] == 0 and by[3043]["other_flag"] == 1
    assert by[3043]["other_reason"] == "saturation:r_knee_act-" and by[3043]["gate_stat"] == 0.5
    assert by[3044]["other_reason"] == "short:500" and by[3044]["fall"] == 1
    assert by[3045]["other_reason"] == "short:150" and by[3045]["gate_stat"] is None
    assert by[3045]["tail3_cmd_window_mean"] is None
    assert by[3046]["tailup_flag"] == 0 and by[3046]["tail3_share_cmd_ge_0p99"] == pytest.approx(0.1)
    assert by[3046]["other_flag"] == 0  # 0.10 is not above the bar
    assert by[3042]["other_stat"] == 0.0 and by[3042]["other_actuator"] == ""  # tail_3's upper side never counts
    assert by[3047]["tailup_flag"] == 1 and by[3047]["other_flag"] == 1 and by[3047]["fall"] == 1
    assert by[3047]["other_reason"] == "short:900"
    assert by[3048]["tailup_flag"] == 1 and by[3048]["other_flag"] == 1
    assert by[3048]["gate_actuator"] == "tail_3_pitch_act" and by[3048]["gate_stat"] == 1.0
    assert by[3048]["other_reason"] == "saturation:r_knee_act-" and by[3048]["other_stat"] == 0.5
    assert by[3049]["tailup_flag"] == 0 and by[3049]["other_flag"] == 1
    assert by[3049]["other_reason"] == "saturation:tail_3_pitch_act-" and by[3049]["other_stat"] == 1.0
    assert agg["tailup_flags"] == 3 and agg["other_flags"] == 6 and agg["falls"] == 3
    assert agg["short_episodes"] == 3
    assert agg["other_reasons"] == ("3043:saturation:r_knee_act-;3044:short:500;3045:short:150;3047:short:900;"
                                    "3048:saturation:r_knee_act-;3049:saturation:tail_3_pitch_act-")
    assert agg["class_single"] == "RAIL"
    # CMD: mean over the 19 episodes that reach the window of each episode's window-mean tail_3 command
    exp = [1.0, 0.5, 0.5, (80 * 0.995 + 720 * 0.5) / 800, 1.0, 1.0, -1.0] + [0.5] * 12
    assert agg["cmd"] == pytest.approx(np.mean(exp), abs=1e-12)
    assert agg["others_max_frac99"] == pytest.approx(1.0 / 19, abs=1e-12)
    # pinned = |tail_3 command| >= 0.99, pooled over steps: 800 + 700 + 800 steps at +1.0, 800 at -1.0, 80 at 0.995
    assert agg["pinned_step_raw_mean"] == pytest.approx((2300 - 800 + 80 * 0.995) / 3180, abs=1e-12)
    assert agg["n_episodes"] == 20 and agg["seeds"] == "3042-3061"


def test_other_flag_never_reads_the_tail_up_flag(tmp_path):
    """Episode-level independence: putting tail_3 at +1 on every window step where it is not at its lower rail turns
    episodes into TAIL-UP episodes and leaves every OTHER flag, statistic and reason as it was."""
    _, base, problems = M.reduce_panel(make_trace(tmp_path / "base"))
    assert problems == []

    def tail_up(k, cmd):
        free = cmd[SETTLE:, T3] > -0.99
        cmd[SETTLE:, T3][free] = 1.0

    _, raised, problems = M.reduce_panel(make_trace(tmp_path / "raised", edit=tail_up))
    assert problems == []
    assert sum(f["tailup_flag"] for f in raised) > sum(f["tailup_flag"] for f in base)
    for b, r in zip(base, raised, strict=True):
        assert (b["other_flag"], b["other_stat"], b["other_reason"]) == (r["other_flag"], r["other_stat"],
                                                                         r["other_reason"]), b["seed"]


def test_class_single():
    assert M.class_single(1, 0.1) == "RAIL"
    assert M.class_single(0, 0.60) == "PARKED"
    assert M.class_single(0, 0.5999) == "LOW"
    assert M.class_single(0, None) == "LOW"


def test_reduce_refuses_wrong_seeds_or_probe(tmp_path):
    d = make_trace(tmp_path / "p")
    rep = json.loads((d / "episode_trace.json").read_text())
    rep["holds"] = [{"name": "tail_3_pitch_act", "value": 0.0}]
    rep["seeds"] = rep["seeds"][:-1] + [4000]
    (d / "episode_trace.json").write_text(json.dumps(rep))
    _, _, problems = M.reduce_panel(d)
    assert any("deterministic" in p for p in problems) and any("seeds" in p for p in problems)


def test_append_row_lock_header_and_duplicate(tmp_path):
    p = tmp_path / "panels.csv"
    row = {k: "x" for k in M.PANEL_COLUMNS}
    row.update(name="s42_b60_r101", label="final")
    M.append_row(p, row)
    row2 = dict(row, label="5111808")
    M.append_row(p, row2)
    lines = p.read_text().splitlines()
    assert lines[0] == ",".join(M.PANEL_COLUMNS) and len(lines) == 3
    with pytest.raises(SystemExit):
        M.append_row(p, row)
    assert len(p.read_text().splitlines()) == 3


def test_append_row_verify_runs_under_the_lock_and_refuses(tmp_path):
    p = tmp_path / "panels.csv"
    row = {k: "x" for k in M.PANEL_COLUMNS}
    row.update(name="s42_b60_r101", label="final")

    def gone():
        raise SystemExit(4)

    with pytest.raises(SystemExit):
        M.append_row(p, row, verify=gone)
    assert not p.exists() or p.read_text() == ""
    M.append_row(p, row, verify=lambda: None)
    assert len(p.read_text().splitlines()) == 2


def test_retire_crashed_moves_run_panels_and_rows(tmp_path):
    runs, panels, csv_path = tmp_path / "runs", tmp_path / "panels", tmp_path / "panels.csv"
    name = "s80_control_r202"
    (runs / name).mkdir(parents=True)
    (runs / name / "stage1_5111808_steps.zip").write_text("attempt 1")
    (panels / name / "5111808").mkdir(parents=True)
    (panels / name / "5111808" / "flags.csv").write_text("x")
    other = "s80_b60_r202"
    for n, lb in ((name, "4325376"), (other, "4325376"), (name, "5111808")):
        row = {k: "x" for k in M.PANEL_COLUMNS}
        row.update(name=n, label=lb)
        M.append_row(csv_path, row)
    out = M.retire_crashed(name, runs=runs, panels_dir=panels, panels_csv=csv_path)
    assert out["attempt"] == f"{name}_crashed1" and sorted(out["rows_moved"]) == ["4325376", "5111808"]
    assert not (runs / name).exists() and (runs / f"{name}_crashed1" / "stage1_5111808_steps.zip").is_file()
    assert not (panels / name).exists() and (panels / f"{name}_crashed1" / "5111808" / "flags.csv").is_file()
    rows = M.read_panels(csv_path)
    assert [(r["name"], r["label"]) for r in rows] == [(other, "4325376")]
    with open(tmp_path / M.CRASHED_CSV_NAME, newline="") as fh:
        ledger = list(csv.DictReader(fh))
    assert [(r["attempt"], r["label"]) for r in ledger] == [(f"{name}_crashed1", "4325376"),
                                                            (f"{name}_crashed1", "5111808")]
    # The rerun's labels are pending again (no row), and a second crash takes k = 2 with no panel folder.
    (runs / name).mkdir()
    out2 = M.retire_crashed(name, runs=runs, panels_dir=panels, panels_csv=csv_path)
    assert out2["attempt"] == f"{name}_crashed2" and out2["rows_moved"] == [] and out2["panel_dir"] is None
    with pytest.raises(SystemExit):
        M.retire_crashed(name, runs=runs, panels_dir=panels, panels_csv=csv_path)  # nothing left to retire


def test_gzip_records_sha_first(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    (d / "episode_trace_seed3042.csv").write_text("a,b\n1,2\n")
    want = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    M.gzip_csvs(d)
    assert not (d / "episode_trace_seed3042.csv").exists()
    assert (d / "csv_sha256.txt").read_text() == f"{want}  episode_trace_seed3042.csv\n"
    with gzip.open(d / "episode_trace_seed3042.csv.gz", "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == want


def test_tree_digest_sees_any_change(tmp_path):
    r = tmp_path / "copy"
    (r / "a").mkdir(parents=True)
    (r / "a" / "x.py").write_text("1")
    (r / "TREE_DIGEST.txt").write_text("ignored")
    d0 = M.tree_digest(r)
    (r / "TREE_DIGEST.txt").write_text("still ignored")
    assert M.tree_digest(r) == d0
    (r / "a" / "x.py").write_text("2")
    assert M.tree_digest(r) != d0
    (r / "a" / "x.py").write_text("1")
    (r / "a" / "__pycache__").mkdir()
    (r / "a" / "__pycache__" / "x.pyc").write_text("")
    assert M.tree_digest(r) != d0


def test_parse_name_and_labels():
    assert M.parse_name("s80_b90_r303") == ("s80", "b90", 303)
    for bad in ("s80_b90_r304", "s81_b90_r101", "s42_bound_r101", "start_s42"):
        with pytest.raises(SystemExit):
            M.parse_name(bad)
    assert len(M.all_names()) == 18
    assert M.checkpoint_pair("s42_b60_r101", "final")[1].name == "stage1_final_vecnorm.pkl"
    assert M.checkpoint_pair("s42_b60_r101", "5111808")[0].name == "stage1_5111808_steps.zip"


def test_measure_imports_no_torch_or_environments():
    import subprocess

    code = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('x','{TOOLS / 'measure_tailscreen.py'}');"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print(any(k=='torch' or k.startswith('environments') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
