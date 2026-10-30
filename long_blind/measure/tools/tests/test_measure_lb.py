"""Unit tests of measure_lb.py's own glue (the rules themselves are the round's pinned code, validated on the round's
three runs by validate_round.py).  Run with --basetemp inside LB/measure/work (nothing is written outside LB):

    PY -I -B -m pytest -q -p no:cacheprovider --basetemp LB/measure/work/pytest_tmp LB/measure/tools/tests
"""
import gzip
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_lb", TOOLS / "measure_lb.py")
ML = importlib.util.module_from_spec(spec)
sys.modules["measure_lb"] = ML
spec.loader.exec_module(ML)
C, LG, CP, SC = ML.lane_b()
G = 131072
S1 = ML.S1


@pytest.fixture(autouse=True)
def _inside_lb(tmp_path):
    assert str(tmp_path).startswith(ML.MEASURE + "/work"), f"pytest's basetemp must be under LB/measure/work: {tmp_path}"


def make_run(root: Path, E: int, *, status="ended", zip_n=None, stop_reason="end", hop=False, suffix="", meta=True,
             S=2097152):
    s = root / "01_stance"
    (s / "models").mkdir(parents=True, exist_ok=True)
    (s / f"progress.json{suffix}").write_text(json.dumps({"status": status, "timesteps": E, "written_at": "x"}))
    with zipfile.ZipFile(s / "models" / f"stage1_final.zip{suffix}", "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": E if zip_n is None else zip_n}))
    if meta:
        (root / f"run_meta.json{suffix}").write_text(json.dumps({"stop_reason": stop_reason, "status": "done", "S": S}))
    if hop:
        (s / f"hop_watch.json{suffix}").write_text("{}")
    return root


# ---------------------------------------------------------------------------------------------------- pins and paths
def test_pins_hold():
    assert ML.check_pins() == []


def test_lane_b_modules_are_the_pinned_files():
    for mod, name in ((C, "r10round_common.py"), (LG, "ledger_r10round.py"), (CP, "complete_r10round.py"),
                      (SC, "score_r10round.py")):
        assert Path(mod.__file__).resolve() == Path(ML.LANE_B, name).resolve()
        assert ML.sha256_file(mod.__file__) == ML.PINS[f"{ML.LANE_B}/{name}"]


def test_writes_only_under_measure(tmp_path):
    with pytest.raises(ML.Refusal):
        ML.require_under("/tmp/elsewhere/x.json")
    with pytest.raises(ML.Refusal):
        ML.require_under(ML.LB + "/tools/x.json")
    assert ML.require_under(tmp_path / "x.json")


def test_slots_flag_is_limited_to_two():
    assert ML.main(["classify", "/nonexistent", "--out-root", ML.MEASURE + "/work/x", "--slots", "3"]) == 2


# ---------------------------------------------------------------------------------------------------- C5 ending
def test_finished(tmp_path):
    e = ML.read_ending(C, make_run(tmp_path, ML.FINAL), "")
    assert e["ending"] == "FINISHED" and e["E"] == ML.FINAL and e["run_meta_S"] == 2097152


def test_finished_with_copy_suffix_and_no_run_meta(tmp_path):
    e = ML.read_ending(C, make_run(tmp_path, ML.FINAL, suffix=".copy1", meta=False), ".copy1")
    assert e["ending"] == "FINISHED" and e["run_meta_stop_reason"] is None


def test_stopped_by_collapse_and_by_hop_watch(tmp_path):
    assert ML.read_ending(C, make_run(tmp_path / "a", 5_000_000 // 8192 * 8192, stop_reason="collapse"),
                          "")["ending"] == "STOPPED"
    assert ML.read_ending(C, make_run(tmp_path / "b", 4_000_000 // 8192 * 8192, stop_reason="other", hop=True),
                          "")["ending"] == "STOPPED"
    assert ML.read_ending(C, make_run(tmp_path / "c", 8_000_000 // 8192 * 8192, stop_reason="hop_watch"),
                          "")["ending"] == "STOPPED"


@pytest.mark.parametrize("kw", [dict(E=6_000_000 // 8192 * 8192, stop_reason="other"),
                                dict(E=ML.FINAL, status="running"),
                                dict(E=ML.FINAL, zip_n=ML.FINAL - 8192),
                                dict(E=ML.FINAL, stop_reason="collapse"),
                                dict(E=ML.FINAL + 8192)])
def test_not_classified(tmp_path, kw):
    E = kw.pop("E")
    with pytest.raises(ML.Refusal):
        ML.read_ending(C, make_run(tmp_path, E, **kw), "")


# ---------------------------------------------------------------------------------------------------- outcome-blind
def test_blind_guard(tmp_path, monkeypatch):
    lb = tmp_path / "lb"
    runs = lb / "runs"
    for n in ("control_1", "critic_blind_tail3_2"):
        (runs / n / "01_stance").mkdir(parents=True)
    (lb / "elsewhere_in_lb" / "x").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    monkeypatch.setattr(ML, "LB", str(lb))
    monkeypatch.setattr(ML, "LB_RUNS", str(runs))
    monkeypatch.setattr(ML, "LB_SCREEN", str(lb / "SCREEN.json"))
    monkeypatch.setattr(ML, "LB_LOST", str(lb / "lost.json"))
    assert ML.blind_guard(tmp_path / "outside")["applies"] is False
    with pytest.raises(ML.Refusal, match="unreadable"):
        ML.blind_guard(runs / "control_1")
    with pytest.raises(ML.Refusal, match="unreadable"):         # anywhere under LB, not only LB/runs
        ML.blind_guard(lb / "elsewhere_in_lb" / "x")
    tmp_path = lb
    (tmp_path / "SCREEN.json").write_text(json.dumps({"queue": [["control", 1], ["critic_blind_tail3", 2]]}))
    (runs / "control_1" / "check.json").write_text(json.dumps({"passed": True, "mode": "continuation"}))
    with pytest.raises(ML.Refusal, match="critic_blind_tail3_2"):
        ML.blind_guard(runs / "control_1")
    (runs / "critic_blind_tail3_2" / "check.json").write_text(json.dumps({"passed": False, "mode": "continuation"}))
    with pytest.raises(ML.Refusal):
        ML.blind_guard(runs / "control_1")
    (tmp_path / "lost.json").write_text(json.dumps({"lost": {"critic_blind_tail3_2": "crashed 3 times"}}))
    g = ML.blind_guard(runs / "control_1")
    assert g["applies"] is True and g["lost"] == ["critic_blind_tail3_2"]


# ---------------------------------------------------------------------------------------------------- window
def test_window_reading():
    S = 2097152
    res = dict(evals={S - G: dict(any=0), S: dict(any=3), S + G: dict(any=6)},
               any_flagged=[S, S + G, S + 2 * G, S + 3 * G], tail_flagged=[S + 2 * G, S + 3 * G],
               any_alarms=[S + G, S + 2 * G, S + 3 * G], tail_alarms=[S + 3 * G], needs_resolution=[])
    w = ML.window_reading(C, res, S, ML.FINAL)
    assert w["any_alarms"] == [S + 2 * G, S + 3 * G] and w["straddling_S"] == [S + G]
    assert w["tail_alarms"] == [S + 3 * G] and w["before_S"]["any_flagged"] == [S]
    assert w["before_S"]["over_bar_episodes"] == 3


# ---------------------------------------------------------------------------------------------------- habit summary
HEADER = ["timesteps", "episode", "length", "screen_v2_clean", "screen_v2_reasons", "saturated_actuator",
          "saturated_actuator_fraction", "saturated_actuator_positive_share", "saturated_actuator_ties",
          "actuators_saturated", "tail_3_pitch_act_command_mean"]


def ledger_rows(over: dict):
    """over: {step: (actuator name, n episodes over the bar, positive share)}; other rows under the bar."""
    rows = []
    for k in range(1, 92):
        s = k * G
        name, n, pos = over.get(s, (None, 0, None))
        for ep in range(10):
            hit = ep < n
            rows.append(dict(timesteps=str(s), episode=str(ep), length="1000", screen_v2_clean="",
                             screen_v2_reasons="", saturated_actuator=name if hit else "",
                             saturated_actuator_fraction="0.5" if hit else "0.0",
                             saturated_actuator_positive_share=str(pos) if hit else "",
                             saturated_actuator_ties="0", actuators_saturated="1" if hit else "0",
                             tail_3_pitch_act_command_mean="0.1"))
    return rows


def m1_record(cmd_by_index: dict, over=None):
    over = over or {}
    eps = []
    for i in range(20):
        acts = {}
        for name, (n, sign) in over.items():
            acts[name] = {"over_bar": i < n, "sign": sign if i < n else None}
        eps.append({"seed": 3042 + i, "full_horizon": True, "actuators": acts})
    return {"cmd_all_actuators": {str(k): v for k, v in cmd_by_index.items()}, "episodes": eps}


def test_habit_summary_shows_a_non_tail_habit():
    over = {42 * G: ("neck_pitch_act", 6, 0.0), 43 * G: ("neck_pitch_act", 7, 0.0), 44 * G: ("neck_pitch_act", 2, 0.0),
            60 * G: ("tail_3_pitch_act", 1, 1.0)}
    rows = ledger_rows(over)
    res = C.m5(HEADER, rows)
    assert res["any_alarms"] == [43 * G] and res["tail_alarms"] == []
    classes = {"any_alarms": [{"step": 43 * G, "actuators": {"neck_pitch_act": 13}}]}
    js = [4194304, 5242880, 6291456, 8388608, 10485760]
    recs = {"5242880": m1_record({0: -0.80, 14: 0.2}), "6291456": m1_record({0: -0.92, 14: 0.2},
                                                                              {"neck_pitch_act": (18, "-")}),
            "4194304": m1_record({0: -0.1}), "8388608": m1_record({0: -0.1}), "10485760": m1_record({0: -0.39}),
            "final": m1_record({0: -0.2})}
    h = ML.habit_summary(C, HEADER, rows, res, classes, recs, None, 0, ML.FINAL, js, 10485760)
    neck = h["per_actuator"]["neck_pitch_act"]
    assert neck["ledger_over_total"] == 15 and neck["ledger_over_signs"] == {"-": 15}
    assert neck["own_alarms"] == [43 * G] and neck["any_alarms_naming"] == [43 * G]
    assert neck["any_alarms_sole"] == [43 * G] and neck["flagged_evaluations"] == [42 * G, 43 * G]
    assert neck["habit_sites"] == ["L", "T", "D"]
    assert neck["m1"]["6291456"]["over_bar_full_minus"] == 18
    tail3 = h["per_actuator"]["tail_3_pitch_act"]
    assert tail3["ledger_over_total"] == 1 and tail3["habit_sites"] == [] and tail3["ledger_over_signs"] == {"+": 1}
    assert h["habit_sites"] == {"neck_pitch_act": ["L", "T", "D"]}


def test_habit_drive_by_mean_of_r1_and_r2_only():
    rows = ledger_rows({})
    res = C.m5(HEADER, rows)
    recs = {"10485760": m1_record({13: -0.55}), "final": m1_record({13: -0.70})}
    h = ML.habit_summary(C, HEADER, rows, res, {"any_alarms": []}, recs, None, 0, ML.FINAL, [10485760], 10485760)
    assert h["per_actuator"]["tail_2_pitch_act"]["habit_sites"] == ["D"]          # mean -0.625 in sign -
    recs = {"10485760": m1_record({13: -0.55}), "final": m1_record({13: 0.70})}
    h = ML.habit_summary(C, HEADER, rows, res, {"any_alarms": []}, recs, None, 0, ML.FINAL, [10485760], 10485760)
    assert h["per_actuator"]["tail_2_pitch_act"]["habit_sites"] == []


# ---------------------------------------------------------------------------------------------------- pruning, jobs
def test_prune_keeps_over_bar_episodes_and_merges(tmp_path):
    d = tmp_path / "panels" / "final" / "m1"
    d.mkdir(parents=True)
    for s in (3042, 3043, 3044):
        with gzip.open(d / f"episode_trace_seed{s}.csv.gz", "wb") as fh:
            fh.write(b"step\n0\n")
    rec = {"episodes": [{"seed": 3042}, {"seed": 3043}, {"seed": 3044}],
           "any_actuator_over_bar_episodes": [{"seed": 3043, "actuators": [14]}]}
    r = ML.prune_csvs(d, rec)
    assert r["removed"] == 2 and r["kept"] == 1
    assert sorted(p.name for p in d.glob("*.gz")) == ["episode_trace_seed3043.csv.gz"]
    r2 = ML.prune_csvs(d, rec)
    assert r2["removed"] == 0
    j = json.loads((d.parent / "m1_pruned.json").read_text())
    assert j["removed"] == ["episode_trace_seed3042.csv.gz", "episode_trace_seed3044.csv.gz"]


def test_jobs_for_reuses_finished_and_refused_measurements(tmp_path):
    out = tmp_path / "run"
    prep = dict(out=out, name="run", finished=True, labels=["4194304", "final"], stage_config="sc",
                pairs={"4194304": dict(zip="z", pkl="p"), "final": dict(zip="zf", pkl="pf")})
    (out / "panels" / "final" / "m3").mkdir(parents=True)
    (out / "panels" / "final" / "m3" / "m3_summary.json").write_text("{}")
    (out / "panels" / "4194304" / "m1").mkdir(parents=True)
    (out / "panels" / "4194304" / "m1" / "m1_refused.json").write_text("{}")
    js = {(j["label"], j["kind"]): j for j in ML.jobs_for(prep, True)}
    assert js[("final", "m3")]["outcome"] == "ALREADY MEASURED"
    assert js[("4194304", "m1")]["outcome"] == "ALREADY REFUSED BY THE MEASURING TOOL"
    assert "outcome" not in js[("final", "m1")]
    stopped = dict(prep, finished=False)
    assert ("final", "m3") not in {(j["label"], j["kind"]) for j in ML.jobs_for(stopped, True)}
    assert all(j.get("outcome") for j in ML.jobs_for(dict(prep, out=tmp_path / "other"), False))


# ---------------------------------------------------------------------------------------------------- M5 on real data
def test_m5_cli_on_the_round_s80_ledger_agrees_with_the_in_process_reading(tmp_path):
    led = Path(S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s80/copies/01_stance/eval_episodes.csv.copy1")
    header, rows = C.read_ledger(led)
    res = C.m5(header, rows)
    c6 = LG.c6_check(header, rows, ML.FINAL)
    cli = ML.run_m5_cli(dict(out=tmp_path / "s80", ledger=str(led), E=ML.FINAL))
    assert ML.m5_agrees(cli, res, c6) == []
    assert len(res["any_alarms"]) == 6 and res["tail_alarms"] == [] and c6["passed"] and c6["n_grid"] == 91
    assert res["any_alarms"][0] == 5505024


def test_prefix_check_on_the_parent_against_itself(tmp_path):
    p = ML.prefix_check(Path(ML.PARENT_LEDGER), None, 2097152)
    assert p["checked"] and p["rows_equal"] and p["rows_run"] == 160 and p["header_equal"]
    assert ML.prefix_check(Path(ML.PARENT_LEDGER), None, 0)["checked"] is False


def test_panels_from_another_folder_are_a_validation_reading_only(tmp_path):
    with pytest.raises(ML.Refusal, match="validation reading only"):
        ML.classify_one(dict(out=tmp_path / "o", run_dir=ML.LB + "/runs/control_1"), [], {}, None, False,
                        panels_dir=tmp_path, filed_run_id="x")
    with pytest.raises(ML.Refusal, match="validation reading only"):
        ML.classify_one(dict(out=tmp_path / "o", run_dir="/elsewhere/run"), [], {}, None, True,
                        panels_dir=tmp_path, filed_run_id="x")


def test_machine_count_is_an_integer_and_skips_this_tools_wrappers():
    assert isinstance(ML.Slots.machine_sims(), int)
    assert "measure_r10round.py" in ML.WRAPPERS and "measure_lb.py" in ML.WRAPPERS
