"""LB measurement validation: compare measure_lb.py's classification of the r10 round's three runs (made from the F2
copies, read only) with the round's own reading.  Report only; reads only; writes VALIDATION.txt / validation.json
under LB/measure/validate.

    PY -I -B validate_round.py [--out-root LB/measure/validate]

References (read only):
  SCORE   = S2/r10round/f2/score.json (the round's scorer output, STATUS FINAL; sha256 cb380a7e...), per_seed classes;
  PANELS  = S1/foot/stance6/panels/<RUN_ID>/<label>/<m1|m3>/ (the round's own M1/M3 outputs, the C7 measurements);
  RUNLOG line 711 (S1/foot/stance4/live/run_log.txt) and the task's statements (EXPECT below).
Compared: every key of each run's classes (score_r10round.classes_of output, JSON-normalised the same way); per
label the M1 per-step CSV sha256 (csv_sha256.txt, 20 lines), the per-episode summaries of episode_trace.json, lane A's
cmd_a / tail_flags / cmd_all_actuators; at final the M3 verdict, failures, per-heading n_clean / clean_lcb /
full-horizon fraction / reward mean and the per-episode panel CSV's sha256.
"""
from __future__ import annotations

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
MEASURE = S2 + "/r10round/long_blind/measure"
SCORE = S2 + "/r10round/f2/score.json"
SCORE_SHA256 = "cb380a7e816e519cce267fd4a7c00a7d4528730a388e39b3678b95d235c01474"
PANELS = S1 + "/foot/stance6/panels"
RUNS = ("stance6_r10LTSB_s42", "stance6_r10LTSB_s60", "stance6_r10LTSB_s80")
# The task's statements of the round's reading (RUNLOG line 711)
EXPECT = {
    "stance6_r10LTSB_s42": dict(cls="RAIL", kind="RAIL-LEDGER-PERSISTENT", tail_actuator="tail_3_pitch_act+",
                                first_tail_alarm=3932160, n_tail_alarms=22, flagged_evaluations=26, gate="FAIL",
                                gate_hop_or_fall_all_six=True, outcome="NEGATIVE"),
    "stance6_r10LTSB_s60": dict(cls="PARKED", gate="PASS", outcome="NEGATIVE", n_tail_alarms=0, n_any_alarms=0),
    "stance6_r10LTSB_s80": dict(cls="CLEAR", gate="PASS", outcome="POSITIVE", n_any_alarms=6,
                                any_alarm_actuators={"neck_pitch_act"}, n_tail_alarms=0),
}
HEADINGS = (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(o):
    return json.loads(json.dumps(o, default=str))


def compare_classes(mine: dict, ref: dict) -> dict:
    keys = sorted(set(mine) | set(ref))
    diff = {k: dict(mine=mine.get(k), round=ref.get(k)) for k in keys if norm(mine.get(k)) != norm(ref.get(k))}
    return dict(keys=len(keys), equal=not diff, differing=diff)


def compare_m1(run, label, out_root) -> dict:
    mine = out_root / run / "panels" / label / "m1"
    ref = Path(PANELS) / run / label / "m1"
    r = dict(label=label)
    a, b = (mine / "csv_sha256.txt").read_text(), (ref / "csv_sha256.txt").read_text()
    r["csv_sha256_lines"] = len(a.splitlines())
    r["csv_sha256_equal"] = a == b
    ja, jb = json.loads((mine / "episode_trace.json").read_text()), json.loads((ref / "episode_trace.json").read_text())
    r["episodes_equal"] = norm(ja.get("episodes")) == norm(jb.get("episodes"))
    r["trace_json_keys_differing"] = sorted(k for k in set(ja) | set(jb) if norm(ja.get(k)) != norm(jb.get(k)))
    sa, sb = json.loads((mine / "m1_summary.json").read_text()), json.loads((ref / "m1_summary.json").read_text())
    for k in ("cmd_a", "tail_flags", "fall_flags", "cmd_all_actuators", "tail_flags_full_horizon_by_actuator_sign",
              "any_actuator_over_bar_episodes", "full_horizon_episodes", "episodes_without_window"):
        r[f"{k}_equal"] = norm(sa.get(k)) == norm(sb.get(k))
    r["physics_revision"] = sa.get("physics_revision")
    r["copy_commit"] = (sa.get("copy_seal") or {}).get("commit")
    r["checkpoint_sha256_equal"] = (sa.get("checkpoint") or {}).get("sha256") == (sb.get("checkpoint") or {}).get(
        "sha256")
    r["trace_seconds"] = (sa.get("trace_run") or {}).get("seconds")
    r["all_equal"] = all(v for k, v in r.items() if k.endswith("_equal"))
    return r


def compare_m3(run, out_root) -> dict:
    mine = out_root / run / "panels" / "final" / "m3"
    ref = Path(PANELS) / run / "final" / "m3"
    sa, sb = json.loads((mine / "m3_summary.json").read_text()), json.loads((ref / "m3_summary.json").read_text())
    r = dict(verdict=sa.get("verdict"), round_verdict=sb.get("verdict"), verdict_equal=sa.get("verdict") ==
             sb.get("verdict"), failures_equal=norm(sa.get("failures")) == norm(sb.get("failures")),
             headings_equal=norm(sa.get("headings")) == norm(sb.get("headings")),
             label=sa.get("label"), physics_revision=sa.get("physics_revision"),
             copy_commit=(sa.get("copy_seal") or {}).get("commit"),
             checkpoint_sha256_equal=(sa.get("checkpoint") or {}).get("sha256") == (sb.get("checkpoint") or {}).get(
                 "sha256"),
             task_sha256_equal=sa.get("task_sha256") == sb.get("task_sha256"),
             gate_sha256_equal=sa.get("gate_sha256") == sb.get("gate_sha256"),
             certificate_seconds=(sa.get("certificate_run") or {}).get("seconds"),
             round_certificate_seconds=(sb.get("certificate_run") or {}).get("seconds"))
    pa, pb = mine / "stance_heading_panel_selected.csv", ref / "stance_heading_panel_selected.csv"
    if pa.is_file() and pb.is_file():
        r["panel_csv_sha256_equal"] = sha256_file(pa) == sha256_file(pb)
    ra = json.loads((mine / "stance_heading_report.json").read_text())
    rb = json.loads((ref / "stance_heading_report.json").read_text())
    r["report_keys_differing"] = sorted(k for k in set(ra) | set(rb) if norm(ra.get(k)) != norm(rb.get(k)))
    r["report_headings_equal"] = norm(ra.get("headings")) == norm(rb.get("headings"))
    r["all_equal"] = all(v for k, v in r.items() if k.endswith("_equal"))
    return r


def expectations(run, c: dict) -> list[tuple[str, bool, object]]:
    e = EXPECT[run]
    out = [("class", c["cls"] == e["cls"], c["cls"]), ("gate", c["gate"] == e["gate"], c["gate"]),
           ("outcome", c["outcome"] == e["outcome"], c["outcome"])]
    if "kind" in e:
        out.append(("kind", c["kind"] == e["kind"], c["kind"]))
    if "n_tail_alarms" in e:
        out.append(("TAIL ALARM count", len(c["tail_alarms"]) == e["n_tail_alarms"], len(c["tail_alarms"])))
    if "first_tail_alarm" in e:
        out.append(("first TAIL ALARM", (c["tail_alarms"] or [None])[0] == e["first_tail_alarm"],
                    (c["tail_alarms"] or [None])[0]))
    if "flagged_evaluations" in e:
        out.append(("tail-flagged evaluations", c["flagged_evaluations"] == e["flagged_evaluations"],
                    c["flagged_evaluations"]))
    if "tail_actuator" in e:
        acts = sorted({k for d in c["tail_alarm_detail"] for k in d["actuators"]})
        out.append(("TAIL ALARM actuator/sign", acts == [e["tail_actuator"]], acts))
    if e.get("gate_hop_or_fall_all_six"):
        why = c.get("gate_why") or []
        ok = len(why) == 6 and all("hop_or_fall_episodes 2 > 1" in w for w in why) and all(
            any(w.startswith(f"heading {h}") for w in why) for h in ("-90", "-45", "+0", "+45", "+90", "+180"))
        out.append(("GATE FAIL hop_or_fall 2 > 1 at all six headings", ok, why))
    if "n_any_alarms" in e:
        out.append(("ANY ALARM count", len(c["any_alarms"]) == e["n_any_alarms"], len(c["any_alarms"])))
    if "any_alarm_actuators" in e:
        acts = {k for x in c["any_alarms"] for k in x["actuators"]}
        out.append(("ANY ALARM actuators", acts == e["any_alarm_actuators"], sorted(acts)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=MEASURE + "/validate")
    ap.add_argument("--classes-only", action="store_true",
                    help="compare the classes and the stated facts only (a re-read of the round's own panels)")
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    a = ap.parse_args(argv)
    out_root = Path(a.out_root)
    if sha256_file(SCORE) != SCORE_SHA256:
        print(f"REFUSED: {SCORE} is not at its sha256 {SCORE_SHA256}")
        return 2
    score = json.loads(Path(SCORE).read_text())
    ref = {x["run"]: x["classes"] for x in score["per_seed"]}
    res, L, ok_all = {}, [], True
    for run in a.runs:
        mfile = out_root / run / "measure_lb.json"
        if not mfile.is_file():
            L.append(f"{run}: no measure_lb.json")
            ok_all = False
            continue
        m = json.loads(mfile.read_text())
        c = m.get("classes")
        if c is None:
            L.append(f"{run}: not classified: {m.get('status')}")
            ok_all = False
            continue
        cc = compare_classes(c, ref[run])
        m1 = [] if a.classes_only else [compare_m1(run, lab, out_root) for lab in m["labels"]]
        m3 = dict(all_equal=True, skipped=True) if a.classes_only else compare_m3(run, out_root)
        ex = expectations(run, c)
        ok = cc["equal"] and all(x["all_equal"] for x in m1) and m3["all_equal"] and all(x[1] for x in ex)
        ok_all &= ok
        res[run] = dict(classes=cc, m1=m1, m3=m3, expectations=[dict(what=w, ok=o, value=v) for w, o, v in ex],
                        all_ok=ok, costs=m.get("costs"))
        L.append(f"{run}: {'REPRODUCED' if ok else 'DIFFERS'}")
        L.append(f"  classes: {cc['keys']} keys compared with score.json; equal {cc['equal']}"
                 + ("" if cc["equal"] else f"; differing {sorted(cc['differing'])}"))
        for w, o, v in ex:
            L.append(f"  {'OK ' if o else 'NO '} {w}: {json.dumps(v, default=str)[:300]}")
        for x in m1:
            L.append(f"  M1 {x['label']:>9}: csv sha256 {x['csv_sha256_lines']} lines equal {x['csv_sha256_equal']}; "
                     f"episodes equal {x['episodes_equal']}; cmd_a {x['cmd_a_equal']}; tail_flags "
                     f"{x['tail_flags_equal']}; all {x['all_equal']}; trace json keys differing "
                     f"{x['trace_json_keys_differing']}; {x['trace_seconds']} s")
        if a.classes_only:
            continue
        L.append(f"  M3 final: verdict {m3['verdict']} (round {m3['round_verdict']}); failures equal "
                 f"{m3['failures_equal']}; headings equal {m3['headings_equal']}; panel csv equal "
                 f"{m3.get('panel_csv_sha256_equal')}; report keys differing {m3['report_keys_differing']}; "
                 f"{m3['certificate_seconds']} s (round {m3['round_certificate_seconds']} s); all {m3['all_equal']}")
    L.append(f"ALL REPRODUCED: {ok_all}")
    text = "\n".join(L) + "\n"
    print(text, end="")
    (out_root / "VALIDATION.txt").write_text(text)
    (out_root / "validation.json").write_text(json.dumps(dict(score=SCORE, score_sha256=SCORE_SHA256, runs=res,
                                                              all_reproduced=ok_all), indent=1, default=str) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
