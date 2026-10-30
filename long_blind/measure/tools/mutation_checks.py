"""In-memory mutation checks of measure_lb.py's glue on the validation outputs (read only; writes only its own
report under LB/measure/validate).  Each check changes one input in memory and states what the round's rules must
give; nothing is simulated and no measurement folder is touched.

    PY -I -B mutation_checks.py [--out-root LB/measure/validate]
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("measure_lb", TOOLS / "measure_lb.py")
ML = importlib.util.module_from_spec(spec)
sys.modules["measure_lb"] = ML
spec.loader.exec_module(ML)
F2 = ML.S1 + "/foot/stance6/fetch/F2"


def prep_of(C, LG, CP, run, out_root, start=None):
    p = ML.prepare(C, LG, CP, Path(F2) / run / "copies", ".copy1", Path(out_root), start)
    p["m5cli"] = dict(json=json.loads((Path(out_root) / run / "m5.json").read_text()))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=ML.MEASURE + "/validate")
    ap.add_argument("--round-panels", action="store_true",
                    help="read the round's own M1/M3 panels (S1/foot/stance6/panels/<RUN_ID>, read only)")
    ap.add_argument("--runs", nargs="*", default=["stance6_r10LTSB_s42", "stance6_r10LTSB_s60", "stance6_r10LTSB_s80"])
    a = ap.parse_args(argv)
    C, LG, CP, SC = ML.lane_b()
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {json.dumps(detail, default=str)[:400]}", flush=True)

    for run in a.runs:
        p = prep_of(C, LG, CP, run, a.out_root)
        pd = Path(ML.S1 + "/foot/stance6/panels") / run if a.round_panels else Path(a.out_root) / run / "panels"
        fr = run if a.round_panels else None
        kw = dict(panels_dir=pd, filed_run_id=fr)
        base = ML.classify_one(p, [], p["m5cli"], None, False, **kw)
        saved = json.loads((Path(a.out_root) / run / "classes.json").read_text())
        check(f"{run} baseline re-read equals the saved classes.json", base["classes"] == saved,
              dict(cls=base["classes"]["cls"], gate=base["classes"]["gate"], outcome=base["classes"]["outcome"]))
        # (1) a trace bound to another pair: the two pairs' bytes swapped -> binding fails, no class (C7)
        q = dict(p, files=dict(p["files"]))
        za, pa = ML.pair_rel(C, "4194304")
        zb, pb = ML.pair_rel(C, "5242880")
        q["files"][za], q["files"][zb] = p["files"][zb], p["files"][za]
        r = ML.classify_one(q, [], p["m5cli"], None, False, **kw)
        check(f"{run} swapped pair bytes -> NOT CLASSIFIED", r["classes"] is None and
              len(r["c7"]["measured_on_another_file"]) == 2, r["c7"]["measured_on_another_file"][:1])
        # (2) a label C7 names without a measurement -> NOT CLASSIFIED (C7 PENDING, never a class)
        q = dict(p, labels=p["labels"] + ["2359296"])        # a 262,144 pair nobody traced
        r = ML.classify_one(q, [], p["m5cli"], None, False, **kw)
        check(f"{run} unmeasured C7 label -> NOT CLASSIFIED", r["classes"] is None and r["c7"]["missing"], r["c7"][
            "missing"][:1])
        # (3) the same records read as a STOPPED run: no certificate, GATE FAIL (STOPPED); RAIL stays RAIL, else UNREAD
        q = dict(p, stopped=True, finished=False)
        r = ML.classify_one(q, [], p["m5cli"], None, False, **kw)
        c = r["classes"]
        want = "RAIL" if base["classes"]["cls"] == "RAIL" else "UNREAD"
        check(f"{run} read as STOPPED -> {want}, GATE FAIL, NEGATIVE",
              c["cls"] == want and c["gate"] == "FAIL" and c["outcome"] == "NEGATIVE" and "STOPPED" in str(c["gate_why"]),
              dict(cls=c["cls"], gate=c["gate"], why=c["gate_why"], reasons=c["reasons"]))
        # (4) the certificate's verdict flipped in memory -> the gate and the outcome follow (classes_of reads it)
        q = dict(p)
        r0 = ML.classify_one(q, [], p["m5cli"], None, False, **kw)
        if r0["classes"]["gate"] == "PASS":
            orig = SC.classes_of
            panels = CP.PanelLocator(pd, "{label}/{kind}")
            cert, _ = panels.certificate("final")
            traces = {lab: panels.trace(lab)[0] for lab in p["labels"]}
            comp = dict(m5=p["res"], traces=traces, r1=p["r1"], resolution=p["resolution"], js=p["js"],
                        stopped=False, E=p["E"], certificate=dict(copy.deepcopy(cert), passed=False,
                                                                  failures=["synthetic: heading +0 n_clean 30 < 32"]))
            c = orig(comp)
            check(f"{run} certificate flipped to FAIL -> GATE FAIL, NEGATIVE", c["gate"] == "FAIL" and
                  c["outcome"] == "NEGATIVE", dict(gate=c["gate"], outcome=c["outcome"], reasons=c["reasons"]))
    # (5) s42 read as a continuation from S = 2,097,152: the window (S, E] keeps every alarm, none straddles S
    p = prep_of(C, LG, CP, "stance6_r10LTSB_s42", a.out_root, start=2097152)
    pd = Path(ML.S1 + "/foot/stance6/panels/stance6_r10LTSB_s42") if a.round_panels else Path(a.out_root) / "stance6_r10LTSB_s42" / "panels"
    r = ML.classify_one(p, [], p["m5cli"], None, False, panels_dir=pd,
                        filed_run_id="stance6_r10LTSB_s42" if a.round_panels else None)
    w = r["window"]
    check("s42 window (2,097,152, E]: 22 TAIL ALARMs, 22 ANY ALARMs, none straddling S, nothing before S",
          len(w["tail_alarms"]) == 22 and len(w["any_alarms"]) == 22 and not w["straddling_S"] and
          w["before_S"]["over_bar_episodes"] == 0, dict(t=len(w["tail_alarms"]), a=len(w["any_alarms"]),
                                                        straddling=w["straddling_S"], before=w["before_S"]))
    check("s42 prefix check against its own F2 ledger", r["prefix"].get("rows_equal") is True, r["prefix"])
    ok = all(x["ok"] for x in checks)
    out = Path(a.out_root) / "mutation_checks.json"
    ML.write_json(out, dict(utc=ML.utcnow(), tool_sha256=ML.sha256_file(ML.__file__), checks=checks, all_pass=ok))
    print(f"ALL PASS: {ok} ({sum(x['ok'] for x in checks)} of {len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
