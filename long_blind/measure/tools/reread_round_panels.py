"""Validation (read only, no simulation): measure_lb.py's classification path on the round's OWN M1/M3 measurements
(S1/foot/stance6/panels/<RUN_ID>/<label>/<m1|m3>/, made at F2 by the same pinned measure_r10round.py on the same F2
pairs), with M5 run afresh on the F2 ledger copy.  Used where a fresh M1/M3 cannot be made (the 3 GiB disk floor of
measure_r10round.py) and as a second check where it can.  Writes only <out-root>/<RUN_ID>/ (measure_lb.json,
classes.json, habit.json, report.txt, m5.json, m5.txt).

    PY -I -B reread_round_panels.py [RUN_ID ...] [--out-root LB/measure/validate/reread_round]
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("measure_lb", TOOLS / "measure_lb.py")
ML = importlib.util.module_from_spec(spec)
sys.modules["measure_lb"] = ML
spec.loader.exec_module(ML)
F2 = ML.S1 + "/foot/stance6/fetch/F2"
ROUND_PANELS = ML.S1 + "/foot/stance6/panels"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["stance6_r10LTSB_s42", "stance6_r10LTSB_s60", "stance6_r10LTSB_s80"])
    ap.add_argument("--out-root", default=ML.MEASURE + "/validate/reread_round")
    a = ap.parse_args(argv)
    problems = ML.check_pins()
    if problems:
        print("REFUSED: pins:", problems)
        return 2
    C, LG, CP, SC = ML.lane_b()
    out_root = ML.require_under(Path(a.out_root))
    rc = 0
    for run in a.runs:
        p = ML.prepare(C, LG, CP, Path(F2) / run / "copies", ".copy1", out_root, None)
        p["m5cli"] = ML.run_m5_cli(p)
        r = ML.classify_one(p, [], p["m5cli"], None, False, panels_dir=Path(ROUND_PANELS) / run, filed_run_id=run)
        ML.write_json(p["out"] / "measure_lb.json", r)
        text = ML.render(r)
        ML.write_text(p["out"] / "report.txt", text + "\n")
        if r.get("classes") is not None:
            ML.write_json(p["out"] / "classes.json", r["classes"])
            ML.write_json(p["out"] / "habit.json", r["habit"])
        else:
            rc = 2
        print(text, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
