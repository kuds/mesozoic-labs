"""Adapter lane: the scorer's dry runs on PS-shaped roots under <PS>/criteria_work (criteria VALIDATIONS):
  dry_line1   synthetic: the control arm does not climb (median rise 0.065) while the phase arm is far lower -> line 1
  dry_line2   synthetic: control rises +0.30..+0.33, phase +0.00..+0.03 -> line 2 (and every secondary exercised with
              loop rows at E: SEC-CMD 2/70, SEC-MARGIN 24/70, SEC-MECH 1/70, SEC-TAIL2 3/70)
  dry_line3   synthetic: interleaved primaries (p > 0.05) with the line-2 secondaries -> line 3
  dry_parent  the parent's own markers (PS V3's reference read) as every continuation, both arms -> line 3, p 70/70
Each root is built with the scorer tests' own helpers (tools/tests/test_score_bs.py) and scored by the real CLI
(score_bs.main --root <root>), which writes <root>/score/score.{json,txt}.  Writes only under <PS>/criteria_work/dry_*.
    cd /tmp && TMPDIR=<PS>/criteria_work/adapt_tmp/tmpdir PY -I -B adapt_dry_runs.py
"""
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
PS = Path("/home/user/work/phase_screen")
WORK = PS / "criteria_work"
spec = importlib.util.spec_from_file_location("test_score_bs_dry", str(PS / "tools/tests/test_score_bs.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
SB = T.SB
B_S, f32 = T.B_S, T.f32

roots = {k: WORK / k for k in ("dry_line1", "dry_line2", "dry_line3", "dry_parent")}
for r in roots.values():
    assert not r.exists(), f"{r} exists: not overwritten"
T.Screen(roots["dry_line1"]).all([f32(B_S + x) for x in (0.05, 0.06, 0.07, 0.90)],
                                 [f32(B_S - 0.2 + x / 100) for x in range(4)], per_seed=T._per_seed_main())
T._main_screen(roots["dry_line2"])
T.Screen(roots["dry_line3"]).all([0.5 + x / 128 for x in (26, 19, 32, 23)], [0.5 + x / 128 for x in (27, 22, 24, 31)],
                                 per_seed=T._per_seed_main())
T.build_parent_root(roots["dry_parent"])
for key, root in roots.items():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = SB.main(["--root", str(root)])
    res = json.loads((root / "score" / "score.json").read_text())
    secs = {s["key"]: (s["verdict"], f'{s["p_filled"]["count"]}/{s["p_filled"]["total"]}', s["notes"],
                       sorted({str(c) for c in s["causes"].values() if c})[:1]) for s in res["secondaries"]}
    print(f"{key}: exit {code} status {res['status']} line {res['line']} ({res['line_key']}) p_filled "
          f"{res['p_filled']['count']}/{res['p_filled']['total']} p_as_counted "
          f"{res['p_as_counted']['count']}/{res['p_as_counted']['total']} control_rise {res['control_rise']!r} "
          f"informative {res['informative']} additions {res['additions']}")
    for k, v in secs.items():
        print(f"    {k}: {v}")
    print(f"    provisional items: {len(res['provisional'])}; first: {res['provisional'][:2]}")
