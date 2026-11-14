"""The faster tracker's merge check (``STAGE1_PLAN.md`` decision 21: "merged only after the fit, once it reproduces
recorded evaluations bit for bit").

Run from hardware/trex_form_study:

    python -m designlab.stage1.replay evals --body 60-N-T-10 --seed 42 [--lines 0,5,13] [--run stage1/runs/2026-11-08]
    python -m designlab.stage1.replay digests --baseline designlab.stage1_base --body 60-N-T-10 [--lines 0,5]

* ``evals``: each chosen line of ``<run>/<body>/tune/<seed>/evals.jsonl`` (the fit's log, read only) evaluated again
  with this package (``tune.Evaluator``, the line's gains and episode): its J, S, P, falls and the 66 rounded terms
  must equal the logged ones exactly (``EVALS_RULE``).
* ``digests``: the same evaluations with this package and with a baseline package (a copy of designlab/stage1 at the
  commit to compare with, importable as ``designlab.<name>``, e.g. made with ``git show <commit>:<path>`` per file into
  ``designlab/stage1_base/``), every physics step's sha256 of qpos, qvel, ctrl and sensordata compared
  (``DIGESTS_RULE``; ``mujoco.mj_step`` is wrapped while they run), and their full results (wall-clock fields aside).

The exit status is 0 only if every line matches.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import tune

EVALS_RULE = ("each evaluation of the log run again (tune.Evaluator.evaluate on the line's gains, from_unit(x), and "
              "episode seed): J, S, P and falls equal as floats and integers, the rounded terms equal as a list")
DIGESTS_RULE = ("each evaluation run with this package and with the baseline package: sha256 of qpos, qvel, ctrl and "
                "sensordata after every physics step equal, step by step, and the evaluation's results equal (the "
                "wall-clock fields left out)")
WALL_KEYS = {"wall_s", "steps_per_s"}


def _lines(path: Path, which) -> list[tuple[int, dict]]:
    rows = path.read_text().splitlines()
    idx = range(len(rows)) if which is None else which
    return [(i, json.loads(rows[i])) for i in idx if i < len(rows)]


def check_evals(body: str, seed: int, run: Path, lines=None, log=print) -> dict:
    """``evals`` (module docstring)."""
    ev = tune.Evaluator(body, Path(run) / body)
    out = []
    for i, rec in _lines(Path(run) / body / "tune" / str(int(seed)) / "evals.jsonl", lines):
        r = ev.evaluate(tune.from_unit(np.asarray(rec["x"], float)), rec["episode"])
        same = dict(J=r["J"] == rec["J"], S=r["S"] == rec["S"], P=r["P"] == rec["P"], falls=r["falls"] == rec["falls"],
                    terms=r["terms"] == rec["terms"])
        out.append(dict(line=i, gen=rec["gen"], i=rec["i"], episode=rec["episode"], J=r["J"], logged_J=rec["J"],
                        identical=all(same.values()), fields=same))
        log(f"{body} seed {seed} line {i} (gen {rec['gen']} member {rec['i']}): J {r['J']!r} logged {rec['J']!r} "
            f"-> {'identical' if out[-1]['identical'] else 'DIFFERENT ' + str(same)}")
    return dict(rule=EVALS_RULE, body=body, seed=int(seed), lines=out, identical=all(x["identical"] for x in out))


def _canon(x):
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items() if k not in WALL_KEYS}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, np.ndarray):
        return _canon(x.tolist())
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


class _Digests:
    """``mujoco.mj_step`` wrapped: after every step, sha256 of qpos, qvel, ctrl and sensordata into ``rows``."""

    def __init__(self):
        self.rows = []
        self._orig = mujoco.mj_step

    def __enter__(self):
        orig, rows = self._orig, self.rows

        def step(m, d, nstep=1):
            orig(m, d, nstep)
            rows.append(b"".join(hashlib.sha256(a.tobytes()).digest() for a in (d.qpos, d.qvel, d.ctrl,
                                                                                  d.sensordata)))

        mujoco.mj_step = step
        return self

    def __exit__(self, *exc):
        mujoco.mj_step = self._orig
        return False


def check_digests(body: str, seed: int, run: Path, baseline: str, lines=None, log=print) -> dict:
    """``digests`` (module docstring)."""
    base_tune = importlib.import_module(f"{baseline}.tune")
    evs = {"this": tune.Evaluator(body, Path(run) / body), "baseline": base_tune.Evaluator(body, Path(run) / body)}
    mods = {"this": tune, "baseline": base_tune}
    out = []
    for i, rec in _lines(Path(run) / body / "tune" / str(int(seed)) / "evals.jsonl", lines):
        res, dig = {}, {}
        for name in ("this", "baseline"):
            with _Digests() as dg:
                r = evs[name].evaluate(mods[name].from_unit(np.asarray(rec["x"], float)), rec["episode"])
            res[name], dig[name] = _canon(r), dg.rows
        a, b = dig["this"], dig["baseline"]
        first = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
        row = dict(line=i, episode=rec["episode"], steps=len(a), baseline_steps=len(b),
                   steps_identical=bool(first is None and len(a) == len(b)), first_differing_step=first,
                   results_identical=res["this"] == res["baseline"])
        row["identical"] = row["steps_identical"] and row["results_identical"]
        out.append(row)
        log(f"{body} line {i}: {len(a)} physics steps; per-step digests "
            f"{'identical' if row['steps_identical'] else f'DIFFER from step {first}'}; results "
            f"{'identical' if row['results_identical'] else 'DIFFER'}")
    return dict(rule=DIGESTS_RULE, body=body, seed=int(seed), baseline=baseline, lines=out,
                identical=all(x["identical"] for x in out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage1.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("evals", "digests"))
    ap.add_argument("--body", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default=str(tune.RUN), help="run directory (default: stage1/runs/2026-11-08)")
    ap.add_argument("--lines", default=None, help="comma-separated line numbers of evals.jsonl (default: every line)")
    ap.add_argument("--baseline", default=None, help="digests: the baseline package, e.g. designlab.stage1_base")
    ap.add_argument("--json", default=None, help="write the result here")
    a = ap.parse_args(argv)
    lines = None if a.lines is None else [int(x) for x in a.lines.split(",")]
    if a.what == "evals":
        doc = check_evals(a.body, a.seed, Path(a.run), lines)
    else:
        if not a.baseline:
            raise SystemExit("digests needs --baseline <package>")
        doc = check_digests(a.body, a.seed, Path(a.run), a.baseline, lines)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1))
    print(f"{a.what}: {'all identical' if doc['identical'] else 'DIFFERENCES'} ({len(doc['lines'])} evaluations)")
    return 0 if doc["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
