"""Summarize the smoke lane's judge steps (reads only): each step's exit status against the expected one (the job
name's _expect<N> suffix), its last log line (the judge's final message), and for each written cell the binding's
validation fields and the probe's labels (pilot4_score.py's own label_episode and seed_outcome, imported from the
pinned file), L, plant and episode lengths.

    python -I summarize_judge.py JUDGE_DIR TOOLS OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True


def main() -> int:
    jd, tools, out = (Path(a).resolve() for a in sys.argv[1:4])
    spec = importlib.util.spec_from_file_location("pilot4_score_labels", tools / "pilot4_score.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    steps = []
    for line in (jd / "status.txt").read_text().splitlines():
        m = re.match(r"(\S+) rc=(\d+) elapsed_s=(\d+)", line)
        if not m:
            continue
        name, rc, el = m.group(1), int(m.group(2)), int(m.group(3))
        exp = re.search(r"_expect(\d+)$", name)
        stem = name[: exp.start()] if exp else name
        logs = sorted((jd / "logs").glob(f"{stem}*.log")) or sorted((jd / "logs").glob(f"{stem.split('_')[0]}_*.log"))
        last = None
        for lg in logs:
            if lg.stem.startswith(stem[:2]) and lg.is_file():
                lines = [ln for ln in lg.read_text().splitlines() if ln.strip()]
                if lines and lg.stem in name:
                    last = lines[-1][:1500]
        steps.append({"step": name, "rc": rc, "expected": int(exp.group(1)) if exp else None,
                      "as_expected": (rc == int(exp.group(1))) if exp else (rc == 0), "elapsed_s": el,
                      "last_line": last})
    cells = {}
    for b in sorted(jd.rglob("binding.json")):
        cell = b.parent
        bind = json.loads(b.read_text())
        doc = json.loads((cell / "probe.json").read_text())
        rows, meta = doc["episodes"], doc["meta"]
        labels = [sc.label_episode(r) for r in rows]
        counts = Counter(labels)
        cells[str(cell.relative_to(jd))] = {
            "binding": {k: bind.get(k) for k in ("run_id", "seed", "steps", "head", "branch", "checkpoint_sha256",
                                                 "vecnorm_sha256", "validation_only", "expected_budget",
                                                 "cadence_checked", "text_anchors_checked", "off_schedule",
                                                 "final_fallback", "final_fallback_record", "resume_step",
                                                 "shaping_inert", "run_end", "lineage", "env_kwargs_drift",
                                                 "probe_env_kwargs_drift", "curriculum_cadence", "plant",
                                                 "hyperparameters_sha256", "task_sha256", "commit_note")},
            "labels": dict(counts), "seed_outcome": sc.seed_outcome(counts) if hasattr(sc, "seed_outcome")
            else None,
            "episodes": len(rows), "lengths": dict(Counter(int(r.get("length", -1) or -1)
                                                           for r in rows)),
            "meta": {k: meta.get(k) for k in meta if any(s in k for s in ("leg_length", "plant", "seed", "settle",
                                                                            "episodes", "stage", "checkpoint"))},
        }
    rep = {"steps": steps, "all_as_expected": all(s["as_expected"] for s in steps), "cells": cells}
    Path(out).write_text(json.dumps(rep, indent=1, default=str))
    for s in steps:
        print(f"{s['step']}: rc {s['rc']} ({'as expected' if s['as_expected'] else 'NOT AS EXPECTED'}) "
              f"{s['elapsed_s']} s | {s['last_line']}")
    for k, c in cells.items():
        print(k, c["labels"], c["seed_outcome"], c["meta"].get("leg_length_m"), c["binding"]["validation_only"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
