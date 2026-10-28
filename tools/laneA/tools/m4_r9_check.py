"""M4's reproduction check on r9 (validation only; the task's "check M4 against a judged r9 checkpoint so it
reproduces the judge's numbers").  judge_r10ltsb.py's own code, with its BINDING CONSTANTS set to REPL's r9 values
(physics r9, seeds 42/52/60/70/80 with the sidecar for 70 and 80, HEAD cf80ecb1, collapse reference 4457.4, cadence
524,288, no K3-panel sigma check: the r9 tree has no k3_panel), judges a checkpoint that REPL's judge_stance5r.py
judged, on a sealed git archive of cf80ecb1 (the r9 tree), and the two judge files are compared value by value.

    PY -I -B m4_r9_check.py RUN_ID STEPS SEED R9_COPY RUN_DIR OUT_DIR

The production binding refuses an r9 checkpoint (V5); this harness sets the module attributes in its own process
only, and its output names the binding "r9-REPL-validation" (no scorer reads it).  PASS = every value equal except the
identity and timing fields listed in IDENTITY_KEYS / TIMING_KEYS (paths, schema, the tool's own sha256 and version,
the clone record, seconds).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
VERSION = "m4_r9_check 1.0 (2026-10-26)"
HERE = Path(__file__).resolve().parent
REPL_JUDGED = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/"
               "replication/judged")
REPL_RUNS = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/runs"
R9 = {"BINDING_NAME": "r9-REPL-validation", "PLANT_PHYSICS_REVISION": 9,
      "PLANT_PHYSICS_SHA256": "sha256:3d68aec73fbc4642681ed601bedaf3524459d693d6a7afce036bb14bac872d67",
      "SEEDS": (42, 52, 60, 70, 80), "SIDECAR_SEEDS": (70, 80), "CHECKPOINT_CADENCE": 524288,
      "K3_PANEL_SIGMA_CHECK": False}
R9_HEAD = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"
REPL_CLONE = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/fix/scratch_copband"
IDENTITY_KEYS = {"schema", "judge", "judge_sha256", "argv", "clone", "git", "expect_head", "modules", "python",
                 "run_dir", "stage_dir", "model", "vecnorm", "version", "binding", "checkpoint_physics_revision",
                 "k3_panel_sigma", "copy_tree_digest_after", "branch_expected", "modules_all_under_clone"}
TIMING_KEYS = {"judge_seconds", "panel_seconds", "seconds"}


def diff(a, b, path="", out=None):
    """Every key path where a and b differ (values compared exactly; floats as JSON numbers)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in TIMING_KEYS or (path == "" and k in IDENTITY_KEYS):
                continue
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing on one side"))
                continue
            diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"lengths {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b, strict=False)):  # a length difference is reported above
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, f"{a!r} vs {b!r}"[:300]))
    return out


def main() -> int:
    run_id, steps, seed, r9_copy, run_dir, out_dir = sys.argv[1:7]
    print(VERSION, flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"judge_{steps}.json"
    spec = importlib.util.spec_from_file_location("judge_r9binding", HERE / "judge_r10ltsb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in R9.items():
        setattr(mod, k, v)
    mod.ARMS = {"LTSB": {**mod.ARMS["LTSB"], "head": R9_HEAD, "branch": "claude/screen-trex-r9-armlts-copband",
                         "collapse_peak_floor_reference": 4457.4}}
    if not target.exists():
        sys.argv = ["judge_r10ltsb.py", run_dir, "--arm", "LTSB", "--seed", str(seed), "--steps", steps, "--clone",
                    r9_copy, "--out", str(target)]
        rc = mod.main()
        if rc:
            return rc
    mine = json.loads(target.read_text())
    ref_p = Path(REPL_JUDGED) / run_id / f"judge_{steps}.json"
    ref = json.loads(ref_p.read_text())
    # the run folder's and the clone's paths appear inside a few strings (descriptions, the mirror map's path);
    # compared with REPL's paths substituted (the r9 tree is a git archive of REPL's clone at the same commit)
    text = json.dumps(mine).replace(os.path.realpath(run_dir), f"{REPL_RUNS}/{run_id}")
    text = text.replace(os.path.realpath(r9_copy), REPL_CLONE)
    mine_n = json.loads(text)
    d = diff(mine_n, ref)
    res = {"tool": VERSION, "run_id": run_id, "steps": steps, "seed": int(seed), "mine": str(target),
           "mine_sha256": __import__("hashlib").sha256(target.read_bytes()).hexdigest(), "repl": str(ref_p),
           "binding": mine.get("binding"), "identity_keys_skipped": sorted(IDENTITY_KEYS),
           "timing_keys_skipped": sorted(TIMING_KEYS), "differences": [{"path": p, "what": w} for p, w in d],
           "model_sha256_equal": mine.get("model_sha256") == ref.get("model_sha256"),
           "vecnorm_sha256_equal": mine.get("vecnorm_sha256") == ref.get("vecnorm_sha256"),
           "clone_head": (mine.get("git") or {}).get("head"), "repl_clone_head": (ref.get("git") or {}).get("head"),
           "headline": {k: [(mine.get("panel") or {}).get(k), (ref.get("panel") or {}).get(k)]
                        for k in ("verdict", "n_clean", "n_full_horizon", "mean_reward", "statue_mean_reward")}}
    for key in ("noise_matched_own", "noise_matched_control_joint"):
        res["headline"][key + ".falls"] = [(mine.get(key) or {}).get("falls"), (ref.get(key) or {}).get("falls")]
    res["PASS"] = not d and res["model_sha256_equal"] and res["vecnorm_sha256_equal"] and res["clone_head"] == R9_HEAD
    (out / f"compare_{run_id}_{steps}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("run_id", "steps", "PASS", "headline")}))
    print(f"{len(d)} differing value(s)" + ("" if not d else ": " + "; ".join(f"{p} {w}" for p, w in d[:10])))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
