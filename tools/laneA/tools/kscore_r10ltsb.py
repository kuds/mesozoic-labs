"""K1-K4 of one r10 run from judge_r10ltsb.py's files: M4's scoring (criteria lean v5, HOW IT IS MEASURED, M4:
"scored with score_stance3.py's code unchanged but for its judge identity (JUDGE_SCHEMA r10ltsb-judge/v1, ARM_ENV LTSB,
HEADS the branch HEAD, as score_stance5.py rebinds them) and its r9 plant check, which reads the pinned r10 plant
(physics_revision 10, physics_sha256 6a47a0eb...) in place of r9's").  Report only (D-7); never a class, statistic or
reading input.

    PY -I -B kscore_r10ltsb.py JUDGED_ROOT RUN_ID SEED [--approved APPROVED | --expect-head SHA] [--json OUT.json]
    PY -I -B kscore_r10ltsb.py --panels <ROUND>/panels RUN_ID SEED [...]
        (fixer, RA-2: the layout lane A's runner and judge caller write, <panels>/<RUN_ID>/<label>/m4/judge_<steps>.json
         with label the step or "final"; a read-only flat view of symlinks is handed to score_stance3's load_cell)
    PY -I kscore_r10ltsb.py --reproduce-repl [--json OUT.json]    # the scoring check on REPL's s70/s80 judge files

score_stance3.py (sha256 pinned) is loaded by path as its own module instance, as score_stance5.py loads it, and the
judge identity is rebound, as score_stance5.py rebinds it for its band arm (JUDGE_SCHEMA, ARM_ENV, HEADS): schema
r10ltsb-judge/v1, arm LTSB (the r9 LTSB arm's [env] keys, the port's), HEAD the branch HEAD.  No tool holds the
branch HEAD (criteria lean v5 VALIDATIONS): it is read from the approval record (<ROUND>/APPROVED "branch_head",
--approved) and checked by r10m_common.approved_branch_head (beside this file: its only parent cf9402bc, only
stance.toml comment lines changed); --expect-head names another HEAD for a validation (recorded).  Its joint-noise
check (validity_issues) reads frozen_params.json's criteria.H4.control_sigma; seed 80's entry is added from the
replication sidecar (vector and source as judge_r10ltsb.py records them), as score_stance5r.py adds 70/80.

A-3 (decided at RUNLOG line 657): score_stance3.validity_issues' plant check ("plant ... is not r9" unless
plant_identity is physics_revision 9 with R9_PHYSICS_SHA256) reads the pinned r10 plant in place of r9's: the r9
issue is dropped and "plant <rev> <sha> is not r10" is written unless plant_identity is physics_revision 10 with
physics_sha256 the pin.  Nothing else of score_stance3 changes.  Beside it (fixer, RA-3; the text's "every M1-M4
output records the model's physics_revision and the runner refuses anything but 10") a file whose
checkpoint_physics_revision is not 10 is held invalid.  score_run() returns that one table, "m4", from score_stance3's
own load_cell / run_criteria / k1_failure_modes.

--reproduce-repl: a third instance with REPL's identity (stance5r-judge/v1, arm LTSB, HEAD cf80ecb1, the sidecar's 70
and 80 added) scores REPL's judge files of stance5_LTSB_s70 and stance5_LTSB_s80 and compares each run's criteria and
K1 failure modes with REPL's score_final.json (the replication's score of record): equal = this module drives
score_stance3's code as score_stance5r.py does.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

sys.dont_write_bytecode = True
VERSION = "kscore_r10ltsb 1.2 (2026-10-26; fixer: RA-2, RA-3; v5: A-3 rebound plant check, the approved HEAD)"
SP = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
SC3_PATH = SP + "/foot/stance3/judge/score_stance3.py"
SC3_SHA256 = "2c3879d3a83d453159773d6fed7cacdfbe88e5337c83605834b14377588cc66a"
FROZEN_PARAMS = SP + "/foot/symmetry/design/frozen_params.json"
FROZEN_PARAMS_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
SIDECAR = SP + "/foot/stance5/replication/sigma_s70_s80.json"
SIDECAR_SHA256 = "89778b3f81f83f05d4c15423b3658593dfaa767388213044c50a72e2d96b8522"
REPL_SCORE_FINAL = SP + "/foot/stance5/replication/scores/score_final.json"
REPL_JUDGED = SP + "/foot/stance5/replication/judged"
VIEW_TMP = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/caches"   # RA-2 views

ARM_ENV_LTSB = {"leg_home_pose_weight": 1.0, "leg_home_pose_tolerance": 0.05, "stance_stagger_weight": 0.30,
                "stance_stagger_tolerance_m": 0.045, "stance_cop_band_weight": 0.30}   # score_stance5.ARM_ENV_LTSB
IDENTITIES = {
    "r10ltsb": {"schema": "r10ltsb-judge/v1", "head": None,      # the branch HEAD: from the approval record (v5)
                "sidecar_seeds": ("80",), "plant_revision": 10,
                "plant_sha256": "sha256:6a47a0eb78164b9738752e2cf928aa0a6fd3ef98fc953eddc9bcb2eca366da4d"},
    "repl": {"schema": "stance5r-judge/v1", "head": "cf80ecb14d1630ce3b4dcdea28a05835d143470b",
             "sidecar_seeds": ("70", "80"), "plant_revision": 9,
             "plant_sha256": "sha256:3d68aec73fbc4642681ed601bedaf3524459d693d6a7afce036bb14bac872d67"},
}


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_stance3(name: str):
    """A fresh module instance of the pinned score_stance3.py (refuses any other file content)."""
    got = sha256_file(SC3_PATH)
    if got != SC3_SHA256:
        raise SystemExit(f"{SC3_PATH} sha256 {got} != the pinned {SC3_SHA256}")
    spec = importlib.util.spec_from_file_location(name, SC3_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r9_plant_issue(d: dict) -> str:
    """The exact issue string score_stance3.validity_issues writes for a non-r9 plant."""
    plant = d.get("plant_identity") or {}
    return f"plant {plant.get('physics_revision')} {plant.get('physics_sha256')} is not r9"


def r10_plant_issue(d: dict, ident: dict) -> str | None:
    """A-3: the rebound plant check, score_stance3's r9 check with the pinned r10 plant in place of r9's (None when the
    recorded plant_identity is it)."""
    plant = d.get("plant_identity") or {}
    if plant.get("physics_sha256") != ident["plant_sha256"] or plant.get("physics_revision") != ident["plant_revision"]:
        return f"plant {plant.get('physics_revision')} {plant.get('physics_sha256')} is not r{ident['plant_revision']}"
    return None


def checkpoint_revision_issue(d: dict, ident: dict) -> str | None:
    """RA-3 (kept): a judge file whose checkpoint_physics_revision is not the plant's revision (the runner refuses
    anything but 10)."""
    if d.get("checkpoint_physics_revision") == ident["plant_revision"]:
        return None
    return (f"checkpoint_physics_revision {d.get('checkpoint_physics_revision')!r} is not {ident['plant_revision']} "
            "(the runner refuses anything but 10)")


def _load_common():
    spec = importlib.util.spec_from_file_location("r10m_common_kscore", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "r10m_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC = _load_common()


def resolve_head(identity: str, *, approved: str | None = None, expect_head: str | None = None,
                 repo: str | None = None) -> dict:
    """The HEAD HEADS["LTSB"] is rebound to: the identity's own (REPL's cf80ecb1), else --expect-head (a validation,
    recorded), else the branch HEAD from the approval record, checked by r10m_common.approved_branch_head in the
    run-branch clone (*repo* for a test)."""
    if IDENTITIES[identity]["head"]:
        return {"head": IDENTITIES[identity]["head"], "basis": f"identity {identity}"}
    if expect_head:
        return {"head": expect_head, "basis": "--expect-head (a validation)"}
    try:
        h = RC.approved_branch_head(approved, repo)
    except RC.Refusal as exc:
        raise SystemExit(f"refusing (nothing scored): {exc.message}") from None
    return {"head": h["head"], "basis": "the approval record", "approval": h}


def make_instance(identity: str, head: str):
    """score_stance3's code with the judge identity rebound (JUDGE_SCHEMA, ARM_ENV, HEADS = *head*) and, for an r10
    identity, its r9 plant check rebound to the pinned r10 plant (A-3), RA-3's checkpoint check beside it.  REPL's
    identity (r9) keeps score_stance3's own plant check.  Nothing else of score_stance3 changes."""
    ident = IDENTITIES[identity]
    sc = load_stance3(f"score_stance3_for_{identity}")
    sc.JUDGE_SCHEMA = ident["schema"]
    sc.ARM_ENV = {**sc.ARM_ENV, "LTSB": dict(ARM_ENV_LTSB)}
    sc.HEADS = {**sc.HEADS, "LTSB": head}
    if ident["plant_revision"] != 9:
        original = sc.validity_issues

        def validity_issues(d, **kwargs):
            bad, soft = original(d, **kwargs)
            bad = [b for b in bad if b != r9_plant_issue(d)]        # the r9 check, replaced by the r10 one (A-3)
            extra = [x for x in (r10_plant_issue(d, ident), checkpoint_revision_issue(d, ident)) if x]
            return bad + extra, soft

        sc.validity_issues = validity_issues
    return sc


def frozen_with_sidecar(seeds) -> dict:
    """frozen_params.json (pinned) with the sidecar's (pinned) control_sigma entries for *seeds* added."""
    if sha256_file(FROZEN_PARAMS) != FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{FROZEN_PARAMS} is not the pinned file")
    if sha256_file(SIDECAR) != SIDECAR_SHA256:
        raise SystemExit(f"{SIDECAR} is not the pinned file")
    with open(FROZEN_PARAMS) as fh:
        frozen = json.load(fh)
    with open(SIDECAR) as fh:
        side = json.load(fh)
    if side.get("source_sha256") != FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{SIDECAR} was not computed from the pinned frozen_params.json")
    out = copy.deepcopy(frozen)
    cs = out["criteria"]["H4"]["control_sigma"]
    for s in seeds:
        if s in cs:
            raise SystemExit(f"frozen_params.json already holds control_sigma[{s}]")
        cs[s] = {"source": side["control_sigma"][s]["source"], "vector": list(side["control_sigma"][s]["vector"])}
    return out


def score_with(sc, root: str, run: str, seed: int, frozen: dict, head: str) -> dict:
    notes: list = []
    cells, missing, invalid = {}, [], []
    for label, steps in sc.J:
        c, why, entry = sc.load_cell(root, run, "LTSB", seed, steps, label, expect_head=head, frozen=frozen,
                                     validation=False, needs_noise=label in sc.NOISE_SLOTS, notes=notes)
        cells[label] = c
        if why == "missing":
            missing.append(label)
        elif why == "invalid":
            invalid.append({"label": label, **(entry or {})})
    return {"criteria": sc.run_criteria(cells), "k1_failure_modes": sc.k1_failure_modes(cells), "missing": missing,
            "invalid": invalid, "notes": notes}


def panels_view(panels: str, run: str, view_root: str) -> dict:
    """RA-2: <view_root>/<run>/judge_<steps>.json symlinked to <panels>/<run>/<label>/m4/judge_<steps>.json for each J
    slot whose file exists (label = the step, or "final").  Returns {steps: source path or None}."""
    sc = load_stance3("score_stance3_for_layout")
    os.makedirs(os.path.join(view_root, run))
    linked = {}
    for _label, steps in sc.J:
        src = os.path.join(panels, run, str(steps), "m4", f"judge_{steps}.json")
        if os.path.isfile(src):
            os.symlink(os.path.realpath(src), os.path.join(view_root, run, f"judge_{steps}.json"))
            linked[str(steps)] = os.path.realpath(src)
        else:
            linked[str(steps)] = None
    return linked


def score_run(root: str, run: str, seed: int, *, layout: str = "flat", approved: str | None = None,
              expect_head: str | None = None) -> dict:
    """M4's table for one r10 run (report only; A-3).  layout "panels": root is <ROUND>/panels (RA-2)."""
    ident = IDENTITIES["r10ltsb"]
    head = resolve_head("r10ltsb", approved=approved, expect_head=expect_head)
    frozen = frozen_with_sidecar(ident["sidecar_seeds"])
    out = {"tool": VERSION, "score_stance3": {"path": SC3_PATH, "sha256": SC3_SHA256,
                                              "rebound": ["JUDGE_SCHEMA", "ARM_ENV", "HEADS", "the r9 plant check (A-3)"]},
           "run": run, "seed": seed, "judged_root": root, "layout": layout, "head": head["head"],
           "head_basis": head}
    os.makedirs(VIEW_TMP, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kscore_view_", dir=VIEW_TMP) as tmp:
        if layout == "panels":
            out["panels_files"] = panels_view(root, run, tmp)
            read_root = tmp
        else:
            read_root = root
        out["m4"] = score_with(make_instance("r10ltsb", head["head"]), read_root, run, seed, frozen, head["head"])
    return out


def reproduce_repl() -> dict:
    """REPL's s70/s80 J files through this module's REPL instance, against REPL's score_final.json."""
    ident = IDENTITIES["repl"]
    frozen = frozen_with_sidecar(ident["sidecar_seeds"])
    sc = make_instance("repl", ident["head"])
    with open(REPL_SCORE_FINAL) as fh:
        record = json.load(fh)
    res = {"tool": VERSION, "score_final": REPL_SCORE_FINAL, "score_final_sha256": sha256_file(REPL_SCORE_FINAL),
           "runs": {}, "equal": True}
    for run, seed in (("stance5_LTSB_s70", 70), ("stance5_LTSB_s80", 80)):
        got = score_with(sc, REPL_JUDGED, run, seed, frozen, ident["head"])
        rec = record["new"][run]
        same_c = json.loads(json.dumps(got["criteria"])) == rec["criteria"]
        same_k = json.loads(json.dumps(got["k1_failure_modes"])) == rec["k1_failure_modes"]
        res["runs"][run] = {"criteria_equal": same_c, "k1_failure_modes_equal": same_k, "missing": got["missing"],
                            "invalid": got["invalid"], "HOLDS": got["criteria"]["HOLDS"],
                            "failed": got["criteria"]["failed"], "recorded_HOLDS": rec["criteria"]["HOLDS"]}
        res["equal"] = res["equal"] and same_c and same_k
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("judged_root", nargs="?")
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--reproduce-repl", action="store_true")
    ap.add_argument("--panels", action="store_true", help="JUDGED_ROOT is <ROUND>/panels (RA-2)")
    ap.add_argument("--approved", default=None, help="the approval record holding the branch HEAD (default "
                    "<ROUND>/APPROVED)")
    ap.add_argument("--expect-head", default=None, help="validation only: the HEAD the judge files must name")
    ap.add_argument("--json")
    a = ap.parse_args()
    print(VERSION, flush=True)
    if a.reproduce_repl:
        res = reproduce_repl()
    else:
        if not (a.judged_root and a.run_id and a.seed in (60, 42, 80)):
            ap.error("JUDGED_ROOT RUN_ID SEED (60, 42 or 80)")
        res = score_run(a.judged_root, a.run_id, a.seed, layout="panels" if a.panels else "flat",
                        approved=a.approved, expect_head=a.expect_head)
    text = json.dumps(res, indent=1, default=str)
    if a.json:
        with open(a.json, "x") as fh:
            fh.write(text + "\n")
    print(text[:4000])
    return 0 if (not a.reproduce_repl or res["equal"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
