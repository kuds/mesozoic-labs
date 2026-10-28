"""Derive judge_r10ltsb.py from REPL's judge_stance5r.py (sha256 pinned) by exact, counted text replacements, and
write the unified diff beside it (judge_diff.txt), as REPL's judge_diff.txt records judge_stance5r.py's derivation.

    PY -I -B make_judge_r10ltsb.py            # writes judge_r10ltsb.py and judge_diff.txt here (refused if they exist)

Each replacement must match exactly once in the source, or nothing is written.  The binding constants live in one
block (BINDING CONSTANTS); m4_r9_check.py swaps that block's module attributes for REPL's r9 values.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
VERSION = "make_judge_r10ltsb 1.1 (2026-10-26; v5: the branch HEAD from the approval record)"
HERE = Path(__file__).resolve().parent
SRC = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/"
           "replication/judge/judge_stance5r.py")
SRC_SHA256 = "16d39f7dcb14a8b6f94061331933df8f1dcee76aa988479e293b132bf8ac05ae"
OUT = HERE / "judge_r10ltsb.py"
DIFF = HERE / "judge_diff.txt"

HEADER_OLD = '''"""Judge ONE checkpoint of a T. rex stance screen run (arms L, LT, LTS, LTSB; physics r9) against its branch clone (scratch only).

    S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
    cd $S/foot/stance5/replication && env ISO_WT=<clone> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl \\
        python3 $S/foot/stance3/tools/iso.py judge/judge_stance5r.py RUN_DIR --arm {L,LT,LTS,LTSB} \\
        --seed {42,52,60,70,80} --steps STEPS --out OUT.json [MODE]

'''
HEADER_NEW = '''"""Judge ONE checkpoint of a T. rex stance run of the r10 round (arm LTSB; physics r10) on the round's measuring copy (scratch only).

    S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
    env OMP_NUM_THREADS=1 MPLCONFIGDIR=$S2/caches/mpl nice -n 10 $S2/venv/bin/python -I -B judge_r10ltsb.py RUN_DIR \\
        --arm LTSB --seed {60,42,80} --steps STEPS --clone <measuring copy> --out OUT.json [MODE]

judge_r10ltsb.py (M4 of the r10 round's criteria, S2/r10round/lean/v5/criteria_r10round_lean_v5.txt: "REPL's
judge_stance5r.py rebound to r10, the branch HEAD, its task and statue, the per-seed sigma") is a copy of
judge_stance5r.py (sha256 16d39f7d...) changed only in its binding (BINDING CONSTANTS below) and where the binding has
to reach the code (judge_diff.txt lists every changed line; make_judge_r10ltsb.py derives this file):
  (1) the plant: physics r10 (PLANT_PHYSICS_REVISION 10, PLANT_PHYSICS_SHA256), checked by Ctx._check_arm as r9 was;
      and, before anything is loaded, the checkpoint's own recorded physics_revision (read from the zip's data member,
      nothing unpickled): anything but 10 is refused (the criteria's HOW IT IS MEASURED; V5);
  (2) the code: the round's measuring copy (a sealed git archive of the branch HEAD, --clone, required) instead of a
      git clone.  copy_state() returns the record git_state() returned (head = the archived commit, status "" while the
      re-computed tree digest equals the seal's), so every binding that reads it is unchanged; the HEAD the copy must
      be is the branch HEAD, which this file never holds (criteria lean v5 VALIDATIONS): it is read at run time from
      the approval record (<ROUND>/APPROVED "branch_head", --approved) and checked by r10m_common.approved_branch_head
      beside this file (its only parent the header's parent cf9402bc, only stance.toml comment lines changed), unless
      --expect-head names a HEAD for a validation; the copy is checked again after judging;
  (3) the arm: LTSB only, its [env] keys as the r9 LTSB arm's (the port's), collapse reference 4470.4 (D20 (a)), the
      checkpoint cadence 262,144 (D3 (a));
  (4) the seeds 60, 42 and 80 (D2).  Their joint-space sigma (THIS12 lines 132-136): 60 and 42 frozen_params.json's
      control_sigma, 80 the replication sidecar's control_sigma["80"], read exactly as judge_stance5r.py reads them;
      so that the offline judge and the in-run panel use the same vector per seed (the criteria's ARMS AND SEEDS), the
      vector must also equal the node the in-run panel reads (K3_PANEL_SIGMA_KEY) in the branch's copy of the sidecar,
      resolved by the measuring copy's own k3_panel.read_joint_sigma, else refused; that node's sha256 is recorded;
  (5) the schema r10ltsb-judge/v1, the log prefix [judge_r10ltsb], a printed version string, the binding's name.
Everything else, below included, is judge_stance5r.py's text and code unchanged (read "the clone" as the measuring
copy, and the r9 plant check as the binding's).  It is scored with score_stance3.py's code unchanged but for its
judge identity and its r9 plant check, which reads the pinned r10 plant (criteria lean v5 M4; kscore_r10ltsb.py).

'''

REPLACEMENTS = [
    ("header", HEADER_OLD, HEADER_NEW),
    ("schema",
     'SCHEMA = "stance5r-judge/v1"\n',
     'SCHEMA = "r10ltsb-judge/v1"\nVERSION = "judge_r10ltsb 1.1 (2026-10-26; v5: the branch HEAD from the approval record)"\n'),
    ("seeds",
     "SEEDS = (42, 52, 60, 70, 80)\n",
     "SEEDS = (60, 42, 80)  # r10 round: D2\n"),
    ("sidecar seeds",
     "SIDECAR_SEEDS = (70, 80)\n",
     "SIDECAR_SEEDS = (80,)  # r10 round: seed 80 reads the sidecar's control_sigma (THIS12 lines 132-136)\n"
     "# ---------------------------------------------------------------------------------- BINDING CONSTANTS (r10 round)\n"
     "#: m4_r9_check.py sets these module attributes (with SEEDS, SIDECAR_SEEDS, ARMS and CHECKPOINT_CADENCE) to REPL's\n"
     "#: r9 values to show the code reproduces judge_stance5r.py's numbers on r9 (validation only; its output names that\n"
     "#: binding).\n"
     'BINDING_NAME = "r10ltsb"\n'
     "PLANT_PHYSICS_REVISION = 10\n"
     'PLANT_PHYSICS_SHA256 = "sha256:6a47a0eb78164b9738752e2cf928aa0a6fd3ef98fc953eddc9bcb2eca366da4d"\n'
     "#: The branch HEAD the runs train is never held here (criteria lean v5 VALIDATIONS): main() reads it from the\n"
     "#: approval record (approved_branch_head) unless --expect-head names one; m4_r9_check.py sets ARMS' head.\n"
     'BRANCH_HEAD = None\n'
     'BRANCH_NAME = "claude/screen-trex-r10-armltsb"\n'
     "SEAL_NAME = \".r10m_seal.json\"  # the measuring copy's seal (r10m_common.py)\n"
     "K3_PANEL_SIGMA_CHECK = True\n"
     'K3_SIGMA_REPO_REL = "docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json"\n'
     'K3_SIGMA_REPO_SHA256 = "939285e30f377e696fef7bdcc30d6d61786e18c593ae39a1b50c936e3822a0ab"\n'
     'K3_PANEL_SIGMA_KEY = {60: "source_vectors.60", 42: "source_vectors.42", 80: "control_sigma.80"}\n'),
    ("arms",
     None,  # the whole ARMS block, located by its first and last lines (below)
     'ARMS = {\n'
     '    # r10 round (criteria WHAT IS TRAINED): arm LTSB\'s [env] keys as the r9 LTSB arm\'s (the port carries them); the\n'
     '    # code is the round\'s measuring copy (--clone, required); "head" the branch HEAD; collapse reference 4470.4.\n'
     '    "LTSB": {"clone": None, "head": BRANCH_HEAD, "branch": BRANCH_NAME,\n'
     '             "env": {**ARM_ENV_COMMON, "leg_home_pose_weight": 1.0, "leg_home_pose_tolerance": 0.05,\n'
     '                     "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045,\n'
     '                     "stance_cop_band_weight": 0.30},\n'
     '             "collapse_peak_floor_reference": 4470.4},\n'
     '}\n'),
    ("cadence",
     "CHECKPOINT_CADENCE = 524288\n",
     "CHECKPOINT_CADENCE = 262144  # r10 round: D3 (a)\n"),
    ("plant check",
     '''        if plant.get("physics_sha256") != R9_PHYSICS_SHA256 or plant.get("physics_revision") != 9:
            problems.append(f"plant {plant.get('physics_revision')} {plant.get('physics_sha256')} is not r9")
''',
     '''        if plant.get("physics_sha256") != PLANT_PHYSICS_SHA256 or plant.get("physics_revision") != PLANT_PHYSICS_REVISION:
            problems.append(f"plant {plant.get('physics_revision')} {plant.get('physics_sha256')} is not "
                            f"r{PLANT_PHYSICS_REVISION}")
'''),
    ("copy_state",
     '''# ------------------------------------------------------------------------------------------------ helpers
''',
     '''def _tree_digest(root: str) -> tuple[str, int]:
    """r10m_common.tree_digest: sha256 over the sorted 'relpath<TAB>sha256' lines of every file but the seal."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel == SEAL_NAME:
                continue
            if os.path.islink(p):
                lines.append(f"{rel}\\tsymlink:{os.readlink(p)}")
            else:
                lines.append(f"{rel}\\t{sha256_file(p)[len('sha256:'):]}")
    lines.sort()
    return hashlib.sha256(("\\n".join(lines) + "\\n").encode()).hexdigest(), len(lines)


def copy_state(clone: str) -> dict:
    """r10 round: the measuring copy's seal, its tree digest re-computed; the record git_state() returned for a clone
    (head = the archived commit; status "" while the digest equals the seal's, else the mismatch)."""
    seal_p = os.path.join(clone, SEAL_NAME)
    if not os.path.isfile(seal_p):
        raise SystemExit(f"{clone} is not a sealed measuring copy ({SEAL_NAME} missing)")
    with open(seal_p) as fh:
        seal = json.load(fh)
    digest, n = _tree_digest(clone)
    ok = digest == seal.get("tree_digest") and n == seal.get("files")
    return {"head": seal.get("commit"), "status": "" if ok else f"tree digest {digest} ({n} files) != sealed "
            f"{seal.get('tree_digest')} ({seal.get('files')} files)", "branch": f"(git archive {seal.get('commit')})",
            "tree_digest": digest, "seal": seal}


def checkpoint_physics_revision(zip_path: str) -> dict:
    """r10 round: the checkpoint's recorded plant identity, read from the zip's data member as JSON (nothing unpickled)."""
    import zipfile

    try:
        with zipfile.ZipFile(zip_path) as z:
            data = json.loads(z.read("data"))
        plant = data.get("_mesozoic_plant_identity")
    except Exception as exc:  # noqa: BLE001 - recorded; refused by the caller
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}", "physics_revision": None}
    plant = plant if isinstance(plant, dict) else {}
    return {"readable": True, "physics_revision": plant.get("physics_revision"),
            "physics_sha256": plant.get("physics_sha256")}


def approved_branch_head(approved=None, repo=None) -> dict:
    """r10 round (criteria lean v5 VALIDATIONS: no tool holds the header commit): the branch HEAD from the approval
    record, checked by r10m_common.approved_branch_head (beside this file) in the run-branch clone (*repo* for a
    test); refused otherwise."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("r10m_common_judge", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "r10m_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return mod.approved_branch_head(approved, repo)
    except mod.Refusal as exc:
        raise SystemExit(f"refusing (nothing judged): {exc.message}") from None


def k3_panel_sigma(clone: str, seed: int, vector) -> dict:
    """r10 round: the node the in-run K3 panel reads for *seed* (K3_PANEL_SIGMA_KEY) in the branch's copy of the
    sidecar, resolved by the copy's own k3_panel.read_joint_sigma; refused unless it equals *vector* exactly."""
    from environments.shared import k3_panel

    path = os.path.join(clone, K3_SIGMA_REPO_REL)
    sha = sha256_file(path)
    if sha != "sha256:" + K3_SIGMA_REPO_SHA256:
        raise SystemExit(f"{path} sha256 {sha} != the pinned {K3_SIGMA_REPO_SHA256}")
    rec = k3_panel.read_joint_sigma(path, K3_PANEL_SIGMA_KEY[int(seed)])
    if rec.vector is None:
        raise SystemExit(f"the in-run panel's sigma for seed {seed} cannot be read: {rec.problem}")
    if [float(v) for v in rec.vector] != [float(v) for v in vector]:
        raise SystemExit(f"the in-run panel's sigma {K3_PANEL_SIGMA_KEY[int(seed)]} differs from the judge's sigma for "
                         f"seed {seed}: the offline judge and the panel must use the same vector")
    return {"file": path, "file_sha256": sha, "key": rec.key, "path_in_file": rec.path_in_file, "sha256": rec.sha256,
            "equal_to_judge_vector": True}


# ------------------------------------------------------------------------------------------------ helpers
'''),
    ("seed help",
     '''                    help="the run's training seed (selects the control sigma: frozen_params for 42/52/60, the sidecar for 70/80)")
''',
     '''                    help="the run's training seed (selects the control sigma: frozen_params for 60/42, the sidecar for 80)")
'''),
    ("clone help",
     '''    ap.add_argument("--clone", default=None, help="default: the arm's clone")
''',
     '''    ap.add_argument("--clone", default=None, help="the round's measuring copy (required)")
    ap.add_argument("--approved", default=None,
                    help="the approval record holding the branch HEAD (default <ROUND>/APPROVED; criteria lean v5)")
'''),
    ("placeholder",
     '''    if spec["collapse_peak_floor_reference"] is None or (spec["head"] is None and not a.expect_head):
''',
     '''    if spec["collapse_peak_floor_reference"] is None:
'''),
    ("approved head",
     '''    expect_head = a.expect_head or spec["head"]
''',
     '''    expect_head = a.expect_head or spec["head"]
    approval = None
    if expect_head is None:
        approval = approved_branch_head(a.approved)          # r10 round: the branch HEAD from the approval record
        expect_head = approval["head"]
'''),
    ("clone required",
     '''    clone = a.clone or spec["clone"]
''',
     '''    clone = a.clone or spec["clone"]
    print(f"[judge_r10ltsb] {VERSION}; binding {BINDING_NAME}", flush=True)
    if not clone:
        raise SystemExit("pass --clone <the round's measuring copy>")
'''),
    ("physics refusal",
     '''    missing = [p for p in (zp, vp) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"checkpoint missing: {missing}")
''',
     '''    missing = [p for p in (zp, vp) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"checkpoint missing: {missing}")
    ck_phys = checkpoint_physics_revision(zp)
    if ck_phys.get("physics_revision") != PLANT_PHYSICS_REVISION:
        raise SystemExit(f"refusing (nothing written): {zp} records physics_revision {ck_phys.get('physics_revision')!r} "
                         f"(physics_sha256 {ck_phys.get('physics_sha256')!r}), not {PLANT_PHYSICS_REVISION}")
'''),
    ("git -> copy",
     '''    git = git_state(clone)
    if git["status"] or git["head"] != expect_head:
        raise SystemExit(f"the judge runs the clean clone at {expect_head} (got {git['head']}, dirty={bool(git['status'])})")
    print(f"[judge_stance5r] arm {a.arm} clone {clone} HEAD {git['head']} ({git['branch']}); mode {mode_name}; "
          f"environments = {modules['environments']}", flush=True)
''',
     '''    git = copy_state(clone)
    if git["status"] or git["head"] != expect_head:
        raise SystemExit(f"the judge runs the clean clone at {expect_head} (got {git['head']}, dirty={bool(git['status'])})")
    print(f"[judge_r10ltsb] arm {a.arm} clone {clone} HEAD {git['head']} ({git['branch']}); mode {mode_name}; "
          f"environments = {modules['environments']}", flush=True)
    k3_sigma = k3_panel_sigma(clone, a.seed, ctrl["vector"]) if K3_PANEL_SIGMA_CHECK else None
'''),
    ("record identity",
     '''        "control_sigma_file": sigma_file, "control_sigma_file_sha256": sigma_sha,
''',
     '''        "control_sigma_file": sigma_file, "control_sigma_file_sha256": sigma_sha,
        "version": VERSION, "binding": BINDING_NAME, "checkpoint_physics_revision": ck_phys.get("physics_revision"),
        "k3_panel_sigma": k3_sigma, "branch_head_approval": approval,
'''),
    ("copy after",
     '''    record["modules_all_under_clone"] = modules_under(clone)
''',
     '''    record["modules_all_under_clone"] = modules_under(clone)
    after = copy_state(clone)
    if after["status"] or after["tree_digest"] != git["tree_digest"]:
        raise SystemExit(f"the measuring copy changed while judging: {after['status']}")
    record["copy_tree_digest_after"] = after["tree_digest"]
'''),
    ("log prefix",
     '''            print(f"[judge_stance5r] {exc.code}", file=sys.stderr, flush=True)
''',
     '''            print(f"[judge_r10ltsb] {exc.code}", file=sys.stderr, flush=True)
'''),
]


def main() -> int:
    print(VERSION, flush=True)
    raw = SRC.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != SRC_SHA256:
        raise SystemExit(f"{SRC} sha256 {got} != the pinned {SRC_SHA256}")
    text = raw.decode("utf-8")
    for name, old, new in REPLACEMENTS:
        if name == "arms":
            start = text.index("ARMS = {\n")
            end = text.index("PPO_SYMMETRY = ")
            block = text[start:end]
            if text.count("ARMS = {\n") != 1 or not block.rstrip().endswith("}"):
                raise SystemExit("the ARMS block was not found exactly once")
            text = text[:start] + new + text[end:]
            continue
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"replacement {name!r}: matched {n} times (need exactly 1); nothing written")
        text = text.replace(old, new)
    for path in (OUT, DIFF):
        if path.exists():
            raise SystemExit(f"{path} exists; delete it to derive again")
    OUT.write_text(text)
    diff = difflib.unified_diff(raw.decode("utf-8").splitlines(keepends=True), text.splitlines(keepends=True),
                                fromfile=str(SRC), tofile=str(OUT))
    DIFF.write_text("".join(diff))
    print(f"wrote {OUT} (sha256 {hashlib.sha256(text.encode()).hexdigest()}) and {DIFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
