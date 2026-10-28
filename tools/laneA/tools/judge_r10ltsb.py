"""Judge ONE checkpoint of a T. rex stance run of the r10 round (arm LTSB; physics r10) on the round's measuring copy (scratch only).

    S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
    env OMP_NUM_THREADS=1 MPLCONFIGDIR=$S2/caches/mpl nice -n 10 $S2/venv/bin/python -I -B judge_r10ltsb.py RUN_DIR \
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

judge_stance5r.py (the stance5 replication, criteria SP/foot/stance5/replication/criteria.txt) is a copy of
judge_stance5.py (sha256 d9286724...) changed only so that (1) --seed also takes 70 and 80, the replication seeds, whose
joint-space control sigma is read from the sidecar SP/foot/stance5/replication/sigma_s70_s80.json (sha256 pinned,
CONTROL_SIGMA_SIDECAR_SHA256; the element-wise mean of frozen_params.json control_sigma 42, 52, 60) while seeds 42, 52
and 60 read frozen_params.json exactly as before, and the JSON records which file the sigma came from; (2) its schema is
stance5r-judge/v1 and its log prefix [judge_stance5r].  Everything else, below included, is judge_stance5.py's text and
code unchanged (where it names score_stance5.py, the replication's scorer is score_stance5r.py).

judge_stance4.py plus arm LTSB (stance5: the stance4 recipe, L+T + the stance stagger penalty, plus the per-foot CoP
band, stance_cop_band_weight 0.30; clone fix/scratch_copband, branch claude/screen-trex-r9-armlts-copband) and the
schema stance5-judge/v1 (criteria SP/foot/stance5/criteria.txt).  Arms L, LT and LTS, the bars, the slots, the K
statistics and every binding check are unchanged.  Arm LTSB's collapse reference is 4457.4 (criteria; 3756.8 + the LTSB
statue gain 700.5512, measured on the config change, SP/foot/stance5/build/statue_LTSB.json).  Its HEAD is the config
commit cf80ecb14d1630ce3b4dcdea28a05835d143470b, filled in ARMS after that commit;
while it is None the judge refuses arm LTSB unless --expect-head is given.

Band readouts (arm LTSB only; report only, never in a criterion): for an arm whose stance [env] turns the band on, a
read-only wrapper on each recorded env instance's step (installed when the recorder attaches, after run_panel's plant
validation, as SP/foot/stance3/night/diag_split/roll_split.py installs its info capture) keeps the band's info keys of
every step.  Each panel and statue episode then carries "band": its window (recorder index steps 200 .. L-1; the band's
gate opens on the window's first step) guard firing (guard 0, on the ramp 0 < guard < 1, at 1 = <s>_cop_band_counted),
per-foot mean band quality and digit share, the foot-steps whose whole-foot CoP lies outside [-0.6, +0.4], whose pad
CoP lies outside [-0.65, +0.65] and whose pad excess binds (exceeds the whole-foot excess), the load share in
[0.25, 0.35) and below 0.25, and the band paid over the episode (the sum of reward_stance_cop_band).  The panel and the
statue each get "band_summary" over their full-horizon episodes.  The definitions are
SP/foot/stance5/build/scripts/statue_ref_s5.py's, so the statue's summary reproduces statue_LTSB.json's guard and band
counts.  For arms whose [env] leaves the band off nothing is wrapped: their code path is judge_stance4.py's.

RUN_DIR is a notebook run directory with the notebook's layout (RUN_DIR/01_stance/models/stage1_{N}_steps.zip and
stage1_vecnormalize_{N}_steps.pkl, stage1_final.zip and stage1_final_vecnorm.pkl, best_model / robust_best_model and their
_vecnorm.pkl; RUN_DIR/01_stance/{plant_identity.json, stage_config.json, task_fingerprint.json}).  --arm picks the
branch clone and its pinned HEAD (L: fix/screen_s3_L, claude/screen-trex-r9-arml-12m; LT: fix/screen_s3_LT,
claude/screen-trex-r9-armlt; LTS: fix/screen_s4_LTS, claude/screen-trex-r9-armlt-stagger; LTSB: fix/scratch_copband,
claude/screen-trex-r9-armlts-copband); --clone / --expect-head override them.  The clone's stance stage must carry the arm's
[env] options, [ppo] symmetry keys and collapse reference, on the r9 plant, or the judge refuses before rolling.

--steps: the criteria's judged checkpoints J are 4194304, 5242880, 6291456, 8388608, 10485760 and final
(stage1_final.zip, 12,001,280 steps); 1048576, 2097152, 3145728, 7340032, 9437184 and 11534336 are judged like J,
report only.  For the report-only selection comparison
(S0-S3) the judge also takes robust_best, best and any other periodic checkpoint (a multiple of 524,288); with
--smoke-episodes any multiple of 131,072 (the CPU smoke runs' checkpoints).  score_stance5.py reads full-mode files as
JUDGED/<RUN_ID>/judge_<steps>.json (RUN_ID stance5_LTSB_s52, stance5_LTSB_s60, stance5_LTSB_s42; <steps> as given to
--steps); write the report-only modes under other names (e.g. judge_quick_<steps>.json, judge_tail_final.json).

Ported from the r10 screen's judge (docs/investigations/stance_2026_10/r10_screen/judge_r10.py); what is the same:
  * isolation: `environments` imports from the clone only (run it through tools/iso.py with ISO_WT=<clone>, or bare:
    the judge then removes the editable finder and the main checkout itself); asserted for every loaded module;
  * the panel: reporting.stance_report._stance_v2_report on the clone's stance stage, its [env] kwargs exactly as the
    stage loads them, StanceV2Thresholds from its own [curriculum] (the r9 bars), 40 episodes on seeds 3042-3081
    through the floor-truth recorder, deterministic at heading 0, the zero-action statue rolled on the same seeds under
    the same env, the verdict from evaluate_stance_v2_gate; PR #620's per-foot pad telemetry per episode;
  * the H4 noise tests, arithmetic unchanged: 20 spawns 3042-3061, numpy default_rng(episode seed),
      noise_matched_control_joint  sigma = frozen_params criteria.H4.control_sigma[seed].vector, JOINT space:
                                   a = clip(clip(mu) + sigma z)
      noise_matched_own            sigma = the checkpoint's own exp(log_std), POLICY space: a = clip(mu + sigma z)
    a fall = an episode short of the 1000-step horizon;
  * the bindings, each refused on a mismatch before anything is rolled: RUN_DIR/01_stance/plant_identity.json and the
    checkpoint's own plant identity against the clone's plant, the checkpoint's recorded seed against --seed, its
    mesozoic_symmetry record (mirror_map_sha256 included) against the record the clone's [ppo] keys and committed
    mirror map give, its and the run's task_sha256 against the task the panel rolls, and stage_config.json's env keys,
    v2 bars, [ppo] symmetry keys and git_commit against the clone, and its run block's n_envs (criteria: 4),
    hyperparameters_sha256 (the digest the clone's stance stage regenerates, which must be the committed
    sha256:054ab483...) and seed (against --seed).  Validation-only relaxations (--no-symmetry-check, --task-binding
    report, --allow-missing-records, --allow-seed-mismatch, --allow-commit-mismatch) are recorded; score_stance5.py
    refuses them.  The checkpoint's num_timesteps is recorded beside the slot's (12,001,280 for final, the step for a
    periodic slot) and a mismatch is printed; score_stance5.py refuses such a file.
What changed against the r10 judge:
  * r9 rails (criteria K2a): the pad's max_sole_cop_fore_aft > 0.80 counted over all 40 episodes, a non-full-horizon
    episode counting as over (the r10 whole-foot rail max_foot_cop_fore_aft is kept per episode, report only);
  * per episode, policy panel and statue alike: each foot's signed whole-foot CoP <label>_foot_cop_fore_aft (the pad
    telemetry's loaded-step window mean, span half-lengths; criteria K2b: outside [-0.929, +0.571]), the same in mm
    ahead of that foot's MTP III joint, the both-feet whole-foot CoP x_c (per step the load-weighted mean of the loaded
    feet's whole-foot CoP ahead of their MTP III, then the window mean; criteria K2c), and the COM ahead of the mean
    MTP III anchor along the pelvis heading (com_mtp3_head, ported from foot/tonight/cop_statistic/scripts/cop_roll.py),
    plus the COM ahead of the mean foot reference site (the frame of eval_episodes.csv's com_fore_aft_mean_m, for the
    K5 readout's statue reference).  The per-step readings come from a wrapper around SubstepContactRecorder.end_step
    that only reads mjData after env.step;
  * report-only modes (one per invocation, each its own JSON):
      --quick         the v2 panel (and statue) on 20 episodes, seeds 3042-3061; no noise test
      --common-noise  40 episodes, seeds 3042-3081, a = clip(mu + sigma z) with sigma = exp(log_std) of scratch arm L
                      s60 robust_best (sha256 pinned) and z from default_rng(SeedSequence(seed, spawn_key=(1,))) (the
                      noise battery's common random numbers); falls only
      --tail          the 32 forward-start seeds of foot/tonight/noise_robustness/work/tail_seeds.txt (sha256 pinned),
                      one deterministic episode each, policy and statue; falls only
      --fresh         the v2 panel (and statue) on 40 fresh seeds 7042-7081; no noise test
  * --smoke-episodes N (development only) shortens every panel to N episodes and marks the JSON "smoke".
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

SCHEMA = "r10ltsb-judge/v1"
VERSION = "judge_r10ltsb 1.1 (2026-10-26; v5: the branch HEAD from the approval record)"
MAIN_CHECKOUT = "/home/user/mesozoic-labs"
SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
FROZEN_PARAMS = SCRATCH + "/foot/symmetry/design/frozen_params.json"
#: The r9 screen verdict's params_sha256 (foot/symmetry/screen/analysis/verdict.json): the frozen file is that one.
FROZEN_PARAMS_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
COMMON_SIGMA_ZIP = SCRATCH + "/foot/symmetry/screen/runs/sym_lean_settled_s60/models/robust_best_model.zip"
COMMON_SIGMA_SHA256 = "c99f555ce7a36ac4654f71d2a412010e37cd4f64412d81e595262112526c276c"
TAIL_SEEDS_FILE = SCRATCH + "/foot/tonight/noise_robustness/work/tail_seeds.txt"
TAIL_SEEDS_SHA256 = "8406594b2cebe802b31ee6fe18285366b943c72526ffc926e295675c281e5de4"
R9_PHYSICS_SHA256 = "sha256:3d68aec73fbc4642681ed601bedaf3524459d693d6a7afce036bb14bac872d67"
STAGE_SUBDIR = "01_stance"
SEEDS = (60, 42, 80)  # r10 round: D2
#: stance5r: the replication seeds' joint-space control sigma (replication criteria; maintainer decision 4 of 2026-10-22).
CONTROL_SIGMA_SIDECAR = SCRATCH + "/foot/stance5/replication/sigma_s70_s80.json"
CONTROL_SIGMA_SIDECAR_SHA256 = "89778b3f81f83f05d4c15423b3658593dfaa767388213044c50a72e2d96b8522"
SIDECAR_SEEDS = (80,)  # r10 round: seed 80 reads the sidecar's control_sigma (THIS12 lines 132-136)
# ---------------------------------------------------------------------------------- BINDING CONSTANTS (r10 round)
#: m4_r9_check.py sets these module attributes (with SEEDS, SIDECAR_SEEDS, ARMS and CHECKPOINT_CADENCE) to REPL's
#: r9 values to show the code reproduces judge_stance5r.py's numbers on r9 (validation only; its output names that
#: binding).
BINDING_NAME = "r10ltsb"
PLANT_PHYSICS_REVISION = 10
PLANT_PHYSICS_SHA256 = "sha256:6a47a0eb78164b9738752e2cf928aa0a6fd3ef98fc953eddc9bcb2eca366da4d"
#: The branch HEAD the runs train is never held here (criteria lean v5 VALIDATIONS): main() reads it from the
#: approval record (approved_branch_head) unless --expect-head names one; m4_r9_check.py sets ARMS' head.
BRANCH_HEAD = None
BRANCH_NAME = "claude/screen-trex-r10-armltsb"
SEAL_NAME = ".r10m_seal.json"  # the measuring copy's seal (r10m_common.py)
K3_PANEL_SIGMA_CHECK = True
K3_SIGMA_REPO_REL = "docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json"
K3_SIGMA_REPO_SHA256 = "939285e30f377e696fef7bdcc30d6d61786e18c593ae39a1b50c936e3822a0ab"
K3_PANEL_SIGMA_KEY = {60: "source_vectors.60", 42: "source_vectors.42", 80: "control_sigma.80"}
#: The arms: clone, pinned HEAD (the build's commits) and the stance stage values the clone must carry.
ARM_ENV_COMMON = {"lateral_lean_weight": 0.25, "lateral_lean_tolerance_m": 0.015, "leg_home_pose_reference": "settled",
                  "leg_home_pose_broad_fraction": 0.25, "leg_home_pose_broad_scale": 6.0, "reset_noise_scale": 0.05}
ARMS = {
    # r10 round (criteria WHAT IS TRAINED): arm LTSB's [env] keys as the r9 LTSB arm's (the port carries them); the
    # code is the round's measuring copy (--clone, required); "head" the branch HEAD; collapse reference 4470.4.
    "LTSB": {"clone": None, "head": BRANCH_HEAD, "branch": BRANCH_NAME,
             "env": {**ARM_ENV_COMMON, "leg_home_pose_weight": 1.0, "leg_home_pose_tolerance": 0.05,
                     "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045,
                     "stance_cop_band_weight": 0.30},
             "collapse_peak_floor_reference": 4470.4},
}
PPO_SYMMETRY = {"symmetry": "mirror_kl", "symmetry_kappa": 0.3, "symmetry_sigma_floor": 0.01, "symmetry_mirror_critic": True}
#: criteria.txt: n_envs 4; 12,000,000 steps (final 12,001,280); the committed PPO schedules, i.e. the stance
#: hyperparameters_sha256.PPO of both heads' configs/digest_snapshot.generated.txt.
N_ENVS = 4
FINAL_TIMESTEPS = 12001280
HP_SHA256 = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"
J_STEPS = (4194304, 5242880, 6291456, 8388608, 10485760, "final")
REPORT_STEPS = (1048576, 2097152, 3145728, 7340032, 9437184, 11534336)  # k x 1,048,576 not in J
NAMED_CHECKPOINTS = ("final", "best", "robust_best")
CHECKPOINT_CADENCE = 262144  # r10 round: D3 (a)
SMOKE_CADENCE = 131072
PANEL_SEED, PANEL_EPISODES = 3042, 40
QUICK_SEED, QUICK_EPISODES = 3042, 20
FRESH_SEED, FRESH_EPISODES = 7042, 40
NOISE_SEED, NOISE_EPISODES = 3042, 20
COMMON_SEED, COMMON_EPISODES = 3042, 40
#: The fixed rails of criteria.txt (K2a, K2b, K4), recorded per episode.
PAD_COP_BAR = 0.80           # K2a: max_sole_cop_fore_aft > 0.80, all episodes, a non-full-horizon one counting as over
FOOT_COP_BAR = 0.80          # the r10 whole-foot rail, report only on r9
K2B_LO, K2B_HI = -0.929, 0.571   # K2b: either foot's <label>_foot_cop_fore_aft outside [lo, hi] (span half-lengths)
R9_X_AT_W0_M, R9_X_PER_W_M = 0.060, 0.070   # criteria: on r9, x = 60 + 70 w mm ahead of that foot's MTP III
ROLL_TILT_DEG = 2.0          # K4: max_sole_tilt_deg > 2.0
ROLL_LIFT_M = 0.004          #     or max_sole_corner_lift_m > 0.004
#: The band readouts (criteria PRE-REGISTERED READOUTS; report only): the band's own edges and ramp at their code defaults
#: (base_env 8i), checked against the env class's defaults for a band-on arm.
BAND_WHOLE = (-0.6, 0.4)     # stance_cop_band_lower / _upper: the whole-foot band, span half-lengths
BAND_PAD = (-0.65, 0.65)     # stance_cop_band_pad_lower / _upper: the pad band, pad half-lengths
BAND_SHARE_FULL_PRICE, BAND_SHARE_MIN = 0.25, 0.35   # the guard's ramp (stance_cop_band_full_price_load_share / _min_)
BAND_DEFAULTS = {"stance_cop_band_lower": -0.6, "stance_cop_band_upper": 0.4, "stance_cop_band_pad_lower": -0.65,
                 "stance_cop_band_pad_upper": 0.65, "stance_cop_band_tolerance": 0.357,
                 "stance_cop_band_full_price_load_share": 0.25, "stance_cop_band_min_load_share": 0.35}
BAND_FEET = ("r", "l")
BAND_PER_FOOT = ("foot_cop", "pad_cop", "load_share", "digit_share", "guard", "counted", "quality")
BAND_KEYS = tuple(f"{s}_cop_band_{k}" for s in BAND_FEET for k in BAND_PER_FOOT) + ("stance_cop_band_quality",
                                                                                   "reward_stance_cop_band")
#: (trace, per-step readings) of every recorded episode, in the order the recorder closed them.
TRACES: list = []
_STEP_READINGS: dict = {}
_STEP_GEOMETRY: dict = {}
#: Band readouts (band-on arms only): per env instance the band info of each step, and per closed trace its steps.
_BAND_STEPS: dict = {}
BAND_BY_TRACE: dict = {}
BAND_CAPTURE = {"on": False}


# ------------------------------------------------------------------------------------------------ isolation
def isolate(clone: str) -> dict[str, str]:
    """Import `environments` from *clone* only: under tools/iso.py it is already imported (asserted to be the clone's);
    bare, the editable finder, the main checkout and this directory are removed and the clone is put first."""
    import importlib

    wt = os.path.realpath(clone)
    here = os.path.dirname(os.path.realpath(__file__))
    loaded = [n for n in sys.modules if n == "environments" or n.startswith("environments.")]
    if loaded:
        origin = os.path.realpath(getattr(sys.modules.get("environments"), "__file__", "") or "")
        if not origin.startswith(wt + "/"):
            raise RuntimeError(f"environments was imported from {origin} before the judge ran, not from {wt}")
    else:
        sys.meta_path[:] = [f for f in sys.meta_path
                            if "__editable__" not in (getattr(f, "__module__", "") or type(f).__module__)]
        sys.path_hooks[:] = [h for h in sys.path_hooks if "__editable__" not in getattr(h, "__module__", "")]
        sys.path[:] = [p for p in sys.path
                       if "__editable__" not in p and os.path.realpath(p or ".") not in (MAIN_CHECKOUT, here)]
        sys.path_importer_cache.clear()
        sys.path.insert(0, wt)
    paths = {}
    for name in ("environments", "environments.shared.reporting.stance_report", "environments.shared.curriculum.stance_gate_v2",
                 "environments.shared.policy_loading", "environments.shared.gait.recorder", "environments.shared.gait.pad_telemetry",
                 "environments.shared.gait.balance_telemetry", "environments.tyrannosaurus_rex.base_env"):
        module = importlib.import_module(name)
        origin = os.path.realpath(module.__file__)
        if not origin.startswith(wt + "/"):
            raise RuntimeError(f"{name} imported from {origin}, not from {wt}")
        paths[name] = origin
    return paths


def modules_under(clone: str) -> int:
    """Every loaded environments.* module resolves under *clone* (raises otherwise); returns how many."""
    wt = os.path.realpath(clone)
    n = 0
    for name, m in sorted(sys.modules.items()):
        if name == "environments" or name.startswith("environments."):
            f = getattr(m, "__file__", None)
            if f is None:
                continue
            if not os.path.realpath(f).startswith(wt + "/"):
                raise RuntimeError(f"{name} resolves to {os.path.realpath(f)}, outside {wt}")
            n += 1
    return n


def git_state(clone: str) -> dict[str, str]:
    """HEAD and the porcelain status of *clone* (read-only: --no-optional-locks, so the index is never refreshed)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", "-C", clone, *args], capture_output=True, text=True,
                              check=True).stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--porcelain"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def _tree_digest(root: str) -> tuple[str, int]:
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
                lines.append(f"{rel}\tsymlink:{os.readlink(p)}")
            else:
                lines.append(f"{rel}\t{sha256_file(p)[len('sha256:'):]}")
    lines.sort()
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


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
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def finite_or_none(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def med(values):
    import numpy as np

    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else None


def mean(values):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(v) / len(v) if v else None


def parse_steps(text: str):
    return text if text in NAMED_CHECKPOINTS else int(text)


def checkpoint_paths(models: str, steps) -> tuple[str, str, str]:
    """(stem, model zip, VecNormalize pickle) of a checkpoint in the notebook's models directory."""
    if steps == "final":
        return "stage1_final", os.path.join(models, "stage1_final.zip"), os.path.join(models, "stage1_final_vecnorm.pkl")
    if steps in ("best", "robust_best"):
        stem = f"{steps}_model"
        return stem, os.path.join(models, f"{stem}.zip"), os.path.join(models, f"{stem}_vecnorm.pkl")
    stem = f"stage1_{int(steps)}_steps"
    return stem, os.path.join(models, f"{stem}.zip"), os.path.join(models, f"stage1_vecnormalize_{int(steps)}_steps.pkl")


def slot_kind(steps) -> str:
    if steps in J_STEPS:
        return "J"
    if steps in REPORT_STEPS:
        return "report (judged like J)"
    return "selection / report"


# ------------------------------------------------------------------------------------------------ the stage
class Ctx:
    """The clone's stance stage, its config as loaded, and the repo objects every panel needs."""

    def __init__(self, arm: str) -> None:
        from environments.shared.cli import _resolve_stage_ref
        from environments.shared.config import hyperparameters_sha256, load_all_stages, load_stage_config
        from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds
        from environments.shared.gait.morphology import Morphology
        from environments.shared.gait.pad_telemetry import PadLayout
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.reporting import stance_report as SR
        from environments.shared.species_names import require_model_id

        self.SR = SR
        self.arm = arm
        self.mid = require_model_id("tyrannosaurus_rex", "anatomical")
        configs = load_all_stages(self.mid)
        self.stage = _resolve_stage_ref("stance", configs, self.mid)
        self.cfg = configs[self.stage]
        assert self.cfg == load_stage_config(self.mid, "stance")
        assert not any(k.startswith("experiment_") for k in self.cfg["env_kwargs"]), "no scratch experiment keys"
        self.curriculum = self.cfg["curriculum_kwargs"]
        self.env_kwargs = dict(self.cfg["env_kwargs"])
        self.ppo_kwargs = dict(self.cfg.get("ppo_kwargs") or {})
        # The recipe digest train_base records in stage_config.json's run block, regenerated from the clone's stage.
        self.hp_sha256 = hyperparameters_sha256(self.cfg, "ppo")
        self.horizon = int(self.env_kwargs.get("max_episode_steps", 1000))
        self.control_dt = float(self.env_kwargs.get("timestep", 0.002)) * int(self.env_kwargs.get("frame_skip", 5))
        self.thresholds = StanceV2Thresholds.from_curriculum(self.curriculum)
        self.thresholds.validate_settle_window(self.control_dt)
        self.settle = int(self.thresholds.settle_steps)
        self.plant_identity = current_plant_identity(self.mid)
        self.env_class = SR.model_env_class(self.mid)
        probe = self.env_class(**self.env_kwargs)
        try:
            self.morphology = Morphology.from_env(probe, self.mid)
            self.layout = PadLayout.from_morphology(self.morphology)
        finally:
            probe.close()
        self.feet = [f.label for f in self.layout.feet]
        self.arm_check = self._check_arm()

    def _check_arm(self) -> dict:
        """The clone's stance stage is the arm's (refuses otherwise): [env] options, [ppo] symmetry, reference, plant."""
        spec = ARMS[self.arm]
        env = {k: self.env_kwargs.get(k) for k in spec["env"]}
        ppo = {k: self.ppo_kwargs.get(k) for k in PPO_SYMMETRY}
        ref = self.curriculum.get("collapse_peak_floor_reference")
        plant = self.plant_identity.to_dict()
        problems = [f"[env] {k} = {env[k]!r}, the arm needs {v!r}" for k, v in spec["env"].items() if env[k] != v]
        problems += [f"[ppo] {k} = {ppo[k]!r}, the arm needs {v!r}" for k, v in PPO_SYMMETRY.items() if ppo[k] != v]
        if ref != spec["collapse_peak_floor_reference"]:
            problems.append(f"collapse_peak_floor_reference {ref!r} != {spec['collapse_peak_floor_reference']}")
        if plant.get("physics_sha256") != PLANT_PHYSICS_SHA256 or plant.get("physics_revision") != PLANT_PHYSICS_REVISION:
            problems.append(f"plant {plant.get('physics_revision')} {plant.get('physics_sha256')} is not "
                            f"r{PLANT_PHYSICS_REVISION}")
        if "max_sole_cop_fore_aft" not in self.thresholds.declared():
            problems.append("the stance stage declares no pad CoP bar (not the r9 stance stage)")
        if problems:
            raise SystemExit(f"the clone's stance stage is not arm {self.arm}'s: " + "; ".join(problems))
        return {"arm": self.arm, "env": env, "ppo_symmetry": ppo, "collapse_peak_floor_reference": ref,
                "physics_revision": plant.get("physics_revision"), "physics_sha256": plant.get("physics_sha256")}


# ------------------------------------------------------------------------------------------------ per-step readings
def _geometry(rec) -> dict:
    """Per recorder: the MTP III joints, soles, sole axes, root body and the feet's reference sites (cop_roll.py's)."""
    import mujoco
    import numpy as np

    key = id(rec)
    g = _STEP_GEOMETRY.get(key)
    if g is not None and g["model"] is rec._env.model:
        return g
    m = rec._env.model
    feet = rec._morphology.feet
    mtp3 = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f.label}_toe_d3_joint") for f in feet]
    assert all(j >= 0 for j in mtp3), mtp3
    root = int(rec._morphology.root_body)
    free = int(m.body_jntadr[root])
    assert free >= 0 and int(m.jnt_type[free]) == int(mujoco.mjtJoint.mjJNT_FREE), "the root has no free joint"
    qadr = int(m.jnt_qposadr[free])
    g = {"model": m, "mtp3": np.asarray(mtp3, dtype=np.int64), "sole": [int(f.sole_geom) for f in feet],
         "axis": [int(f.sole_forward_axis) for f in feet], "sign": [float(f.sole_forward_sign) for f in feet],
         "ankle_forward": [float(f.ankle_forward_m) for f in feet], "root": root, "quat": slice(qadr + 3, qadr + 7),
         "sites": np.asarray([int(f.primary_site) for f in feet], dtype=np.int64)}
    _STEP_GEOMETRY[key] = g
    return g


def _heading_xy(quat):
    """cop_roll.py's heading: the normalised horizontal projection of the root's x axis (wxyz quaternion)."""
    import numpy as np

    w, x, y, z = (float(v) for v in quat)
    f = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    return f / np.linalg.norm(f)


def _step_reading(rec) -> tuple:
    """After env.step: (COM ahead of the mean MTP III anchor along the heading [cop_roll.py com_mtp3_head],
    COM ahead of the mean foot reference site along the root's heading [balance_telemetry's frame], per foot the MTP III
    anchor ahead of the ankle axis along the sole [cop_roll.py mtp3_ahead_ankle])."""
    import numpy as np

    from environments.shared.gait.balance_telemetry import heading_frames

    g = _geometry(rec)
    d = rec._env.data
    com = np.asarray(d.subtree_com[g["root"]], dtype=np.float64)
    anchors = np.asarray(d.xanchor[g["mtp3"]], dtype=np.float64)
    head = _heading_xy(d.qpos[g["quat"]])
    com_mtp3_head = float(head @ (com[:2] - anchors[:, :2].mean(axis=0)))
    forward = heading_frames(np.asarray(d.xquat[g["root"]], dtype=np.float64)[None])[0][0]
    feet_point = np.asarray(d.site_xpos[g["sites"]], dtype=np.float64).mean(axis=0)
    com_site = float(forward @ (com[:2] - feet_point[:2]))
    mt = []
    for k, sole in enumerate(g["sole"]):
        R3 = np.asarray(d.geom_xmat[sole], dtype=np.float64).reshape(3, 3)
        local = R3.T @ (anchors[k] - np.asarray(d.geom_xpos[sole], dtype=np.float64))
        mt.append(float(g["sign"][k] * local[g["axis"][k]] - g["ankle_forward"][k]))
    return com_mtp3_head, com_site, tuple(mt)


def install_trace_capture() -> None:
    from environments.shared.gait.recorder import SubstepContactRecorder

    original_step = SubstepContactRecorder.end_step
    original_episode = SubstepContactRecorder.end_episode

    def end_step(self, *args, **kwargs):
        original_step(self, *args, **kwargs)
        _STEP_READINGS.setdefault(id(self), []).append(_step_reading(self))

    def end_episode(self, *args, **kwargs):
        trace = original_episode(self, *args, **kwargs)
        readings = _STEP_READINGS.pop(id(self), [])
        assert len(readings) == int(trace.length), (len(readings), trace.length)
        TRACES.append((trace, readings))
        return trace

    SubstepContactRecorder.end_step = end_step
    SubstepContactRecorder.end_episode = end_episode


def install_band_capture() -> None:
    """Band-on arms only (report only).  Instance-level, read-only wrappers, installed when a recorder attaches (after
    run_panel's plant validation, which hashes the env class's methods; roll_split.py's method): the attached env's step
    keeps its band info keys, and the recorder's end_episode files them under the closed trace (BAND_BY_TRACE)."""
    from environments.shared.gait.recorder import SubstepContactRecorder

    original_attach = SubstepContactRecorder.attach
    original_episode = SubstepContactRecorder.end_episode

    def attach(self, *args, **kwargs):
        rec = original_attach(self, *args, **kwargs)
        env = self._env
        if not getattr(env, "_judge5_band_wrapped", False):
            original_step = env.step

            def step(action):
                out = original_step(action)
                info = out[4]
                _BAND_STEPS.setdefault(id(env), []).append({k: float(info[k]) for k in BAND_KEYS if k in info})
                return out

            env.step = step
            env._judge5_band_wrapped = True
        _BAND_STEPS.pop(id(env), None)
        return rec

    def end_episode(self, *args, **kwargs):
        trace = original_episode(self, *args, **kwargs)
        rows = _BAND_STEPS.pop(id(self._env), [])
        assert len(rows) == int(trace.length), (len(rows), trace.length)
        BAND_BY_TRACE[id(trace)] = rows
        return trace

    SubstepContactRecorder.attach = attach
    SubstepContactRecorder.end_episode = end_episode
    BAND_CAPTURE["on"] = True


def _band_excess(v: float, lo: float, hi: float) -> float:
    return max(lo - v, 0.0) + max(v - hi, 0.0)


def band_episode(ctx: Ctx, trace) -> dict | None:
    """One episode's band readouts over the window [settle, L) (report only; statue_ref_s5.py's definitions: the info
    keys as the env writes them, a reading on an unloaded foot written 0.0).  None when the band is not captured."""
    rows = BAND_BY_TRACE.get(id(trace))
    if rows is None:
        return None
    L = int(trace.length)
    assert len(rows) == L, (len(rows), L)
    paid = [r.get("reward_stance_cop_band", 0.0) for r in rows]
    w = rows[ctx.settle:L]
    out = {"length": L, "window": [ctx.settle, L], "window_steps": len(w), "window_foot_steps": 2 * len(w),
           "paid_episode": float(sum(paid)), "paid_window": float(sum(paid[ctx.settle:L])),
           "paid_before_window": float(sum(paid[:ctx.settle])),
           "window_steps_paying": sum(1 for v in paid[ctx.settle:L] if v < 0.0),
           "guard_0": 0, "guard_ramp": 0, "guard_1": 0, "counted": 0, "counted_equals_guard_ge_1": True,
           "whole_out": 0, "pad_out": 0, "pad_binds": 0, "share_ramp_025_035": 0, "share_below_025": 0,
           "share_min": None, "guard_below_1_before_window": 0,
           "quality_mean": {}, "digit_share_mean": {}, "stance_cop_band_quality_mean": None}
    for r in rows[:ctx.settle]:
        out["guard_below_1_before_window"] += sum(1 for s in BAND_FEET if r[f"{s}_cop_band_guard"] < 1.0)
    shares = []
    for r in w:
        for s in BAND_FEET:
            g, c = r[f"{s}_cop_band_guard"], r[f"{s}_cop_band_counted"]
            out["guard_0"] += int(g == 0.0)
            out["guard_ramp"] += int(0.0 < g < 1.0)
            out["guard_1"] += int(g >= 1.0)
            out["counted"] += int(c == 1.0)
            if (c == 1.0) != (g >= 1.0):
                out["counted_equals_guard_ge_1"] = False
            wh = _band_excess(r[f"{s}_cop_band_foot_cop"], *BAND_WHOLE)
            pd = _band_excess(r[f"{s}_cop_band_pad_cop"], *BAND_PAD)
            out["whole_out"] += int(wh > 0.0)
            out["pad_out"] += int(pd > 0.0)
            out["pad_binds"] += int(pd > wh)
            share = r[f"{s}_cop_band_load_share"]
            shares.append(share)
            out["share_ramp_025_035"] += int(BAND_SHARE_FULL_PRICE <= share < BAND_SHARE_MIN)
            out["share_below_025"] += int(share < BAND_SHARE_FULL_PRICE)
    out["share_min"] = min(shares) if shares else None
    for s in BAND_FEET:
        out["quality_mean"][s] = mean(r[f"{s}_cop_band_quality"] for r in w)
        out["digit_share_mean"][s] = mean(r[f"{s}_cop_band_digit_share"] for r in w)
    out["stance_cop_band_quality_mean"] = mean(r["stance_cop_band_quality"] for r in w)
    return out


def band_summary(rows: list[dict]) -> dict | None:
    """The panel's (or statue's) band readouts over its full-horizon episodes (report only)."""
    import numpy as np

    have = [r for r in rows if r.get("band") is not None]
    if not have:
        return None
    full = [r for r in have if r["reached_horizon"]]
    counts = ("window_foot_steps", "guard_0", "guard_ramp", "guard_1", "counted", "whole_out", "pad_out", "pad_binds",
              "share_ramp_025_035", "share_below_025", "window_steps_paying", "guard_below_1_before_window")
    out = {"episodes": len(have), "full_horizon_episodes": len(full),
           "full_horizon": {k: sum(r["band"][k] for r in full) for k in counts}}
    out["full_horizon"]["counted_equals_guard_ge_1"] = all(r["band"]["counted_equals_guard_ge_1"] for r in full)
    out["full_horizon"]["share_min"] = min((r["band"]["share_min"] for r in full if r["band"]["share_min"] is not None),
                                           default=None)
    paid = np.asarray([r["band"]["paid_episode"] for r in full], dtype=np.float64)
    out["paid_per_full_horizon_episode"] = None if not paid.size else {
        "mean": float(paid.mean()), "median": float(np.median(paid)), "worst": float(paid.min()),
        "least": float(paid.max()), "episodes_paying_over_5": int(np.sum(paid < -5.0))}
    out["paid_per_episode_all_mean"] = float(np.mean([r["band"]["paid_episode"] for r in have]))
    out["paid_by_seed"] = {str(r["seed"]): r["band"]["paid_episode"] for r in have}
    for s in BAND_FEET:
        out.setdefault("quality_mean_full_horizon", {})[s] = mean(r["band"]["quality_mean"][s] for r in full)
        out.setdefault("digit_share_mean_full_horizon", {})[s] = mean(r["band"]["digit_share_mean"][s] for r in full)
    return out


def episode_cop(ctx: Ctx, trace, readings: list) -> dict:
    """K2b and K2c inputs of one episode over the window [settle, T): per foot the signed whole-foot CoP (span units, as
    the pad telemetry's <label>_foot_cop_fore_aft; and m ahead of MTP III), the both-feet whole-foot CoP x_c, the COM
    ahead of the mean MTP III anchor and ahead of the mean foot reference site (all None for an empty window)."""
    import numpy as np

    from environments.shared.gait.stance_metrics import whole_foot_cop

    T = int(trace.length)
    out = {"window": [ctx.settle, T], "foot_cop_mm_ahead_mtp3": {}, "mtp3_ahead_ankle_m": {}, "foot_frame": {},
           "xc_both_feet_m": None, "com_mtp3_head_m": None, "com_fore_aft_footsite_m": None, "x_vs_w_max_abs_m": None}
    if T <= ctx.settle:
        return out
    w = slice(ctx.settle, T)
    ch = np.array([r[0] for r in readings], dtype=np.float64)
    cs = np.array([r[1] for r in readings], dtype=np.float64)
    mt = np.array([r[2] for r in readings], dtype=np.float64)                  # (T, F)
    spanned, wcop, loaded = whole_foot_cop(trace)
    x = np.asarray(trace.foot_cop, dtype=np.float64) - mt                      # (T, F) m ahead of MTP III
    force = np.where(loaded, np.asarray(trace.leg_floor_mean, dtype=np.float64), 0.0)
    total = force.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xc_t = np.where(total > 0, np.nansum(np.where(loaded, x, 0.0) * force, axis=1) / np.maximum(total, 1e-9), np.nan)
    xw, lw, ww = x[w], loaded[w], wcop[w]
    span = np.asarray(trace.foot_support_span, dtype=np.float64).reshape(-1, 2)
    diffs = []
    for k, label in enumerate(ctx.feet):
        mine = lw[:, k]
        xk = float(xw[mine, k].mean()) if mine.any() else None
        out["foot_cop_mm_ahead_mtp3"][label] = None if xk is None else 1000.0 * xk
        out["mtp3_ahead_ankle_m"][label] = float(mt[w, k].mean())
        mid, half = float(span[k].mean()), float((span[k, 1] - span[k, 0]) / 2.0)
        out["foot_frame"][label] = {"span_mid_ahead_ankle_m": mid, "span_half_length_m": half,
                                    "x_at_w0_m": mid - float(mt[w, k].mean())}
        if mine.any():  # the criteria's r9 frame, x = 60 + 70 w mm, against the direct window mean
            diffs.append(abs(xk - (R9_X_AT_W0_M + R9_X_PER_W_M * float(ww[mine, k].mean()))))
    finite = xc_t[w][np.isfinite(xc_t[w])]
    out["xc_both_feet_m"] = float(finite.mean()) if finite.size else None
    out["com_mtp3_head_m"] = float(ch[w].mean())
    out["com_fore_aft_footsite_m"] = float(cs[w].mean())
    out["x_vs_w_max_abs_m"] = max(diffs) if diffs else None
    return out


def pad_rows(ctx: Ctx, traces: list) -> list[dict]:
    from environments.shared.gait.pad_telemetry import episode_pad_telemetry

    return [{k: finite_or_none(v) for k, v in episode_pad_telemetry(t, ctx.layout, settle_steps=ctx.settle).items()}
            for t, _ in traces]


def check_pad_identities(ctx: Ctx, seed, pad: dict, sole, foot) -> None:
    """The telemetry's per-foot gate statistics reproduce the episode metrics bit for bit (PR #620 / #622)."""
    gate = [g for g in (pad.get(f"{f}_sole_cop_fore_aft_gate") for f in ctx.feet) if g is not None]
    if sole is not None:
        assert gate and max(gate) == sole, (seed, "pad", gate, sole)
    fgate = [g for g in (pad.get(f"{f}_foot_cop_fore_aft_gate") for f in ctx.feet) if g is not None]
    if foot is not None:
        assert fgate and max(fgate) == foot, (seed, "foot", fgate, foot)


def roll_rail(full: bool, tilt, lift) -> bool:
    return bool(full and ((tilt is not None and tilt > ROLL_TILT_DEG) or (lift is not None and lift > ROLL_LIFT_M)))


def k2a_over(full: bool, sole) -> bool:
    """K2a: a non-full-horizon episode, or (unmeasured or) the pad's max_sole_cop_fore_aft > 0.80."""
    return bool((not full) or sole is None or sole > PAD_COP_BAR)


def k2b_outside(ctx: Ctx, full: bool, pad: dict) -> bool:
    """K2b: a non-full-horizon episode, or either foot's signed whole-foot CoP unmeasured or outside [-0.929, +0.571]."""
    if not full:
        return True
    for f in ctx.feet:
        v = pad.get(f"{f}_foot_cop_fore_aft")
        if v is None or v < K2B_LO or v > K2B_HI:
            return True
    return False


def add_k2_fields(ctx: Ctx, row: dict, trace, readings: list) -> None:
    pad = row["pad"]
    cop = episode_cop(ctx, trace, readings)
    row["foot_cop_fore_aft"] = {f: pad.get(f"{f}_foot_cop_fore_aft") for f in ctx.feet}
    row["foot_cop_mm_ahead_mtp3"] = cop["foot_cop_mm_ahead_mtp3"]
    row["xc_both_feet_m"] = cop["xc_both_feet_m"]
    row["com_mtp3_head_m"] = cop["com_mtp3_head_m"]
    row["com_fore_aft_footsite_m"] = cop["com_fore_aft_footsite_m"]
    row["cop_frame"] = {k: cop[k] for k in ("window", "mtp3_ahead_ankle_m", "foot_frame", "x_vs_w_max_abs_m")}
    row["k2a_over"] = k2a_over(row["reached_horizon"], row.get("max_sole_cop_fore_aft"))
    row["k2b_outside"] = k2b_outside(ctx, row["reached_horizon"], pad)


def episode_rows(ctx: Ctx, evidence: list[dict], traces: list) -> list[dict]:
    """One row per panel episode (the v2 evidence row's classification and metrics, the trace's termination)."""
    from environments.shared.curriculum.stance_gate_v2 import is_hop_or_fall, reason_key

    pads = pad_rows(ctx, traces)
    rows = []
    for ev, pad, (tr, readings) in zip(evidence, pads, traces, strict=True):
        assert int(tr.length) == int(ev["length"]), (tr.length, ev["length"])
        sole, foot = finite_or_none(ev.get("max_sole_cop_fore_aft")), finite_or_none(ev.get("max_foot_cop_fore_aft"))
        check_pad_identities(ctx, ev["seed"], pad, sole, foot)
        reasons = ev.get("reasons")
        full = bool(ev["reached_horizon"])
        tilt, lift = finite_or_none(ev.get("max_sole_tilt_deg")), finite_or_none(ev.get("max_sole_corner_lift_m"))
        row = {
            "seed": int(ev["seed"]), "length": int(ev["length"]), "reached_horizon": full, "full_horizon": full,
            "clean": ev.get("clean"), "reasons": reasons,
            "failing_rails": None if reasons is None else sorted({reason_key(r) for r in reasons}),
            "hop_or_fall": None if reasons is None else bool(is_hop_or_fall(reasons)),
            "reward": finite_or_none(ev["reward"]),
            "termination_reason": getattr(tr, "termination_reason", None),
            "max_sole_cop_fore_aft": sole, "max_foot_cop_fore_aft": foot,
            "max_sole_tilt_deg": tilt, "max_sole_corner_lift_m": lift,
            "roll_rail": roll_rail(full, tilt, lift),
            "pad": pad,
        }
        add_k2_fields(ctx, row, tr, readings)
        if BAND_CAPTURE["on"]:
            row["band"] = band_episode(ctx, tr)
        rows.append(row)
    return rows


def summarize_episodes(ctx: Ctx, rows: list[dict]) -> dict:
    """Falls, the K2a / K2b counts, the K2c medians, the roll rail, the r10 whole-foot rail, per-foot statistics."""
    full = [r for r in rows if r["reached_horizon"]]

    def over(r, key, bar):
        v = r.get(key)
        return v is not None and v > bar

    xs = [r.get("xc_both_feet_m") for r in full]
    cm = [r.get("com_mtp3_head_m") for r in full]
    out = {
        "n": len(rows), "n_full_horizon": len(full),
        "falls": sum(1 for r in rows if not r["reached_horizon"]),
        "fall_seeds": [r["seed"] for r in rows if not r["reached_horizon"]],
        "fall_lengths": [r["length"] for r in rows if not r["reached_horizon"]],
        # K2a (criteria): every episode; a non-full-horizon or unmeasured one counts as over
        "k2a_count": sum(1 for r in rows if r.get("k2a_over")),
        "k2a_seeds": [r["seed"] for r in rows if r.get("k2a_over")],
        # K2b (criteria): every episode; a non-full-horizon or unmeasured one counts as outside
        "k2b_count": sum(1 for r in rows if r.get("k2b_outside")),
        "k2b_seeds": [r["seed"] for r in rows if r.get("k2b_outside")],
        # K2c inputs (criteria: panel median of the both-feet whole-foot CoP; the scorer takes the full-horizon median)
        "xc_median_full_horizon_m": med(xs), "xc_mean_full_horizon_m": mean(xs),
        "com_mtp3_head_median_full_horizon_m": med(cm), "com_mtp3_head_mean_full_horizon_m": mean(cm),
        "xc_minus_com_max_abs_full_horizon_m": max((abs(a - b) for a, b in zip(xs, cm, strict=True) if a is not None and b is not None),
                                                   default=None),
        "com_fore_aft_footsite_mean_full_horizon_m": mean(r.get("com_fore_aft_footsite_m") for r in full),
        "com_fore_aft_footsite_median_full_horizon_m": med(r.get("com_fore_aft_footsite_m") for r in full),
        "x_vs_w_max_abs_m": max((r["cop_frame"]["x_vs_w_max_abs_m"] for r in rows
                                 if (r.get("cop_frame") or {}).get("x_vs_w_max_abs_m") is not None), default=None),
        # the pad rail as r9's H3a counted it (all episodes, measured over the bar), and over the full-horizon ones
        "cop_gt_bar_gate": sum(1 for r in rows if over(r, "max_sole_cop_fore_aft", PAD_COP_BAR)),
        "cop_gt_bar_gate_full_horizon": sum(1 for r in full if over(r, "max_sole_cop_fore_aft", PAD_COP_BAR)),
        "cop_unmeasured": sum(1 for r in rows if r.get("max_sole_cop_fore_aft") is None),
        "gate_cop_median": med(r.get("max_sole_cop_fore_aft") for r in rows),
        # the r10 whole-foot rail, report only on r9
        "foot_cop_gt_bar": sum(1 for r in rows if over(r, "max_foot_cop_fore_aft", FOOT_COP_BAR)),
        "foot_cop_gt_bar_full_horizon": sum(1 for r in full if over(r, "max_foot_cop_fore_aft", FOOT_COP_BAR)),
        "roll_rail_full_horizon": sum(1 for r in rows if r.get("roll_rail")),
        "reward_mean_full": (sum(r["reward"] for r in full) / len(full)) if full and all(r["reward"] is not None for r in full) else None,
        "per_foot": {},
    }
    for foot in ctx.feet:
        def p(r, key, foot=foot):
            return (r.get("pad") or {}).get(f"{foot}_{key}")
        out["per_foot"][foot] = {
            "signed_cop_median": med(p(r, "sole_cop_fore_aft") for r in full),
            "gate_cop_median": med(p(r, "sole_cop_fore_aft_gate") for r in full),
            "digit_share_median": med(p(r, "digit_share") for r in full),
            "foot_signed_cop_mean": mean(p(r, "foot_cop_fore_aft") for r in full),
            "foot_signed_cop_median": med(p(r, "foot_cop_fore_aft") for r in full),
            "foot_signed_cop_min": min((v for v in (p(r, "foot_cop_fore_aft") for r in full) if v is not None), default=None),
            "foot_signed_cop_max": max((v for v in (p(r, "foot_cop_fore_aft") for r in full) if v is not None), default=None),
            "foot_cop_mm_ahead_mtp3_median": med((r.get("foot_cop_mm_ahead_mtp3") or {}).get(foot) for r in full),
            "foot_gate_cop_median": med(p(r, "foot_cop_fore_aft_gate") for r in full),
        }
    return out


# ------------------------------------------------------------------------------------------------ panels
STATUE_ROW_KEYS = ("seed", "length", "reached_horizon", "clean", "reasons", "reward", "termination_reason",
                   "max_foot_cop_fore_aft", "max_sole_cop_fore_aft", "max_sole_tilt_deg", "max_sole_corner_lift_m",
                   "roll_rail", "pad", "foot_cop_fore_aft", "foot_cop_mm_ahead_mtp3", "xc_both_feet_m", "com_mtp3_head_m",
                   "com_fore_aft_footsite_m", "k2a_over", "k2b_outside")


def v2_panel(ctx: Ctx, *, model_path: str, vecnorm_path: str, episodes: int, seed: int) -> dict:
    """The stance_quality/v2 report (repo code) on *episodes* episodes from *seed*, the statue on the same seeds, and
    the per-episode pad telemetry and K2 statistics of both."""
    SR = ctx.SR
    predict, note = SR._load_policy(model_path, vecnorm_path, lambda: ctx.env_class(**ctx.env_kwargs),
                                    plant_identity=ctx.plant_identity, allow_legacy_plant=False)
    description = f"{os.path.basename(model_path)} — {note}"
    TRACES.clear()
    BAND_BY_TRACE.clear()
    t0 = time.monotonic()
    report = SR._stance_v2_report(
        ctx.mid, ctx.stage, curriculum=ctx.curriculum, env_class=ctx.env_class, env_kwargs=ctx.env_kwargs,
        thresholds=ctx.thresholds, predict=predict, description=description, zero_action=False,
        model_path=model_path, vecnorm_path=vecnorm_path, episodes=episodes, seed=seed,
        horizon=ctx.horizon, control_dt=ctx.control_dt, plant_identity=ctx.plant_identity, allow_legacy_plant=False,
    )
    seconds = time.monotonic() - t0
    assert report["statue"] is not None and not report["statue"]["reused_policy_panel"], "the statue panel must be rolled"
    assert len(TRACES) == 2 * episodes, (len(TRACES), 2 * episodes)
    pol_traces, st_traces = TRACES[:episodes], TRACES[episodes:]
    rows = episode_rows(ctx, report["episode_evidence"], pol_traces)
    st_rows = episode_rows(ctx, report["statue"]["episode_evidence"], st_traces)
    statue_summary = summarize_episodes(ctx, st_rows)
    statue_summary["reference"] = report["statue"]["reference"]
    if BAND_CAPTURE["on"]:
        statue_summary["band_summary"] = band_summary(st_rows)
    st_keys = STATUE_ROW_KEYS + (("band",) if BAND_CAPTURE["on"] else ())
    statue_summary["terminations"] = {}
    for r in st_rows:
        statue_summary["terminations"][str(r["termination_reason"])] = statue_summary["terminations"].get(str(r["termination_reason"]), 0) + 1
    result = report["result"]
    cf = result.get("criterion_failures", {})
    mean_reward, statue_reward = result["mean_reward"], result.get("statue_mean_reward")
    return {
        "policy": report["policy"], "seeds": [seed, seed + episodes - 1], "episodes": episodes,
        "panel_seconds": seconds,
        "verdict": "PASS" if report["passed"] else "FAIL", "passed": report["passed"],
        "n_clean": result["n_clean"], "clean_lcb": result["clean_lcb"], "clean_fraction": result["clean_fraction"],
        "n_full_horizon": result["n_full_horizon"], "full_horizon_fraction": result.get("full_horizon_fraction"),
        "mean_reward": mean_reward, "statue_mean_reward": statue_reward,
        "reward_ratio_to_statue": (mean_reward / statue_reward) if statue_reward else None,
        "hop_or_fall_episodes": result.get("hop_or_fall_episodes"),
        "gate_failures": report["failures"],
        "failing_rails": {k: v for k, v in cf.items() if v},
        "criterion_failures": cf,
        "terminations": report["terminations"],
        "summary": summarize_episodes(ctx, rows),
        "statue_on_same_seeds": statue_summary,
        "statue_episodes_detail": [{k: r.get(k) for k in st_keys} for r in st_rows],
        "scored_gate_kind": report["scored_gate_kind"], "stage_gate_kind": report["gate_kind"],
        "schema": report["schema"], "task_sha256": report["task_sha256"], "handoff": report["handoff"],
        "thresholds": report["thresholds"], "measurement_sha256": report["measurement_sha256"],
        "result": result,
        "episodes_detail": rows,
        **({"band_summary": band_summary(rows)} if BAND_CAPTURE["on"] else {}),
    }


def load_checkpoint(ctx: Ctx, model_path: str, vecnorm_path: str):
    from environments.shared.policy_loading import load_sb3_checkpoint

    model, normalizer, _ = load_sb3_checkpoint(model_path, vecnorm_path, lambda: ctx.env_class(**ctx.env_kwargs),
                                               guess_sidecar=False, allow_unnormalized=False,
                                               plant_identity=ctx.plant_identity, allow_legacy_plant=False)
    return model, normalizer


def noise_matched(ctx: Ctx, model, normalizer, *, sigma_arg: str, sigma, sigma_source: str, space: str) -> dict:
    """The r9 judge's noise_matched (scale None), arithmetic unchanged: mean action + iid N(0, sigma^2), numpy RNG
    default_rng(episode seed); space "joint" clips the command first and adds the noise after it, "policy" adds it to
    the raw mean; the sum is clipped to the action bounds either way."""
    assert space in ("policy", "joint"), space
    import numpy as np
    import torch
    from stable_baselines3.common.policies import BaseModel

    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder
    from environments.shared.gait.stance_metrics import episode_stance_metrics

    env = ctx.env_class(**ctx.env_kwargs)
    low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
    pol = model.policy
    own = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
    sigma = own if sigma is None else np.asarray(sigma, dtype=np.float64)
    assert sigma.shape == own.shape, (sigma.shape, own.shape)
    vmean, vvar = np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var)
    eps, clip_obs = normalizer.epsilon, normalizer.clip_obs
    rec = SubstepContactRecorder(env, Morphology.from_env(env, ctx.mid)).attach()
    TRACES.clear()
    rows = []
    t0 = time.monotonic()
    try:
        for i in range(NOISE_EPISODES):
            seed = NOISE_SEED + i
            rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=seed)
            steps, reward = 0, 0.0
            while True:
                o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
                with torch.no_grad():
                    f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
                    mu = pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)
                if space == "joint":
                    cmd = np.clip(mu, low, high)
                    act = np.clip(cmd + sigma * rng.standard_normal(cmd.shape), low, high)
                else:
                    act = np.clip(mu + sigma * rng.standard_normal(mu.shape), low, high)
                obs, r, term, trunc, info = env.step(act)
                rec.end_step(act, float(r))
                reward += float(r)
                steps += 1
                if term or trunc:
                    tr = rec.end_episode(terminated=bool(term), truncated=bool(trunc),
                                         termination_reason=str(info.get("termination_reason", "")))
                    break
            m = episode_stance_metrics(tr, settle_steps=ctx.settle)
            rows.append({"seed": seed, "length": steps, "reached_horizon": steps >= ctx.horizon, "reward": reward,
                         "termination_reason": getattr(tr, "termination_reason", None),
                         "max_sole_cop_fore_aft": finite_or_none(m.max_sole_cop_fore_aft),
                         "max_foot_cop_fore_aft": finite_or_none(getattr(m, "max_foot_cop_fore_aft", None)),
                         "max_sole_tilt_deg": finite_or_none(m.max_sole_tilt_deg),
                         "max_sole_corner_lift_m": finite_or_none(m.max_sole_corner_lift_m)})
    finally:
        rec.detach()
        env.close()
    seconds = time.monotonic() - t0
    assert len(TRACES) == NOISE_EPISODES, len(TRACES)
    for row, pad, (tr, readings) in zip(rows, pad_rows(ctx, TRACES), TRACES, strict=True):
        check_pad_identities(ctx, row["seed"], pad, row["max_sole_cop_fore_aft"], row["max_foot_cop_fore_aft"])
        row["pad"] = pad
        row["roll_rail"] = roll_rail(row["reached_horizon"], row["max_sole_tilt_deg"], row["max_sole_corner_lift_m"])
        add_k2_fields(ctx, row, tr, readings)
    out = summarize_episodes(ctx, rows)
    out.update({
        "sigma": sigma_arg, "sigma_source": sigma_source, "sigma_vector": sigma.tolist(), "sigma_mean": float(sigma.mean()),
        "space": space, "noise_added": "after the command clip (joint space)" if space == "joint" else "to the raw mean (policy space)",
        "policy_std_mean": float(own.mean()), "action_scale": None,
        "seeds": [NOISE_SEED, NOISE_SEED + NOISE_EPISODES - 1], "rng": "numpy default_rng(episode seed)",
        "cop_gt_0p8_noise_py": sum(1 for r in rows if (r["max_sole_cop_fore_aft"] or 0) > 0.8),
        "panel_seconds": seconds, "episodes_detail": rows,
    })
    return out


def _zip_sigma(path: str):
    """exp(log_std) read from a checkpoint zip's policy.pth (the noise battery's _sigma_vector)."""
    import io
    import zipfile

    import numpy as np
    import torch

    with zipfile.ZipFile(path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    return np.exp(sd["log_std"].numpy().astype(np.float64))


def rolled_falls(ctx: Ctx, predict, seeds: list[int]) -> list[dict]:
    """Each seed one heading-0 episode through stance_report.run_panel (floor truth): length, termination, reward."""
    rows = []
    for s in seeds:
        TRACES.clear()
        res = ctx.SR.run_panel(ctx.mid, predict=predict, episodes=1, seed=s, settle_steps=ctx.settle, horizon=ctx.horizon,
                               env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity, control_dt=ctx.control_dt,
                               floor_truth=True, spawn_yaw=None)
        m = res["stance_metrics"][0]
        assert len(TRACES) == 1
        rows.append({"seed": int(s), "length": int(m.length), "reached_horizon": int(m.length) >= ctx.horizon,
                     "termination_reason": getattr(TRACES[0][0], "termination_reason", None),
                     "reward": float(res["rewards"][0]),
                     "max_sole_cop_fore_aft": finite_or_none(getattr(m, "max_sole_cop_fore_aft", None))})
    return rows


def common_noise(ctx: Ctx, model, normalizer, *, episodes: int) -> dict:
    """Falls under the common noise: a = clip(mu + sigma z), sigma = exp(log_std) of scratch arm L s60 robust_best, z from
    default_rng(SeedSequence(entropy=episode seed, spawn_key=(1,))) (the noise battery's common random numbers),
    rolled through stance_report.run_panel exactly as foot/tonight/noise_robustness/scripts/noise_panel.py did."""
    import numpy as np
    import torch

    sha = sha256_file(COMMON_SIGMA_ZIP)
    if sha != "sha256:" + COMMON_SIGMA_SHA256:
        raise SystemExit(f"{COMMON_SIGMA_ZIP} sha256 {sha} != the pinned {COMMON_SIGMA_SHA256}")
    noise_std = _zip_sigma(COMMON_SIGMA_ZIP)
    pol = model.policy
    assert not getattr(pol, "use_sde", False)
    low = np.asarray(model.action_space.low, dtype=np.float32)
    high = np.asarray(model.action_space.high, dtype=np.float32)
    vmean = np.asarray(normalizer.obs_rms.mean, dtype=np.float64)
    vvar = np.asarray(normalizer.obs_rms.var, dtype=np.float64)
    eps_n, clip_o = float(normalizer.epsilon), float(normalizer.clip_obs)

    class CommonSigma:
        def __init__(self):
            self.index = -1
            self.rng = None

        def reset(self):  # run_panel calls it once per episode, right after env.reset(seed=seed+index)
            self.index += 1
            self.rng = np.random.default_rng(np.random.SeedSequence(entropy=COMMON_SEED + self.index, spawn_key=(1,)))

        def __call__(self, obs):
            normalized = np.clip((obs - vmean) / np.sqrt(vvar + eps_n), -clip_o, clip_o).astype(np.float32)
            with torch.no_grad():
                mu = pol.get_distribution(torch.as_tensor(normalized[None])).distribution.mean.cpu().numpy()[0]
            z = self.rng.standard_normal(mu.shape[0]).astype(np.float32)
            return np.clip(mu + noise_std.astype(np.float32) * z, low, high)

    predict = CommonSigma()
    TRACES.clear()
    t0 = time.monotonic()
    res = ctx.SR.run_panel(ctx.mid, predict=predict, episodes=episodes, seed=COMMON_SEED, settle_steps=ctx.settle,
                           horizon=ctx.horizon, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity,
                           control_dt=ctx.control_dt, floor_truth=True, spawn_yaw=None)
    seconds = time.monotonic() - t0
    assert predict.index == episodes - 1 and len(TRACES) == episodes, (predict.index, len(TRACES))
    rows = [{"seed": COMMON_SEED + i, "length": int(m.length), "reached_horizon": int(m.length) >= ctx.horizon,
             "termination_reason": getattr(TRACES[i][0], "termination_reason", None), "reward": float(res["rewards"][i])}
            for i, m in enumerate(res["stance_metrics"])]
    falls = [r["seed"] for r in rows if not r["reached_horizon"]]
    return {"sigma_source": COMMON_SIGMA_ZIP, "sigma_source_sha256": "sha256:" + COMMON_SIGMA_SHA256,
            "sigma_vector": noise_std.tolist(), "sigma_mean": float(noise_std.mean()), "space": "policy",
            "rng": "numpy default_rng(SeedSequence(entropy=episode seed, spawn_key=(1,)))",
            "seeds": [COMMON_SEED, COMMON_SEED + episodes - 1], "n": episodes, "falls": len(falls), "fall_seeds": falls,
            "reference": "scratch arm L s60 6M: 38/40 falls (foot/tonight/noise_robustness/work/summary.txt)",
            "panel_seconds": seconds, "episodes_detail": rows}


# ------------------------------------------------------------------------------------------------ run bindings
def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def norm(x):
    """JSON-normalised (tuples -> lists, numpy -> python) for comparing a loaded config with a recorded one."""
    return json.loads(json.dumps(jsonable(x)))


def run_bindings(ctx: Ctx, a, stage_dir: str, model_path: str, panel_task: str, git: dict) -> dict:
    """Bind the run's records and the checkpoint to the clone (see the module docstring); raise on a hard mismatch."""
    from environments.shared.task_fingerprint import read_checkpoint_attribute

    checks: dict = {"relaxed": {"no_symmetry_check": bool(a.no_symmetry_check), "task_binding": a.task_binding,
                                "allow_missing_records": bool(a.allow_missing_records),
                                "allow_seed_mismatch": bool(a.allow_seed_mismatch),
                                "allow_commit_mismatch": bool(a.allow_commit_mismatch)}}
    hard: list[str] = []
    clone_plant = ctx.plant_identity.to_dict()
    pij = os.path.join(stage_dir, "plant_identity.json")
    if os.path.exists(pij):
        run_plant = read_json(pij)
        checks["run_plant_identity"] = run_plant
        ok = run_plant.get("physics_sha256") == clone_plant["physics_sha256"]
        checks["plant_physics_sha256"] = {"run": run_plant.get("physics_sha256"), "clone": clone_plant["physics_sha256"], "match": ok}
        checks["plant_identity_differences"] = sorted(k for k in set(run_plant) | set(clone_plant) if run_plant.get(k) != clone_plant.get(k))
        if not ok:
            hard.append(f"{pij} physics_sha256 {run_plant.get('physics_sha256')} != the clone's {clone_plant['physics_sha256']}")
    else:
        checks["plant_physics_sha256"] = {"run": None, "clone": clone_plant["physics_sha256"], "match": None}
        (checks.setdefault("records_missing", [])).append("plant_identity.json")
        if not a.allow_missing_records:
            hard.append(f"{pij} missing (required)")
    ck_plant = read_checkpoint_attribute(model_path, "_mesozoic_plant_identity")
    checks["checkpoint_plant_physics_sha256"] = {"checkpoint": (ck_plant or {}).get("physics_sha256"),
                                                 "match": (ck_plant or {}).get("physics_sha256") == clone_plant["physics_sha256"]}
    if not checks["checkpoint_plant_physics_sha256"]["match"]:
        hard.append(f"checkpoint physics_sha256 {(ck_plant or {}).get('physics_sha256')} != the clone's")
    ck_seed = read_checkpoint_attribute(model_path, "seed")
    checks["seed"] = {"label": int(a.seed), "checkpoint": ck_seed, "match": ck_seed == int(a.seed)}
    if ck_seed != int(a.seed) and not a.allow_seed_mismatch:
        hard.append(f"the checkpoint records seed {ck_seed}, the judge was told {a.seed}")
    checks["checkpoint_num_timesteps"] = read_checkpoint_attribute(model_path, "num_timesteps")
    # criteria.txt: final is stage1_final at 12,001,280 steps; a periodic slot's checkpoint holds its step.  Recorded,
    # not refused here (an early stop's final may still be judged for a report); score_stance5.py refuses a mismatch.
    want_ts = FINAL_TIMESTEPS if a.steps == "final" else a.steps if isinstance(a.steps, int) else None
    checks["checkpoint_num_timesteps_expected"] = want_ts
    checks["checkpoint_num_timesteps_match"] = None if want_ts is None else checks["checkpoint_num_timesteps"] == want_ts
    if checks["checkpoint_num_timesteps_match"] is False:
        print(f"WARNING: {model_path} holds num_timesteps {checks['checkpoint_num_timesteps']}, slot {a.steps} needs "
              f"{want_ts:,}: score_stance5.py will refuse this file", flush=True)
    recorded = read_checkpoint_attribute(model_path, "mesozoic_symmetry")
    sym = {"recorded": recorded}
    try:
        from environments.shared.symmetry import CHECKPOINT_ATTRIBUTE, MirrorMap, pop_symmetry_config, symmetry_mismatch

        assert CHECKPOINT_ATTRIBUTE == "mesozoic_symmetry"
        config = pop_symmetry_config(dict(ctx.ppo_kwargs))
        mm = MirrorMap.load(ctx.mid, ctx.plant_identity)
        wanted = config.checkpoint_record(mm) if config is not None else None
        sym.update({"clone_mirror_map_sha256": f"sha256:{mm.sha256}", "clone_mirror_map_path": mm.path, "wanted": wanted,
                    "mirror_map_sha256_match": (recorded or {}).get("mirror_map_sha256") == f"sha256:{mm.sha256}",
                    "mismatch": symmetry_mismatch(recorded, wanted)})
    except Exception as exc:  # noqa: BLE001 - recorded, and refused below unless --no-symmetry-check
        sym.update({"clone_mirror_map_sha256": None, "error": f"{type(exc).__name__}: {exc}", "mirror_map_sha256_match": None,
                    "mismatch": "not checked"})
    checks["symmetry"] = sym
    if not a.no_symmetry_check and not (sym.get("mirror_map_sha256_match") is True and sym.get("mismatch") is None):
        hard.append(f"mesozoic_symmetry {recorded!r} does not match the clone ({sym.get('mismatch')}; "
                    f"clone mirror map {sym.get('clone_mirror_map_sha256')})")
    ck_task = (read_checkpoint_attribute(model_path, "mesozoic_task_fingerprint") or {}).get("task_sha256")
    tfj = os.path.join(stage_dir, "task_fingerprint.json")
    run_task = read_json(tfj).get("task_sha256") if os.path.exists(tfj) else None
    if not os.path.exists(tfj):
        (checks.setdefault("records_missing", [])).append("task_fingerprint.json")
        if not a.allow_missing_records:
            hard.append(f"{tfj} missing (required)")
    checks["task"] = {"panel": panel_task, "checkpoint": ck_task, "run": run_task,
                      "checkpoint_match": ck_task == panel_task, "run_match": None if run_task is None else run_task == panel_task}
    if a.task_binding == "require":
        if ck_task != panel_task:
            hard.append(f"the checkpoint's task_sha256 {ck_task} != the panel's {panel_task}")
        if run_task is not None and run_task != panel_task:
            hard.append(f"{tfj} task_sha256 {run_task} != the panel's {panel_task}")
    scj = os.path.join(stage_dir, "stage_config.json")
    if os.path.exists(scj):
        sc = read_json(scj)
        # config.save_stage_config writes the env's full kwargs: the env class's constructor defaults overlaid with the
        # stage's [env] keys.  Every key the clone's stance [env] sets must be recorded with its value, and every other
        # recorded key must be the clone env class's own default.
        import inspect

        recorded_env = sc.get("reward_weights") or {}
        defaults = {n: q.default for n, q in inspect.signature(ctx.env_class).parameters.items()
                    if q.default is not inspect.Parameter.empty}
        extra = sorted(set(recorded_env) - set(ctx.env_kwargs))
        env_diff = sorted(k for k in ctx.env_kwargs if norm(recorded_env.get(k, "<absent>")) != norm(ctx.env_kwargs[k]))
        env_diff += [f"{k} (not a default)" for k in extra if k not in defaults or norm(recorded_env[k]) != norm(defaults[k])]
        env_ok = not env_diff
        from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds

        try:
            run_bars = StanceV2Thresholds.from_curriculum(sc.get("curriculum") or {}).declared()
        except Exception as exc:  # noqa: BLE001
            run_bars = {"error": str(exc)}
        bars_ok = norm(run_bars) == norm(ctx.thresholds.declared())
        hp = sc.get("hyperparameters") or {}
        sym_keys = {k: hp.get(k) for k in PPO_SYMMETRY}
        sym_ok = sym_keys == {k: ctx.ppo_kwargs.get(k) for k in sym_keys}
        commit = sc.get("git_commit")
        run_block = sc.get("run") or {}
        run_seed, run_n_envs, run_hp = run_block.get("seed"), run_block.get("n_envs"), run_block.get("hyperparameters_sha256")
        checks["stage_config"] = {"env_kwargs_match": env_ok, "gate_bars_match": bars_ok, "symmetry_keys": sym_keys,
                                  "symmetry_keys_match": sym_ok, "git_commit": commit, "clone_head": git["head"],
                                  "git_commit_match": commit == git["head"], "env_kwargs_differences": env_diff,
                                  "env_default_keys_recorded": [k for k in extra if k in defaults and norm(recorded_env[k]) == norm(defaults[k])],
                                  "run_seed": run_seed, "run_seed_match": run_seed == int(a.seed),
                                  "run_n_envs": run_n_envs, "n_envs_expected": N_ENVS, "n_envs_match": run_n_envs == N_ENVS,
                                  "run_hyperparameters_sha256": run_hp, "hyperparameters_sha256_clone": ctx.hp_sha256,
                                  "hyperparameters_sha256_committed": HP_SHA256,
                                  "hyperparameters_sha256_match": run_hp == ctx.hp_sha256 == HP_SHA256}
        if a.task_binding == "require":
            for ok, what in ((env_ok, "env kwargs"), (bars_ok, "stance_quality/v2 bars"), (sym_ok, "[ppo] symmetry keys")):
                if not ok:
                    hard.append(f"{scj}: {what} differ from the clone's stance stage")
            if commit != git["head"] and not a.allow_commit_mismatch:
                hard.append(f"{scj}: git_commit {commit} != clone HEAD {git['head']} (--allow-commit-mismatch to judge anyway)")
            if run_n_envs != N_ENVS:
                hard.append(f"{scj}: run n_envs {run_n_envs} != {N_ENVS} (criteria.txt)")
            if not run_hp == ctx.hp_sha256 == HP_SHA256:
                hard.append(f"{scj}: run hyperparameters_sha256 {run_hp} != the clone's stance recipe {ctx.hp_sha256} "
                            f"(committed {HP_SHA256})")
        if run_seed != int(a.seed) and not a.allow_seed_mismatch:
            hard.append(f"{scj}: run seed {run_seed} != --seed {a.seed}")
    else:
        (checks.setdefault("records_missing", [])).append("stage_config.json")
        if not a.allow_missing_records:
            hard.append(f"{scj} missing (required)")
    checks["hard_failures"] = hard
    return checks


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="notebook run directory (holds 01_stance/)")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS), help="L, LT (stance3), LTS (stance4) or LTSB (stance5): picks the clone and HEAD")
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS,
                    help="the run's training seed (selects the control sigma: frozen_params for 60/42, the sidecar for 80)")
    ap.add_argument("--steps", type=parse_steps, required=True,
                    help="4194304 5242880 6291456 8388608 10485760 final (J), 1048576 2097152 3145728 7340032 9437184 "
                         "11534336 (report), best / robust_best / another multiple of 524288 (selection comparison)")
    ap.add_argument("--out", required=True, help="the judge JSON to write (never overwritten)")
    ap.add_argument("--clone", default=None, help="the round's measuring copy (required)")
    ap.add_argument("--approved", default=None,
                    help="the approval record holding the branch HEAD (default <ROUND>/APPROVED; criteria lean v5)")
    ap.add_argument("--expect-head", default=None, help="the clone HEAD the judge must run (clean tree); default: the arm's")
    ap.add_argument("--stage-subdir", default=STAGE_SUBDIR)
    ap.add_argument("--frozen-params", default=FROZEN_PARAMS)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="report only: the v2 panel on 20 episodes, seeds 3042-3061")
    mode.add_argument("--common-noise", action="store_true", help="report only: 40 episodes under the s60 robust_best sigma; falls")
    mode.add_argument("--tail", action="store_true", help="report only: the 32 forward-start seeds, deterministic; falls")
    mode.add_argument("--fresh", action="store_true", help="report only: the v2 panel on fresh seeds 7042-7081")
    ap.add_argument("--no-noise", action="store_true", help="development only: skip the H4 noise tests of a full judge")
    ap.add_argument("--smoke-episodes", type=int, default=None, help="development only: N episodes per panel (JSON marked smoke)")
    # validation-only relaxations (recorded in the JSON; score_stance5.py refuses them)
    ap.add_argument("--no-symmetry-check", action="store_true", help="validation only: do not require the mesozoic_symmetry match")
    ap.add_argument("--task-binding", choices=("require", "report"), default="require",
                    help="validation only ('report'): record but do not require the task / stage_config binding")
    ap.add_argument("--allow-missing-records", action="store_true", help="validation only: stage_config / task_fingerprint / plant_identity may be absent")
    ap.add_argument("--allow-seed-mismatch", action="store_true", help="validation only: the checkpoint's recorded seed may differ from --seed")
    ap.add_argument("--allow-commit-mismatch", action="store_true",
                    help="validation only: stage_config.json git_commit may differ from the clone HEAD (score_stance5.py refuses it)")
    a = ap.parse_args()
    spec = ARMS[a.arm]
    if spec["collapse_peak_floor_reference"] is None:
        raise SystemExit(f"arm {a.arm}: PLACEHOLDER constants not filled in ARMS (head {spec['head']}, collapse reference "
                         f"{spec['collapse_peak_floor_reference']}); fill them after the config commit")
    clone = a.clone or spec["clone"]
    print(f"[judge_r10ltsb] {VERSION}; binding {BINDING_NAME}", flush=True)
    if not clone:
        raise SystemExit("pass --clone <the round's measuring copy>")
    expect_head = a.expect_head or spec["head"]
    approval = None
    if expect_head is None:
        approval = approved_branch_head(a.approved)          # r10 round: the branch HEAD from the approval record
        expect_head = approval["head"]
    mode_name = "quick" if a.quick else "common_noise" if a.common_noise else "tail" if a.tail else "fresh" if a.fresh else "full"
    steps = a.steps
    if isinstance(steps, int):
        cadence = SMOKE_CADENCE if a.smoke_episodes else CHECKPOINT_CADENCE
        if steps <= 0 or steps % cadence:
            raise SystemExit(f"--steps {steps} is not a checkpoint of the {cadence:,}-step cadence")
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    if not os.path.isdir(os.path.dirname(out)):
        raise SystemExit(f"output directory {os.path.dirname(out)} does not exist")
    run = os.path.abspath(a.run_dir)
    stage_dir = os.path.join(run, a.stage_subdir)
    stem, zp, vp = checkpoint_paths(os.path.join(stage_dir, "models"), steps)
    missing = [p for p in (zp, vp) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"checkpoint missing: {missing}")
    ck_phys = checkpoint_physics_revision(zp)
    if ck_phys.get("physics_revision") != PLANT_PHYSICS_REVISION:
        raise SystemExit(f"refusing (nothing written): {zp} records physics_revision {ck_phys.get('physics_revision')!r} "
                         f"(physics_sha256 {ck_phys.get('physics_sha256')!r}), not {PLANT_PHYSICS_REVISION}")
    n_panel = {"full": PANEL_EPISODES, "quick": QUICK_EPISODES, "fresh": FRESH_EPISODES, "common_noise": COMMON_EPISODES,
               "tail": None}[mode_name]
    panel_seed = {"full": PANEL_SEED, "quick": QUICK_SEED, "fresh": FRESH_SEED, "common_noise": COMMON_SEED, "tail": None}[mode_name]
    global NOISE_EPISODES
    if a.smoke_episodes:
        n_panel = int(a.smoke_episodes) if n_panel is not None else None
        NOISE_EPISODES = int(a.smoke_episodes)
    with open(a.frozen_params, "rb") as fh:
        fp_sha = hashlib.sha256(fh.read()).hexdigest()
    if fp_sha != FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{a.frozen_params} sha256 {fp_sha} != the r9 verdict's params_sha256 {FROZEN_PARAMS_SHA256}")
    frozen = read_json(a.frozen_params)
    h4 = frozen["criteria"]["H4"]
    if a.seed in SIDECAR_SEEDS:  # stance5r: 70 and 80 read the pinned sidecar; 42, 52 and 60 read frozen_params as before
        with open(CONTROL_SIGMA_SIDECAR, "rb") as fh:
            sc_raw = fh.read()
        sigma_file, sigma_sha = CONTROL_SIGMA_SIDECAR, hashlib.sha256(sc_raw).hexdigest()
        if sigma_sha != CONTROL_SIGMA_SIDECAR_SHA256:
            raise SystemExit(f"{CONTROL_SIGMA_SIDECAR} sha256 {sigma_sha} != the pinned {CONTROL_SIGMA_SIDECAR_SHA256}")
        sidecar = json.loads(sc_raw.decode("utf-8"))
        if sidecar.get("source_sha256") != FROZEN_PARAMS_SHA256:
            raise SystemExit(f"{CONTROL_SIGMA_SIDECAR} was not computed from the pinned frozen_params ({sidecar.get('source_sha256')})")
        ctrl = sidecar["control_sigma"][str(a.seed)]
    else:
        sigma_file, sigma_sha = os.path.realpath(a.frozen_params), fp_sha
        ctrl = h4["control_sigma"][str(a.seed)]
    assert h4["episodes_per_seed"] == 20 and h4["spawn_range"] == [3042, 3061] and h4["space"] == "joint", h4
    tail_seeds = None
    if a.tail:
        sha = sha256_file(TAIL_SEEDS_FILE)
        if sha != "sha256:" + TAIL_SEEDS_SHA256:
            raise SystemExit(f"{TAIL_SEEDS_FILE} sha256 {sha} != the pinned {TAIL_SEEDS_SHA256}")
        with open(TAIL_SEEDS_FILE) as fh:
            tail_seeds = [int(s) for s in fh.read().strip().split(",") if s.strip()]
        if a.smoke_episodes:
            tail_seeds = tail_seeds[: int(a.smoke_episodes)]

    t0 = time.monotonic()
    modules = isolate(clone)
    git = copy_state(clone)
    if git["status"] or git["head"] != expect_head:
        raise SystemExit(f"the judge runs the clean clone at {expect_head} (got {git['head']}, dirty={bool(git['status'])})")
    print(f"[judge_r10ltsb] arm {a.arm} clone {clone} HEAD {git['head']} ({git['branch']}); mode {mode_name}; "
          f"environments = {modules['environments']}", flush=True)
    k3_sigma = k3_panel_sigma(clone, a.seed, ctrl["vector"]) if K3_PANEL_SIGMA_CHECK else None
    import torch

    torch.set_num_threads(1)
    install_trace_capture()
    ctx = Ctx(a.arm)
    band_on = float(ctx.env_kwargs.get("stance_cop_band_weight", 0.0) or 0.0) > 0.0
    band_defaults = None
    if band_on:  # report-only band readouts; the band's own edges must be the readouts' (code defaults, not set in [env])
        import inspect

        sig = inspect.signature(ctx.env_class).parameters
        band_defaults = {k: ctx.env_kwargs.get(k, sig[k].default) for k in BAND_DEFAULTS}
        if band_defaults != BAND_DEFAULTS:
            raise SystemExit(f"the band's settings {band_defaults} are not the readouts' {BAND_DEFAULTS}")
        install_band_capture()
    record = {
        "schema": SCHEMA, "judge": os.path.realpath(__file__), "judge_sha256": sha256_file(os.path.realpath(__file__)),
        "argv": sys.argv[1:], "arm": a.arm, "branch_expected": spec["branch"], "clone": os.path.realpath(clone), "git": git,
        "expect_head": expect_head, "modules": modules, "python": sys.version.split()[0], "torch_threads": torch.get_num_threads(),
        "mode": mode_name, "run_dir": run, "stage_dir": stage_dir, "seed": int(a.seed), "steps": steps,
        "slot_kind": slot_kind(steps), "checkpoint": stem,
        "model": zp, "model_sha256": sha256_file(zp), "vecnorm": vp, "vecnorm_sha256": sha256_file(vp),
        "smoke": bool(a.smoke_episodes), "noise_run": mode_name == "full" and not a.no_noise,
        "stage": ctx.stage, "stage_gate_kind": ctx.curriculum.get("gate_kind"), "feet": ctx.feet, "settle_steps": ctx.settle,
        "horizon": ctx.horizon, "thresholds_declared": ctx.thresholds.declared(), "arm_check": ctx.arm_check,
        "arm_env_kwargs": {k: ctx.env_kwargs.get(k) for k in ARMS[a.arm]["env"]},
        "ppo_symmetry_keys": {k: v for k, v in ctx.ppo_kwargs.items() if k.startswith("symmetry")},
        "rails": {"k2a_pad_cop_bar": PAD_COP_BAR, "k2b_signed_foot_cop": [K2B_LO, K2B_HI], "foot_cop_bar_report": FOOT_COP_BAR,
                  "roll_tilt_deg": ROLL_TILT_DEG, "roll_lift_m": ROLL_LIFT_M},
        "panel_seeds": None if n_panel is None else [panel_seed, panel_seed + n_panel - 1],
        "noise_seeds": [NOISE_SEED, NOISE_SEED + NOISE_EPISODES - 1],
        "plant_identity": ctx.plant_identity.to_dict(),
        "frozen_params": os.path.realpath(a.frozen_params), "frozen_params_sha256": fp_sha,
        "control_sigma_file": sigma_file, "control_sigma_file_sha256": sigma_sha,
        "version": VERSION, "binding": BINDING_NAME, "checkpoint_physics_revision": ck_phys.get("physics_revision"),
        "k3_panel_sigma": k3_sigma, "branch_head_approval": approval,
        "band_readouts": {"captured": band_on, "settings": band_defaults, "whole_band": list(BAND_WHOLE),
                          "pad_band": list(BAND_PAD), "share_ramp": [BAND_SHARE_FULL_PRICE, BAND_SHARE_MIN],
                          "report_only": True},
    }
    from environments.shared.task_fingerprint import stage_task_fingerprint

    panel_task = stage_task_fingerprint(ctx.mid, ctx.stage, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity)["task_sha256"]
    record["task_sha256_rolled"] = panel_task
    record["checks"] = run_bindings(ctx, a, stage_dir, zp, panel_task, git)
    print(json.dumps({"checks": {k: record["checks"].get(k) for k in ("plant_physics_sha256", "seed", "task", "records_missing",
                                                                      "checkpoint_num_timesteps",
                                                                      "checkpoint_num_timesteps_match")},
                      "symmetry_mirror_map_match": record["checks"]["symmetry"].get("mirror_map_sha256_match"),
                      "stage_config": {k: (record["checks"].get("stage_config") or {}).get(k) for k in
                                       ("env_kwargs_match", "gate_bars_match", "symmetry_keys_match", "git_commit_match",
                                        "run_seed_match", "n_envs_match", "hyperparameters_sha256_match")}},
                     default=str), flush=True)
    if record["checks"]["hard_failures"]:
        raise SystemExit("refusing (nothing written): " + " | ".join(record["checks"]["hard_failures"]))

    if mode_name in ("full", "quick", "fresh"):
        record["panel"] = v2_panel(ctx, model_path=zp, vecnorm_path=vp, episodes=n_panel, seed=panel_seed)
        p = record["panel"]
        assert p["task_sha256"] == panel_task, (p["task_sha256"], panel_task)
        s, st = p["summary"], p["statue_on_same_seeds"]
        def r_mm(v):
            return None if v is None else round(1000 * v, 2)

        print(json.dumps({"panel": p["verdict"], "clean": p["n_clean"], "lcb": round(p["clean_lcb"], 4), "full": p["n_full_horizon"],
                          "hop_or_fall": p["hop_or_fall_episodes"], "k2a": s["k2a_count"], "k2b": s["k2b_count"],
                          "k2b_seeds": s["k2b_seeds"], "xc_med_mm": r_mm(s["xc_median_full_horizon_m"]),
                          "statue_xc_med_mm": r_mm(st["xc_median_full_horizon_m"]),
                          "roll_rail": s["roll_rail_full_horizon"], "reward": round(p["mean_reward"], 1),
                          "statue": round(p["statue_mean_reward"] or float("nan"), 1), "statue_falls": st["fall_seeds"],
                          "seconds": round(p["panel_seconds"], 1)}), flush=True)
        if band_on:
            for who, bs in (("panel", p.get("band_summary")), ("statue", st.get("band_summary"))):
                fh = (bs or {}).get("full_horizon") or {}
                pp = (bs or {}).get("paid_per_full_horizon_episode") or {}
                print(json.dumps({f"band_{who}": {k: fh.get(k) for k in ("window_foot_steps", "guard_0", "guard_ramp",
                                                                         "counted", "whole_out", "pad_out", "pad_binds",
                                                                         "share_min")}
                                  | {"paid_mean": None if not pp else round(pp["mean"], 4),
                                     "paid_median": None if not pp else round(pp["median"], 4),
                                     "paid_worst": None if not pp else round(pp["worst"], 4)}}), flush=True)
    if mode_name == "full" and not a.no_noise:
        model, normalizer = load_checkpoint(ctx, zp, vp)
        record["noise_matched_own"] = noise_matched(ctx, model, normalizer, sigma_arg="own", sigma=None, sigma_source="own", space="policy")
        record["noise_matched_control_joint"] = noise_matched(ctx, model, normalizer, sigma_arg="control", sigma=ctrl["vector"],
                                                              sigma_source=ctrl["source"], space="joint")
        for key in ("noise_matched_own", "noise_matched_control_joint"):
            n = record[key]
            print(json.dumps({key: {"falls": n["falls"], "fall_seeds": n["fall_seeds"], "sigma_mean": round(n["sigma_mean"], 4),
                                    "k2a_all": n["k2a_count"], "seconds": round(n["panel_seconds"], 1)}}), flush=True)
    if mode_name == "common_noise":
        model, normalizer = load_checkpoint(ctx, zp, vp)
        record["common_noise"] = common_noise(ctx, model, normalizer, episodes=n_panel)
        c = record["common_noise"]
        print(json.dumps({"common_noise": {"falls": c["falls"], "of": c["n"], "seconds": round(c["panel_seconds"], 1)}}), flush=True)
    if mode_name == "tail":
        SR = ctx.SR
        predict, note = SR._load_policy(zp, vp, lambda: ctx.env_class(**ctx.env_kwargs), plant_identity=ctx.plant_identity,
                                        allow_legacy_plant=False)
        t1 = time.monotonic()
        pol_rows = rolled_falls(ctx, predict, tail_seeds)
        st_rows = rolled_falls(ctx, SR._zero_action_predict(ctx.env_class, ctx.env_kwargs), tail_seeds)
        record["tail"] = {"seeds_file": TAIL_SEEDS_FILE, "seeds_sha256": "sha256:" + TAIL_SEEDS_SHA256, "seeds": tail_seeds,
                          "policy_note": note, "n": len(tail_seeds),
                          "falls": sum(1 for r in pol_rows if not r["reached_horizon"]),
                          "fall_seeds": [r["seed"] for r in pol_rows if not r["reached_horizon"]],
                          "statue_falls": sum(1 for r in st_rows if not r["reached_horizon"]),
                          "statue_fall_seeds": [r["seed"] for r in st_rows if not r["reached_horizon"]],
                          "reference": "statue 0, scratch arm L s60 6M 3, s52 6M 7, s42 6M 26 of 32 (noise_robustness summary.txt)",
                          "seconds": time.monotonic() - t1, "episodes_detail": pol_rows, "statue_episodes_detail": st_rows}
        print(json.dumps({"tail": {k: record["tail"][k] for k in ("falls", "fall_seeds", "statue_falls")}}), flush=True)
    record["modules_all_under_clone"] = modules_under(clone)
    after = copy_state(clone)
    if after["status"] or after["tree_digest"] != git["tree_digest"]:
        raise SystemExit(f"the measuring copy changed while judging: {after['status']}")
    record["copy_tree_digest_after"] = after["tree_digest"]
    record["judge_seconds"] = time.monotonic() - t0
    with open(out, "x") as fh:
        json.dump(jsonable(record), fh, indent=1)
    print(json.dumps({"wrote": out, "seconds": round(record["judge_seconds"], 1)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:  # a refusal's reason reaches the log even when a wrapper (tools/iso.py) swallows the exit
        if isinstance(exc.code, str):
            print(f"[judge_r10ltsb] {exc.code}", file=sys.stderr, flush=True)
        raise
