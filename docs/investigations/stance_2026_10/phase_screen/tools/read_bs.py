"""Phase screen (PS): the reader.  Report only: it never trains; the loop markers simulate (deterministic rollouts and
stochastic on-policy rollouts with frozen statistics, no update, no optimizer).  PS = BS2's read_bs.py (sha256
06f4173d...) with PS's root, arms and validation seeds (bs_common.py) and the critic_phase reading: at every pair of a
critic_phase run the actor's mask is installed and the critic is fed the phase of each state's own step counter (REF:
the bank's kept step + 1 = 200, settled; OWN: the snapshot's counter, 500-1000, settled; ONPOLICY: each state's counter
recorded in the collection), with the slots-zero reading ("raw") and the all-settled reading beside it; at the start
copy (S) the arm's load transform is applied first (the critic's phase columns zeroed, the statistics' slots reseeded:
bs_hooks), every later pair must carry the frozen slots (mean 0 / var 1).  CHECK also reads critic_phase's run_meta
(the hook's verified transform, the probe's phase record, the split of the phase evidence before / after the first
update).  BS2 = BS's read_bs.py (sha256 04036e53...)
with the window from BS2's bs_common.py, the parent's F2 pairs in [S, E] derived (2,097,152 and 3,145,728), their
reproduction targets, and the loop cache keyed on this file's own sha256 (and its two modules') as well (BS's process
lens P3: an entry can never outlive the reader that wrote it).

    PY -I -B read_bs.py --check RUN
        CHECK (the runner reads only its exit code).  Reads RUN/run_meta.json, RUN/arm_evidence.jsonl and four column
        groups of RUN/01_stance/policy_probe.npz (update_timesteps, ent_coef, learning_rate, obs_rms_mean/var/count;
        allow_pickle False) and the parent's probe.  Writes RUN/check.json (pass/fail per item; no marker values).
        Exit 0 pass, 4 fail, 2 input error.
    PY -I -B read_bs.py RUN [--out DIR] [--no-onpolicy] [--no-own]
        The markers of one finished run (a continuation under <BS>/runs/, refused until every SCREEN.json entry is
        decided: outcome-blind; or a validation run under <BS>/validate/).  Needs a current passing check.json.
        Writes DIR/markers.json and DIR/markers.txt (default DIR <BS>/reads/<name>/, or RUN/read/ for a validation
        run), and DIR/m5.json, DIR/m5.txt.  Exit 0 read, 2 input error.
    PY -I -B read_bs.py --reference [--out DIR] [--no-onpolicy] [--no-own]
        The same markers on the PARENT (r10 stance6 s42, F2 copies, read only) over (S, E]: its probe and ledger rows
        and its two F2 checkpoints in [S, E] (2,097,152 and 3,145,728; F2 holds every 1,048,576).  Doubles as the
        reader's reproduction check against the early-warning (ew_r10.jsonl), early-bias (eb_realized.json), its
        verifier's (vew_analyze.txt) and the warning-margin (warning_margin/NOTES.txt) values (REPRO below).

THE MARKERS (what a screen would score; the criteria choose; nothing here is a verdict):
  M-PROBE  per update row with update_timesteps in (S, T): the probe bank's deterministic tail_3 command
           (bank_action_now[:, i, 14], the raw Gaussian mean) on the two statue_settled rows (i = 0, 5) and their mean;
           its 8-row trailing mean (the early-warning T1 form) and the first rows where that crosses 0.10 / 0.14 /
           0.15 / 0.16; the critic's value on those rows (bank_value_now: the run's own critic, so the BLINDED one under
           critic_blind_tail3); the tail_3 normaliser (obs_rms_mean[:, 6], obs_rms_var[:, 6]); std and rollout mean of
           action 14.
  M-LEDGER the pinned M5 (ledger_r10round.py, unchanged, --E T) on RUN/01_stance/eval_episodes.csv: ANY / TAIL
           flagged evaluations and alarms in (S, T] (an alarm straddling S not counted), C6; and directly from the
           ledger per evaluation in (S, T]: CMD_L = the mean over its episodes of tail_3_pitch_act_command_mean, the
           episodes over the bar (saturated_actuator_fraction > 0.10), those naming a tail actuator (11-14) and
           tail_3; the first evaluation with a tail over-bar episode (FOB), with CMD_L >= 0.30, >= 0.60 (PARK),
           >= 0.90 (RAIL) (the early-warning ew_events definitions).
  M-LOOP   at each saved pair in [S, T] (the start copy, the periodic pairs S + 262,144 k, and the final pair at T):
           (a) the early-warning study's fixed-state tail sweep (ew_common, verified by verify_early_warning): obs[6]
               set to 0, 0.25, 0.5, 0.75, 1.0 deg, everything else fixed; least-squares slope per state, median over
               states; G = slope x 5.16; c0 = the median command at 0 deg; FP = c0 / (1 - G); on OWN settled states
               (deterministic rollouts, seeds 5001-5010, 1,000 steps, raw observations after steps 500, 600, ..., 1000)
               and on REF (the probe bank's two statue_settled rows); own14 = the median episode mean of the clipped
               command over steps 501-1000;
           (b) the verifier's additions (verify_early_warning/vew_analyze.txt sections B, C, G; its NOTES.txt does not
               exist): G_H over 0-0.5 deg; G_L, the local slope at each state's own angle +-0.25 deg; the margins to the
               rail b*_P = 1 - G - c0 and b*_L = (1 - G_L)(1 - own14); the clip and linearity diagnostics;
           (c) the critic's slope on observation 6, the early-bias probe's method (eb_realized.py sens()): autograd
               d V / d z6 x (1 / sqrt(var6 + eps)) x pi / 180, per degree of tail_3 angle, mean (and SD) over the
               states whose z6 is inside the clip, on the probe's ON-POLICY states (eb_common.build: a 4-env DummyVecEnv
               under the pair's statistics frozen, the model loaded with the stage's [ppo] kwargs and enable_mirror_kl,
               one collect_rollouts of 2,048 steps per env, seeds 9301 and 9302), and also on OWN and REF; for a
               critic_blind_tail3 run both the EFFECTIVE critic (the arm's hook installed: exactly 0 by construction)
               and the raw value MLP are reported; for a critic_phase run (PS) the EFFECTIVE critic is fed each
               state's phase, "raw" is the critic with the phase slots zero and "settled" feeds every state the
               settled phase (1, 1, 1); the record "phase" says what was fed.
Every simulation runs from the sealed code copy bs_common.CODE (BS's, read only), OMP/MKL 1 thread; checkpoints are
opened read only.  A pair's loop markers are cached under <BS>/reads/_cache/ by (this file's sha256, the sha256 of
bs_common.py and bs_hooks.py, zip sha256, pkl sha256, critic form, read form): cache_key(); an entry whose recorded
reader digests differ from this reader's is refused, never used.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
VERSION = "read_bs 2.0 (2026-10-31; phase screen from s42 2,097,152; derived from read_bs 1.1 of bias screen 2)"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = sys.modules.get("bs_common") or _load("bs_common")
H = sys.modules.get("bs_hooks") or _load("bs_hooks")

EXIT_OK, EXIT_INPUT, EXIT_CHECK_FAIL = C.EXIT_OK, C.EXIT_REFUSED, C.EXIT_CHECK_FAIL          # 0, 2, 4
CHECK_SCHEMA = "bs-check/1"
MARKERS_SCHEMA = "bs-markers/1"
PROBE_REL = f"{C.STAGE_DIR}/policy_probe.npz"
LEDGER_REL = f"{C.STAGE_DIR}/eval_episodes.csv"
MODELS_REL = f"{C.STAGE_DIR}/models"
OBS_RMS_COLUMNS = ("obs_rms_mean", "obs_rms_var", "obs_rms_count")
GRID = C.EVAL_FREQ
FLAG_ROWS = 5                                   # M5's ANY / TAIL flag: >= 5 of an evaluation's rows (ECS read_ecs)
BAR = 0.10                                      # over the bar: saturated_actuator_fraction > 0.10 (ew_events)
TAIL_ACTS = ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
CMD_LINES = (0.30, 0.60, 0.90)                  # CMD_L >= 0.3, PARK, RAIL (ew_events)
PROBE_LINES = (0.10, 0.14, 0.15, 0.16)          # the 8-row trailing statue_settled mean (ew NOTES T1)
TRAIL = 8
M5_TIMEOUT_S = 900
# The early-warning method (ew_common.py; verify_early_warning/vew_measure.py)
PLANT = 5.16
TAIL_P = (0.0, 0.25, 0.5, 0.75, 1.0)
TAIL_H = (0.0, 0.25, 0.5)
OWN_SEEDS = tuple(range(5001, 5011))
HORIZON = 1000
SNAP_STEPS = (500, 600, 700, 800, 900, 1000)
ONPOLICY_SEEDS = (9301, 9302)                   # eb_realized.py SEEDS
F2_EVERY = 1_048_576                            # the F2 fetch holds the parent's pairs at every 1,048,576 (FMAN)
# The parent's F2 pairs in [S, E] (the reference read's M-LOOP pairs): 2,097,152 and 3,145,728 for BS2.
REFERENCE_PAIRS = tuple(s for s in range(F2_EVERY, C.E + 1, F2_EVERY) if s >= C.S)
# Reproduction targets for --reference (parent s42 at its F2 pairs in [S, E]; [measured] by their studies; paths under
# S2/com_revisit/notes/):
#   ew_*   early_warning/ew_r10.jsonl line 3 (2,097,152) and line 1 (3,145,728), full precision: own.primary.G,
#          own.primary.c0, own14_win_median, ref_statue_settled.primary.G;
#   eb_*   early_bias/eb_realized.json lines 88 / 91 (s42_2097152_3145728.sens_N) and 140 / 143
#          (s42_3145728_3670016.sens_N): dV_dq3_per_deg (compared), loop_gain_G_5p16 (recorded only: another method);
#   vew_*  verify_early_warning/vew_analyze.txt section C lines 64 / 62 (G_H, G_L) and section G lines 160 / 158
#          (b*_P, b*_L), printed to 3 decimals;
#   wm_*   warning_margin/NOTES.txt lines 20 / 21 (the b*_P "(verifier)" column), printed to 3 decimals.
REPRO = {
    2097152: {"ew_own": {"G": -0.6363920116424561, "c0": 0.23771291971206665, "own14": 0.14250202003121376},
              "ew_ref_G": -0.6194893866330384, "eb_dV_dq3_per_deg": 0.02142920531332493,
              "eb_G": -0.6060763597488403, "vew_G_H": -0.625, "vew_G_L": -0.648, "vew_b_star_P": 1.399,
              "vew_b_star_L": 1.413, "wm_b_star_P": 1.399},
    3145728: {"ew_own": {"G": 0.3489333643913269, "c0": 0.4114188253879547, "own14": 0.6013715491890907},
              "ew_ref_G": 0.32630227160453795, "eb_dV_dq3_per_deg": 0.05854005739092827,
              "eb_G": 0.25905266404151917, "vew_G_H": 0.354, "vew_G_L": 0.259, "vew_b_star_P": 0.240,
              "vew_b_star_L": 0.296, "wm_b_star_P": 0.240},
}


class InputError(Exception):
    """An input the reader cannot read as required (exit 2)."""


# The reader's own digests, for the loop cache (P3): this file, and the two modules it loads (bs_common.py: the pins
# and grid; bs_hooks.py: the critic-blind class the EFFECTIVE critic slope installs).
READER_SHA256 = C.sha256_file(Path(__file__).resolve())
DEPS_SHA256 = hashlib.sha256((C.sha256_file(TOOLS / "bs_common.py") + C.sha256_file(TOOLS / "bs_hooks.py"))
                             .encode()).hexdigest()


def cache_key(zip_sha256, pkl_sha256, *, blind, onpolicy, own, phase=False, start=False, reader=READER_SHA256,
              deps=DEPS_SHA256):
    """The loop cache's file stem: the reader's own sha256 and its modules' digest first (an entry never outlives the
    reader that wrote it: BS's process lens P3), then the pair and the read's form (pure).  PS: the form 'phase' (a
    critic_phase run's reading), with '_S' when the pair is read as the start copy (the arm's load transform
    applied)."""
    if blind and phase:
        raise ValueError("a read is blind or phase, not both")
    form = "blind" if blind else ("phase" + ("_S" if start else "") if phase else "plain")
    return (f"r{reader[:16]}_d{deps[:8]}_{zip_sha256[:16]}_{pkl_sha256[:16]}_{form}_"
            f"{'on' if onpolicy else 'off'}{'_own' if own else ''}")


def cache_entry_problem(row, reader=READER_SHA256, deps=DEPS_SHA256):
    """None if a cache entry was written by this reader (its recorded digests), else the reason (pure)."""
    if not isinstance(row, dict):
        return "not a JSON object"
    if row.get("reader_sha256") != reader or row.get("deps_sha256") != deps:
        return (f"written by reader {str(row.get('reader_sha256'))[:16]} / modules {str(row.get('deps_sha256'))[:8]}, "
                f"not this reader {reader[:16]} / {deps[:8]}")
    return None


# ================================================================================================ small helpers
def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def require_under(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    if p != r and r not in p.parents:
        raise InputError(f"{p} is not under {r} (the reader writes only under the screen's folder)")
    return p


def read_json_file(path):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise InputError(f"{path} is not a JSON object")
    return d, hashlib.sha256(raw).hexdigest()


def run_kind(run_dir, root):
    run, root = Path(run_dir).resolve(), Path(root).resolve()
    if run.parent == root / "runs":
        return "continuation"
    if (root / "validate") in run.parents:
        return "validation"
    return None


def finite_or_none(v):
    v = float(v)
    return v if math.isfinite(v) else None


def load_probe(path, keys):
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise InputError(f"{path}: no {missing}")
            return {k: np.asarray(z[k]) for k in keys}
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


def parent_probe(keys):
    path = C.f2_record("policy_probe.npz")
    want = C.START["records"]["policy_probe.npz"]
    got = C.sha256_file(path)
    if got != want:
        raise InputError(f"parent probe {path}: sha256 {got} != FMAN's {want}")
    return path, got, load_probe(path, keys)


def window_rows(steps, S, T):
    return [i for i, u in enumerate(steps.tolist()) if S < int(u) < T]


# ================================================================================================ CHECK
def probe_check(run_cols, parent_cols, arm, S, T, start_stats):
    """R4 (as read_ecs) for every arm: rows at S + 8,192 k, k = 1..(T - S)/8,192 - 1, none at or after T; ent_coef
    and learning_rate equal (float32) to the parent's F2 probe at the same step (no arm changes them).  Then the
    observation statistics the probe recorded per row (float32): frozen_obs_norm, every row equal to the start pkl's
    (as float32); the other arms, moved from it by the last row.  (failures, details: no marker values)."""
    import numpy as np

    fails = []
    n_exp = C.expected_update_rows(S, T)
    steps = np.asarray(run_cols["update_timesteps"]).astype(np.int64)
    idx = window_rows(steps, S, T)
    got_steps = [int(steps[i]) for i in idx]
    exp_steps = [S + C.ROLLOUT * k for k in range(1, n_exp + 1)]
    late = [int(u) for u in steps.tolist() if int(u) >= T]
    det = dict(rows_expected=n_exp, rows_in_window=len(idx), n_rows_at_or_after_T=len(late),
               seeded_rows_at_or_before_S=sum(1 for u in steps.tolist() if int(u) <= S))
    if len(idx) != n_exp or got_steps != exp_steps:
        fails.append(f"probe: {len(idx)} rows in ({S}, {T}) not at S + 8,192 k, k = 1..{n_exp}")
    if late:
        fails.append(f"probe: {len(late)} rows at or after T = {T}: an update on the end rollout")
    if fails:
        return fails, det
    pst = np.asarray(parent_cols["update_timesteps"]).astype(np.int64).tolist()
    where = {}
    for i, u in enumerate(pst):
        where.setdefault(int(u), []).append(i)
    p_idx = []
    for s in exp_steps:
        if len(where.get(s, [])) != 1:
            raise InputError(f"parent probe has {len(where.get(s, []))} rows at update_timesteps {s}")
        p_idx.append(where[s][0])
    for col in ("ent_coef", "learning_rate"):
        a = np.asarray(run_cols[col], dtype=np.float32)[idx]
        b = np.asarray(parent_cols[col], dtype=np.float32)[p_idx]
        bad = [i for i in range(n_exp) if not a[i] == b[i]]
        det[f"{col}_mismatches"] = len(bad)
        if bad:
            fails.append(f"probe: {col} differs from the parent's F2 probe at {len(bad)} of {n_exp} rows")
    mean0 = np.asarray(start_stats.obs_rms.mean, dtype=np.float32)
    var0 = np.asarray(start_stats.obs_rms.var, dtype=np.float32)
    cnt0 = float(start_stats.obs_rms.count)
    m = np.asarray(run_cols["obs_rms_mean"], dtype=np.float32)[idx]
    v = np.asarray(run_cols["obs_rms_var"], dtype=np.float32)[idx]
    c = np.asarray(run_cols["obs_rms_count"], dtype=np.float64)[idx]
    same = [bool(np.array_equal(m[i], mean0) and np.array_equal(v[i], var0) and c[i] == cnt0) for i in range(n_exp)]
    det["obs_rms_rows_equal_start"] = int(sum(same))
    if arm == "frozen_obs_norm" and not all(same):
        fails.append(f"probe: frozen_obs_norm: {n_exp - sum(same)} rows' observation statistics differ from the start")
    if arm != "frozen_obs_norm" and same[-1]:
        fails.append(f"probe: {arm}: the observation statistics at the last row equal the start pair's (frozen?)")
    return fails, det


def evidence_check(run, meta, arm, n_expected):
    """The per-rollout evidence: arm_evidence.jsonl's lines, its summary recomputed and equal to run_meta's, and
    bs_hooks.evidence_problems on it."""
    fails = []
    path = Path(run) / C.EVIDENCE
    try:
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError) as exc:
        return [f"evidence: {path}: {exc}"], {}
    summ = H.summarize(lines)
    if json.loads(json.dumps(summ)) != meta.get("arm_evidence"):
        fails.append("evidence: run_meta's arm_evidence summary is not the summary of arm_evidence.jsonl")
    steps = [e.get("num_timesteps") for e in lines]
    want = [C.S + C.ROLLOUT * k for k in range(1, n_expected + 1)]
    if steps != want:
        fails.append(f"evidence: {len(steps)} rollout ends recorded, not one per update row at S + 8,192 k")
    fails += H.evidence_problems(arm, summ, n_expected=n_expected)
    return fails, {"n": len(lines), "problems": list(fails)}


def check_meta(meta, kind, folder):
    """run_meta.json's part of CHECK.  (failures, identity)."""
    fails = []
    get = meta.get
    if get("schema") != C.RUN_SCHEMA:
        fails.append(f"run_meta schema {get('schema')!r} != {C.RUN_SCHEMA!r}")
    p = C.parse_run_name(folder)
    if kind == "continuation" and (p is None or p["suffix"]):
        raise InputError(f"{folder}: not a continuation's own folder name (a crashed / void folder is never read)")
    arm, seed = get("arm"), get("seed")
    if p is not None and (p["arm"], p["seed"]) != (arm, seed):
        fails.append(f"folder {folder} names {p['arm']} {p['seed']}, run_meta {arm} {seed}")
    if arm not in C.ARMS:
        fails.append(f"run_meta arm {arm!r} is not one of {C.ARMS}")
        return fails, dict(arm=arm, seed=seed, S=None, E=None, T=None, end=None, stop_reason=get("stop_reason"))
    if not isinstance(seed, int) or isinstance(seed, bool):
        fails.append(f"run_meta seed {seed!r}")
    elif kind == "continuation":
        d, sp = C.load_screen(str(Path(C.BS) / "SCREEN.json"))
        if sp or (arm, seed) not in C.queue_of(d):
            fails.append(f"({arm}, {seed}) is not an entry of a valid SCREEN.json ({sp})")
    elif seed not in C.VALIDATION_SEEDS:
        fails.append(f"validation run seed {seed} is not a validation seed")
    if get("S") != C.S or get("E") != C.E:
        fails.append(f"run_meta S, E = {get('S')!r}, {get('E')!r}; expected {C.S}, {C.E}")
    if get("name") != C.run_name(arm, seed if isinstance(seed, int) else 0):
        fails.append(f"run_meta name {get('name')!r}")
    want_mode = "continuation" if kind == "continuation" else "no_update"
    if get("mode") != want_mode:
        fails.append(f"run_meta mode {get('mode')!r}: a {kind} run must be {want_mode!r}")
    if get("status") != "done" or get("exit") != 0:
        fails.append(f"run_meta status {get('status')!r} exit {get('exit')!r} (done, 0 expected)")
    code = get("code") or {}
    if code.get("tree_digest") != C.TREE_DIGEST or code.get("commit") != C.COMMIT:
        fails.append(f"code {code.get('tree_digest')!r} / {code.get('commit')!r} not the pinned copy")
    if get("code_seal_after") != "unchanged" or get("modules_outside_code"):
        fails.append("the code seal after the run or the modules' origin failed")
    st = C.START
    sp = get("start_pair") or {}
    if sp.get("zip_sha256") != st["zip_sha256"] or sp.get("pkl_sha256") != st["pkl_sha256"]:
        fails.append("start pair sha256 not FMAN's")
    pr = get("parent_records") or {}
    for rec in C.PARENT_RECORDS:
        if pr.get(rec) != st["records"][rec]:
            fails.append(f"parent record {rec} sha256 not FMAN's")
    want_ppo = dict(C.ARM_PPO_EDIT[arm]) or "unchanged"
    if get("arm_ppo") != want_ppo:
        fails.append(f"arm {arm} [ppo] edit {get('arm_ppo')!r} != {want_ppo!r}")
    if get("arm_change") != C.ARM_CHANGE[arm]:
        fails.append("run_meta arm_change is not bs_common.ARM_CHANGE[arm]")
    pa = get("ppo_applied") or {}
    if pa.get("gae_lambda") != C.ARM_GAE_LAMBDA[arm] or get("gae_lambda_after_load") != C.ARM_GAE_LAMBDA[arm]:
        fails.append(f"gae_lambda applied {pa.get('gae_lambda')!r} / after load {get('gae_lambda_after_load')!r}")
    if get("ent_coef_after_load") != C.ENT_COEF_AFTER_LOAD:
        fails.append(f"ent_coef after load {get('ent_coef_after_load')!r} != {C.ENT_COEF_AFTER_LOAD}")
    aa = get("arm_applied") or {}
    want_cls = {"critic_blind_tail3": ("CriticBlindTail3MlpExtractor", "SymmetricRunningMeanStd"),
                "frozen_obs_norm": ("MlpExtractor", "FrozenObsSymmetricRunningMeanStd"),
                "critic_phase": ("CriticPhaseMlpExtractor", "PhaseSlotsSymmetricRunningMeanStd")}.get(
        arm, ("MlpExtractor", "SymmetricRunningMeanStd"))
    if aa.get("arm") != arm or (aa.get("mlp_extractor_class"), aa.get("obs_rms_class")) != want_cls:
        fails.append(f"arm_applied {aa.get('arm')!r} {aa.get('mlp_extractor_class')!r} {aa.get('obs_rms_class')!r} "
                     f"!= {arm} {want_cls}")
    fails += phase_meta_problems(meta, arm, kind)
    if get("arm_state_end") != "in force":
        fails.append(f"the arm's hook at the end: {get('arm_state_end')!r}")
    if bool(get("obs_rms_end_equal_start")) != (arm == "frozen_obs_norm"):
        fails.append(f"observation statistics at the end equal the start's: {get('obs_rms_end_equal_start')!r}")
    v5 = get("recipe_v5")
    if not isinstance(v5, dict) or v5.get("problems") != []:
        fails.append("run_meta recipe_v5 (V5 before learn()) not passed")
    if get("torch_threads") != C.TORCH_THREADS:
        fails.append(f"torch_threads {get('torch_threads')!r} != {C.TORCH_THREADS}")
    T, target, stop = get("T"), get("target"), get("stop_reason")
    if not isinstance(T, int) or isinstance(T, bool):
        fails.append(f"run_meta T {T!r} is not an integer")
        T = None
    sae, col = get("stop_at_end") or {}, get("collapse") or {}
    if col.get("found") != 1 or col.get("patience") != C.COLLAPSE_PATIENCE:
        fails.append(f"the collapse stop is not in the run with patience {C.COLLAPSE_PATIENCE}")
    end = C.E if kind == "continuation" else target
    if kind == "continuation" and target != C.E:
        fails.append(f"run_meta target {target!r} != E = {C.E}")
    if kind == "validation" and (not isinstance(target, int) or isinstance(target, bool) or target <= C.S):
        fails.append(f"run_meta target {target!r} is not a step after S")
        end = None
    if T is not None and end is not None:
        if stop == "end":
            if T != end or sae.get("flag") is not True:
                fails.append(f"stop reason 'end' at T = {T} (target {end}, flag {sae.get('flag')})")
        elif stop == "collapse":
            if not C.S < T < end or col.get("tripped") is not True:
                fails.append(f"stop reason 'collapse' at T = {T} without the collapse state")
        else:
            fails.append(f"stop reason {stop!r}: neither 'end' nor 'collapse'")
        if (T - C.S) % C.ROLLOUT or T <= C.S:
            fails.append(f"T - S = {T - C.S} is not a positive multiple of {C.ROLLOUT}")
    d0, d1 = get("digest_start") or {}, get("digest_end") or {}
    if kind == "continuation":
        if not d0.get("params") or d0.get("params") == d1.get("params"):
            fails.append("continuation: the parameter digest did not change: no update made")
        n0, n1 = d0.get("n_updates"), d1.get("n_updates")
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in (n0, n1)) or n1 <= n0:
            fails.append(f"continuation: n_updates {n0!r} -> {n1!r} did not grow")
    else:
        for k in ("params", "adam"):
            if not d0.get(k) or d0.get(k) != d1.get(k):
                fails.append(f"no-update run: {k} digest changed")
        if T is not None and end is not None and T > C.S and get("train_calls") != C.expected_update_rows(C.S, T):
            fails.append(f"no-update run: train_calls {get('train_calls')!r}")
    return fails, dict(arm=arm, seed=seed, S=C.S, E=C.E, T=T, end=end, stop_reason=stop)


def phase_meta_problems(meta, arm, kind):
    """PS: critic_phase's run_meta records (the verified load transform, the phase definition, the probe feeding the
    bank's phase, the evidence's split: one rollout before the first update in a continuation, all of them in a
    no-update run); any other arm: none of them (pure)."""
    fails = []
    aa, pp, pd = meta.get("arm_applied") or {}, meta.get("probe_phase") or {}, meta.get("phase_definition")
    ev = meta.get("arm_evidence") or {}
    want_def = {"slots": list(C.PHASE_SLOTS), "features": list(C.PHASE_FEATURES), "settle_steps": C.SETTLE_STEPS,
                "gamma": C.PHASE_GAMMA}
    d0, dl = meta.get("digest_start") or {}, meta.get("digest_load") or {}
    if arm != "critic_phase":
        if pd is not None or pp.get("installed") is not False or aa.get("transform_verified") is not None:
            fails.append(f"{arm}: phase records present (definition {pd is not None}, probe {pp.get('installed')!r}, "
                         f"transform {aa.get('transform_verified')!r})")
        if not dl or {k: dl.get(k) for k in ("params", "adam", "vecnormalize")} != {
                k: d0.get(k) for k in ("params", "adam", "vecnormalize")}:
            fails.append(f"{arm}: the hook changed the loaded state (digest_load != digest_start)")
        return fails
    if not isinstance(pd, dict) or {k: pd.get(k) for k in want_def} != want_def:
        fails.append(f"critic_phase: phase_definition {pd!r} != {want_def}")
    if aa.get("transform_verified") is not True or aa.get("phase_slots") != list(C.PHASE_SLOTS):
        fails.append("critic_phase: the load transform was not verified")
    if not dl or dl.get("adam") != d0.get("adam") or dl.get("params") == d0.get("params"):
        fails.append("critic_phase: digest_load / digest_start do not show the transform (Adam equal, params changed)")
    if pp.get("installed") is not True or pp.get("class") != "PhaseFedPolicyProbeCallback" or not pp.get("k") or \
            any(k < C.SETTLE_STEPS for k in pp.get("k")) or pp.get("phase_fed") != [[1.0, 1.0, 1.0]] * len(pp["k"]):
        fails.append(f"critic_phase: the policy probe does not feed the settled bank's phase ({pp})")
    n = ev.get("n")
    z, u = ev.get("phase_n_zero_init"), ev.get("phase_n_updated")
    if kind == "continuation" and (z != 1 or u != (n - 1 if isinstance(n, int) else None)):
        fails.append(f"critic_phase: {z} rollouts before the first update and {u} after (1 and {n} - 1 expected)")
    if kind != "continuation" and (z != n or u != 0):
        fails.append(f"critic_phase: a no-update run with {z} zero-init and {u} updated rollouts (all zero-init)")
    return fails


def check_run(run_dir, root=None):
    """CHECK.  Writes <run>/check.json.  (exit code, record)."""
    root = Path(root or C.BS).resolve()
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise InputError(f"no run folder {run}")
    kind = run_kind(run, root)
    if kind is None:
        raise InputError(f"{run} is neither <BS>/runs/<name> nor under <BS>/validate/")
    require_under(run / C.CHECK_JSON, root)
    fails, sources, items = [], {}, {}
    meta_path = run / C.RUN_META
    try:
        meta, msha = read_json_file(meta_path)
        sources["run_meta.json"] = [str(meta_path), msha]
    except InputError as exc:
        meta = None
        fails.append(f"run_meta.json: {exc}")
    ident = {}
    if meta is not None:
        mf, ident = check_meta(meta, kind, run.name)
        items["run_meta"] = list(mf)
        fails += mf
        S, T = ident.get("S"), ident.get("T")
        if S is not None and T is not None and T > S and (T - S) % C.ROLLOUT == 0 and ident["arm"] in C.ARMS:
            n_exp = C.expected_update_rows(S, T)
            ef, edet = evidence_check(run, meta, ident["arm"], n_exp)
            items["evidence"] = ef
            fails += ef
            ppath, psha, pcols = parent_probe(C.PROBE_CHECK_COLUMNS)
            sources["parent policy_probe.npz"] = [ppath, psha]
            probe_path = run / PROBE_REL
            try:
                rcols = load_probe(probe_path, (*C.PROBE_CHECK_COLUMNS, *OBS_RMS_COLUMNS))
                sources["policy_probe.npz"] = [str(probe_path), C.sha256_file(probe_path)]
            except InputError as exc:
                rcols = None
                fails.append(f"policy_probe.npz: {exc}")
            if rcols is not None:
                start_stats = pkl_stats(run / MODELS_REL / C.periodic_names(S)[1])
                pf, pdet = probe_check(rcols, pcols, ident["arm"], S, T, start_stats)
                items["probe"] = pf
                items["probe_detail"] = pdet
                fails += pf
            sources[C.EVIDENCE] = [str(run / C.EVIDENCE), C.sha256_file(run / C.EVIDENCE)
                                   if (run / C.EVIDENCE).is_file() else None]
        else:
            fails.append("evidence and probe not checked: T, S or the arm do not allow it")
    rec = dict(schema=CHECK_SCHEMA, version=VERSION, tool_sha256=C.sha256_file(Path(__file__)), utc=utcnow(),
               run=str(run), name=(meta or {}).get("name"), mode=kind, passed=not fails, failures=fails,
               items=items, sources=sources, **{k: ident.get(k) for k in ("arm", "seed", "S", "E", "T", "end",
                                                                           "stop_reason")})
    atomic_write_text(run / C.CHECK_JSON, json.dumps(rec, indent=1, default=str) + "\n")
    return (EXIT_OK if not fails else EXIT_CHECK_FAIL), rec


def pkl_stats(path):
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - the run's own start pkl (sha256-checked copy)


def check_is_current(run, check):
    src = check.get("sources") or {}
    for key, rel in (("run_meta.json", C.RUN_META), ("policy_probe.npz", PROBE_REL), (C.EVIDENCE, C.EVIDENCE)):
        if key not in src:
            return f"check.json names no {key}"
        try:
            now = C.sha256_file(Path(run) / rel)
        except OSError as exc:
            return f"{rel}: {exc}"
        if now != src[key][1]:
            return f"{rel} changed since check.json"
    return None


def load_lost(path):
    path = Path(path)
    if not path.exists():
        return {}
    d, _ = read_json_file(path)
    if d.get("schema") != C.LOST_SCHEMA or not isinstance(d.get("lost"), dict):
        raise InputError(f"{path}: schema {d.get('schema')!r}")
    return dict(d["lost"])


def decided(root=None):
    """(all decided, [undecided names]) over SCREEN.json's queue: a passing continuation check.json, or lost.json."""
    root = Path(root or C.BS)
    d, problems = C.load_screen(str(root / "SCREEN.json"))
    if problems:
        return False, [f"SCREEN.json: {p}" for p in problems]
    lost = load_lost(root / "lost.json")
    und = []
    for arm, seed in C.queue_of(d):
        name = C.run_name(arm, seed)
        if name in lost:
            continue
        try:
            c, _ = read_json_file(root / "runs" / name / C.CHECK_JSON)
        except InputError:
            und.append(name)
            continue
        if c.get("passed") is not True or c.get("mode") != "continuation":
            und.append(name)
    return not und, und


# ================================================================================================ M-PROBE
def probe_markers(path, S, T, label):
    import numpy as np

    keys = ("update_timesteps", "bank_action_now", "bank_value_now", "obs_rms_mean", "obs_rms_var",
            "std_per_actuator", "rollout_mu_mean", "bank_labels")
    cols = load_probe(path, keys)
    labels = [str(x) for x in cols["bank_labels"]]
    ss = list(C.BANK_STATUE_SETTLED)
    if [labels[i] for i in ss] != ["statue_settled", "statue_settled"]:
        raise InputError(f"{path}: bank rows {ss} are {[labels[i] for i in ss]}, not statue_settled")
    steps = np.asarray(cols["update_timesteps"]).astype(np.int64)
    idx = [i for i, u in enumerate(steps.tolist()) if S < int(u) < T]
    a = np.asarray(cols["bank_action_now"], dtype=np.float64)[idx][:, ss, C.ACT_TAIL3]
    v = np.asarray(cols["bank_value_now"], dtype=np.float64)[idx][:, ss]
    mean_ss = a.mean(axis=1)
    trail = [float(np.mean(mean_ss[max(0, i - TRAIL + 1): i + 1])) if i >= TRAIL - 1 else None
             for i in range(len(mean_ss))]
    st = [int(steps[i]) for i in idx]
    first = {}
    for line in PROBE_LINES:
        hit = [st[i] for i, x in enumerate(trail) if x is not None and x > line]
        first[f"{line:.2f}"] = hit[0] if hit else None
    var6 = np.asarray(cols["obs_rms_var"], dtype=np.float64)[idx][:, C.OBS_TAIL3]
    return dict(
        file=str(path), sha256=C.sha256_file(path), label=label, S=S, T=T, n_rows=len(idx), steps=st,
        tail3_statue_settled=[[float(x) for x in row] for row in a], tail3_statue_settled_mean=mean_ss.tolist(),
        trailing8_mean=trail, trailing8_first_above=first,
        trailing8_max=max((x for x in trail if x is not None), default=None),
        tail3_ss_mean_first_last=[float(mean_ss[0]), float(mean_ss[-1])] if len(idx) else None,
        value_statue_settled_mean=v.mean(axis=1).tolist(),
        obs6_mean_deg=np.degrees(np.asarray(cols["obs_rms_mean"], dtype=np.float64)[idx][:, C.OBS_TAIL3]).tolist(),
        obs6_sd_deg=np.degrees(np.sqrt(var6)).tolist(),
        std14=np.asarray(cols["std_per_actuator"], dtype=np.float64)[idx][:, C.ACT_TAIL3].tolist(),
        rollout_mu14_mean=np.asarray(cols["rollout_mu_mean"], dtype=np.float64)[idx][:, C.ACT_TAIL3].tolist())


# ================================================================================================ M-LEDGER
C6_TAIL_RE = re.compile(r"^evaluation (\d+) episode .*: TAIL input missing \(")


def run_m5(ledger, T, out_dir):
    """The pinned M5, unchanged, as a subprocess: (M5 JSON, sha256, argv)."""
    out_dir = Path(out_dir)
    for path, want in C.PIN_SHA256.items():
        if path in (C.M5, C.M5_COMMON) and C.sha256_file(path) != want:
            raise InputError(f"{path} is not at its pin")
    jpath = out_dir / "m5.json"
    jpath.unlink(missing_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    argv = [C.PY, "-I", "-B", C.M5, str(ledger), "--E", str(int(T)), "--json", str(jpath)]
    try:
        p = subprocess.run(argv, cwd="/tmp", env=env, capture_output=True, text=True, timeout=M5_TIMEOUT_S,
                           check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise InputError(f"M5 did not run: {exc}") from exc
    atomic_write_text(out_dir / "m5.txt", p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else ""))
    if p.returncode != 0:
        raise InputError(f"M5 exit {p.returncode}: {(p.stdout + p.stderr)[-400:]}")
    m5, jsha = read_json_file(jpath)
    return m5, jsha, argv


def m5_window(m5, S, T):
    """M5's flags read in (S, T] (read_ecs.window_reading's rules, reduced)."""
    per = {int(k): v for k, v in (m5.get("per_evaluation") or {}).items()}
    for key in ("any_flagged", "tail_flagged", "any_alarms", "tail_alarms"):
        if not isinstance(m5.get(key), list):
            raise InputError(f"M5 JSON has no {key} list")
    c6 = m5.get("c6") or {}
    if c6.get("E") != T:
        raise InputError(f"M5's C6 was run with E = {c6.get('E')!r}, not T = {T}")

    def inwin(s):
        return S < int(s) <= T
    counting = [f for f in c6.get("failures", []) if not C6_TAIL_RE.match(str(f))]
    return dict(
        any_flagged=sorted(int(s) for s in m5["any_flagged"] if inwin(s)),
        tail_flagged=sorted(int(s) for s in m5["tail_flagged"] if inwin(s)),
        any_alarms=[int(s) for s in m5["any_alarms"] if S < int(s) - GRID and int(s) <= T],
        tail_alarms=[int(s) for s in m5["tail_alarms"] if S < int(s) - GRID and int(s) <= T],
        alarms_straddling_S=[int(s) for s in m5["any_alarms"] + m5["tail_alarms"] if int(s) - GRID <= S < int(s)],
        c6_passed=bool(c6.get("passed")), c6_counting_failures=counting[:20], n_c6_counting_failures=len(counting),
        per_evaluation={s: {k: per[s].get(k) for k in ("n", "any", "tail", "unres", "any_by", "tail_by")}
                        for s in sorted(per) if inwin(s)})


def ledger_direct(ledger, S, T):
    """Per evaluation in (S, T], from the ledger's own columns (ew_events.per_eval's definitions)."""
    with open(ledger, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        try:
            t = float(r.get("timesteps"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and t == int(t) and S < int(t) <= T:
            by.setdefault(int(t), []).append(r)

    def num(r, k):
        try:
            v = float(r.get(k))
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None
    per = {}
    for s, rs in sorted(by.items()):
        cmd = [num(r, "tail_3_pitch_act_command_mean") for r in rs]
        cmd = [c for c in cmd if c is not None]
        over = [r for r in rs if (num(r, "saturated_actuator_fraction") or 0.0) > BAR]
        per[s] = dict(n=len(rs), cmd_l=sum(cmd) / len(cmd) if cmd else None, cmd_l_max=max(cmd) if cmd else None,
                      n_cmd=len(cmd), n_over=len(over),
                      n_tail_over=sum(1 for r in over if r.get("saturated_actuator") in TAIL_ACTS),
                      n_t3_over=sum(1 for r in over if r.get("saturated_actuator") == "tail_3_pitch_act"),
                      over_actuators=sorted({str(r.get("saturated_actuator")) for r in over}))
    first = {"FOB_tail": next((s for s, e in per.items() if e["n_tail_over"] > 0), None),
             "first_over_any": next((s for s, e in per.items() if e["n_over"] > 0), None),
             "first_t3_over": next((s for s, e in per.items() if e["n_t3_over"] > 0), None)}
    for line in CMD_LINES:
        first[f"cmd_l_ge_{line:.2f}"] = next((s for s, e in per.items() if e["cmd_l"] is not None
                                              and e["cmd_l"] >= line), None)
    expected = [S + GRID * k for k in range(1, (T - S) // GRID + 1)]
    return dict(per_evaluation=per, firsts=first, evaluations_absent=[s for s in expected if s not in per],
                cmd_l_max=max((e["cmd_l"] for e in per.values() if e["cmd_l"] is not None), default=None),
                cmd_l_last=per[max(per)]["cmd_l"] if per else None)


def truncate_ledger(src, dst, T):
    """A copy of a ledger with the rows whose timesteps are <= T (header and cell text kept); returns dst."""
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    ti = head.index("timesteps")
    keep = []
    for r in body:
        try:
            ok = float(r[ti]) <= T
        except (ValueError, IndexError):
            ok = True                                    # kept: M5's C6 then names it off the grid
        if ok:
            keep.append(r)
    with open(require_under(dst, C.BS), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        w.writerows(keep)
    return Path(dst)


def ledger_markers(ledger, S, T, out_dir):
    m5, jsha, argv = run_m5(ledger, T, out_dir)
    return dict(file=str(ledger), sha256=C.sha256_file(ledger), m5_argv=argv, m5_json_sha256=jsha,
                m5=m5_window(m5, S, T), direct=ledger_direct(ledger, S, T))


# ================================================================================================ M-LOOP
def setup_simulation(tmp_name):
    """Before importing numpy/torch/the code: threads 1, TMPDIR under <BS>/tmp, the copy first on sys.path."""
    tmp = Path(C.TMP) / tmp_name
    (tmp / "mpl").mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "TMPDIR": str(tmp), "MPLCONFIGDIR": str(tmp / "mpl"), "GITHUB_SHA": C.COMMIT})
    import tempfile

    tempfile.tempdir = None
    code = os.path.realpath(C.CODE)
    if sys.path[:1] != [code]:
        sys.path.insert(0, code)
    import environments

    origin = os.path.realpath(environments.__file__)
    if not origin.startswith(code + os.sep):
        raise InputError(f"environments imported from {origin}, not the code copy {code}")
    import torch

    torch.set_num_threads(1)
    return origin


def _ls_slopes(xs, ys):
    import numpy as np

    xs = np.asarray(xs, dtype=np.float64)
    if xs.ndim == 1:
        xs = np.repeat(xs[:, None], ys.shape[1], axis=1)
    xc = xs - xs.mean(axis=0)
    return (xc * (ys - ys.mean(axis=0))).sum(axis=0) / (xc ** 2).sum(axis=0)


class Net:
    """A loaded pair (ew_common.Net): SB3's own load and normaliser (frozen); mu14 = the raw Gaussian mean."""

    def __init__(self, zip_path, pkl_path, factory):
        import numpy as np

        from environments.shared.policy_loading import load_sb3_checkpoint

        self.model, self.norm, _ = load_sb3_checkpoint(str(zip_path), str(pkl_path), factory)
        self.policy = self.model.policy
        rms = self.norm.obs_rms
        self.mean = np.asarray(rms.mean, dtype=np.float64)
        self.var = np.asarray(rms.var, dtype=np.float64)
        self.eps = float(self.norm.epsilon)
        self.std = np.sqrt(self.var + self.eps)
        self.clip = float(self.norm.clip_obs)

    def refresh_stats(self):
        """PS: re-read the normaliser's statistics (after critic_phase's start transform reseeded slots 60-62)."""
        import numpy as np

        rms = self.norm.obs_rms
        self.mean = np.asarray(rms.mean, dtype=np.float64)
        self.var = np.asarray(rms.var, dtype=np.float64)
        self.std = np.sqrt(self.var + self.eps)

    def z(self, X):
        import numpy as np

        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]

    def mu(self, X):
        import numpy as np
        import torch

        nb = self.norm.normalize_obs(np.asarray(X, dtype=np.float32))
        stored = dict(vars(self.policy.action_dist))
        try:
            with torch.no_grad():
                d = self.policy.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32)))
                return d.distribution.mean.numpy().astype(np.float64)
        finally:
            vars(self.policy.action_dist).clear()
            vars(self.policy.action_dist).update(stored)


def _set_t3(X, deg):
    import numpy as np

    Y = np.array(X, dtype=np.float64, copy=True)
    Y[:, C.OBS_TAIL3] = np.radians(deg)
    return Y


def tail_sweeps(net, X):
    """(a) and (b): the 0-1 deg primary sweep (P), the 0-0.5 deg half (H), the local +-0.25 deg slope (L)."""
    import numpy as np

    out = {"n_states": int(len(X))}
    if not len(X):
        return out
    P = np.asarray(TAIL_P)
    ys = np.array([net.mu(_set_t3(X, d))[:, C.ACT_TAIL3] for d in P])
    zt = np.array([net.z(_set_t3(X, d))[:, C.OBS_TAIL3] for d in P])
    sl = _ls_slopes(P, ys)
    G = float(np.median(sl)) * PLANT
    c0 = float(np.median(ys[0]))
    lo = float(np.median((ys[2] - ys[0]) / 0.5))
    hi = float(np.median((ys[4] - ys[2]) / 0.5))
    out["P"] = {"G": G, "c0": c0, "FP": c0 / (1.0 - G) if G < 1.0 else None, "b_star": 1.0 - G - c0,
                "slope_per_deg": float(np.median(sl)), "G_q25": float(np.quantile(sl, 0.25)) * PLANT,
                "G_q75": float(np.quantile(sl, 0.75)) * PLANT, "frac_states_slope_pos": float(np.mean(sl > 0)),
                "z_tail_max": float(zt.max()), "z_tail_min": float(zt.min()),
                "n_points_clipped": int((np.abs(zt) >= net.clip).sum()),
                "lin_ratio_hi_lo": hi / lo if abs(lo) > 1e-9 else None,
                "mu14_at_states": float(np.median(net.mu(X)[:, C.ACT_TAIL3]))}
    Hh = np.asarray(TAIL_H)
    ysH = np.array([net.mu(_set_t3(X, d))[:, C.ACT_TAIL3] for d in Hh])
    out["H"] = {"G": float(np.median(_ls_slopes(Hh, ysH))) * PLANT, "c0": float(np.median(ysH[0]))}
    th = np.degrees(np.asarray(X, dtype=np.float64)[:, C.OBS_TAIL3])
    degs = np.array([th - 0.25, th, th + 0.25])
    ysL = np.array([net.mu(_set_t3(X, degs[i]))[:, C.ACT_TAIL3] for i in range(3)])
    out["L"] = {"G": float(np.median(_ls_slopes(degs, ysL))) * PLANT, "theta_deg_median": float(np.median(th))}
    return out


def with_phase(X, K):
    """PS: X with the phase of each state's step counter K in slots 60-62 (float64 copy)."""
    import numpy as np

    Y = np.array(X, dtype=np.float64, copy=True)
    Y[:, list(C.PHASE_SLOTS)] = H.phase_matrix(K)
    return Y


def critic_slope(policy, norm, X, K=None):
    """(c) eb_realized.sens(): mean (SD) over in-clip states of dV/dz6 x 1/sqrt(var6 + eps) x pi/180, per degree.
    PS: with K (the states' step counters) the critic is fed their phase in slots 60-62."""
    import numpy as np
    import torch

    x = np.asarray(X if K is None else with_phase(X, K), dtype=np.float32)
    z = torch.as_tensor(np.asarray(norm.normalize_obs(x), dtype=np.float32), dtype=torch.float32).requires_grad_(True)
    v = policy.predict_values(z).sum()
    gv = torch.autograd.grad(v, z)[0][:, C.OBS_TAIL3].numpy()
    scale = 1.0 / math.sqrt(float(norm.obs_rms.var[C.OBS_TAIL3]) + float(norm.epsilon))
    clipped = np.abs(np.asarray(norm.normalize_obs(x))[:, C.OBS_TAIL3]) >= float(norm.clip_obs)
    k = math.pi / 180.0
    g = (gv * scale * k)[~clipped]
    with torch.no_grad():
        vm = float(policy.predict_values(torch.as_tensor(np.asarray(norm.normalize_obs(x), dtype=np.float32))).mean())
    return {"dV_dq3_per_deg": float(g.mean()) if len(g) else None, "dV_dq3_per_deg_sd": float(g.std()) if len(g)
            else None, "n": int(len(x)), "n_clipped": int(clipped.sum()), "V_mean": vm,
            "tail3_deg_mean": float(np.degrees(x[:, C.OBS_TAIL3]).mean())}


def own_rollouts(net, env, seeds=OWN_SEEDS):
    """ew_measure.rollouts: deterministic 1,000-step episodes; snapshots after steps 500..1000; own14 window means."""
    import numpy as np

    eps, X, K = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        own14 = []
        term = trunc = False
        k = 0
        for k in range(1, HORIZON + 1):
            a = net.act(obs)
            own14.append(float(np.clip(a[C.ACT_TAIL3], -1, 1)))
            obs, _r, term, trunc, _ = env.step(a)
            if k in SNAP_STEPS:
                X.append(np.array(obs, dtype=np.float64, copy=True))
                K.append(int(env.unwrapped._step_count))          # PS: the snapshot's own counter (= k)
            if term or trunc:
                break
        w = own14[500:1000]
        eps.append({"seed": seed, "steps": k, "terminated": bool(term),
                    "own14_win": float(np.mean(w)) if len(own14) > 500 else None,
                    "frac99_win": float(np.mean(np.abs(np.asarray(w)) >= 0.99)) if len(own14) > 500 else None})
    return eps, np.array(X, dtype=np.float64), np.array(K, dtype=np.int64)


def onpolicy_states(zip_path, pkl_path, seed):
    """eb_common.build + eb_realized's Rec: the raw observations of one collect_rollouts (4 envs x 2,048 steps) of the
    pair's policy (stochastic) under its statistics frozen; envs seeded seed + rank; torch/np seeded seed."""
    import copy

    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.config import load_stage_config
    from environments.shared.mirror_kl_ppo import MirrorKLPPO, enable_mirror_kl
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.policy_loading import load_sb3_model
    from environments.shared.species_registry import env_class
    from environments.shared.symmetry import MirrorMap, pop_symmetry_config
    from environments.shared.train_base import _prepare_alg_kwargs, create_vec_env

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    cfg = copy.deepcopy(load_stage_config(mid, "stance"))

    class _Species:
        model_id = mid
        env_class = None

    species = _Species()
    species.env_class = env_class(mid)
    ppo = copy.deepcopy(cfg["ppo_kwargs"])
    sym = pop_symmetry_config(ppo)
    venv = create_vec_env(species, {1: {"env_kwargs": dict(cfg["env_kwargs"])}}, 1, C.N_ENVS, seed=seed,
                          use_subproc=False, algorithm="ppo", gamma=float(ppo["gamma"]))
    vn = VecNormalize.load(str(pkl_path), venv.venv)
    vn.training = False
    vn.norm_reward = True
    alg_kwargs, _tb, _g = _prepare_alg_kwargs({"ppo_kwargs": copy.deepcopy(cfg["ppo_kwargs"])}, "ppo", 0,
                                              Path(os.environ["TMPDIR"]) / "_tb_unused", use_tensorboard=False)
    pop_symmetry_config(alg_kwargs)
    alg_kwargs.pop("policy_kwargs", None)
    model = load_sb3_model(str(zip_path), algorithm="ppo", env=vn, device="cpu", **alg_kwargs)
    model = enable_mirror_kl(model, sym, MirrorMap.load(mid, current_plant_identity(mid)))
    assert isinstance(model, MirrorKLPPO)
    vn.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.set_logger(Logger(folder=None, output_formats=[]))
    model._last_obs = None

    class Rec(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.raw = []
            self.k = []

        def _on_rollout_start(self):
            self.prev = vn.get_original_obs().copy()
            self.prev_k = [int(e.unwrapped._step_count) for e in vn.venv.envs]   # PS: read only

        def _on_step(self):
            self.raw.append(self.prev.copy())
            self.k.append(list(self.prev_k))
            self.prev = vn.get_original_obs().copy()
            self.prev_k = [int(e.unwrapped._step_count) for e in vn.venv.envs]
            return True

    rec = Rec()
    _t, cb = model._setup_learn(model.n_steps * model.n_envs, rec, False, "bs_read", False)
    if not model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps):
        raise InputError("the on-policy collection stopped early")
    raw = np.concatenate(rec.raw, 0)
    ks = np.asarray(rec.k, dtype=np.int64).reshape(-1)
    vn.close()
    return raw, ks


def phase_prepare(net, *, start):
    """PS, a critic_phase pair: at the start copy the arm's load transform (bs_hooks: the critic's phase columns := 0,
    the statistics' slots := mean 0 / var 1); at any later pair the slots must already be mean 0 / var 1 (written by
    the arm); then the actor's mask.  Returns the record."""
    import torch
    from stable_baselines3.common.torch_layers import MlpExtractor

    w_vf = net.policy.mlp_extractor.value_net[0].weight
    lo, hi = C.PHASE_SLOTS[0], C.PHASE_SLOTS[-1] + 1
    rec = {"start_transform": bool(start), "value_phase_cols_max_abs_read": float(w_vf[:, lo:hi].abs().max())}
    if type(net.policy.mlp_extractor) is not MlpExtractor:
        raise InputError(f"the loaded mlp_extractor is {type(net.policy.mlp_extractor).__name__}")
    if start:
        with torch.no_grad():
            w_vf[:, lo:hi] = 0.0
        H.reseed_phase_slots(net.norm.obs_rms)
        net.refresh_stats()
    elif not H.phase_slots_frozen(net.norm.obs_rms):
        raise InputError("a critic_phase pair after S whose statistics' slots 60-62 are not mean 0 / var 1")
    net.policy.mlp_extractor.__class__ = H.critic_phase_class(MlpExtractor)
    rec["value_phase_cols_max_abs_used"] = float(w_vf[:, lo:hi].abs().max())
    return rec


def loop_markers_one(zip_path, pkl_path, *, blind, onpolicy, own, env, factory, bank_ss, phase=False, start=False,
                     bank_k=None):
    """M-LOOP at one pair.  blind: the critic_blind_tail3 hook is installed for the EFFECTIVE critic (both reported).
    phase (PS): critic_phase's reading (phase_prepare; the critic fed each state's phase; "raw" with the slots zero;
    "settled" with every state's phase (1, 1, 1)); start: the pair is the start copy (S)."""
    import numpy as np
    from stable_baselines3.common.torch_layers import MlpExtractor

    t0 = time.process_time()
    net = Net(zip_path, pkl_path, factory)
    phase_rec = phase_prepare(net, start=start) if phase else None
    out = {"zip": str(zip_path), "zip_sha256": C.sha256_file(zip_path), "pkl": str(pkl_path),
           "pkl_sha256": C.sha256_file(pkl_path), "num_timesteps": int(net.model.num_timesteps),
           "t3_norm_mean_deg": math.degrees(net.mean[C.OBS_TAIL3]),
           "t3_norm_sd_deg": math.degrees(net.std[C.OBS_TAIL3]),
           "critic_form": "blind+raw" if blind else ("phase+raw+settled" if phase else "plain")}
    out["REF"] = tail_sweeps(net, bank_ss)
    sets = {"REF": bank_ss}
    ks = {"REF": np.asarray(bank_k if bank_k is not None else [C.SETTLE_STEPS] * len(bank_ss), dtype=np.int64)}
    if own:
        eps, X, K_own = own_rollouts(net, env)
        sets["OWN"] = X
        ks["OWN"] = K_own
        w = [e["own14_win"] for e in eps if e["own14_win"] is not None]
        out["OWN"] = tail_sweeps(net, X)
        out["OWN"].update(own14_median=float(np.median(w)) if w else None, episodes=eps,
                          n_terminated=int(sum(e["terminated"] for e in eps)),
                          n_over_bar=int(sum((e["frac99_win"] or 0) > 0.10 for e in eps)))
        if out["OWN"].get("L") and w:
            out["OWN"]["b_star_L"] = (1.0 - out["OWN"]["L"]["G"]) * (1.0 - float(np.median(w)))
    if onpolicy:
        got = [onpolicy_states(zip_path, pkl_path, s) for s in ONPOLICY_SEEDS]
        sets["ONPOLICY"] = np.concatenate([g[0] for g in got], 0)
        ks["ONPOLICY"] = np.concatenate([g[1] for g in got], 0)
    crit = {}
    for name, X in sets.items():
        crit[name] = {"raw": critic_slope(net.policy, net.norm, X)}
    if phase:
        for name, X in sets.items():
            crit[name]["effective"] = critic_slope(net.policy, net.norm, X, K=ks[name])
            crit[name]["settled"] = critic_slope(net.policy, net.norm, X, K=np.full(len(X), C.SETTLE_STEPS))
        mu_plain, mu_fed = net.mu(bank_ss), net.mu(with_phase(bank_ss, ks["REF"]))
        phase_rec.update(
            actor_phase_invariance_max_abs=float(np.max(np.abs(mu_fed - mu_plain))),
            fed={n: {"k_min": int(k.min()), "k_max": int(k.max()), "n": int(len(k)),
                     "n_presettle": int((k < C.SETTLE_STEPS).sum())} for n, k in ks.items()},
            rule={"effective": "each state's own phase: REF kept step + 1, OWN the snapshot's counter, ONPOLICY the "
                               "counter recorded in the collection", "raw": "slots 60-62 zero (no phase input)",
                  "settled": "every state at k = 200: (1, 1, 1)"})
        out["phase"] = phase_rec
    elif blind:
        net.policy.mlp_extractor.__class__ = H.critic_blind_class(MlpExtractor)
        for name, X in sets.items():
            crit[name]["effective"] = critic_slope(net.policy, net.norm, X)
        net.policy.mlp_extractor.__class__ = MlpExtractor
    else:
        for name in sets:
            crit[name]["effective"] = crit[name]["raw"]
    out["critic_slope_obs6"] = crit
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    return out


def saved_pairs(run_dir, T):
    """[(steps, zip, pkl)] in [S, T]: the start copy, the periodic pairs, and the final pair at T."""
    models = Path(run_dir) / MODELS_REL
    out = []
    for s in (C.S, *[x for x in C.SAVED_STEPS if x < T]):
        z, p = (models / n for n in C.periodic_names(s))
        if z.is_file() and p.is_file():
            out.append((s, z, p))
    z, p = models / C.FINAL_ZIP, models / C.FINAL_PKL
    if z.is_file() and p.is_file():
        out.append((T, z, p))
    return out


def loop_markers(pairs, *, blind, onpolicy, own, tag, phase=False):
    """M-LOOP over pairs, with the cache under <BS>/reads/_cache/.  phase (PS): critic_phase's reading; the pair at
    steps S is the start copy (the arm's load transform applied)."""
    import numpy as np

    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])

    def factory():
        return env_class(mid)(**ctor)
    env = factory()
    if env.unwrapped.model.actuator(C.ACT_TAIL3).name != "tail_3_pitch_act":
        raise InputError("action 14 is not tail_3_pitch_act")
    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        bank_ss = np.asarray(z["bank_raw_obs"], dtype=np.float64)[list(C.BANK_STATUE_SETTLED)]
        bank_k = np.asarray(z["bank_steps"], dtype=np.int64)[list(C.BANK_STATUE_SETTLED)] + 1     # PS: k = 200
    cache = require_under(Path(C.READS) / "_cache", C.BS)
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for steps, z, p in pairs:
        key = cache_key(C.sha256_file(z), C.sha256_file(p), blind=blind, onpolicy=onpolicy, own=own, phase=phase,
                        start=phase and steps == C.S)
        cpath = cache / f"{key}.json"
        if cpath.is_file():
            row = json.loads(cpath.read_text())
            bad = cache_entry_problem(row)
            if bad:
                raise InputError(f"loop cache entry {cpath}: {bad}")
            row["cache"] = str(cpath)
        else:
            row = loop_markers_one(z, p, blind=blind, onpolicy=onpolicy, own=own, env=env, factory=factory,
                                   bank_ss=bank_ss, phase=phase, start=phase and steps == C.S, bank_k=bank_k)
            row.update(reader_sha256=READER_SHA256, deps_sha256=DEPS_SHA256, reader_version=VERSION)
            atomic_write_text(cpath, json.dumps(row) + "\n")
        row["steps"] = steps
        rows.append(row)
        print(f"[{tag}] {steps}: " + loop_line(row), flush=True)
    env.close()
    return rows


def loop_line(row):
    def g(*ks):
        d = row
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def f(x, fmt="+.3f"):
        return "  n/a " if x is None else format(x, fmt)
    return (f"OWN G {f(g('OWN', 'P', 'G'))} c0 {f(g('OWN', 'P', 'c0'))} own14 {f(g('OWN', 'own14_median'))} "
            f"G_L {f(g('OWN', 'L', 'G'))} b* {f(g('OWN', 'P', 'b_star'))} | REF G {f(g('REF', 'P', 'G'))} c0 "
            f"{f(g('REF', 'P', 'c0'))} | dV/dq3 on-policy eff "
            f"{f(g('critic_slope_obs6', 'ONPOLICY', 'effective', 'dV_dq3_per_deg'), '+.5f')} "
            f"raw {f(g('critic_slope_obs6', 'ONPOLICY', 'raw', 'dV_dq3_per_deg'), '+.5f')} own eff "
            f"{f(g('critic_slope_obs6', 'OWN', 'effective', 'dV_dq3_per_deg'), '+.5f')}")


# ================================================================================================ the reads
def render(markers):
    L = [f"BS MARKERS {markers['label']}  ({markers['version']}; {markers['utc']})",
         f"window (S, T] = ({markers['S']:,}, {markers['T']:,}]; arm {markers.get('arm')}"]
    pm = markers.get("probe")
    if pm:
        L.append(f"M-PROBE {pm['n_rows']} rows; statue_settled tail_3 mean first/last "
                 f"{pm['tail3_ss_mean_first_last']}; "
                 f"8-row trailing max {pm['trailing8_max']}; first above {pm['trailing8_first_above']}")
    lm = markers.get("ledger")
    if lm:
        d, m = lm["direct"], lm["m5"]
        L.append(f"M-LEDGER CMD_L max {d['cmd_l_max']} last {d['cmd_l_last']}; firsts {d['firsts']}; absent "
                 f"{d['evaluations_absent']}")
        L.append(f"  M5: ANY flagged {m['any_flagged']} TAIL flagged {m['tail_flagged']} ANY alarms {m['any_alarms']} "
                 f"TAIL alarms {m['tail_alarms']}; C6 counting failures {m['n_c6_counting_failures']}")
        for s, e in d["per_evaluation"].items():
            cmd = e["cmd_l"] if e["cmd_l"] is None else round(e["cmd_l"], 3)
            L.append(f"    {int(s):>9}  n {e['n']:>2}  CMD_L {cmd!s:>7}"
                     f"  over {e['n_over']}  tail {e['n_tail_over']}  t3 {e['n_t3_over']}")
    for row in markers.get("loop") or []:
        L.append(f"M-LOOP {row['steps']:>9}: " + loop_line(row))
    for k, v in (markers.get("reproduction") or {}).items():
        L.append(f"REPRO {k}: {v}")
    return "\n".join(L) + "\n"


def full_read(run_dir, out=None, *, onpolicy=True, own=True):
    root = Path(C.BS).resolve()
    run = Path(run_dir).resolve()
    kind = run_kind(run, root)
    if kind is None:
        raise InputError(f"{run} is neither <BS>/runs/<name> nor under <BS>/validate/")
    check, _ = read_json_file(run / C.CHECK_JSON)
    if check.get("passed") is not True:
        raise InputError(f"{run}/check.json does not pass")
    stale = check_is_current(run, check)
    if stale:
        raise InputError(f"check.json is not current: {stale}")
    if kind == "continuation":
        ok, und = decided(root)
        if not ok:
            raise InputError(f"outcome-blind: not every SCREEN.json continuation is decided ({und})")
    meta, _ = read_json_file(run / C.RUN_META)
    S, T, arm = C.S, int(meta["T"]), meta["arm"]
    out = require_under(Path(out) if out else (root / "reads" / run.name if kind == "continuation"
                                               else run / "read"), root)
    out.mkdir(parents=True, exist_ok=True)
    markers = dict(schema=MARKERS_SCHEMA, version=VERSION, tool_sha256=C.sha256_file(Path(__file__)), utc=utcnow(),
                   label=run.name, run=str(run), kind=kind, arm=arm, seed=meta.get("seed"), S=S, T=T,
                   stop_reason=meta.get("stop_reason"))
    markers["probe"] = probe_markers(run / PROBE_REL, S, T, run.name)
    markers["ledger"] = ledger_markers(run / LEDGER_REL, S, T, out)
    setup_simulation(f"read_{run.name}")
    markers["probe_phase"] = meta.get("probe_phase")
    markers["loop"] = loop_markers(saved_pairs(run, T), blind=(arm == "critic_blind_tail3"), onpolicy=onpolicy,
                                   own=own, tag=run.name, phase=(arm == "critic_phase"))
    atomic_write_text(out / "markers.json", json.dumps(markers, indent=1, default=str) + "\n")
    atomic_write_text(out / "markers.txt", render(markers))
    return markers


def reference_read(out=None, *, onpolicy=True, own=True):
    """The parent's markers over (S, E] and its F2 pairs in [S, E] (REFERENCE_PAIRS), with the reproduction check."""
    root = Path(C.BS).resolve()
    out = require_under(Path(out) if out else root / "reads" / "parent_s42", root)
    out.mkdir(parents=True, exist_ok=True)
    S, T = C.S, C.E
    st = C.START
    for rel, want in ((st["zip_rel"], st["zip_sha256"]), (st["pkl_rel"], st["pkl_sha256"])):
        if C.sha256_file(C.f2_file(rel)) != want:
            raise InputError(f"{rel}: not at FMAN's sha256")
    if C.sha256_file(C.f2_record("eval_episodes.csv")) != st["records"]["eval_episodes.csv"]:
        raise InputError("the parent ledger is not at FMAN's sha256")
    markers = dict(schema=MARKERS_SCHEMA, version=VERSION, tool_sha256=C.sha256_file(Path(__file__)), utc=utcnow(),
                   label="parent stance6_r10LTSB_s42 (F2, read only)", kind="reference", arm="parent", S=S, T=T)
    markers["probe"] = probe_markers(C.f2_record("policy_probe.npz"), S, T, "parent")  # the same rows as a run
    # the parent's ledger runs to 12M: M5 (with its C6 up to E) reads a copy truncated to timesteps <= E (read_ecs's
    # V4 form), written under the output folder
    markers["ledger"] = ledger_markers(truncate_ledger(C.f2_record("eval_episodes.csv"), out / "eval_episodes_le_E.csv",
                                                       T), S, T, out)
    setup_simulation("read_parent_s42")
    pairs = [(s, C.f2_file(f"{C.STAGE_DIR}/models/{C.periodic_names(s)[0]}"),
              C.f2_file(f"{C.STAGE_DIR}/models/{C.periodic_names(s)[1]}")) for s in REFERENCE_PAIRS]
    markers["loop"] = loop_markers(pairs, blind=False, onpolicy=onpolicy, own=own, tag="parent")
    rep = {}
    for row in markers["loop"]:
        want = REPRO.get(row["steps"])
        if not want:
            continue
        r = {}
        if own and "OWN" in row:
            r["ew_own_G"] = [row["OWN"]["P"]["G"], want["ew_own"]["G"]]
            r["ew_own_c0"] = [row["OWN"]["P"]["c0"], want["ew_own"]["c0"]]
            r["ew_own14"] = [row["OWN"]["own14_median"], want["ew_own"]["own14"]]
            r["vew_G_H"] = [row["OWN"]["H"]["G"], want["vew_G_H"]]
            r["vew_G_L"] = [row["OWN"]["L"]["G"], want["vew_G_L"]]
            r["vew_b_star_P"] = [row["OWN"]["P"]["b_star"], want["vew_b_star_P"]]
            r["vew_b_star_L"] = [row["OWN"].get("b_star_L"), want["vew_b_star_L"]]
            r["wm_b_star_P"] = [row["OWN"]["P"]["b_star"], want["wm_b_star_P"]]
        r["ew_ref_G"] = [row["REF"]["P"]["G"], want["ew_ref_G"]]
        if onpolicy:
            r["eb_dV_dq3_per_deg"] = [row["critic_slope_obs6"]["ONPOLICY"]["raw"]["dV_dq3_per_deg"],
                                      want["eb_dV_dq3_per_deg"]]
        pairs_ = dict(r)
        r["max_abs_diff_ew"] = max(abs(a - b) for k, (a, b) in pairs_.items() if k.startswith("ew_"))
        printed = [abs(a - b) for k, (a, b) in pairs_.items() if k.startswith(("vew_", "wm_")) and a is not None]
        if printed:
            r["max_abs_diff_printed_3dp"] = max(printed)          # against values printed to 3 decimals: <= 5e-4
        if onpolicy:
            a, b = r["eb_dV_dq3_per_deg"]
            r["rel_diff_eb"] = abs(a - b) / abs(b)
        rep[str(row["steps"])] = r
    markers["reproduction"] = rep
    atomic_write_text(out / "markers.json", json.dumps(markers, indent=1, default=str) + "\n")
    atomic_write_text(out / "markers.txt", render(markers))
    return markers


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-onpolicy", dest="onpolicy", action="store_false")
    ap.add_argument("--no-own", dest="own", action="store_false")
    ns = ap.parse_args(argv)
    print(VERSION, flush=True)
    try:
        if ns.check:
            if not ns.run:
                raise InputError("--check needs RUN")
            code, rec = check_run(ns.run)
            print(json.dumps({"check": "pass" if rec["passed"] else "fail", "failures": rec["failures"]}, indent=1))
            return code
        if ns.reference:
            reference_read(ns.out, onpolicy=ns.onpolicy, own=ns.own)
            return EXIT_OK
        if not ns.run:
            raise InputError("RUN, --check RUN or --reference")
        full_read(ns.run, ns.out, onpolicy=ns.onpolicy, own=ns.own)
        return EXIT_OK
    except InputError as exc:
        print(f"INPUT ERROR: {exc}", flush=True)
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
