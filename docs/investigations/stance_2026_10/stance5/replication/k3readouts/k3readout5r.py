"""k3readout5r.py (REPORT ONLY): the stance5 replication's K3 mechanism readouts, as fixed in
R/readouts_addendum_k3.txt (that file governs; R = SP/foot/stance5/replication, SP = the scratchpad).  The readouts enter
no verdict, no decision line and no tie-break.

Run it through the runner (k3readout_one5r.sh RUN STEPS [--arm LTSB|LTSBr]), or by hand exactly as the runner does:

    cd R/k3readouts && ISO_WT=SP/fix/scratch_copband OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        PYTHONPYCACHEPREFIX=SP/caches/pyc MPLCONFIGDIR=SP/caches/mpl \
        nice -n 19 python3 SP/foot/stance3/tools/iso.py k3readout5r.py --arm LTSBr --run stance5_LTSB_s70 --steps final

Arms (--arm):
  LTSBr (default, the replication): judge module R/judge/judge_stance5r.py (sha256 pinned below), runs stance5_LTSB_s70 /
        stance5_LTSB_s80 under R/runs, judge files R/judged/<RUN>/judge_<steps>.json.  The judge sigma of seeds 70 / 80 is
        read by judge_stance5r.main's own sidecar branch (its lines 1221-1236, re-executed here statement for statement
        on the module's own constants CONTROL_SIGMA_SIDECAR, CONTROL_SIGMA_SIDECAR_SHA256 and FROZEN_PARAMS_SHA256):
        the sidecar's sha256 must equal the pinned one and its source_sha256 the pinned frozen_params; the vector must
        equal the judge file's noise_matched_control_joint.sigma_vector exactly, and the judge file must name the same
        sidecar file and sha256 (control_sigma_file / control_sigma_file_sha256).
  LTSB  (validation and reference): judge module SP/foot/stance5/judge/judge_stance5.py, runs stance5_LTSB_s42/s52/s60
        under SP/foot/stance5/runs and .../judged; sigma = frozen_params criteria.H4.control_sigma[seed].vector, read as
        k3diag/verify_1/vroll.py lines 93-95 read it (frozen_params sha256 checked against J.FROZEN_PARAMS_SHA256).

What is rolled (one fresh env per condition, the judge's recorder SubstepContactRecorder(env, Morphology) attached, 20
spawns J.NOISE_SEED .. +J.NOISE_EPISODES-1 = 3042..3061, rng = numpy default_rng(episode seed), z = rng.standard_normal(15)
drawn every step, obs normalised with the checkpoint's VecNormalize stats, mu = action_net(mlp_extractor.forward_actor(
extract_features(obs))), cmd = clip(mu); a fall = an episode shorter than the horizon (1000)):
  noise conditions, act = clip(cmd + transform(n)), n = sigma * z  (verify_1/vroll.py lines 35-64 and 115-136):
    full       n                                     (must reproduce the judge file: per-episode seed, length,
                                                      termination_reason, recorded as matches_judge; and per-episode
                                                      reward within 1e-6 as v2_probe.py line 182 required, recorded
                                                      as reward_matches_judge; valid needs both)
    hip_pitch  n masked to actuators 3, 7 (r/l hip pitch)
    nohp       n masked to every actuator except 3, 7
    hpD        d = (n_r - n_l)/2 applied as (d, -d) on 3, 7; zero elsewhere   (real amplitude, differential part)
    hpC        c = (n_r + n_l)/2 applied as (c, c) on 3, 7; zero elsewhere    (real amplitude, common part)
  kick probe (k3diag/rollouts/scripts/k3roll.py lines 57-62 and 214-225, conditions pulse_anti and pulse_anti20):
    deterministic (the noise mask is all zeros: act = clip(cmd + 0 * z), i.e. the clipped mean), and on the control
    steps 400 .. 419 (nsteps counted from 0 before env.step; 20 steps x control_dt 0.01 s = 0.2 s, starting 4.0 s in)
    the kick is ADDED to that command and clipped again: act[3] = clip(act[3] + a_r), act[7] = clip(act[7] + a_l);
      pulse_anti   (a_r, a_l) = (+0.3, -0.3)        pulse_anti20   (a_r, a_l) = (+0.2, -0.2)
    falls of 20 over the same spawns (k3pulse.py: falls = length < 1000).  Also recorded (not part of k3pulse's
    count): falls_before_kick, the episodes that ended at or before step 400 (L <= 400: the kick was never applied).
Readouts:
  response   (k3diag/verify_2/v2_probe.py lines 101-139 and v2_probe_ana.py): during "full", at every visited state, a
             batch of the raw observation and six copies with the hip-pitch angle channels nudged (obs index
             jnt_qposadr - 7 of the r / l hip-pitch joints: q_r + h, q_l - h and q_r - h, q_l + h, h = 0.01 rad; velocity
             channels (nq - 7) + dofadr - 6: +/- 0.05 rad/s; common: both + h / both - h) goes through the same
             normalisation and actor (float32 batch); Jq = ((mu_r - mu_l)(+h) - (mu_r - mu_l)(-h)) / (4 h) per rad of
             q_r - q_l.  The probe passes never touch the env or the rng.  Aggregate: steps 200 .. L-1 of survivors and
             200 .. L-151 of falls, episodes with fewer than 61 such steps skipped, pooled; the median is the readout
             (also mean, p10/p90, per-episode-median range; Jv per rad/s; Jq_common).
  own sigma  exp(model.policy.log_std) of the checkpoint J.load_checkpoint returns, per actuator; sigma_judge / sigma_own
             per actuator; the hip-pitch value is v2_probe's own_sigma_hp = mean(own[3], own[7]).
  stagger    the env's info["foot_stagger_m"] (heading-frame r_foot - l_foot fore-aft offset) read after each env.step,
             as v2_probe line 160 reads it; per noise condition: sd over window steps 200 .. L-1 pooled over the 20
             episodes (the addendum's readout, sd_mm), the per-episode max |stagger| over the same window, and, for
             comparison with k3diag, the sd over v2_probe_ana's window (200 .. L-1 survivors, 200 .. L-151 falls) and the
             sd of q_r - q_l (the hip-pitch angles the policy saw) over that window.  sd_mm includes the topple steps of
             falling episodes, so whenever a condition has falls inside its window (falls_in_window > 0) it partly
             measures the falls; sd_mm_v2_window stops falls 150 steps before the end and is the figure comparable with
             k3diag's 48-50 mm.  The summary line prints both and falls_in_window.

Refusals (exit 3; a refused JSON with the reasons is written as OUT_ROOT/<RUN>/refused_<steps>.json, never overwritten):
the judge module is not the pinned file; the judge file is missing, foreign (schema, mode, seed, steps, arm, smoke, or
a recorded judge_sha256 other than the pinned judge module's) or incomplete; the checkpoint is missing; the model or
VecNormalize sha256 differs from the judge file's; the sigma source fails its sha256 checks or the sigma differs from the
judge file's; the clone is not clean at its expected HEAD (or not the judge file's HEAD).  If "full" does not reproduce
the judge file exactly the output is still written, marked "valid": false ("matches_judge": false for seed / length /
termination, "reward_matches_judge": false for a reward difference of 1e-6 or more) and the tool exits 4.

Exit codes: 0 a valid output written; 2 bad arguments or a K3R_TEST_ROOT outside R/k3readouts (nothing written);
3 refused (refused JSON written; with --check-only nothing is written); 4 an output written but not valid ("full" did
not reproduce the judge file, the clone changed during the rolls, or a smoke / test-root run); 1 a crash.  Readers
select outputs by their "valid" field, never by file name (an invalid first output keeps the k3r_<steps>.json name).

Output: OUT_ROOT/<RUN>/k3r_<steps>.json (OUT_ROOT = R/k3readouts/out); an existing output is never overwritten: the
next free k3r_<steps>.<n>.json is written instead and the tool says so.  One summary line starting "K3R " on stdout.
"addendum_slot" (top level and in the summary line) says whether the run and slot are one the addendum names:
"readout" (LTSBr s70 / s80 at 6291456 and final), "reference" (LTSB s42 at 6291456, 7340032 and final; s52 and s60
finals) or "no" (any other slot: a measurement, but not one of the addendum's readouts; the tool prints a NOTE).
--check-only runs every identity check (and loads the checkpoint) and writes nothing.  --smoke N (development only)
rolls N episodes per condition and writes under R/k3readouts/selftest/out (never under out/), marked smoke and invalid.
K3R_TEST_ROOT (self-tests only; must resolve inside R/k3readouts, not R/k3readouts itself, out/ or logs/, else exit 2):
judged files are read from $K3R_TEST_ROOT/judged/<RUN>/ and every output goes under $K3R_TEST_ROOT/out; the JSON
records it and is never valid.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

SCRATCH = "scratchpad"
S5 = SCRATCH + "/foot/stance5"
R = S5 + "/replication"
HERE = R + "/k3readouts"
OUT_ROOT = HERE + "/out"
SMOKE_ROOT = HERE + "/selftest/out"
ADDENDUM = R + "/readouts_addendum_k3.txt"
ADDENDUM_SHA256 = "7a79e9f66091e8fcdcc53eba2248df3cc2f5b6a7ad3db6bc66d88bb4e58d2eaa"
CLONE = SCRATCH + "/fix/scratch_copband"
CLONE_HEAD = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"
JUDGE_ARM = "LTSB"  # both modes judge arm LTSB (the band arm) on the same clone
SIDECAR_SHA256_PIN = "89778b3f81f83f05d4c15423b3658593dfaa767388213044c50a72e2d96b8522"
FROZEN_PARAMS_SHA256_PIN = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
MODES = {
    # addendum_slots: the slots R/readouts_addendum_k3.txt names for each run ("WHERE AND WHEN": the replication's
    # readouts at 6,291,456 and final; the stance5 reference values at s42 6,291,456, 7,340,032 and final, s52 and s60 finals)
    "LTSBr": {"judge": R + "/judge/judge_stance5r.py",
              "judge_sha256": "16d39f7dcb14a8b6f94061331933df8f1dcee76aa988479e293b132bf8ac05ae",
              "module": "judge_stance5r", "root": R, "schema": "stance5r-judge/v1",
              "runs": {"stance5_LTSB_s70": 70, "stance5_LTSB_s80": 80}, "addendum_role": "readout",
              "addendum_slots": {"stance5_LTSB_s70": ("6291456", "final"), "stance5_LTSB_s80": ("6291456", "final")}},
    "LTSB": {"judge": S5 + "/judge/judge_stance5.py",
             "judge_sha256": "d9286724550b83eccfd11e2ceba634cfca847f1e52e03988f9c486c6db6bda13",
             "module": "judge_stance5", "root": S5, "schema": "stance5-judge/v1",
             "runs": {"stance5_LTSB_s42": 42, "stance5_LTSB_s52": 52, "stance5_LTSB_s60": 60}, "addendum_role": "reference",
             "addendum_slots": {"stance5_LTSB_s42": ("6291456", "7340032", "final"), "stance5_LTSB_s52": ("final",),
                                "stance5_LTSB_s60": ("final",)}},
}
EXIT_OK, EXIT_ARGS, EXIT_REFUSED, EXIT_INVALID = 0, 2, 3, 4
REWARD_TOL = 1e-6  # v2_probe.py line 182: matches_judge also required max |reward - judge reward| < 1e-6
ACT = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
       "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
       "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]
RHP, LHP = 3, 7
NAMED = {"hip_pitch": [3, 7], "nohp": [i for i in range(15) if i not in (3, 7)]}  # vroll.py NAMED
NOISE_CONDS = ("full", "hip_pitch", "nohp", "hpD", "hpC")
PULSE = {"pulse_anti": (0.3, -0.3), "pulse_anti20": (0.2, -0.2)}  # k3roll.py PULSE (the two kicks of the addendum)
PULSE_STEPS = (400, 420)  # k3roll.py PULSE_STEPS: control steps 400 .. 419
CONDS = NOISE_CONDS + tuple(PULSE)
H_POS, H_VEL = 0.01, 0.05  # v2_probe.py
W0 = 200  # window start (v2_probe_ana.py W0; the addendum's stagger window)
FALL_TRIM = 150  # v2_probe_ana.py: falls' window ends at L - 150 (exclusive)
MIN_EXTRA = 60  # v2_probe_ana.py: an episode needs hi > W0 + 60
DEFINITIONS = {
    "full": "act = clip(cmd + sigma*z) (the judge's noise_matched, space joint)",
    "hip_pitch": "act = clip(cmd + sigma*z masked to actuators 3, 7)",
    "nohp": "act = clip(cmd + sigma*z masked to every actuator except 3, 7)",
    "hpD": "n = sigma*z; d = 0.5*(n[3]-n[7]); act = clip(cmd + e), e[3] = d, e[7] = -d, e = 0 elsewhere",
    "hpC": "n = sigma*z; c = 0.5*(n[3]+n[7]); act = clip(cmd + e), e[3] = c, e[7] = c, e = 0 elsewhere",
    "pulse_anti": "deterministic: act = clip(cmd + 0*z); at control steps 400..419 act[3] = clip(act[3]+0.3), "
                  "act[7] = clip(act[7]-0.3) (0.2 s, k3roll.py pulse_anti)",
    "pulse_anti20": "deterministic: act = clip(cmd + 0*z); at control steps 400..419 act[3] = clip(act[3]+0.2), "
                    "act[7] = clip(act[7]-0.2) (0.2 s, k3roll.py pulse_anti20)",
}
SOURCES = {
    "noise_loop": "SP/foot/stance5/k3diag/verify_1/vroll.py lines 108-147 (judge noise_matched, joint space)",
    "transform": "SP/foot/stance5/k3diag/verify_1/vroll.py lines 35-64 (full, NAMED masks, hpD/hpC)",
    "kick": "SP/foot/stance5/k3diag/rollouts/scripts/k3roll.py lines 57-62 (PULSE, PULSE_STEPS) and 214-225 "
            "(mask zeros, act = clip(cmd + sig_eff*z), the pulse added after the clip and clipped again); falls as "
            "k3pulse.py line 43 (length < 1000)",
    "response": "SP/foot/stance5/k3diag/verify_2/v2_probe.py lines 28, 90-94, 101-105, 117, 120-139 (probe) and "
                "v2_probe_ana.py lines 12 and 19-38 (window, pooled median)",
    "own_sigma": "exp(policy.log_std) as judge noise_matched / v2_probe.py line 86; own_sigma_hp as v2_probe.py line 178",
    "stagger": "info['foot_stagger_m'] after env.step as v2_probe.py line 160; v2 window as v2_probe_ana.py line 21",
    "sigma_LTSB": "SP/foot/stance5/k3diag/verify_1/vroll.py lines 93-95 (frozen_params control_sigma[seed].vector)",
    "sigma_LTSBr": "SP/foot/stance5/replication/judge/judge_stance5r.py main() lines 1221-1236 (frozen_params sha256, "
                   "then for seeds 70/80 the sidecar: sha256 == CONTROL_SIGMA_SIDECAR_SHA256, source_sha256 == "
                   "FROZEN_PARAMS_SHA256, control_sigma[seed]); main line 1340 passes ctrl['vector'] to noise_matched, "
                   "whose line 870 takes np.asarray(sigma, float64)",
}


class Refusal(Exception):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def transform(cond: str, z, sigma):
    """verify_1/vroll.py transform, restricted to the addendum's noise conditions (same arithmetic)."""
    n = sigma * z
    if cond == "full":
        return n
    if cond in NAMED:
        m = np.zeros(15)
        m[NAMED[cond]] = 1
        return n * m
    d = 0.5 * (n[RHP] - n[LHP])
    c = 0.5 * (n[RHP] + n[LHP])
    if cond not in ("hpD", "hpC"):
        raise SystemExit("unknown cond " + cond)
    out = np.zeros(15)
    if cond == "hpD":
        out[RHP], out[LHP] = d, -d
    else:
        out[RHP], out[LHP] = c, c
    return out


def free_path(path: str) -> str:
    """path, or the first free <stem>.<n><ext> when path exists (outputs are never overwritten)."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}.{n}{ext}"):
        n += 1
    return f"{stem}.{n}{ext}"


def write_new(path: str, record: dict) -> str:
    """Write record as JSON at path (or the next free name), atomically and never over an existing file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(record, indent=1, allow_nan=False)
    tmp = os.path.join(os.path.dirname(path), f".tmp_{os.getpid()}_{os.path.basename(path)}")
    with open(tmp, "x") as fh:
        fh.write(data + "\n")
    try:
        while True:
            target = free_path(path)
            try:
                os.link(tmp, target)  # fails if target exists: never overwrites
                break
            except FileExistsError:
                continue
    finally:
        os.unlink(tmp)
    if target != path:
        print(f"[k3readout5r] {path} exists; wrote {target} instead (outputs are never overwritten)", flush=True)
    return target


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def load_judge(mode: dict):
    """Import the pinned judge file read-only (refuses any other file)."""
    path = mode["judge"]
    if not os.path.isfile(path):
        raise Refusal([f"judge module {path} missing"])
    got = sha256_file(path)
    if got != mode["judge_sha256"]:
        raise Refusal([f"judge module {path} sha256 {got} != the pinned {mode['judge_sha256']}"])
    spec = importlib.util.spec_from_file_location(mode["module"], path)
    J = importlib.util.module_from_spec(spec)
    sys.modules[mode["module"]] = J
    spec.loader.exec_module(J)
    return J, got


def judge_sigma(J, arm: str, seed: int, ident: dict):
    """The run's judge sigma through the judge's own code path (see SOURCES)."""
    with open(J.FROZEN_PARAMS, "rb") as fh:
        fp_raw = fh.read()
    fp_sha = hashlib.sha256(fp_raw).hexdigest()
    if fp_sha != J.FROZEN_PARAMS_SHA256 or fp_sha != FROZEN_PARAMS_SHA256_PIN:
        raise Refusal([f"{J.FROZEN_PARAMS} sha256 {fp_sha} != the pinned {J.FROZEN_PARAMS_SHA256}"])
    h4 = json.loads(fp_raw.decode("utf-8"))["criteria"]["H4"]
    if not (h4["episodes_per_seed"] == 20 and h4["spawn_range"] == [3042, 3061] and h4["space"] == "joint"):
        raise Refusal([f"frozen_params H4 is not the 20-episode joint-space test: {h4}"])
    if arm == "LTSBr":
        if seed not in J.SIDECAR_SEEDS:
            raise Refusal([f"seed {seed} is not a sidecar seed {J.SIDECAR_SEEDS}"])
        if J.CONTROL_SIGMA_SIDECAR_SHA256 != SIDECAR_SHA256_PIN:
            raise Refusal([f"judge CONTROL_SIGMA_SIDECAR_SHA256 {J.CONTROL_SIGMA_SIDECAR_SHA256} != {SIDECAR_SHA256_PIN}"])
        # judge_stance5r.main lines 1227-1236, statement for statement
        with open(J.CONTROL_SIGMA_SIDECAR, "rb") as fh:
            sc_raw = fh.read()
        sigma_file, sigma_sha = J.CONTROL_SIGMA_SIDECAR, hashlib.sha256(sc_raw).hexdigest()
        if sigma_sha != J.CONTROL_SIGMA_SIDECAR_SHA256:
            raise Refusal([f"{J.CONTROL_SIGMA_SIDECAR} sha256 {sigma_sha} != the pinned {J.CONTROL_SIGMA_SIDECAR_SHA256}"])
        sidecar = json.loads(sc_raw.decode("utf-8"))
        if sidecar.get("source_sha256") != J.FROZEN_PARAMS_SHA256:
            raise Refusal([f"{J.CONTROL_SIGMA_SIDECAR} was not computed from the pinned frozen_params "
                           f"({sidecar.get('source_sha256')})"])
        ctrl = sidecar["control_sigma"][str(seed)]
    else:
        # vroll.py lines 93-95 (the stance5 judge's frozen_params branch)
        sigma_file, sigma_sha = os.path.realpath(J.FROZEN_PARAMS), fp_sha
        ctrl = h4["control_sigma"][str(seed)]
    sigma = np.asarray(ctrl["vector"], dtype=np.float64)  # judge noise_matched: np.asarray(sigma, dtype=np.float64)
    ident["sigma"] = {"file": sigma_file, "file_sha256": sigma_sha, "source": ctrl["source"], "vector": sigma.tolist(),
                      "frozen_params": os.path.realpath(J.FROZEN_PARAMS), "frozen_params_sha256": fp_sha,
                      "code_path": SOURCES["sigma_" + arm]}
    return sigma, ctrl, sigma_file, sigma_sha


def check_judge_file(jd: dict, mode: dict, arm: str, run: str, seed: int, steps: str, zp: str, vp: str,
                     sigma, ctrl: dict, sigma_file: str, sigma_sha: str, ident: dict) -> list[str]:
    """Every identity comparison against the judge file; returns the refusal reasons (empty = pass)."""
    bad = []
    jn = jd.get("noise_matched_control_joint")
    for key, want in (("schema", mode["schema"]), ("arm", JUDGE_ARM), ("mode", "full"), ("seed", seed), ("smoke", False),
                      ("noise_run", True)):
        if jd.get(key) != want:
            bad.append(f"judge file {key} {jd.get(key)!r} != {want!r}")
    if str(jd.get("steps")) != steps:
        bad.append(f"judge file steps {jd.get('steps')!r} != {steps!r}")
    if jd.get("judge_sha256") != "sha256:" + mode["judge_sha256"]:  # the judge that wrote it must be the pinned module
        bad.append(f"judge file judge_sha256 {jd.get('judge_sha256')!r} != the pinned 'sha256:{mode['judge_sha256']}'")
    if "judge_seconds" not in jd:
        bad.append("judge file incomplete (no judge_seconds)")
    if not isinstance(jn, dict) or len(jn.get("episodes_detail") or []) != 20:
        bad.append("judge file has no complete noise_matched_control_joint (20 episodes)")
        return bad
    git = jd.get("git") or {}
    if git.get("head") != CLONE_HEAD or git.get("status") != "":
        bad.append(f"judge file was judged at {git.get('head')} dirty={bool(git.get('status'))}, not the clean {CLONE_HEAD}")
    msha, vsha = "sha256:" + sha256_file(zp), "sha256:" + sha256_file(vp)
    ident.update({"model": zp, "model_sha256": msha, "vecnorm": vp, "vecnorm_sha256": vsha,
                  "judge_file_model_sha256": jd.get("model_sha256"), "judge_file_vecnorm_sha256": jd.get("vecnorm_sha256")})
    if jd.get("model_sha256") != msha:
        bad.append(f"model sha256 {msha} != the judge file's {jd.get('model_sha256')}")
    if jd.get("vecnorm_sha256") != vsha:
        bad.append(f"vecnorm sha256 {vsha} != the judge file's {jd.get('vecnorm_sha256')}")
    jsig = np.asarray(jn.get("sigma_vector"), dtype=np.float64)
    ident["judge_file_sigma_vector_equal"] = bool(jsig.shape == sigma.shape and np.array_equal(jsig, sigma))
    if not ident["judge_file_sigma_vector_equal"]:
        bad.append(f"sigma differs from the judge file's sigma_vector (max abs diff "
                   f"{float(np.max(np.abs(jsig - sigma))) if jsig.shape == sigma.shape else 'shape'})")
    if jn.get("sigma_source") != ctrl["source"] or jn.get("sigma") != "control" or jn.get("space") != "joint":
        bad.append(f"judge file sigma_source/sigma/space {jn.get('sigma_source')!r}/{jn.get('sigma')!r}/{jn.get('space')!r} "
                   f"!= {ctrl['source']!r}/'control'/'joint'")
    if jn.get("seeds") != [3042, 3061]:
        bad.append(f"judge file noise seeds {jn.get('seeds')} != [3042, 3061]")
    if jd.get("frozen_params_sha256") != ident["sigma"]["frozen_params_sha256"]:
        bad.append(f"judge file frozen_params_sha256 {jd.get('frozen_params_sha256')} != "
                   f"{ident['sigma']['frozen_params_sha256']}")
    if arm == "LTSBr":
        if jd.get("control_sigma_file") != sigma_file or jd.get("control_sigma_file_sha256") != sigma_sha:
            bad.append(f"judge file sigma file {jd.get('control_sigma_file')} {jd.get('control_sigma_file_sha256')} "
                       f"!= {sigma_file} {sigma_sha}")
    return bad


def roll(J, ctx, pol, stats, sigma, cond: str, episodes: int) -> dict:
    """One condition: one env, the judge's recorder, spawns NOISE_SEED.. ; the probe runs in "full" only."""
    import mujoco
    import torch
    from stable_baselines3.common.policies import BaseModel

    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder

    vmean, vvar, eps, clip_obs = stats
    probe = cond == "full"
    pulse = PULSE.get(cond)
    sig_eff = np.zeros(15) if pulse is not None else None  # k3roll: the pulse conditions' mask is all zeros

    def mu_of(raw_batch):  # v2_probe.py mu_of
        o = np.clip((raw_batch - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
        with torch.no_grad():
            f = BaseModel.extract_features(pol, torch.as_tensor(o), pol.pi_features_extractor)
            return pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy().astype(np.float64)

    t0 = time.monotonic()
    env = ctx.env_class(**ctx.env_kwargs)
    u = env.unwrapped
    m = u.model
    low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names == ACT, names
    jid = [int(m.actuator_trnid[i, 0]) for i in (RHP, LHP)]
    qi = [int(m.jnt_qposadr[j]) - 7 for j in jid]
    vi = [(m.nq - 7) + int(m.jnt_dofadr[j]) - 6 for j in jid]
    rec = SubstepContactRecorder(env, Morphology.from_env(env, ctx.mid)).attach()
    E, T = episodes, ctx.horizon
    arr = {k: np.full((E, T), np.nan, dtype=np.float32) for k in ("stagger", "dq")}
    if probe:
        arr.update({k: np.full((E, T), np.nan, dtype=np.float32) for k in ("Jq", "Jv", "Jq_common")})
    lengths, terms, rewards = [], [], []
    try:
        for i in range(E):
            s = J.NOISE_SEED + i
            rng = np.random.default_rng(s)
            obs, _ = env.reset(seed=s)
            for k in (0, 1):  # v2_probe.py line 117: the obs indices are the hip-pitch joint angles
                assert abs(float(obs[qi[k]]) - float(u.data.qpos[qi[k] + 7])) < 1e-5
            n, rew = 0, 0.0
            while True:
                raw = np.asarray(obs, dtype=np.float64)
                if probe:  # v2_probe.py lines 121-137
                    B = np.repeat(raw[None], 7, 0)
                    B[1, qi[0]] += H_POS; B[1, qi[1]] -= H_POS  # noqa: E702
                    B[2, qi[0]] -= H_POS; B[2, qi[1]] += H_POS  # noqa: E702
                    B[3, vi[0]] += H_VEL; B[3, vi[1]] -= H_VEL  # noqa: E702
                    B[4, vi[0]] -= H_VEL; B[4, vi[1]] += H_VEL  # noqa: E702
                    B[5, qi[0]] += H_POS; B[5, qi[1]] += H_POS  # noqa: E702
                    B[6, qi[0]] -= H_POS; B[6, qi[1]] -= H_POS  # noqa: E702
                    MU = mu_of(B.astype(np.float32).astype(np.float64))
                    d = MU[:, RHP] - MU[:, LHP]
                    arr["Jq"][i, n] = (d[1] - d[2]) / (4 * H_POS)
                    arr["Jv"][i, n] = (d[3] - d[4]) / (4 * H_VEL)
                    arr["Jq_common"][i, n] = (d[5] - d[6]) / (2 * H_POS)
                # the judge's own single-row mean (noise_matched; vroll.py lines 121-124)
                o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
                with torch.no_grad():
                    f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
                    mu = pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)
                arr["dq"][i, n] = raw[qi[0]] - raw[qi[1]]
                cmd = np.clip(mu, low, high)
                z = rng.standard_normal(cmd.shape)
                if pulse is None:
                    act = np.clip(cmd + transform(cond, z, sigma), low, high)  # vroll.py line 127
                else:  # k3roll.py lines 222-225
                    act = np.clip(cmd + sig_eff * z, low, high)
                    if PULSE_STEPS[0] <= n < PULSE_STEPS[1]:
                        act[RHP] = np.clip(act[RHP] + pulse[0], low[RHP], high[RHP])
                        act[LHP] = np.clip(act[LHP] + pulse[1], low[LHP], high[LHP])
                obs, r, term, trunc, info = env.step(act)
                rec.end_step(act, float(r))
                arr["stagger"][i, n] = float(info.get("foot_stagger_m", np.nan))
                rew += float(r)
                n += 1
                if term or trunc:
                    rec.end_episode(terminated=bool(term), truncated=bool(trunc),
                                    termination_reason=str(info.get("termination_reason", "")))
                    terms.append(str(info.get("termination_reason", "")))
                    break
            lengths.append(n)
            rewards.append(rew)
    finally:
        rec.detach()
        env.close()
    seeds = [J.NOISE_SEED + k for k in range(E)]
    out = {"definition": DEFINITIONS[cond], "n": E, "falls": int(sum(L < ctx.horizon for L in lengths)),
           "fall_seeds": [s for s, L in zip(seeds, lengths) if L < ctx.horizon], "lengths": lengths, "terms": terms,
           "seconds": round(time.monotonic() - t0, 1)}
    return {"row": out, "rewards": rewards, "arr": arr, "lengths": np.asarray(lengths)}


def v2_slices(lengths, horizon: int):
    """v2_probe_ana.py window per episode: 200 .. L-1 (L >= horizon) or 200 .. L-151; None when hi <= 260."""
    out = []
    for L in lengths:
        hi = int(L) if L >= horizon else int(L) - FALL_TRIM
        out.append(None if hi <= W0 + MIN_EXTRA else slice(W0, hi))
    return out


def stagger_block(res: dict, horizon: int) -> dict:
    stg, dq, L = res["arr"]["stagger"], res["arr"]["dq"], res["lengths"]
    falls_in_window = int(sum(W0 < int(n) < horizon for n in L))  # falls whose topple lies inside 200 .. L-1
    win, absmax = [], []
    for e, n in enumerate(L):
        if n > W0:
            x = stg[e, W0:int(n)].astype(np.float64)
            win.append(x)
            absmax.append(round(1000 * float(np.nanmax(np.abs(x))), 1))
        else:
            absmax.append(None)
    sl = v2_slices(L, horizon)
    v2s = [stg[e, s].astype(np.float64) for e, s in enumerate(sl) if s is not None]
    v2q = [dq[e, s].astype(np.float64) for e, s in enumerate(sl) if s is not None]
    pooled = np.concatenate(win) if win else np.zeros(0)
    return {"window": "steps 200 .. L-1, pooled over the episodes with L > 200",
            "sd_mm": round(1000 * float(np.std(pooled)), 2) if pooled.size else None,
            "n_steps": int(pooled.size),
            "falls_in_window": falls_in_window,
            "sd_mm_note": ("sd_mm includes the topple steps of the falls_in_window falling episodes and partly measures "
                           "the falls; compare sd_mm_v2_window with k3diag" if falls_in_window else
                           "no fall inside the window: sd_mm is fall-free"),
            "absmax_mm_per_episode": absmax,
            "absmax_mm_max": max((a for a in absmax if a is not None), default=None),
            "v2_window": "v2_probe_ana: 200 .. L-1 survivors, 200 .. L-151 falls, episodes with hi <= 260 skipped",
            "sd_mm_v2_window": round(1000 * float(np.std(np.concatenate(v2s))), 2) if v2s else None,
            "dq_sd_mrad_v2_window": round(1000 * float(np.std(np.concatenate(v2q))), 2) if v2q else None,
            "episodes_in_v2_window": len(v2s)}


def response_block(res: dict, horizon: int) -> dict:
    """v2_probe_ana.py reduction of the full condition's probe arrays."""
    sl = v2_slices(res["lengths"], horizon)
    sel = {k: [res["arr"][k][e, s].astype(np.float64) for e, s in enumerate(sl) if s is not None]
           for k in ("Jq", "Jv", "Jq_common")}
    if not sel["Jq"]:
        return {"computable": False, "reason": "no episode has a window of more than 60 steps"}
    Jq = np.concatenate(sel["Jq"])
    epm = [float(np.median(v)) for v in sel["Jq"]]
    return {"computable": True, "units": "d(mu_R - mu_L)/d(q_R - q_L), normalised command per rad",
            "condition": "full (the judge's noise loop)", "h_pos_rad": H_POS, "h_vel_rad_s": H_VEL,
            "window": "steps 200 .. L-1 survivors, 200 .. L-151 falls, episodes with hi <= 260 skipped; pooled",
            "Jq_median": float(np.median(Jq)), "Jq_median_3dp": round(float(np.median(Jq)), 3),
            "Jq_mean": float(Jq.mean()), "Jq_p10_p90": [float(np.percentile(Jq, 10)), float(np.percentile(Jq, 90))],
            "Jq_episode_medians_range": [min(epm), max(epm)],
            "Jv_median_per_rad_s": float(np.median(np.concatenate(sel["Jv"]))),
            "Jq_common_median": float(np.median(np.concatenate(sel["Jq_common"]))),
            "n_steps": int(Jq.size), "n_episodes": len(sel["Jq"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(MODES), default="LTSBr")
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", required=True, help="a multiple of 524288, or final")
    ap.add_argument("--check-only", action="store_true", help="every identity check and the checkpoint load; writes nothing")
    ap.add_argument("--smoke", type=int, default=None, help="development only: N episodes per condition (selftest/out)")
    a = ap.parse_args()
    t_start = time.monotonic()
    started = utcnow()
    mode = MODES[a.arm]
    test_root = os.environ.get("K3R_TEST_ROOT") or None
    if test_root:  # self-tests only: confined to R/k3readouts (never R/k3readouts itself, out/ or logs/)
        here, tr = os.path.realpath(HERE), os.path.realpath(test_root)
        forbidden = [os.path.realpath(OUT_ROOT), os.path.join(here, "logs")]
        if not tr.startswith(here + "/") or any(tr == f or tr.startswith(f + "/") for f in forbidden):
            print(f"[k3readout5r] K3R_TEST_ROOT {test_root} (resolves to {tr}) is not a test directory inside {here} "
                  f"(out/ and logs/ excluded); nothing read or written", file=sys.stderr)
            return EXIT_ARGS
        test_root = tr
    out_root = (os.path.join(test_root, "out") if test_root else SMOKE_ROOT if a.smoke else OUT_ROOT)
    judged_root = os.path.join(test_root, "judged") if test_root else os.path.join(mode["root"], "judged")
    steps = a.steps
    if not (steps == "final" or (steps.isdigit() and int(steps) > 0 and int(steps) % 524288 == 0)):
        print(f"[k3readout5r] bad --steps {steps}", file=sys.stderr)
        return EXIT_ARGS
    if a.run not in mode["runs"]:
        print(f"[k3readout5r] --run {a.run} is not one of arm {a.arm}'s runs {sorted(mode['runs'])}", file=sys.stderr)
        return EXIT_ARGS
    if a.smoke is not None and a.smoke < 1:
        print(f"[k3readout5r] --smoke {a.smoke} must be at least 1", file=sys.stderr)
        return EXIT_ARGS
    seed = mode["runs"][a.run]
    in_addendum = steps in mode["addendum_slots"][a.run]
    addendum_slot = mode["addendum_role"] if in_addendum else "no"
    if not in_addendum:
        print(f"[k3readout5r] NOTE: {a.run} {steps} is not one of the addendum's {mode['addendum_role']} slots "
              f"{list(mode['addendum_slots'][a.run])}; the output is a measurement, not one of the addendum's readouts "
              f"(addendum_slot=no)", flush=True)
    tool = os.path.realpath(__file__)
    runner = os.environ.get("K3R_RUNNER")
    ident = {"tool": tool, "tool_sha256": sha256_file(tool),
             "runner": runner, "runner_sha256": sha256_file(runner) if runner and os.path.isfile(runner) else None,
             "addendum": ADDENDUM, "addendum_sha256": sha256_file(ADDENDUM),
             "arm_mode": a.arm, "judge_arm": JUDGE_ARM, "run": a.run, "seed": seed, "steps": steps,
             "slot_in_addendum": in_addendum, "addendum_slot": addendum_slot,
             "addendum_slots_of_run": list(mode["addendum_slots"][a.run]), "smoke": a.smoke, "test_root": test_root,
             "python": sys.version.split()[0], "numpy": np.__version__, "started_utc": started}
    rdir = os.path.join(out_root, a.run)

    def refuse(reasons: list[str]) -> int:
        rec = {"schema": "stance5r-k3readout-refusal/v1", "report_only": True, "refused": True, "reasons": reasons,
               "identity": ident, "written_utc": utcnow()}
        if a.check_only:
            print("[k3readout5r] REFUSED (check-only, nothing written): " + " | ".join(reasons), flush=True)
        else:
            path = write_new(os.path.join(rdir, f"refused_{steps}.json"), jsonable(rec))
            print("[k3readout5r] REFUSED: " + " | ".join(reasons) + f" (recorded {path})", flush=True)
        print(f"K3R {a.run} {steps} arm={a.arm} addendum_slot={addendum_slot} REFUSED: " + " | ".join(reasons), flush=True)
        return EXIT_REFUSED

    if ident["addendum_sha256"] != ADDENDUM_SHA256:
        return refuse([f"addendum sha256 {ident['addendum_sha256']} != the pinned {ADDENDUM_SHA256}"])
    try:
        J, jsha = load_judge(mode)
    except Refusal as exc:
        return refuse(exc.args[0])
    ident.update({"judge_module": mode["judge"], "judge_module_sha256": jsha})
    judge_file = os.path.join(judged_root, a.run, f"judge_{steps}.json")
    ident["judge_file"] = judge_file
    if not os.path.isfile(judge_file):
        return refuse([f"judge file {judge_file} missing"])
    ident["judge_file_sha256"] = sha256_file(judge_file)
    try:
        jd = J.read_json(judge_file)
    except Exception as exc:  # noqa: BLE001
        return refuse([f"judge file {judge_file} unreadable: {exc}"])
    ident["judge_file_judge_sha256"] = jd.get("judge_sha256")
    stage_dir = os.path.join(mode["root"], "runs", a.run, "01_stance")
    _, zp, vp = J.checkpoint_paths(os.path.join(stage_dir, "models"), J.parse_steps(steps))
    missing = [p for p in (zp, vp) if not os.path.isfile(p)]
    if missing:
        return refuse([f"checkpoint missing: {missing}"])
    # the clone: environments from the clone only, clean at the expected HEAD (the judge's own isolate and git_state)
    spec = J.ARMS[JUDGE_ARM]
    if spec["clone"] != CLONE or spec["head"] != CLONE_HEAD:
        return refuse([f"judge ARMS[{JUDGE_ARM}] clone/head {spec['clone']} {spec['head']} != {CLONE} {CLONE_HEAD}"])
    try:
        modules = J.isolate(CLONE)
    except RuntimeError as exc:
        return refuse([f"isolation: {exc}"])
    git = J.git_state(CLONE)
    ident.update({"clone": CLONE, "git": git, "modules": modules})
    if git["head"] != CLONE_HEAD or git["status"]:
        return refuse([f"clone {CLONE} is not clean at {CLONE_HEAD} (HEAD {git['head']}, dirty={bool(git['status'])})"])
    try:
        sigma, ctrl, sigma_file, sigma_sha = judge_sigma(J, a.arm, seed, ident)
    except Refusal as exc:
        return refuse(exc.args[0])
    bad = check_judge_file(jd, mode, a.arm, a.run, seed, steps, zp, vp, sigma, ctrl, sigma_file, sigma_sha, ident)
    if J.NOISE_SEED != 3042 or J.NOISE_EPISODES != 20:
        bad.append(f"judge NOISE_SEED/NOISE_EPISODES {J.NOISE_SEED}/{J.NOISE_EPISODES} != 3042/20")
    if bad:
        return refuse(bad)

    import torch

    torch.set_num_threads(1)
    ctx = J.Ctx(JUDGE_ARM)
    if ctx.horizon != 1000 or abs(ctx.control_dt - 0.01) > 1e-12:
        return refuse([f"horizon/control_dt {ctx.horizon}/{ctx.control_dt} != 1000/0.01"])
    model, normalizer = J.load_checkpoint(ctx, zp, vp)  # the judge's own loader (VecNormalize applied, deterministic mean)
    pol = model.policy
    own = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
    stats = (np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var), normalizer.epsilon, normalizer.clip_obs)
    ident.update({"torch": torch.__version__, "torch_threads": torch.get_num_threads(), "horizon": ctx.horizon,
                  "control_dt": ctx.control_dt, "noise_seeds": [J.NOISE_SEED, J.NOISE_SEED + J.NOISE_EPISODES - 1],
                  "rng": "numpy default_rng(episode seed); z = standard_normal(15) every step",
                  "recorder": "SubstepContactRecorder(env, Morphology.from_env(env, mid)) attached (the judge's)",
                  "checkpoint_loader": f"{mode['module']}.load_checkpoint", "conditions": list(CONDS),
                  "definitions": DEFINITIONS, "sources": SOURCES})
    own_block = {"source": "exp(model.policy.log_std) of the checkpoint J.load_checkpoint returns",
                 "per_actuator": dict(zip(ACT, own.tolist())), "mean": float(own.mean()),
                 "hip_pitch_mean": float(own[[RHP, LHP]].mean()), "r_hip_pitch": float(own[RHP]), "l_hip_pitch": float(own[LHP]),
                 "judge_over_own_per_actuator": dict(zip(ACT, (sigma / own).tolist())),
                 "judge_over_own_hip_pitch": [float(sigma[RHP] / own[RHP]), float(sigma[LHP] / own[LHP])]}
    if a.check_only:
        print(json.dumps(jsonable({"check_only": "pass", "identity": {k: ident[k] for k in (
            "run", "steps", "arm_mode", "judge_module_sha256", "judge_file", "model_sha256", "vecnorm_sha256",
            "judge_file_sigma_vector_equal", "git")}, "sigma_file": ident["sigma"]["file"], "sigma_sha256": ident["sigma"]["file_sha256"],
            "own_hip_pitch_mean": own_block["hip_pitch_mean"]})), flush=True)
        print(f"K3R {a.run} {steps} arm={a.arm} addendum_slot={addendum_slot} CHECK-ONLY pass (nothing written)", flush=True)
        return EXIT_OK

    episodes = int(a.smoke) if a.smoke else J.NOISE_EPISODES
    jn = jd["noise_matched_control_joint"]
    conds, kicks, stagger, response, timing = {}, {}, {}, None, {}
    matches = reward_ok = None
    for cond in CONDS:
        res = roll(J, ctx, pol, stats, sigma, cond, episodes)
        row = res["row"]
        if cond == "full":
            seeds = [J.NOISE_SEED + k for k in range(episodes)]
            jl = [(r["seed"], r["length"], r["termination_reason"]) for r in jn["episodes_detail"]][:episodes]
            matches = jl == list(zip(seeds, row["lengths"], row["terms"]))
            rdiff = max(abs(r["reward"] - x) for r, x in zip(jn["episodes_detail"], res["rewards"]))
            reward_ok = bool(rdiff < REWARD_TOL)
            row.update({"matches_judge": matches, "judge_falls": jn["falls"], "judge_fall_seeds": jn["fall_seeds"],
                        "reward_max_abs_diff_vs_judge": rdiff, "reward_matches_judge": reward_ok,
                        "reward_tolerance": REWARD_TOL})
            response = response_block(res, ctx.horizon)
        if cond in PULSE:
            row["amplitude_r_l"] = list(PULSE[cond])
            row["pulse_control_steps"] = [PULSE_STEPS[0], PULSE_STEPS[1] - 1]
            row["pulse_seconds"] = (PULSE_STEPS[1] - PULSE_STEPS[0]) * ctx.control_dt
            # not part of k3pulse's count (falls = length < 1000): episodes that ended before the kick was applied
            pre = [s for s, L in zip([J.NOISE_SEED + k for k in range(episodes)], row["lengths"]) if L <= PULSE_STEPS[0]]
            row["falls_before_kick"] = len(pre)
            row["fall_seeds_before_kick"] = pre
            row["falls_before_kick_definition"] = f"length <= {PULSE_STEPS[0]} (ended before control step {PULSE_STEPS[0]})"
            kicks[cond] = row
        else:
            conds[cond] = row
            stagger[cond] = stagger_block(res, ctx.horizon)
        timing[cond] = row["seconds"]
        print(json.dumps({"cond": cond, "falls": row["falls"], "fall_seeds": row["fall_seeds"], "seconds": row["seconds"],
                          **({"matches_judge": matches, "reward_matches_judge": reward_ok} if cond == "full" else {})}),
              flush=True)
        if cond == "full" and not (matches and reward_ok):
            print("[k3readout5r] WARNING: full does not reproduce the judge file (matches_judge="
                  f"{matches}, reward_matches_judge={reward_ok}); the output will be marked invalid", flush=True)
    J.modules_under(CLONE)
    git_after = J.git_state(CLONE)
    clean_after = git_after["head"] == CLONE_HEAD and not git_after["status"]
    valid = bool(matches) and bool(reward_ok) and clean_after and not a.smoke and not test_root
    timing["total_seconds"] = round(time.monotonic() - t_start, 1)
    ident["finished_utc"] = utcnow()
    ident["git_after"] = git_after
    full, sf = conds["full"], stagger["full"]
    summary = (f"K3R {a.run} {steps} arm={a.arm} addendum_slot={addendum_slot} valid={valid} "
               f"full={full['falls']}/{full['n']} (judge {jn['falls']}, matches_judge={matches}, "
               f"reward_matches_judge={reward_ok}) hip_pitch={conds['hip_pitch']['falls']} nohp={conds['nohp']['falls']} "
               f"hpD={conds['hpD']['falls']} hpC={conds['hpC']['falls']} kick0.3={kicks['pulse_anti']['falls']} "
               f"kick0.2={kicks['pulse_anti20']['falls']} "
               f"kick_falls_before_kick={kicks['pulse_anti']['falls_before_kick']}/{kicks['pulse_anti20']['falls_before_kick']} "
               f"response={response.get('Jq_median_3dp') if response else None}/rad "
               f"own_hp={own_block['hip_pitch_mean']:.4f} stagger_sd={sf['sd_mm']}mm "
               f"stagger_sd_v2_window={sf['sd_mm_v2_window']}mm stagger_falls_in_window={sf['falls_in_window']} "
               f"({timing['total_seconds']:.0f}s)")
    record = {"schema": "stance5r-k3readout/v1", "report_only": True,
              "note": "post-hoc report-only readouts (R/readouts_addendum_k3.txt); no verdict, decision line or tie-break",
              "valid": valid, "matches_judge": matches, "reward_matches_judge": reward_ok, "clone_clean_after": clean_after,
              "addendum_slot": addendum_slot,
              "identity": ident, "falls_of": episodes, "conditions": conds, "kick": kicks, "response": response,
              "own_sigma": own_block, "stagger": stagger, "timing_seconds": timing, "summary": summary}
    path = write_new(os.path.join(rdir, f"k3r_{steps}.json"), jsonable(record))
    print(summary + f" -> {path}", flush=True)
    return EXIT_OK if valid else EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
