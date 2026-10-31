"""vg_tool: the critic's error V - G by episode-step bucket and the tail's exploration noise, on the phase screen's
COUNTED continuations' final pairs (REPORT ONLY: outside the screen's reading; never a line, never a secondary).
SPEC.txt (this folder) is the pre-registration; every choice below is stated there.

    PY -I -B vg_sigma.py --root <PS> --out DIR [--plan]
        --root is required (no default: even the guard opens <PS>/score/score.json, outcome-bearing while the screen
        runs, so a command without --pair or --root refuses before opening anything).
        Refuses (exit 2, nothing under <PS>/runs or <PS>/reads opened, listed or statted) unless
        <PS>/score/score.json exists with status "final", names <PS> as its root, and its classes equal
        <PS>/SCREEN.json's queue in order with every entry decided (COUNTED, STOPPED or LOST).  Then it measures the
        shared start pair S (the F2 copies at FMAN sha256; both critic forms, which must agree) and each COUNTED
        continuation's final pair (stage1_final.zip / stage1_final_vecnorm.pkl at E), and reads the tail sigma at S,
        each saved pair and E.  --plan: the guard and the list of what would be read, nothing else.
    PY -I -B vg_sigma.py --pair ZIP PKL --form {plain,phase,blind} [--phase-start] --out DIR
                         [--episodes N] [--seed B] [--label L] [--expect-steps T]
        One explicit pair (validation; no screen root).  --phase-start: the pair is critic_phase's start copy, read
        with the arm's load transform (read_bs.phase_prepare(start=True)); the plain form is read too and must agree.

What it writes: DIR/vg_sigma.json, DIR/vg_sigma.txt (deterministic: no clock values), DIR/run_info.json (UTC
times, CPU seconds, per-episode cost); its TMPDIR / MPLCONFIGDIR is DIR/_tmp (removed at the end).  It refuses a DIR
inside the screen's root or <PS>, or one that already holds vg_sigma.json.
Exit 0 measured; 3 measured with a PROBLEM (listed); 2 refused or input error.

How (SPEC.txt EPISODES, DEFINITIONS): the phase screen's own tools are imported read only from <PS>/tools
(read_bs.py, which loads bs_common.py and bs_hooks.py): read_bs.Net loads the pair as the reader does,
read_bs.phase_prepare applies critic_phase's start transform / checks the frozen slots and installs the actor mask,
read_bs.with_phase feeds each state's phase phi(k) exactly as read_bs.critic_slope does, bs_hooks.critic_blind_class
is BS2's blinded critic (--pair --form blind only).  The episodes are collected with read_bs.onpolicy_states' build
(the stage config, create_vec_env with 4 envs, the pair's VecNormalize frozen with norm_reward True, load_sb3_model
with the stage's [ppo] kwargs, enable_mirror_kl) and SB3 2.9.0 collect_rollouts' action path (policy forward,
sampled, clipped), stepping the inner DummyVecEnv and normalising with the same VecNormalize object so the raw
time-limit terminal observation is kept.  No optimizer, no update, no learn(); every pair is opened read only.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
VERSION = "vg_sigma 1.1 (2026-10-31; report-only V - G by step bucket and tail sigma for the phase screen)"
SCHEMA = "vg-sigma/1"
PS_ROOT = Path("/home/user/work/phase_screen")
PS_TOOLS = PS_ROOT / "tools"
IMPORTED_TOOLS = ("bs_common.py", "bs_hooks.py", "read_bs.py")
SPEC = HERE / "SPEC.txt"


def _import_ps_tools():
    """read_bs (and through it bs_common, bs_hooks) from <PS>/tools, read only (no bytecode is written)."""
    if str(PS_TOOLS) not in sys.path:
        sys.path.insert(0, str(PS_TOOLS))
    import read_bs

    if Path(read_bs.__file__).resolve() != PS_TOOLS / "read_bs.py":
        raise ImportError(f"read_bs imported from {read_bs.__file__}, not {PS_TOOLS}")
    return read_bs


R = _import_ps_tools()
C, H = R.C, R.H

# ------------------------------------------------------------------------------------------------ the pre-registration
BUCKETS = ((0, 50), (50, 150), (150, 200), (200, 250), (250, 1000))     # [lo, hi); the last one [250, 1000]
BUCKET_NAMES = ("[0,50)", "[50,150)", "[150,200)", "[200,250)", "[250,1000]")
HORIZON = 1000                                  # stance max_episode_steps (base_env._is_truncated)
N_ENVS = 4
N_EPISODES = 32                                 # per measured pair: the first 8 complete episodes of each of 4 envs
PROD_SEED = 6271                                # --root: envs 6271-6274, torch / numpy 6271 (val/seeds_check.out)
VAL_SEED = 6281                                 # --pair default: envs 6281-6284
ACT_TAIL3, ACT_TAIL2 = 14, 13
ACT_NAMES = {ACT_TAIL3: "tail_3_pitch_act", ACT_TAIL2: "tail_2_pitch_act"}
N_ACT = 15
SIGMA_FLOOR = 0.01                              # stage [ppo] symmetry_sigma_floor (mirror-KL loss only)
GAMMA = 0.98                                    # stage [ppo] gamma
E1_BUCKETS = (2, 3)                             # [150,200), [200,250)
V_LOOP_TOL = 1e-4                               # loop forward values vs the batched plain critic (batching noise)
FORMS = ("plain", "phase", "blind")
ARM_FORM = {"control": "plain", "critic_phase": "phase"}
CRITIC_FORM_MARKERS = {"control": "plain", "critic_phase": "phase+raw+settled"}
DECIDED = ("COUNTED", "STOPPED", "LOST")
EXIT_OK, EXIT_REFUSED, EXIT_PROBLEM = 0, 2, 3


class Refused(Exception):
    """The guard or an input refuses (exit 2)."""


# ================================================================================================ pure functions
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket_of(k):
    """Index into BUCKETS of an observation's step counter k (0 <= k <= 1000; 1000 belongs to [250, 1000])."""
    if isinstance(k, bool) or int(k) != k:
        raise ValueError(f"k {k!r} is not an integer")
    k = int(k)
    if k < 0 or k > HORIZON:
        raise ValueError(f"k {k} outside [0, {HORIZON}]")
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= k < hi:
            return i
    return len(BUCKETS) - 1                     # k == 1000


def buckets_of(ks):
    """bucket_of for an array of counters (vectorised; the unit tests check it against bucket_of)."""
    k = np.asarray(ks)
    if k.size and (not np.issubdtype(k.dtype, np.integer) or k.min() < 0 or k.max() > HORIZON):
        raise ValueError(f"counters outside the integers 0 ... {HORIZON}")
    return np.searchsorted(np.array([hi for _lo, hi in BUCKETS[:-1]]), k, side="right")


def returns_to_go(r_norm, gamma, bootstrap=None):
    """G_t = sum_{j>=t} gamma^(j-t) r_j (+ gamma^(T-t) bootstrap at a time-limit truncation; None after a true
    termination), float64.  The same as folding gamma * V(terminal) into the last reward (SB3's convention)."""
    r = np.asarray(r_norm, dtype=np.float64).reshape(-1)
    G = np.empty_like(r)
    acc = 0.0 if bootstrap is None else float(bootstrap)
    g = float(gamma)
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + g * acc
        G[t] = acc
    return G


def normalised_rewards(vecnorm, raw):
    """The rewards as the training buffer holds them: VecNormalize.normalize_reward (frozen ret_rms) of the env's
    float32 rewards, float32."""
    out = vecnorm.normalize_reward(np.asarray(raw, dtype=np.float32))
    return np.asarray(out, dtype=np.float32)


def _sd(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else None


def bucket_stats(episodes):
    """episodes: [{"k": int array, "V": float array, "G": float array}].  Per bucket: episodes contributing, states,
    the episode-weighted means of V - G and |V - G| with their SE across episodes, the state-pooled means, mean V,
    mean G."""
    out = []
    bis = [buckets_of(np.asarray(ep["k"], dtype=np.int64)) for ep in episodes]
    for b, name in enumerate(BUCKET_NAMES):
        ep_err, ep_abs, all_err, all_V, all_G = [], [], [], [], []
        for ep, bi in zip(episodes, bis, strict=True):
            sel = bi == b
            if not sel.any():
                continue
            e = np.asarray(ep["V"], dtype=np.float64)[sel] - np.asarray(ep["G"], dtype=np.float64)[sel]
            ep_err.append(float(e.mean()))
            ep_abs.append(float(np.abs(e).mean()))
            all_err.append(e)
            all_V.append(np.asarray(ep["V"], dtype=np.float64)[sel])
            all_G.append(np.asarray(ep["G"], dtype=np.float64)[sel])
        n = len(ep_err)
        row = {"bucket": name, "lo": BUCKETS[b][0], "hi": BUCKETS[b][1], "n_episodes": n,
               "n_states": int(sum(len(x) for x in all_err))}
        if n:
            pe = np.concatenate(all_err)
            sd_e, sd_a = _sd(ep_err), _sd(ep_abs)
            row.update(mean_err=float(np.mean(ep_err)), se_err=sd_e / math.sqrt(n) if sd_e is not None else None,
                       mean_abs_err=float(np.mean(ep_abs)),
                       se_abs_err=sd_a / math.sqrt(n) if sd_a is not None else None,
                       pooled_err=float(pe.mean()), pooled_abs_err=float(np.abs(pe).mean()),
                       mean_V=float(np.concatenate(all_V).mean()), mean_G=float(np.concatenate(all_G).mean()))
        else:
            row.update(mean_err=None, se_err=None, mean_abs_err=None, se_abs_err=None, pooled_err=None,
                       pooled_abs_err=None, mean_V=None, mean_G=None)
        out.append(row)
    return out


def median(xs):
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def arm_summary(runs):
    """runs: [{"name", "arm", "seed", "buckets": bucket_stats(...)}] (COUNTED, measured).  Per bucket: each arm's n,
    median and mean of the per-run mean V - G and mean |V - G|; control minus critic_phase (medians); per seed."""
    arms = ("control", "critic_phase")
    out = []
    for b, name in enumerate(BUCKET_NAMES):
        row = {"bucket": name}
        for arm in arms:
            err = [r["buckets"][b]["mean_err"] for r in runs if r["arm"] == arm and r["buckets"][b]["mean_err"]
                   is not None]
            ab = [r["buckets"][b]["mean_abs_err"] for r in runs if r["arm"] == arm
                  and r["buckets"][b]["mean_abs_err"] is not None]
            row[arm] = {"n": len(err), "median_err": median(err), "mean_err": float(np.mean(err)) if err else None,
                        "median_abs_err": median(ab), "mean_abs_err": float(np.mean(ab)) if ab else None}
        c, p = row["control"], row["critic_phase"]
        row["control_minus_phase"] = {
            "median_err": (c["median_err"] - p["median_err"]) if None not in (c["median_err"], p["median_err"])
            else None,
            "median_abs_err": (c["median_abs_err"] - p["median_abs_err"])
            if None not in (c["median_abs_err"], p["median_abs_err"]) else None}
        paired = []
        for seed in sorted({r["seed"] for r in runs}):
            rc = [r for r in runs if r["arm"] == "control" and r["seed"] == seed]
            rp = [r for r in runs if r["arm"] == "critic_phase" and r["seed"] == seed]
            if rc and rp and None not in (rc[0]["buckets"][b]["mean_err"], rp[0]["buckets"][b]["mean_err"]):
                paired.append({"seed": seed, "control_minus_phase_err":
                               rc[0]["buckets"][b]["mean_err"] - rp[0]["buckets"][b]["mean_err"],
                               "control_minus_phase_abs_err":
                               rc[0]["buckets"][b]["mean_abs_err"] - rp[0]["buckets"][b]["mean_abs_err"]})
        row["paired"] = paired
        out.append(row)
    return out


def expectations(summary):
    """SPEC E1 / E2 read on the arm summary: 'consistent', 'not consistent' or 'not assessable' (a description)."""
    out = {}
    for b in E1_BUCKETS:
        c, p = summary[b]["control"]["median_err"], summary[b]["critic_phase"]["median_err"]
        key = f"E1 {BUCKET_NAMES[b]}"
        if c is None or p is None:
            out[key] = "not assessable"
        else:
            out[key] = "consistent" if abs(p) < abs(c) else "not consistent"
    c1, c2, c3 = (summary[b]["control"]["median_err"] for b in (1, 2, 3))
    if None in (c1, c2, c3):
        out["E2 control sign pattern"] = "not assessable"
    else:
        out["E2 control sign pattern"] = "consistent" if (c1 > 0 and c2 > 0 and c3 < 0) else "not consistent"
    return out


def sigma_record(log_std, floor=SIGMA_FLOOR):
    """The tail actuators' sampled noise exp(log_std) (float32 log_std, exp in float64) and the mirror-KL loss's
    floored sigma max(sigma, floor)."""
    ls = np.asarray(log_std, dtype=np.float32).reshape(-1)
    if ls.shape != (N_ACT,):
        raise Refused(f"log_std of shape {ls.shape}, not ({N_ACT},)")
    out = {}
    for name, i in (("tail_3", ACT_TAIL3), ("tail_2", ACT_TAIL2)):
        s = float(np.exp(np.float64(ls[i])))
        out[name] = {"action": i, "actuator": ACT_NAMES[i], "log_std": float(ls[i]), "sigma": s,
                     "sigma_floored": max(s, float(floor)), "floor_binds": bool(s < float(floor))}
    out["floor"] = float(floor)
    return out


def read_log_std(zip_path):
    """log_std (float32 numpy) from the checkpoint zip's policy.pth (torch.load weights_only; nothing else read)."""
    import torch

    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read("policy.pth")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if "log_std" not in state:
        raise Refused(f"{zip_path}: policy.pth has no log_std")
    return state["log_std"].detach().cpu().numpy().astype(np.float32)


# ================================================================================================ the guard
def _read_json(path):
    raw = Path(path).read_bytes()
    try:
        d = json.loads(raw)
    except ValueError as exc:
        raise Refused(f"{path} does not parse ({exc.__class__.__name__})") from exc
    return d, hashlib.sha256(raw).hexdigest()


def guard(root):
    """The refusal guard: opens only <root>/score/score.json and <root>/SCREEN.json.  Returns the plan."""
    root = Path(root).resolve()
    sp = root / "score" / "score.json"
    if not sp.is_file():
        raise Refused(f"{sp} missing: the screen is not scored (no continuation is read)")
    score, score_sha = _read_json(sp)
    if not isinstance(score, dict):
        raise Refused("score.json is not a JSON object")
    if score.get("status") != "final":
        raise Refused(f"score.json status {score.get('status')!r} is not 'final' (no continuation is read)")
    if not score.get("root") or Path(str(score["root"])).resolve() != root:
        raise Refused(f"score.json root {score.get('root')!r} is not {root}")
    screen_path = root / "SCREEN.json"
    screen, problems = C.load_screen(str(screen_path))
    if problems:
        raise Refused("SCREEN.json: " + "; ".join(problems))
    queue = C.queue_of(screen)
    classes = score.get("classes")
    if not isinstance(classes, list) or len(classes) != len(queue):
        raise Refused(f"score.json classes do not list the {len(queue)} queue entries")
    counted, entries = [], []
    for i, (arm, seed) in enumerate(queue):
        c = classes[i]
        name = C.run_name(arm, seed)
        if not isinstance(c, dict) or c.get("name") != name or c.get("arm") != arm or c.get("seed") != seed:
            raise Refused(f"score.json classes[{i}] {c!r} is not the queue entry {name}")
        cls = c.get("cls")
        if cls not in DECIDED:
            raise Refused(f"{name}: class {cls!r} is not decided (COUNTED, STOPPED or LOST)")
        if cls == "COUNTED":
            if c.get("T") != C.E:
                raise Refused(f"{name}: COUNTED with T {c.get('T')!r} != E {C.E}")
            if arm not in ARM_FORM:
                raise Refused(f"{name}: arm {arm!r} has no critic form here")
            counted.append({"name": name, "arm": arm, "seed": seed})
        entries.append({"name": name, "arm": arm, "seed": seed, "cls": cls})
    if not counted:
        raise Refused("no COUNTED continuation to measure")
    return {"root": str(root), "score_json": str(sp), "score_sha256": score_sha, "score_status": score["status"],
            "score_line_key": score.get("line_key"), "screen_json": str(screen_path),
            "screen_sha256": sha256_file(screen_path), "entries": entries, "counted": counted}


def run_paths(root, name):
    models = Path(root) / "runs" / name / C.STAGE_DIR / "models"
    final = (models / C.FINAL_ZIP, models / C.FINAL_PKL)
    saved = [(s, models / C.periodic_names(s)[0]) for s in C.SAVED_STEPS]
    start_copy = models / C.periodic_names(C.S)[0]
    return final, saved, start_copy


# ================================================================================================ simulation
def setup_sim(tmp):
    """Before torch / the code copy: threads 1, TMPDIR and MPLCONFIGDIR under tmp, the code copy first on sys.path
    (read_bs.setup_simulation's steps with this tool's own TMPDIR)."""
    tmp = Path(tmp)
    (tmp / "mpl").mkdir(parents=True, exist_ok=True)
    os.environ.update({"TMPDIR": str(tmp), "MPLCONFIGDIR": str(tmp / "mpl"), "GITHUB_SHA": C.COMMIT})
    import tempfile

    tempfile.tempdir = None
    code = os.path.realpath(C.CODE)
    if sys.path[:1] != [code]:
        sys.path.insert(0, code)
    import environments

    origin = os.path.realpath(environments.__file__)
    if not origin.startswith(code + os.sep):
        raise Refused(f"environments imported from {origin}, not the code copy {code}")
    import torch

    torch.set_num_threads(1)
    return origin


def env_factory():
    """read_bs.loop_markers' factory: the stance env with the code copy's stance env_kwargs."""
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])

    def factory():
        return env_class(mid)(**ctor)
    return factory


def build_collection(zip_path, pkl_path, seed):
    """read_bs.onpolicy_states' build (4 envs seeded seed + rank; the pair's statistics frozen; norm_reward True; the
    model with the stage's [ppo] kwargs and enable_mirror_kl; vn / torch / numpy seeded seed)."""
    import copy

    import torch
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
    venv = create_vec_env(species, {1: {"env_kwargs": dict(cfg["env_kwargs"])}}, 1, N_ENVS, seed=seed,
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
    if not isinstance(model, MirrorKLPPO):
        raise Refused("the loaded model is not a MirrorKLPPO")
    vn.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return {"cfg": cfg, "ppo": ppo, "sym": sym, "vn": vn, "model": model,
            "mirror_action_perm": [int(x) for x in MirrorMap.load(mid, current_plant_identity(mid)).action.perm]}


def _new_cur():
    return {"k": [], "X": [], "a": [], "r_raw": [], "r_norm": [], "V_loop": []}


def collect(vn, model, n_per_env, max_vec_steps=None):
    """Complete episodes, per env, of the pair's stochastic policy: SB3 2.9.0 collect_rollouts' action path
    (policy forward on the normalised observation, sampled, clipped to the action space) with the inner
    DummyVecEnv stepped and VecNormalize's own normalize_obs / normalize_reward applied (its step_wait with
    training False), so that the raw terminal observation is kept.  Returns (done per env, open per env, steps)."""
    import torch

    policy = model.policy
    policy.set_training_mode(False)
    low, high = model.action_space.low, model.action_space.high
    envs = [e.unwrapped for e in vn.venv.envs]
    n = vn.num_envs
    raw = vn.venv.reset()
    cur = [_new_cur() for _ in range(n)]
    done = [[] for _ in range(n)]
    steps = 0
    while min(len(d) for d in done) < n_per_env:
        if max_vec_steps is not None and steps >= max_vec_steps:
            break
        ks = [int(u._step_count) for u in envs]
        for i in range(n):
            if ks[i] != len(cur[i]["k"]):
                raise Refused(f"env {i}: _step_count {ks[i]} != the episode's step {len(cur[i]['k'])}")
        z = vn.normalize_obs(raw)
        with torch.no_grad():
            actions, values, _logp = policy(torch.as_tensor(z))
        actions = actions.cpu().numpy()
        clipped = np.clip(actions, low, high)
        new_raw, rew, dones, infos = vn.venv.step(clipped)
        rn = normalised_rewards(vn, rew)
        steps += 1
        for i in range(n):
            c = cur[i]
            c["k"].append(ks[i])
            c["X"].append(np.array(raw[i], dtype=np.float32, copy=True))
            c["a"].append(np.array(actions[i], dtype=np.float32, copy=True))
            c["r_raw"].append(np.float32(rew[i]))
            c["r_norm"].append(np.float32(rn[i]))
            c["V_loop"].append(float(values[i, 0]))
            if dones[i]:
                info = infos[i]
                trunc = bool(info.get("TimeLimit.truncated", False))
                tobs = info.get("terminal_observation")
                ep = {"env": i, "k": np.asarray(c["k"], dtype=np.int64), "X": np.stack(c["X"]),
                      "a": np.stack(c["a"]), "r_raw": np.asarray(c["r_raw"], dtype=np.float32),
                      "r_norm": np.asarray(c["r_norm"], dtype=np.float32),
                      "V_loop": np.asarray(c["V_loop"], dtype=np.float64), "trunc": trunc, "term": not trunc,
                      "term_X": None if tobs is None else np.array(tobs, dtype=np.float32, copy=True),
                      "k_term": int(info.get("step", -1)),
                      "monitor_r": float(info.get("episode", {}).get("r", float("nan"))),
                      "monitor_l": int(info.get("episode", {}).get("l", -1))}
                done[i].append(ep)
                cur[i] = _new_cur()
        raw = new_raw
    return done, cur, steps


def critic_values(policy, norm, X, K=None, chunk=8192):
    """V of each state through the pair's critic: read_bs.critic_slope's feeding exactly (float32 states, with the
    phase of each counter in K written by read_bs.with_phase when K is given, then norm.normalize_obs), float64."""
    import torch

    x = np.asarray(X if K is None else R.with_phase(X, K), dtype=np.float32)
    z = np.asarray(norm.normalize_obs(x), dtype=np.float32)
    out = np.empty(len(z), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(z), chunk):
            out[s:s + chunk] = policy.predict_values(torch.as_tensor(z[s:s + chunk]))[:, 0].numpy()
    return out


def _form_values(net, form, X, K, TX, TK, start=False):
    """{"V", "V_term", "V_raw", "V_raw_term", "record"} of one critic form on one Net (forms change the Net in place:
    call plain before phase on the same Net)."""
    from stable_baselines3.common.torch_layers import MlpExtractor

    rec = {}
    if form == "plain":
        if type(net.policy.mlp_extractor) is not MlpExtractor:
            raise Refused(f"plain form on a {type(net.policy.mlp_extractor).__name__}")
        V = critic_values(net.policy, net.norm, X)
        VT = critic_values(net.policy, net.norm, TX) if len(TX) else np.zeros(0)
        return {"V": V, "V_term": VT, "V_raw": V, "V_raw_term": VT, "record": rec}
    if form == "phase":
        try:
            rec = R.phase_prepare(net, start=start)
        except R.InputError as exc:
            raise Refused(f"read_bs.phase_prepare: {exc}") from exc
        V = critic_values(net.policy, net.norm, X, K)
        VT = critic_values(net.policy, net.norm, TX, TK) if len(TX) else np.zeros(0)
        Vr = critic_values(net.policy, net.norm, X)
        VrT = critic_values(net.policy, net.norm, TX) if len(TX) else np.zeros(0)
        rec["slots_frozen_after"] = bool(H.phase_slots_frozen(net.norm.obs_rms))
        rec["max_abs_eff_minus_raw"] = float(np.max(np.abs(V - Vr))) if len(V) else None
        rec["max_abs_eff_minus_raw_term"] = float(np.max(np.abs(VT - VrT))) if len(VT) else None
        return {"V": V, "V_term": VT, "V_raw": Vr, "V_raw_term": VrT, "record": rec}
    if form == "blind":
        if type(net.policy.mlp_extractor) is not MlpExtractor:
            raise Refused(f"blind form on a {type(net.policy.mlp_extractor).__name__}")
        Vr = critic_values(net.policy, net.norm, X)
        VrT = critic_values(net.policy, net.norm, TX) if len(TX) else np.zeros(0)
        net.policy.mlp_extractor.__class__ = H.critic_blind_class(MlpExtractor)
        try:
            V = critic_values(net.policy, net.norm, X)
            VT = critic_values(net.policy, net.norm, TX) if len(TX) else np.zeros(0)
        finally:
            net.policy.mlp_extractor.__class__ = MlpExtractor
        return {"V": V, "V_term": VT, "V_raw": Vr, "V_raw_term": VrT, "record": rec}
    raise Refused(f"unknown critic form {form!r}")


def episode_rows(eps, V, VT, gamma):
    """[{"k", "V", "G"}] per episode: V sliced from the concatenated states, G bootstrapped with this form's V of
    the terminal observation at a time-limit truncation (VT in the order of the truncated episodes), none after a
    termination."""
    n_trunc = sum(1 for ep in eps if ep["trunc"])
    n_states = sum(len(ep["k"]) for ep in eps)
    if n_trunc != len(VT) or n_states != len(V):
        raise Refused(f"episode_rows: {n_states} states / {n_trunc} truncations against {len(V)} values / "
                      f"{len(VT)} terminal values")
    rows, ti, off = [], 0, 0
    for ep in eps:
        T = len(ep["k"])
        boot = None
        if ep["trunc"]:
            boot = float(VT[ti])
            ti += 1
        rows.append({"k": ep["k"], "V": V[off:off + T], "G": returns_to_go(ep["r_norm"], gamma, boot)})
        off += T
    return rows


def _stats_for(eps, V, VT, gamma):
    """bucket_stats of the measured episodes; G with the training gamma only (SPEC: gamma = stage [ppo] gamma)."""
    if gamma != GAMMA:
        raise Refused(f"G with gamma {gamma!r}, not the training gamma {GAMMA}")
    return bucket_stats(episode_rows(eps, V, VT, gamma))


def phase_feed_problems(label, rec, n_term):
    """Self-check of the phase form (rec: _form_values' record): when the critic's phase columns are nonzero, feeding
    the phase must change V on some state and, when an episode was truncated, on some terminal observation; an exact
    equality means the phase was not fed there."""
    used = rec.get("value_phase_cols_max_abs_used")
    if used is None:
        return [f"{label}: phase form without the phase columns' record"]
    out = []
    if used > 0.0:
        if not rec.get("max_abs_eff_minus_raw"):
            out.append(f"{label}: the critic's phase columns are nonzero (max |w| {used:.3g}) but V with the phase "
                       f"equals V with the slots zero on every state (the phase was not fed)")
        if n_term and not rec.get("max_abs_eff_minus_raw_term"):
            out.append(f"{label}: the critic's phase columns are nonzero (max |w| {used:.3g}) but V(terminal) with "
                       f"phi(1000) equals V(terminal) with the slots zero (the terminal phase was not fed)")
    return out


def measure_pair(zip_path, pkl_path, *, forms, seed, n_episodes, label, expect_steps=None, phase_start=False,
                 max_vec_steps=None):
    """Collect n_episodes on-policy episodes of one pair and read V - G in each requested critic form; plus the
    pair's tail sigma.  Returns (record, problems, timing)."""
    if n_episodes % N_ENVS or n_episodes <= 0:
        raise Refused(f"episodes {n_episodes} is not a positive multiple of {N_ENVS}")
    problems = []
    c0, w0 = time.process_time(), time.time()
    b = build_collection(zip_path, pkl_path, seed)
    vn, model = b["vn"], b["model"]
    u0 = vn.venv.envs[0].unwrapped
    names = [u0.model.actuator(i).name for i in range(u0.model.nu)]
    if u0.model.nu != N_ACT or any(names[i] != ACT_NAMES[i] for i in ACT_NAMES):
        raise Refused(f"actuator order: nu {u0.model.nu}, 13/14 = {names[13:15]}")
    gamma = float(b["ppo"]["gamma"])
    if not (gamma == GAMMA == float(model.gamma) == float(vn.gamma)):
        raise Refused(f"gamma: stage {gamma}, model {model.gamma}, VecNormalize {vn.gamma} (expected {GAMMA})")
    if int(u0.max_episode_steps) != HORIZON:
        raise Refused(f"max_episode_steps {u0.max_episode_steps} != {HORIZON}")
    floor_cfg = float(b["sym"].sigma_floor)
    floor_zip = (getattr(model, "mesozoic_symmetry", None) or {}).get("sigma_floor")
    if floor_cfg != SIGMA_FLOOR or (floor_zip is not None and float(floor_zip) != SIGMA_FLOOR):
        raise Refused(f"sigma floor: stage {floor_cfg}, checkpoint {floor_zip} (expected {SIGMA_FLOOR})")
    num_timesteps = int(model.num_timesteps)
    if expect_steps is not None and num_timesteps != int(expect_steps):
        problems.append(f"{label}: num_timesteps {num_timesteps} != expected {expect_steps}")
    units = {"gamma": gamma, "ret_var": float(vn.ret_rms.var), "epsilon": float(vn.epsilon),
             "sigma_r": float(np.sqrt(vn.ret_rms.var + vn.epsilon)), "clip_reward": float(vn.clip_reward),
             "clip_obs": float(vn.clip_obs), "norm_reward": bool(vn.norm_reward), "training": bool(vn.training)}
    done, _open, vec_steps = collect(vn, model, n_episodes // N_ENVS, max_vec_steps=max_vec_steps)
    vn.close()
    c_collect = time.process_time() - c0
    eps = [ep for i in range(N_ENVS) for ep in done[i][:n_episodes // N_ENVS]]
    if len(eps) != n_episodes:
        raise Refused(f"{label}: {len(eps)} complete episodes, not {n_episodes}")
    # episode checks
    for j, ep in enumerate(eps):
        T = len(ep["k"])
        if not np.array_equal(ep["k"], np.arange(T)):
            problems.append(f"{label}: episode {j}: k is not 0 ... {T - 1}")
        if ep["trunc"]:
            if not (T == HORIZON and ep["k_term"] == HORIZON and ep["term_X"] is not None):
                problems.append(f"{label}: episode {j}: a truncation at T {T}, k_term {ep['k_term']}")
        elif T > HORIZON:
            problems.append(f"{label}: episode {j}: terminated after {T} > {HORIZON} steps")
        if ep["monitor_l"] != T:
            problems.append(f"{label}: episode {j}: Monitor length {ep['monitor_l']} != {T}")
    X = np.concatenate([ep["X"] for ep in eps], 0)
    K = np.concatenate([ep["k"] for ep in eps], 0)
    TX = np.stack([ep["term_X"] for ep in eps if ep["trunc"]], 0) if any(ep["trunc"] for ep in eps) \
        else np.zeros((0, X.shape[1]), dtype=np.float32)
    TK = np.array([ep["k_term"] for ep in eps if ep["trunc"]], dtype=np.int64)
    V_loop = np.concatenate([ep["V_loop"] for ep in eps], 0)
    net = R.Net(zip_path, pkl_path, env_factory())
    log_std_zip = read_log_std(zip_path)
    log_std_net = net.policy.log_std.detach().cpu().numpy().astype(np.float32)
    if not np.array_equal(log_std_zip, log_std_net):
        problems.append(f"{label}: policy.pth log_std differs from the loaded policy's")
    rec = {"label": label, "zip": str(zip_path), "zip_sha256": sha256_file(zip_path), "pkl": str(pkl_path),
           "pkl_sha256": sha256_file(pkl_path), "num_timesteps": num_timesteps, "expect_steps": expect_steps,
           "seed": int(seed), "env_seeds": [int(seed) + r for r in range(N_ENVS)], "n_episodes": n_episodes,
           "episodes_per_env": n_episodes // N_ENVS, "units": units,
           "episodes": {"n": len(eps), "truncated": int(sum(ep["trunc"] for ep in eps)),
                        "terminated": int(sum(ep["term"] for ep in eps)),
                        "length_min": int(min(len(ep["k"]) for ep in eps)),
                        "length_max": int(max(len(ep["k"]) for ep in eps)),
                        "states": int(len(K)),
                        "return_raw_mean": float(np.mean([float(np.sum(ep["r_raw"], dtype=np.float64))
                                                          for ep in eps])),
                        "monitor_minus_sum_raw_max_abs": float(max(abs(ep["monitor_r"] - float(
                            np.sum(ep["r_raw"], dtype=np.float64))) for ep in eps))},
           "actuators_13_14": names[13:15], "mirror_action_perm_13_14": b["mirror_action_perm"][13:15],
           "forms": {}, "sigma": sigma_record(log_std_zip)}
    plain_V = None
    for form in forms:
        fv = _form_values(net, form, X, K, TX, TK, start=phase_start and form == "phase")
        st = _stats_for(eps, fv["V"], fv["V_term"], gamma)
        frec = {"buckets": st, "record": fv["record"]}
        if form != "plain":
            st_raw = _stats_for(eps, fv["V_raw"], fv["V_term"], gamma)
            frec["raw_buckets"] = st_raw
            frec["raw_note"] = ("the same critic with the slots / input zero (no phase or tail-blind input); G as the "
                                "effective form's (its bootstrap)")
        dl = float(np.max(np.abs(fv["V_raw"] - V_loop)))
        frec["loop_values_vs_raw_max_abs"] = dl
        if dl > V_LOOP_TOL:
            problems.append(f"{label}: {form}: loop forward values differ from the batched raw critic by {dl:.3g}")
        if form == "phase":
            problems += phase_feed_problems(label, fv["record"], len(TX))
        if form == "plain":
            plain_V = (fv["V"], fv["V_term"])
        elif form == "phase" and plain_V is not None:
            dv = float(np.max(np.abs(fv["V"] - plain_V[0])))
            dvt = float(np.max(np.abs(fv["V_term"] - plain_V[1]))) if len(TX) else 0.0
            frec["max_abs_minus_plain"] = dv
            frec["max_abs_minus_plain_term"] = dvt
            if phase_start and (dv != 0.0 or dvt != 0.0):
                problems.append(f"{label}: the start pair's phase form differs from its plain form (max |dV| {dv})")
        rec["forms"][form] = frec
    c_total = time.process_time() - c0
    timing = {"label": label, "cpu_s_total": round(c_total, 2), "cpu_s_collect": round(c_collect, 2),
              "wall_s_total": round(time.time() - w0, 2), "vec_steps": vec_steps,
              "env_steps": vec_steps * N_ENVS, "states_used": int(len(K)),
              "cpu_ms_per_env_step": round(1000.0 * c_collect / max(1, vec_steps * N_ENVS), 4),
              "cpu_s_per_episode": round(c_total / n_episodes, 3)}
    return rec, problems, timing


def sigma_at(zip_path, label, steps):
    return {"label": label, "steps": int(steps), "zip": str(zip_path), "zip_sha256": sha256_file(zip_path),
            **sigma_record(read_log_std(zip_path))}


# ================================================================================================ output
def _f(x, fmt="+.4f"):
    return "n/a" if x is None else format(x, fmt)


def render_pair(rec):
    L = [f"PAIR {rec['label']}: num_timesteps {rec['num_timesteps']:,}; zip {rec['zip_sha256'][:16]} pkl "
         f"{rec['pkl_sha256'][:16]}; {rec['episodes']['n']} episodes ({rec['episodes']['truncated']} truncated at "
         f"1000, {rec['episodes']['terminated']} terminated), seeds {rec['env_seeds']}; sigma_r "
         f"{rec['units']['sigma_r']:.4f}, gamma {rec['units']['gamma']}"]
    for form, fr in rec["forms"].items():
        variants = [("effective", fr["buckets"])] + ([("raw", fr["raw_buckets"])] if "raw_buckets" in fr else [])
        for vname, rows in variants:
            L.append(f"  {form} critic ({vname}):  bucket       n_ep  states  mean V-G (SE)       mean |V-G| (SE)"
                     f"     pooled V-G  mean V   mean G")
            for r in rows:
                L.append(f"    {r['bucket']:<12} {r['n_episodes']:>4} {r['n_states']:>7}  {_f(r['mean_err'])} "
                         f"({_f(r['se_err'], '.4f')})  {_f(r['mean_abs_err'], '.4f')} ({_f(r['se_abs_err'], '.4f')})"
                         f"  {_f(r['pooled_err'])}  {_f(r['mean_V'], '.3f')}  {_f(r['mean_G'], '.3f')}")
        extra = {k: fr[k] for k in ("loop_values_vs_raw_max_abs", "max_abs_minus_plain", "max_abs_minus_plain_term")
                 if k in fr}
        if fr.get("record"):
            extra["record"] = fr["record"]
        L.append(f"    checks: {json.dumps(extra, sort_keys=True)}")
    s = rec["sigma"]
    L.append(f"  sigma: tail_3 {s['tail_3']['sigma']:.6f} (floored {s['tail_3']['sigma_floored']:.6f}), tail_2 "
             f"{s['tail_2']['sigma']:.6f} (floored {s['tail_2']['sigma_floored']:.6f})")
    return L


def render(result):
    L = [VERSION, f"tool sha256 {result['tool']['sha256']}  SPEC sha256 {result['tool']['spec_sha256']}",
         "REPORT ONLY: outside the phase screen's reading; never a line or a secondary; it changes nothing the "
         "screen reads.",
         f"G: the Monte Carlo return-to-go (gamma {GAMMA}; a time-limit truncation bootstrapped with the form's own "
         "V(terminal), a termination not) in the critic's units under the pair's FROZEN ret_rms; the critic itself was "
         f"trained on GAE({C.ARM_GAE_LAMBDA['control']}) lambda-returns under each rollout's running reward "
         "statistics (SPEC DEFINITIONS).",
         f"mode {result['mode']}"]
    if result.get("guard"):
        g = result["guard"]
        L.append(f"guard: score.json {g['score_sha256'][:16]} status {g['score_status']} (line {g['score_line_key']})"
                 f"; entries " + ", ".join(f"{e['name']} {e['cls']}" for e in g["entries"]))
    L.append(f"settings: {json.dumps(result['settings'], sort_keys=True)}")
    L.append("")
    for rec in result["pairs"]:
        L.extend(render_pair(rec))
        L.append("")
    if result.get("arms"):
        L.append("ARMS (effective critic; per bucket: n runs, median [mean] of the per-run mean V-G and mean |V-G|; "
                 "control - critic_phase of the medians)")
        for row in result["arms"]:
            c, p, d = row["control"], row["critic_phase"], row["control_minus_phase"]
            L.append(f"  {row['bucket']:<12} control n {c['n']} V-G {_f(c['median_err'])} [{_f(c['mean_err'])}] "
                     f"|V-G| {_f(c['median_abs_err'], '.4f')}; critic_phase n {p['n']} V-G {_f(p['median_err'])} "
                     f"[{_f(p['mean_err'])}] |V-G| {_f(p['median_abs_err'], '.4f')}; diff V-G {_f(d['median_err'])} "
                     f"|V-G| {_f(d['median_abs_err'])}")
            if row["paired"]:
                L.append("      per seed control - phase V-G: " + ", ".join(
                    f"{x['seed']} {_f(x['control_minus_phase_err'])}" for x in row["paired"]))
        L.append("EXPECTATIONS (SPEC; a description, not a test): " + "; ".join(
            f"{k}: {v}" for k, v in result["expectations"].items()))
        L.append("")
    if result.get("sigma_table"):
        L.append("TAIL SIGMA (exp(log_std); sampled, no floor; floored = max(sigma, 0.01) of the mirror-KL loss)")
        for row in result["sigma_table"]:
            binds = row["tail_3"]["floor_binds"] or row["tail_2"]["floor_binds"]
            L.append(f"  {row['run']:<22} {row['steps']:>9,}  tail_3 {row['tail_3']['sigma']:.6f}  tail_2 "
                     f"{row['tail_2']['sigma']:.6f}  floor binds {binds}")
        for row in result.get("sigma_arm_medians", []):
            L.append(f"  median {row['arm']:<15} {row['steps']:>9,}  tail_3 {_f(row['tail_3'], '.6f')}  tail_2 "
                     f"{_f(row['tail_2'], '.6f')}  (n {row['n']})")
        L.append("")
    L.append("PROBLEMS: " + ("none" if not result["problems"] else ""))
    L.extend(f"  - {p}" for p in result["problems"])
    L.append(f"STATUS: {result['status']}")
    return L


def tool_record():
    tools = {t: sha256_file(PS_TOOLS / t) for t in IMPORTED_TOOLS}
    return {"version": VERSION, "file": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()),
            "spec": str(SPEC), "spec_sha256": sha256_file(SPEC) if SPEC.is_file() else None,
            "ps_tools_imported": tools, "read_bs_version": R.VERSION}


def settings_record(seed, n_episodes):
    return {"buckets": list(BUCKET_NAMES), "n_episodes": n_episodes, "n_envs": N_ENVS,
            "episodes_per_env": n_episodes // N_ENVS, "seed_base": seed, "gamma": GAMMA, "horizon": HORIZON,
            "tail_actions": {"tail_3": ACT_TAIL3, "tail_2": ACT_TAIL2}, "sigma_floor": SIGMA_FLOOR,
            "se": "sample SD of the per-episode bucket means / sqrt(n episodes)",
            "pre_registered_root_settings": bool(seed == PROD_SEED and n_episodes == N_EPISODES)}


def write_outputs(out, result, info):
    out.mkdir(parents=True, exist_ok=True)
    (out / "vg_sigma.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "vg_sigma.txt").write_text("\n".join(render(result)) + "\n")
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n")


# ================================================================================================ modes
def check_out(out, root=None):
    out = Path(out).resolve()
    for forbidden in [PS_ROOT.resolve()] + ([Path(root).resolve()] if root is not None else []):
        if out == forbidden or forbidden in out.parents:
            raise Refused(f"--out {out} is inside {forbidden}")
    if (out / "vg_sigma.json").exists():
        raise Refused(f"{out}/vg_sigma.json exists (choose a new --out)")
    return out


def run_root(root, out, plan_only=False, *, n_episodes=N_EPISODES, seed=PROD_SEED):
    """--root mode.  n_episodes / seed are the pre-registered N_EPISODES / PROD_SEED from the CLI (which refuses
    overrides); only a validation script calling this function may pass others (recorded in settings)."""
    root = Path(root).resolve()
    out = check_out(out, root)
    plan = guard(root)                       # refuses before anything under runs/ or reads/ is touched
    if plan_only:
        print(json.dumps({"guard": plan, "would_read": {
            c["name"]: {"final": [str(p) for p in run_paths(root, c["name"])[0]],
                        "saved_zips": [str(p) for _s, p in run_paths(root, c["name"])[1]],
                        "start_copy_zip": str(run_paths(root, c["name"])[2]),
                        "markers": str(root / "reads" / c["name"] / "markers.json")} for c in plan["counted"]},
            "start_pair": [C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"])]}, indent=1))
        return EXIT_OK
    t_utc0 = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    origin = setup_sim(tmp)
    problems, timings, inputs = [], [], {}
    try:
        tool = tool_record()
        approved = root / "APPROVED"
        appr = None
        if approved.is_file():
            ad, asha = _read_json(approved)
            appr = {"file": str(approved), "sha256": asha}
            for t in IMPORTED_TOOLS:
                want = (ad.get("tool_sha256") or {}).get(t)
                if want != tool["ps_tools_imported"][t]:
                    problems.append(f"PS tool {t}: sha256 {tool['ps_tools_imported'][t]} != APPROVED's {want}")
            if ad.get("screen_sha256") != plan["screen_sha256"]:
                problems.append(f"SCREEN.json: sha256 {plan['screen_sha256']} != APPROVED's {ad.get('screen_sha256')}")
        else:
            problems.append(f"{approved} missing: the imported PS tools are not checked against it")
        for c in plan["counted"]:                # every final pair present before anything is simulated
            for p in run_paths(root, c["name"])[0]:
                if not p.is_file():
                    raise Refused(f"{c['name']}: final pair file {p} missing")
        s_zip, s_pkl = C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"])
        for p, want in ((s_zip, C.START["zip_sha256"]), (s_pkl, C.START["pkl_sha256"])):
            got = sha256_file(p)
            inputs[str(p)] = got
            if got != want:
                raise Refused(f"start pair {p}: sha256 {got} != FMAN {want}")
        pairs, runs, sigma_rows = [], [], []
        rec, pr, tm = measure_pair(s_zip, s_pkl, forms=("plain", "phase"), seed=seed, n_episodes=n_episodes,
                                   label="S (start pair, F2)", expect_steps=C.S, phase_start=True)
        pairs.append(rec)
        problems += pr
        timings.append(tm)
        s_sigma = sigma_record(read_log_std(s_zip))
        for c in plan["counted"]:
            (fz, fp), saved, start_copy = run_paths(root, c["name"])
            for p in (fz, fp):
                if not p.is_file():
                    raise Refused(f"{c['name']}: final pair file {p} missing")
            sc = sha256_file(start_copy) if start_copy.is_file() else None
            inputs[str(start_copy)] = sc
            if sc != C.START["zip_sha256"]:
                problems.append(f"{c['name']}: its start copy's sha256 {sc} is not the start pair's")
            form = ARM_FORM[c["arm"]]
            rec, pr, tm = measure_pair(fz, fp, forms=(form,), seed=seed, n_episodes=n_episodes,
                                       label=c["name"], expect_steps=C.E, phase_start=False)
            rec.update(run=c["name"], arm=c["arm"], run_seed=c["seed"], steps=C.E)
            inputs[str(fz)], inputs[str(fp)] = rec["zip_sha256"], rec["pkl_sha256"]
            mk = root / "reads" / c["name"] / "markers.json"
            if mk.is_file():
                m, msha = _read_json(mk)
                inputs[str(mk)] = msha
                rows = [r for r in (m.get("loop") or []) if isinstance(r, dict) and r.get("steps") == C.E]
                if len(rows) != 1:
                    problems.append(f"{c['name']}: markers hold {len(rows)} loop rows at E")
                    rec["markers_match"] = False
                else:
                    ok = (rows[0].get("zip_sha256") == rec["zip_sha256"] and rows[0].get("pkl_sha256")
                          == rec["pkl_sha256"] and rows[0].get("critic_form") == CRITIC_FORM_MARKERS[c["arm"]])
                    rec["markers_match"] = ok
                    if not ok:
                        problems.append(f"{c['name']}: the markers' loop row at E names another pair or critic form")
            else:
                rec["markers_match"] = None
                problems.append(f"{c['name']}: {mk} missing (the pair is not cross-checked with the read)")
            pairs.append(rec)
            problems += pr
            timings.append(tm)
            runs.append({"name": c["name"], "arm": c["arm"], "seed": c["seed"],
                         "buckets": rec["forms"][form]["buckets"]})
            sigma_rows.append({"run": c["name"], "arm": c["arm"], "steps": C.S, "zip": s_zip,
                               "zip_sha256": C.START["zip_sha256"], **s_sigma})
            for s, z in saved:
                if z.is_file():
                    row = sigma_at(z, c["name"], s)
                    inputs[str(z)] = row["zip_sha256"]
                    sigma_rows.append({"run": c["name"], "arm": c["arm"], **row})
                else:
                    problems.append(f"{c['name']}: saved pair {z} missing")
            sigma_rows.append({"run": c["name"], "arm": c["arm"], "steps": C.E, "zip": str(fz),
                               "zip_sha256": rec["zip_sha256"], **rec["sigma"]})
        summary = arm_summary(runs)
        sig_med = []
        for arm in ("control", "critic_phase"):
            for s in (C.S, *C.SAVED_STEPS, C.E):
                rows = [r for r in sigma_rows if r["arm"] == arm and r["steps"] == s]
                sig_med.append({"arm": arm, "steps": s, "n": len(rows),
                                "tail_3": median([r["tail_3"]["sigma"] for r in rows]),
                                "tail_2": median([r["tail_2"]["sigma"] for r in rows])})
        result = {"schema": SCHEMA, "version": VERSION, "tool": tool, "mode": "root", "guard": plan,
                  "approved": appr, "environments_origin": origin, "settings": settings_record(seed, n_episodes),
                  "inputs": dict(sorted(inputs.items())), "pairs": pairs, "arms": summary,
                  "expectations": expectations(summary), "sigma_table": sigma_rows, "sigma_arm_medians": sig_med,
                  "problems": problems, "status": "measured with PROBLEMS" if problems else "measured",
                  "report_only": "outside the screen's reading; never a line, a secondary or a PROVISIONAL item"}
        info = {"utc_start": t_utc0, "utc_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timings": timings}
        write_outputs(out, result, info)
        print("\n".join(render(result)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return EXIT_PROBLEM if problems else EXIT_OK


def run_pair(zip_path, pkl_path, form, out, *, phase_start=False, episodes=N_EPISODES, seed=VAL_SEED, label=None,
             expect_steps=None, max_vec_steps=None):
    out = check_out(out)
    if form not in FORMS:
        raise Refused(f"--form {form!r} not one of {FORMS}")
    if phase_start and form != "phase":
        raise Refused("--phase-start needs --form phase")
    for p in (zip_path, pkl_path):
        if not Path(p).is_file():
            raise Refused(f"{p} missing")
    t_utc0 = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    origin = setup_sim(tmp)
    try:
        forms = ("plain", "phase") if phase_start else (form,)
        rec, problems, tm = measure_pair(Path(zip_path), Path(pkl_path), forms=forms, seed=seed, n_episodes=episodes,
                                         label=label or Path(zip_path).name, expect_steps=expect_steps,
                                         phase_start=phase_start, max_vec_steps=max_vec_steps)
        result = {"schema": SCHEMA, "version": VERSION, "tool": tool_record(), "mode": "pair", "guard": None,
                  "environments_origin": origin, "settings": settings_record(seed, episodes),
                  "inputs": {str(zip_path): rec["zip_sha256"], str(pkl_path): rec["pkl_sha256"]}, "pairs": [rec],
                  "problems": problems, "status": "measured with PROBLEMS" if problems else "measured",
                  "report_only": "validation mode (--pair): no screen root"}
        info = {"utc_start": t_utc0, "utc_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timings": [tm]}
        write_outputs(out, result, info)
        print("\n".join(render(result)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return EXIT_PROBLEM if problems else EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="the screen's folder; required without --pair (no default)")
    ap.add_argument("--out", required=True, help="output folder (not inside the root or <PS>)")
    ap.add_argument("--plan", action="store_true", help="--root mode: the guard and the list of inputs only")
    ap.add_argument("--pair", nargs=2, metavar=("ZIP", "PKL"), help="validation: one explicit pair, no root")
    ap.add_argument("--form", choices=FORMS, help="--pair: the critic form")
    ap.add_argument("--phase-start", action="store_true", help="--pair --form phase: the start copy (transform)")
    ap.add_argument("--episodes", type=int, default=None, help="--pair only (default 32)")
    ap.add_argument("--seed", type=int, default=None, help="--pair only (default 6281)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--expect-steps", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        if a.pair:
            if a.root is not None or a.plan:
                raise Refused("--root / --plan are root-mode options, not with --pair")
            if not a.form:
                raise Refused("--pair needs --form")
            return run_pair(a.pair[0], a.pair[1], a.form, a.out, phase_start=a.phase_start,
                            episodes=N_EPISODES if a.episodes is None else a.episodes,
                            seed=VAL_SEED if a.seed is None else a.seed, label=a.label, expect_steps=a.expect_steps)
        if a.episodes is not None or a.seed is not None or a.form or a.phase_start:
            raise Refused("--episodes / --seed / --form / --phase-start are --pair options (--root is pre-registered)")
        if a.root is None:
            raise Refused("--root is required without --pair (no default; nothing opened)")
        return run_root(a.root, a.out, plan_only=a.plan)
    except Refused as exc:
        print(f"{VERSION}\nREFUSED: {exc}")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
