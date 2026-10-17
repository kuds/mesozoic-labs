"""Judge ONE checkpoint of a notebook arm L run (T. rex stance) for the r10 foot + arm L screen (scratch only).

    PYTHONPATH=CLONE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python3 judge_r10.py RUN_DIR --seed {42,52,60} --steps {4194304,5242880,6291456} --out OUT.json

RUN_DIR is a notebook run directory copied locally with the notebook's layout:
    RUN_DIR/01_stance/models/stage1_{N}_steps.zip, RUN_DIR/01_stance/models/stage1_vecnormalize_{N}_steps.pkl,
    RUN_DIR/01_stance/{plant_identity.json, stage_config.json, task_fingerprint.json} (evaluations.npz and
    gate_progress.npz are not read).

Adapted from the r9 symmetry screen's judge (foot/symmetry/screen/harness/judge.py, "the r9 judge"); what is the same:
  * isolation: `environments` imports from CLONE only (the editable finder and every main-checkout path removed, the
    clone first on sys.path; asserted for every loaded environments.* module), BLAS threads 1, torch threads 1;
  * the panel: reporting.stance_report._stance_v2_report -- the code build_stance_gate_report dispatches a
    stance_quality/v2 stage to -- on the clone's stance stage (load_all_stages / _resolve_stage_ref("stance")), its
    [env] kwargs exactly as the stage loads them, StanceV2Thresholds from its own [curriculum] (the clone's bars), 40
    episodes on seeds 3042-3081 through the floor-truth recorder, the policy loaded by stance_report._load_policy
    (policy_loading.load_sb3_checkpoint, the saved VecNormalize statistics frozen, plant identity checked), the
    zero-action statue rolled on the same seeds under the same env, the verdict from evaluate_stance_v2_gate.  The
    stage declares stance_quality_heading/v1 (the six-heading certificate); build_stance_gate_report refuses a
    single-panel report for it, so, as the r9 judge did, the v2 branch is called directly at the panel's single
    heading.  PR #620's per-foot pad telemetry per episode (gait.pad_telemetry, report only);
  * the H4 noise test, arithmetic unchanged: 20 spawns 3042-3061, numpy default_rng(episode seed) draws, the mean
    action of the deterministic policy plus N(0, sigma^2):
      noise_matched_control_joint  sigma = frozen_params criteria.H4.control_sigma[seed].vector (the r9 screen's
                                   1M production control_s{seed} exp(log_std)), JOINT space: a = clip(clip(mu) + sigma z)
      noise_matched_own            sigma = the checkpoint's own exp(log_std), POLICY space: a = clip(mu + sigma z)
    a fall = an episode short of the 1000-step horizon.
What changed against the r9 judge:
  * the clone is a parameter (--clone, default the screen clone fix/screen_r10L, HEAD pinned by --expect-head) and the
    checkpoint is read from a notebook run (RUN_DIR/01_stance/models), not from runs/{arm}_s{seed}/[resume6m/]models;
  * the noise tests load the model and VecNormalize through policy_loading.load_sb3_checkpoint (the r9 judge used
    PPO.load + pickle.load; same parameters, same statistics);
  * the control sigma vector is read from frozen_params.json (the r9 judge read it from the control zip; the frozen
    vector is that zip's exp(log_std) verbatim, and frozen_params.json is checked against the r9 verdict's
    params_sha256);
  * the run's records are bound to the clone: RUN_DIR/01_stance/plant_identity.json physics_sha256 == the clone's
    plant, the checkpoint's mesozoic_symmetry record (mirror_map_sha256 included) == the record the clone's stance
    [ppo] keys and committed mirror map give, the checkpoint's and the run's task fingerprint == the task the panel
    rolls, and stage_config.json's env / gate bars == the clone's.  Each check can be relaxed for validation runs only
    (--no-symmetry-check, --task-binding report, --allow-missing-records, --allow-seed-mismatch), and the JSON says so;
  * the screen's arm L reward terms are ON in the r10 clone's stance [env] (lateral_lean_*, leg_home_pose_reference),
    so the panel, the statue and the noise episodes run with them (reward only; the r9 judge ran the r9 stage's
    production [env], without the arm's experiment_* reward switches);
  * per episode it records clean, full horizon, hop-or-fall (stance_gate_v2.is_hop_or_fall), the failing rails,
    max_foot_cop_fore_aft (the r10 rail), max_sole_cop_fore_aft (the pad; report only on r10), max_sole_tilt_deg,
    max_sole_corner_lift_m, reward and termination; one JSON per checkpoint, never overwritten.

--smoke-episodes N (development only) shortens every panel to N episodes and marks the JSON "smoke"; score_r10.py
refuses smoke files.
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

SCHEMA = "r10screen-judge/v1"
MAIN_CHECKOUT = "/home/user/mesozoic-labs"
SCRATCH = "scratchpad"
DEFAULT_CLONE = SCRATCH + "/fix/screen_r10L"
#: claude/screen-trex-r10-arm-l as pushed (026eb833: physics r10 + arm L on in the stance config).
DEFAULT_HEAD = "026eb833a0c84989a3656f92b6229eab11febade"
FROZEN_PARAMS = SCRATCH + "/foot/symmetry/design/frozen_params.json"
#: The r9 screen verdict's params_sha256 (foot/symmetry/screen/analysis/verdict.json): the frozen file is that one.
FROZEN_PARAMS_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
STAGE_SUBDIR = "01_stance"
SEEDS = (42, 52, 60)
PANEL_SEED, PANEL_EPISODES = 3042, 40
NOISE_SEED, NOISE_EPISODES = 3042, 20
#: The fixed rails of criteria.txt (K2 / K4) and of the r9 screen (frozen_params criteria): recorded per episode.
FOOT_COP_BAR = 0.80          # r10 fore-aft rail: max_foot_cop_fore_aft > 0.80
PAD_COP_BAR = 0.80           # r9 fore-aft rail (pad): max_sole_cop_fore_aft > 0.8 (report only on r10)
ROLL_TILT_DEG = 2.0          # roll rail: max_sole_tilt_deg > 2.0
ROLL_LIFT_M = 0.004          #         or max_sole_corner_lift_m > 0.004
TRACES: list = []


# ------------------------------------------------------------------------------------------------ isolation
def isolate(clone: str) -> dict[str, str]:
    """Put *clone* first on sys.path with the editable finder (and the main checkout, and this directory) removed."""
    import importlib

    wt = os.path.realpath(clone)
    here = os.path.dirname(os.path.realpath(__file__))
    if any(n == "environments" or n.startswith("environments.") for n in sys.modules):
        raise RuntimeError("environments was imported before isolate(); its origin cannot be controlled")
    sys.meta_path[:] = [f for f in sys.meta_path if "__editable__" not in (getattr(f, "__module__", "") or type(f).__module__)]
    sys.path_hooks[:] = [h for h in sys.path_hooks if "__editable__" not in getattr(h, "__module__", "")]
    sys.path[:] = [p for p in sys.path if "__editable__" not in p and os.path.realpath(p or ".") not in (MAIN_CHECKOUT, here)]
    sys.path_importer_cache.clear()
    sys.path.insert(0, wt)
    paths = {}
    for name in ("environments", "environments.shared.reporting.stance_report", "environments.shared.curriculum.stance_gate_v2",
                 "environments.shared.policy_loading", "environments.shared.gait.recorder", "environments.shared.gait.pad_telemetry",
                 "environments.tyrannosaurus_rex.base_env"):
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


class Ctx:
    """The clone's stance stage, its config as loaded, and the repo objects every panel needs."""

    def __init__(self) -> None:
        from environments.shared.cli import _resolve_stage_ref
        from environments.shared.config import load_all_stages, load_stage_config
        from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds
        from environments.shared.gait.morphology import Morphology
        from environments.shared.gait.pad_telemetry import PadLayout
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.reporting import stance_report as SR
        from environments.shared.species_names import require_model_id

        self.SR = SR
        self.mid = require_model_id("tyrannosaurus_rex", "anatomical")
        configs = load_all_stages(self.mid)
        self.stage = _resolve_stage_ref("stance", configs, self.mid)
        self.cfg = configs[self.stage]
        assert self.cfg == load_stage_config(self.mid, "stance")
        assert not any(k.startswith("experiment_") for k in self.cfg["env_kwargs"]), "no scratch experiment keys"
        self.curriculum = self.cfg["curriculum_kwargs"]
        self.env_kwargs = dict(self.cfg["env_kwargs"])
        self.ppo_kwargs = dict(self.cfg.get("ppo_kwargs") or {})
        self.horizon = int(self.env_kwargs.get("max_episode_steps", 1000))
        self.control_dt = float(self.env_kwargs.get("timestep", 0.002)) * int(self.env_kwargs.get("frame_skip", 5))
        self.thresholds = StanceV2Thresholds.from_curriculum(self.curriculum)
        self.thresholds.validate_settle_window(self.control_dt)
        self.settle = int(self.thresholds.settle_steps)
        self.plant_identity = current_plant_identity(self.mid)
        self.env_class = SR.model_env_class(self.mid)
        probe = self.env_class(**self.env_kwargs)
        try:
            self.layout = PadLayout.from_morphology(Morphology.from_env(probe, self.mid))
        finally:
            probe.close()
        self.feet = [f.label for f in self.layout.feet]


def install_trace_capture() -> None:
    from environments.shared.gait.recorder import SubstepContactRecorder

    original = SubstepContactRecorder.end_episode

    def end_episode(self, *args, **kwargs):
        trace = original(self, *args, **kwargs)
        TRACES.append(trace)
        return trace

    SubstepContactRecorder.end_episode = end_episode


def pad_rows(ctx: Ctx, traces: list) -> list[dict]:
    from environments.shared.gait.pad_telemetry import episode_pad_telemetry

    return [{k: finite_or_none(v) for k, v in episode_pad_telemetry(t, ctx.layout, settle_steps=ctx.settle).items()}
            for t in traces]


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


def episode_rows(ctx: Ctx, evidence: list[dict], pads: list[dict], traces: list) -> list[dict]:
    """One row per panel episode (the v2 evidence row's classification and metrics, the trace's termination)."""
    from environments.shared.curriculum.stance_gate_v2 import is_hop_or_fall, reason_key

    rows = []
    for ev, pad, tr in zip(evidence, pads, traces):
        assert int(tr.length) == int(ev["length"]), (tr.length, ev["length"])
        sole, foot = finite_or_none(ev.get("max_sole_cop_fore_aft")), finite_or_none(ev.get("max_foot_cop_fore_aft"))
        check_pad_identities(ctx, ev["seed"], pad, sole, foot)
        reasons = ev.get("reasons")
        full = bool(ev["reached_horizon"])
        tilt, lift = finite_or_none(ev.get("max_sole_tilt_deg")), finite_or_none(ev.get("max_sole_corner_lift_m"))
        rows.append({
            "seed": int(ev["seed"]), "length": int(ev["length"]), "reached_horizon": full, "full_horizon": full,
            "clean": ev.get("clean"), "reasons": reasons,
            "failing_rails": None if reasons is None else sorted({reason_key(r) for r in reasons}),
            "hop_or_fall": None if reasons is None else bool(is_hop_or_fall(reasons)),
            "reward": finite_or_none(ev["reward"]),
            "termination_reason": getattr(tr, "termination_reason", None),
            "max_foot_cop_fore_aft": foot, "max_sole_cop_fore_aft": sole,
            "max_sole_tilt_deg": tilt, "max_sole_corner_lift_m": lift,
            "roll_rail": roll_rail(full, tilt, lift),
            "pad": pad,
        })
    return rows


def summarize_episodes(ctx: Ctx, rows: list[dict]) -> dict:
    """Falls, the two fore-aft CoP rails (all episodes / full horizon), the roll rail, per-foot CoP statistics."""
    full = [r for r in rows if r["reached_horizon"]]

    def over(r, key, bar):
        v = r.get(key)
        return v is not None and v > bar

    out = {
        "n": len(rows), "n_full_horizon": len(full),
        "falls": sum(1 for r in rows if not r["reached_horizon"]),
        "fall_seeds": [r["seed"] for r in rows if not r["reached_horizon"]],
        "fall_lengths": [r["length"] for r in rows if not r["reached_horizon"]],
        # the pad rail (r9's), counted as the r9 judge did (all episodes) and over the full-horizon ones
        "cop_gt_bar_gate": sum(1 for r in rows if over(r, "max_sole_cop_fore_aft", PAD_COP_BAR)),
        "cop_gt_bar_gate_full_horizon": sum(1 for r in full if over(r, "max_sole_cop_fore_aft", PAD_COP_BAR)),
        "cop_unmeasured": sum(1 for r in rows if r.get("max_sole_cop_fore_aft") is None),
        "gate_cop_median": med(r.get("max_sole_cop_fore_aft") for r in rows),
        # the whole-foot rail (r10's)
        "foot_cop_gt_bar": sum(1 for r in rows if over(r, "max_foot_cop_fore_aft", FOOT_COP_BAR)),
        "foot_cop_gt_bar_full_horizon": sum(1 for r in full if over(r, "max_foot_cop_fore_aft", FOOT_COP_BAR)),
        "foot_cop_unmeasured": sum(1 for r in rows if r.get("max_foot_cop_fore_aft") is None),
        "foot_cop_median": med(r.get("max_foot_cop_fore_aft") for r in rows),
        "foot_cop_mean_full_horizon": mean(r.get("max_foot_cop_fore_aft") for r in full),
        "roll_rail_full_horizon": sum(1 for r in rows if r.get("roll_rail")),
        "reward_mean_full": (sum(r["reward"] for r in full) / len(full)) if full and all(r["reward"] is not None for r in full) else None,
        "per_foot": {},
    }
    for foot in ctx.feet:
        def p(r, key):
            return (r.get("pad") or {}).get(f"{foot}_{key}")
        gate = [p(r, "sole_cop_fore_aft_gate") for r in rows]
        out["per_foot"][foot] = {
            "cop_gt_bar": sum(1 for g in gate if g is not None and g > PAD_COP_BAR),
            "cop_gt_bar_full_horizon": sum(1 for r in full if (p(r, "sole_cop_fore_aft_gate") or 0.0) > PAD_COP_BAR),
            "signed_cop_median": med(p(r, "sole_cop_fore_aft") for r in full),
            "gate_cop_median": med(p(r, "sole_cop_fore_aft_gate") for r in full),
            "digit_share_median": med(p(r, "digit_share") for r in full),
            "signed_cop_median_all": med(p(r, "sole_cop_fore_aft") for r in rows),
            # whole-foot CoP (r10; absent fields read None on r9)
            "foot_cop_gt_bar_full_horizon": sum(1 for r in full if (p(r, "foot_cop_fore_aft_gate") or 0.0) > FOOT_COP_BAR),
            "foot_signed_cop_mean": mean(p(r, "foot_cop_fore_aft") for r in full),
            "foot_signed_cop_median": med(p(r, "foot_cop_fore_aft") for r in full),
            "foot_gate_cop_mean": mean(p(r, "foot_cop_fore_aft_gate") for r in full),
            "foot_gate_cop_median": med(p(r, "foot_cop_fore_aft_gate") for r in full),
        }
    return out


# ------------------------------------------------------------------------------------------------ panel
def v2_panel(ctx: Ctx, *, model_path: str, vecnorm_path: str) -> dict:
    """The stance_quality/v2 report (repo code) on the 40-episode panel, plus per-episode pad telemetry."""
    SR = ctx.SR
    predict, note = SR._load_policy(model_path, vecnorm_path, lambda: ctx.env_class(**ctx.env_kwargs),
                                    plant_identity=ctx.plant_identity, allow_legacy_plant=False)
    description = f"{os.path.basename(model_path)} — {note}"
    TRACES.clear()
    t0 = time.monotonic()
    report = SR._stance_v2_report(
        ctx.mid, ctx.stage, curriculum=ctx.curriculum, env_class=ctx.env_class, env_kwargs=ctx.env_kwargs,
        thresholds=ctx.thresholds, predict=predict, description=description, zero_action=False,
        model_path=model_path, vecnorm_path=vecnorm_path, episodes=PANEL_EPISODES, seed=PANEL_SEED,
        horizon=ctx.horizon, control_dt=ctx.control_dt, plant_identity=ctx.plant_identity, allow_legacy_plant=False,
    )
    seconds = time.monotonic() - t0
    assert report["statue"] is not None and not report["statue"]["reused_policy_panel"], "the statue panel must be rolled"
    assert len(TRACES) == 2 * PANEL_EPISODES, (len(TRACES), 2 * PANEL_EPISODES)
    pol_traces, st_traces = TRACES[:PANEL_EPISODES], TRACES[PANEL_EPISODES:]
    rows = episode_rows(ctx, report["episode_evidence"], pad_rows(ctx, pol_traces), pol_traces)
    st_rows = episode_rows(ctx, report["statue"]["episode_evidence"], pad_rows(ctx, st_traces), st_traces)
    statue_summary = summarize_episodes(ctx, st_rows)
    statue_summary["reference"] = report["statue"]["reference"]
    statue_summary["terminations"] = {}
    for r in st_rows:
        statue_summary["terminations"][str(r["termination_reason"])] = statue_summary["terminations"].get(str(r["termination_reason"]), 0) + 1
    result = report["result"]
    cf = result.get("criterion_failures", {})
    mean_reward, statue_reward = result["mean_reward"], result.get("statue_mean_reward")
    return {
        "policy": report["policy"], "seeds": [PANEL_SEED, PANEL_SEED + PANEL_EPISODES - 1], "episodes": PANEL_EPISODES,
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
        "statue_episodes_detail": [{k: r[k] for k in ("seed", "length", "reached_horizon", "reward", "termination_reason",
                                                      "max_foot_cop_fore_aft", "max_sole_cop_fore_aft", "max_sole_tilt_deg",
                                                      "max_sole_corner_lift_m")} for r in st_rows],
        "scored_gate_kind": report["scored_gate_kind"], "stage_gate_kind": report["gate_kind"],
        "schema": report["schema"], "task_sha256": report["task_sha256"], "handoff": report["handoff"],
        "thresholds": report["thresholds"], "measurement_sha256": report["measurement_sha256"],
        "result": result,
        "episodes_detail": rows,
    }


# ------------------------------------------------------------------------------------------------ H4 noise
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
    rows, metrics = [], []
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
            metrics.append(m)
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
    for row, pad in zip(rows, pad_rows(ctx, TRACES)):
        check_pad_identities(ctx, row["seed"], pad, row["max_sole_cop_fore_aft"], row["max_foot_cop_fore_aft"])
        row["pad"] = pad
        row["roll_rail"] = roll_rail(row["reached_horizon"], row["max_sole_tilt_deg"], row["max_sole_corner_lift_m"])
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
    # plant_identity.json (required)
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
    # the checkpoint's own plant identity (load_sb3_checkpoint also refuses a foreign plant)
    ck_plant = read_checkpoint_attribute(model_path, "_mesozoic_plant_identity")
    checks["checkpoint_plant_physics_sha256"] = {"checkpoint": (ck_plant or {}).get("physics_sha256"),
                                                 "match": (ck_plant or {}).get("physics_sha256") == clone_plant["physics_sha256"]}
    if not checks["checkpoint_plant_physics_sha256"]["match"]:
        hard.append(f"checkpoint physics_sha256 {(ck_plant or {}).get('physics_sha256')} != the clone's")
    # seed
    ck_seed = read_checkpoint_attribute(model_path, "seed")
    checks["seed"] = {"label": int(a.seed), "checkpoint": ck_seed, "match": ck_seed == int(a.seed)}
    if ck_seed != int(a.seed) and not a.allow_seed_mismatch:
        hard.append(f"the checkpoint records seed {ck_seed}, the judge was told {a.seed}")
    # mesozoic_symmetry: the record the clone's stance [ppo] keys and committed mirror map give
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
    except Exception as exc:  # noqa: BLE001 - a clone without the repo symmetry module (the r9 clone)
        sym.update({"clone_mirror_map_sha256": None, "error": f"{type(exc).__name__}: {exc}", "mirror_map_sha256_match": None,
                    "mismatch": "not checked"})
    checks["symmetry"] = sym
    if not a.no_symmetry_check and not (sym.get("mirror_map_sha256_match") is True and sym.get("mismatch") is None):
        hard.append(f"mesozoic_symmetry {recorded!r} does not match the clone ({sym.get('mismatch')}; "
                    f"clone mirror map {sym.get('clone_mirror_map_sha256')})")
    # task: the panel's task_sha256 (the clone's stance [env] on the clone's plant) == the trained task
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
    # stage_config.json: env kwargs, gate bars, symmetry keys, commit
    scj = os.path.join(stage_dir, "stage_config.json")
    if os.path.exists(scj):
        sc = read_json(scj)
        # config.save_stage_config writes the env's full kwargs: the env class's constructor defaults overlaid with the
        # stage's [env] keys.  Every key the clone's stance [env] sets must be recorded with its value, and every other
        # recorded key must be the clone env class's own default (so a key the run added, e.g. a scratch experiment_*
        # switch, is a difference).
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
        sym_keys = {k: hp.get(k) for k in ("symmetry", "symmetry_kappa", "symmetry_sigma_floor", "symmetry_mirror_critic")}
        sym_ok = sym_keys == {k: ctx.ppo_kwargs.get(k) for k in sym_keys}
        commit = sc.get("git_commit")
        checks["stage_config"] = {"env_kwargs_match": env_ok, "gate_bars_match": bars_ok, "symmetry_keys": sym_keys,
                                  "symmetry_keys_match": sym_ok, "git_commit": commit, "clone_head": git["head"],
                                  "git_commit_match": commit == git["head"], "env_kwargs_differences": env_diff,
                                  "env_default_keys_recorded": [k for k in extra if k in defaults and norm(recorded_env[k]) == norm(defaults[k])]}
        if a.task_binding == "require":
            for ok, what in ((env_ok, "env kwargs"), (bars_ok, "stance_quality/v2 bars"), (sym_ok, "[ppo] symmetry keys")):
                if not ok:
                    hard.append(f"{scj}: {what} differ from the clone's stance stage")
            if commit and commit != git["head"] and not a.allow_commit_mismatch:
                hard.append(f"{scj}: git_commit {commit} != clone HEAD {git['head']} (--allow-commit-mismatch to judge anyway)")
    else:
        (checks.setdefault("records_missing", [])).append("stage_config.json")
        if not a.allow_missing_records:
            hard.append(f"{scj} missing (required)")
    checks["hard_failures"] = hard
    return checks


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    global PANEL_EPISODES, NOISE_EPISODES
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="notebook run directory (holds 01_stance/)")
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS, help="the run's training seed (selects the frozen control sigma)")
    ap.add_argument("--steps", type=int, required=True, help="checkpoint step count N of models/stage1_{N}_steps.zip")
    ap.add_argument("--out", required=True, help="the judge JSON to write (never overwritten)")
    ap.add_argument("--clone", default=DEFAULT_CLONE)
    ap.add_argument("--expect-head", default=DEFAULT_HEAD, help="the clone HEAD the judge must run (clean tree)")
    ap.add_argument("--stage-subdir", default=STAGE_SUBDIR)
    ap.add_argument("--frozen-params", default=FROZEN_PARAMS)
    ap.add_argument("--no-noise", action="store_true", help="development only: skip the H4 noise tests")
    ap.add_argument("--smoke-episodes", type=int, default=None, help="development only: N episodes per panel (JSON marked smoke)")
    # validation-only relaxations (recorded in the JSON; score_r10.py flags them)
    ap.add_argument("--no-symmetry-check", action="store_true", help="validation only: do not require the mesozoic_symmetry match")
    ap.add_argument("--task-binding", choices=("require", "report"), default="require",
                    help="validation only ('report'): record but do not require the task / stage_config binding")
    ap.add_argument("--allow-missing-records", action="store_true", help="validation only: stage_config / task_fingerprint / plant_identity may be absent")
    ap.add_argument("--allow-seed-mismatch", action="store_true", help="validation only: the checkpoint's recorded seed may differ from --seed")
    ap.add_argument("--allow-commit-mismatch", action="store_true", help="stage_config.json git_commit may differ from the clone HEAD")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    if not os.path.isdir(os.path.dirname(out)):
        raise SystemExit(f"output directory {os.path.dirname(out)} does not exist")
    run = os.path.abspath(a.run_dir)
    stage_dir = os.path.join(run, a.stage_subdir)
    models = os.path.join(stage_dir, "models")
    stem = f"stage1_{int(a.steps)}_steps"
    zp, vp = os.path.join(models, f"{stem}.zip"), os.path.join(models, f"stage1_vecnormalize_{int(a.steps)}_steps.pkl")
    missing = [p for p in (zp, vp) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"checkpoint missing: {missing}")
    if a.smoke_episodes:
        PANEL_EPISODES = NOISE_EPISODES = int(a.smoke_episodes)
    with open(a.frozen_params, "rb") as fh:
        fp_sha = hashlib.sha256(fh.read()).hexdigest()
    if fp_sha != FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{a.frozen_params} sha256 {fp_sha} != the r9 verdict's params_sha256 {FROZEN_PARAMS_SHA256}")
    frozen = read_json(a.frozen_params)
    h4 = frozen["criteria"]["H4"]
    ctrl = h4["control_sigma"][str(a.seed)]
    assert h4["episodes_per_seed"] == 20 and h4["spawn_range"] == [3042, 3061] and h4["space"] == "joint", h4

    t0 = time.monotonic()
    modules = isolate(a.clone)
    git = git_state(a.clone)
    if git["status"] or git["head"] != a.expect_head:
        raise SystemExit(f"the judge runs the clean clone at {a.expect_head} (got {git['head']}, dirty={bool(git['status'])})")
    print(f"[judge_r10] clone {a.clone} HEAD {git['head']} ({git['branch']}); environments = {modules['environments']}", flush=True)
    import torch

    torch.set_num_threads(1)
    install_trace_capture()
    ctx = Ctx()
    record = {
        "schema": SCHEMA, "judge": os.path.realpath(__file__), "judge_sha256": sha256_file(os.path.realpath(__file__)),
        "argv": sys.argv[1:], "clone": os.path.realpath(a.clone), "git": git, "expect_head": a.expect_head,
        "modules": modules, "python": sys.version.split()[0],
        "run_dir": run, "stage_dir": stage_dir, "seed": int(a.seed), "steps": int(a.steps), "checkpoint": stem,
        "model": zp, "model_sha256": sha256_file(zp), "vecnorm": vp, "vecnorm_sha256": sha256_file(vp),
        "smoke": bool(a.smoke_episodes), "noise_run": not a.no_noise,
        "stage": ctx.stage, "stage_gate_kind": ctx.curriculum.get("gate_kind"), "feet": ctx.feet, "settle_steps": ctx.settle,
        "horizon": ctx.horizon, "thresholds_declared": ctx.thresholds.declared(),
        "arm_env_kwargs": {k: ctx.env_kwargs.get(k) for k in ("lateral_lean_weight", "lateral_lean_tolerance_m", "leg_home_pose_reference")},
        "ppo_symmetry_keys": {k: v for k, v in ctx.ppo_kwargs.items() if k.startswith("symmetry")},
        "rails": {"foot_cop_bar": FOOT_COP_BAR, "pad_cop_bar": PAD_COP_BAR, "roll_tilt_deg": ROLL_TILT_DEG, "roll_lift_m": ROLL_LIFT_M},
        "panel_seeds": [PANEL_SEED, PANEL_SEED + PANEL_EPISODES - 1],
        "noise_seeds": [NOISE_SEED, NOISE_SEED + NOISE_EPISODES - 1],
        "plant_identity": ctx.plant_identity.to_dict(),
        "frozen_params": os.path.realpath(a.frozen_params), "frozen_params_sha256": fp_sha,
    }
    # Bound before anything is rolled: the task the panel will roll, derived as _stance_v2_report derives it.
    from environments.shared.task_fingerprint import stage_task_fingerprint

    panel_task = stage_task_fingerprint(ctx.mid, ctx.stage, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity)["task_sha256"]
    record["checks"] = run_bindings(ctx, a, stage_dir, zp, panel_task, git)
    print(json.dumps({"checks": {k: record["checks"].get(k) for k in ("plant_physics_sha256", "seed", "task", "records_missing")},
                      "symmetry_mirror_map_match": record["checks"]["symmetry"].get("mirror_map_sha256_match"),
                      "stage_config": {k: (record["checks"].get("stage_config") or {}).get(k) for k in
                                       ("env_kwargs_match", "gate_bars_match", "symmetry_keys_match", "git_commit_match")}}, default=str), flush=True)
    if record["checks"]["hard_failures"]:
        raise SystemExit("refusing (nothing written): " + " | ".join(record["checks"]["hard_failures"]))
    record["panel"] = v2_panel(ctx, model_path=zp, vecnorm_path=vp)
    p = record["panel"]
    assert p["task_sha256"] == panel_task, (p["task_sha256"], panel_task)
    s = p["summary"]
    print(json.dumps({"panel": p["verdict"], "clean": p["n_clean"], "lcb": round(p["clean_lcb"], 4), "full": p["n_full_horizon"],
                      "hop_or_fall": p["hop_or_fall_episodes"], "foot_cop_fh": s["foot_cop_gt_bar_full_horizon"],
                      "pad_cop_all": s["cop_gt_bar_gate"], "roll_rail": s["roll_rail_full_horizon"],
                      "reward": round(p["mean_reward"], 1), "statue": round(p["statue_mean_reward"] or float("nan"), 1),
                      "seconds": round(p["panel_seconds"], 1)}), flush=True)
    if not a.no_noise:
        model, normalizer = load_checkpoint(ctx, zp, vp)
        record["noise_matched_own"] = noise_matched(ctx, model, normalizer, sigma_arg="own", sigma=None, sigma_source="own", space="policy")
        record["noise_matched_control_joint"] = noise_matched(ctx, model, normalizer, sigma_arg="control", sigma=ctrl["vector"],
                                                              sigma_source=ctrl["source"], space="joint")
        for key in ("noise_matched_own", "noise_matched_control_joint"):
            n = record[key]
            print(json.dumps({key: {"falls": n["falls"], "sigma_mean": round(n["sigma_mean"], 4), "foot_cop_all": n["foot_cop_gt_bar"],
                                    "pad_cop_all": n["cop_gt_bar_gate"], "seconds": round(n["panel_seconds"], 1)}}), flush=True)
    record["modules_all_under_clone"] = modules_under(a.clone)
    record["judge_seconds"] = time.monotonic() - t0
    with open(out, "x") as fh:
        json.dump(jsonable(record), fh, indent=1)
    print(json.dumps({"wrote": out, "seconds": round(record["judge_seconds"], 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
