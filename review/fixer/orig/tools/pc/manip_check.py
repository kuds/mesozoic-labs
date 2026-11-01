"""PC study: THE MANIPULATION CHECK (C8; PC criteria WHAT IS READ) and the phase-fed critic feed (report-only
readers).  Never trains; no optimizer step; writes only its record.

    PY -I -B manip_check.py --copy COPY [--approved APPROVED | --header-parent] --zip Z --pkl P
                            (--stage-config SC | --no-run-records) (--out DIR | --run-id RUN_ID --label LABEL)

THE CHECK, on one fetched pair (its zip and VecNormalize sidecar), on the measuring copy of HEAD (sealed; the runner's
preflight of measure_r10round.py: disk floor, seal and commit, the checkpoint's physics_revision 10, the run's records,
the RUN_ID and label binding, the output outside every run folder; then the seal again):
  BANK: the first 500 observations (k = 0 ... 499; fewer if an episode ends) of two deterministic episodes of the pair's
        own actor, reset seeds 3042 and 3043, the observation normalised with the pair's VecNormalize statistics frozen
        (the trace tool's load_trace_policy, as M1 loads a pair), k = the env's own _step_count at each observation
        (checked against the step index).
  (i)   PHASE IN FORCE: the critic's phase weights (value_net[0].weight[:, 63:66]) are not all zero, and for each of A1's
        five perturbations of the phase (all 0; all 1; +1; all 3; 1 - x) the largest |dV| over the bank is > 0.
  (ii)  ACTOR INDEPENDENCE: under the same perturbations the actor's deterministic mean changes by exactly 0.0 on every
        row, and equals the mean of the 63-value observation alone (the deploy interface).
  (iii) THE CRITIC READS OBSERVATION 6: the largest |dV| over the bank under a +1 shift of the normalised value of
        observation 6 (the tail_3 angle) is > 0.
  VERDICT: "IN FORCE" when (i), (ii) and (iii) hold, else "NOT IN FORCE" with the failing items (a checkpoint whose
  critic reads no phase at all is "NOT IN FORCE").  A crash writes no record (C8 then stays PENDING; rerun on the same
  files).
THE FEED (critic_values): V of a bank with the phase of its step counters appended (critic_phase.CriticPhaseSpec.matrix,
float32 as the training env appends it); "effective" = phi(k), "settled" = phi(1000).  The report-only critic readers
(slope on the tail angle, V - G by step bucket) call it; a stock critic is fed the bank alone.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent
VERSION = "manip_check pc-1 (2026-10-31; PC criteria C8 THE MANIPULATION CHECK; phase-fed critic feed)"
SCHEMA = "pc-manip-check/1"
SEEDS = (3042, 3043)
BANK_LIMIT = 500
ACTOR_DIM = 63
OBS_TAIL3_ANGLE = 6
PERTURBATIONS = ("all 0", "all 1", "+1", "all 3", "1 - x")
SETTLED_K = 1000


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------- pure functions
def perturb(phase, how: str):
    """A1's five perturbations of an (n, 3) phase matrix (numpy)."""
    import numpy as np

    if how == "all 0":
        return np.zeros_like(phase)
    if how == "all 1":
        return np.ones_like(phase)
    if how == "+1":
        return phase + 1
    if how == "all 3":
        return np.full_like(phase, 3)
    if how == "1 - x":
        return 1 - phase
    raise ValueError(how)


def critic_phase_spec(policy):
    """The policy's CriticPhaseSpec, or None for a stock critic."""
    return getattr(policy, "critic_phase", None)


def phase_matrix(spec, ks, dtype="float32"):
    import numpy as np

    return np.asarray(spec.matrix(np.asarray(ks, dtype=np.int64)), dtype=dtype)


def critic_values(policy, bank, ks=None, *, phase=None, mode="effective"):
    """V over a bank of normalised 63-value observations (float32): a phase critic is fed [bank, phase] with phase =
    phi(k) ("effective"), phi(1000) ("settled") or the given matrix; a stock critic the bank alone.  float64 out."""
    import numpy as np
    import torch as th

    spec = critic_phase_spec(policy)
    x = np.asarray(bank, dtype=np.float32)
    if spec is not None:
        if phase is None:
            kk = np.asarray(ks if mode == "effective" else [SETTLED_K] * len(x), dtype=np.int64)
            phase = phase_matrix(spec, kk)
        x = np.concatenate([x, np.asarray(phase, dtype=np.float32)], axis=1)
    with th.no_grad():
        v = policy.predict_values(th.as_tensor(x, device=policy.device))
    return v.detach().cpu().numpy().astype(np.float64).reshape(-1)


def actor_means(policy, bank, phase=None):
    """The actor's deterministic (Gaussian) mean over the bank, with the phase appended when given (float64 out)."""
    import numpy as np
    import torch as th

    x = np.asarray(bank, dtype=np.float32)
    if phase is not None:
        x = np.concatenate([x, np.asarray(phase, dtype=np.float32)], axis=1)
    with th.no_grad():
        d = policy.get_distribution(th.as_tensor(x, device=policy.device))
        mu = d.distribution.mean
    return mu.detach().cpu().numpy().astype(np.float64)


def manipulation_checks(policy, bank, ks) -> dict:
    """(i)-(iii) and the verdict on a bank (module docstring).  policy: an SB3 ActorCriticPolicy."""
    import numpy as np

    spec = critic_phase_spec(policy)
    out = dict(spec=None if spec is None else spec.record(), n_rows=int(len(bank)))
    if spec is None:
        out.update(checks={"(i) phase in force": dict(holds=False, why="the critic reads no phase (a stock critic)")},
                   verdict="NOT IN FORCE", failing=["(i)"])
        return out
    w = policy.mlp_extractor.value_net[0].weight.detach().cpu().numpy()
    w_phase = w[:, ACTOR_DIM:]
    actor_in = int(policy.mlp_extractor.policy_net[0].in_features)
    base_phase = phase_matrix(spec, ks)
    v0 = critic_values(policy, bank, phase=base_phase)
    mu0 = actor_means(policy, bank, base_phase)
    mu63 = actor_means(policy, bank, None)
    dv, dmu = {}, {}
    for how in PERTURBATIONS:
        p = perturb(base_phase, how)
        dv[how] = float(np.max(np.abs(critic_values(policy, bank, phase=p) - v0)))
        dmu[how] = float(np.max(np.abs(actor_means(policy, bank, p) - mu0)))
    shifted = np.array(bank, dtype=np.float32, copy=True)
    shifted[:, OBS_TAIL3_ANGLE] += 1
    dv_obs6 = float(np.max(np.abs(critic_values(policy, shifted, phase=base_phase) - v0)))
    i_holds = bool(np.any(w_phase != 0)) and all(v > 0 for v in dv.values())
    ii_holds = all(v == 0.0 for v in dmu.values()) and bool(np.array_equal(mu63, mu0)) and actor_in == ACTOR_DIM
    iii_holds = dv_obs6 > 0
    checks = {
        "(i) phase in force": dict(holds=i_holds, phase_weights_nonzero=int(np.count_nonzero(w_phase)),
                                   phase_weights_shape=list(w_phase.shape),
                                   phase_weights_max_abs=float(np.max(np.abs(w_phase))) if w_phase.size else None,
                                   max_abs_dV_by_perturbation=dv),
        "(ii) actor independent": dict(holds=ii_holds, max_abs_dmu_by_perturbation=dmu,
                                       mean_63_equals_mean_with_phase=bool(np.array_equal(mu63, mu0)),
                                       actor_first_layer_inputs=actor_in),
        "(iii) critic reads observation 6": dict(holds=iii_holds, max_abs_dV_obs6_plus1=dv_obs6),
    }
    failing = [k.split(" ")[0] for k, v in checks.items() if not v["holds"]]
    out.update(checks=checks, verdict="IN FORCE" if not failing else "NOT IN FORCE", failing=failing,
               v_effective_summary=dict(mean=float(np.mean(v0)), min=float(np.min(v0)), max=float(np.max(v0))))
    return out


def widen_stock_policy(stock_policy, settle_steps: int, gamma: float):
    """A CriticPhaseActorCriticPolicy holding a stock policy's weights exactly, its critic's phase columns zero (main's
    widening rule for a new input), built on the measuring copy's classes (in memory; nothing saved)."""
    import torch as th
    from environments.shared.critic_phase_ppo import CriticPhaseActorCriticPolicy

    params = dict(stock_policy._get_constructor_parameters())
    params.update(critic_phase_settle_steps=int(settle_steps), critic_phase_gamma=float(gamma))
    params["lr_schedule"] = lambda _p: 0.0
    wide = CriticPhaseActorCriticPolicy(**params)
    sd = {k: v.clone() for k, v in stock_policy.state_dict().items()}
    key = "mlp_extractor.value_net.0.weight"
    w = sd[key]
    sd[key] = th.cat([w, th.zeros((w.shape[0], wide.mlp_extractor.critic_phase_width), dtype=w.dtype)], dim=1)
    wide.load_state_dict(sd, strict=True)
    wide.set_training_mode(False)
    return wide


def collect_bank(policy_obj, env, seeds=SEEDS, limit=BANK_LIMIT):
    """(bank (n, 63) float32 normalised, ks, per-episode rows): the first `limit` observations of deterministic episodes
    of the pair's own actor (TracePolicy.observation's frozen normalisation; predict deterministic)."""
    import numpy as np

    rows, ks, per = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        k, n0 = 0, len(rows)
        ended = None
        while k < limit:
            sc = int(env.unwrapped._step_count)
            if sc != k:
                raise RuntimeError(f"seed {seed}: the env's _step_count {sc} != the step index {k}")
            x = policy_obj.observation(obs)
            rows.append(np.asarray(x, dtype=np.float32).reshape(-1))
            ks.append(k)
            action = np.asarray(policy_obj.model.predict(x, deterministic=True)[0])
            obs, _r, term, trunc, _info = env.step(action)
            k += 1
            if term or trunc:
                ended = "terminated" if term else "truncated"
                break
        per.append(dict(seed=int(seed), rows=len(rows) - n0, k_first=0, k_last=k - 1, ended_before_limit=ended))
    return np.stack(rows).astype(np.float32), np.asarray(ks, dtype=np.int64), per


# ---------------------------------------------------------------------------------------------------- the runner
def main(argv=None) -> int:
    MR = _load("measure_r10round")
    R = MR.C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--header-parent", action="store_true")
    ap.add_argument("--approved", default=None)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--stage-config")
    ap.add_argument("--no-run-records", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--run-id")
    ap.add_argument("--label")
    ns = ap.parse_args(argv)
    summary = "manip_check.json"
    try:
        if ns.stage_config and ns.no_run_records:
            raise R.Refusal("--stage-config and --no-run-records exclude each other")
        pf = MR.preflight(ns, "c8", checkpoint=True, pins=None, summary_name=summary)
        copy = Path(ns.copy).resolve()
        sys.path.insert(0, str(copy))
        import environments

        if not Path(environments.__file__).resolve().is_relative_to(copy):
            raise R.Refusal(f"environments imported from {environments.__file__}, not the measuring copy {copy}")
        from environments.shared.config import load_all_stages
        from environments.shared.model_id import ModelId
        from environments.shared.plant_contract import current_plant_identity
        from environments.shared.reporting.episode_trace import load_trace_policy
        from environments.shared.species_registry import env_class as model_env_class
        from environments.shared.stage_manifest import load_stage_manifest

        model_id = ModelId("tyrannosaurus_rex", "anatomical")
        stage = load_stage_manifest(model_id).resolve_behavior("stance").reference
        env_kwargs = dict(load_all_stages(model_id)[stage]["env_kwargs"])
        env_class = model_env_class(model_id)
        tp = load_trace_policy(ns.zip, ns.pkl, lambda: env_class(**env_kwargs),
                               plant_identity=current_plant_identity(model_id))
        env = env_class(**env_kwargs)
        try:
            bank, ks, per = collect_bank(tp, env)
        finally:
            env.close()
        res = manipulation_checks(tp.model.policy, bank, ks)
        after = MR.recheck_copy(ns, pf["seal"])
        out = pf["out"]
        out.mkdir(parents=True, exist_ok=True)
        rec = {"schema": SCHEMA, "tool": os.path.realpath(__file__), "tool_sha256": R.sha256_file(__file__),
               "version": VERSION, "utc": R.utc_now(), "kind": "C8 MANIPULATION CHECK",
               "physics_revision": (pf["checkpoint"] or {}).get("physics_revision"), "checkpoint": pf["checkpoint"],
               "run_records": pf["run_records"], "filed": pf.get("filed"), "copy_seal": pf["seal"],
               "copy_commit_basis": pf.get("copy_commit"), "copy_tree_digest_after": after["tree_digest"],
               "seeds": list(SEEDS), "bank_limit": BANK_LIMIT, "bank_episodes": per, "bank_rows": int(len(bank)),
               "normalization": tp.normalization, "stack": R.stack_versions(), **res}
        R.write_json_x(out / summary, rec)
        print(json.dumps({k: rec[k] for k in ("verdict", "failing", "bank_rows", "bank_episodes", "checks")},
                         indent=1, default=str))
        print(f"[manip_check] wrote {out / summary}")
        return 0
    except R.Refusal as exc:
        print(f"[manip_check] REFUSED: {exc.message}", file=sys.stderr, flush=True)
        return int(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
