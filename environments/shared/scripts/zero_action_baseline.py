"""Score the do-nothing policy on a stage, as the floor every run must clear.

Residual actions are centred on each model's home keyframe, so ``action = 0``
commands exactly the nominal stance.  On a passively stable plant that is a
real policy, and on a balance stage it can be a *strong* one: the T-Rex
stage-1 zero-action score beat both trained policies from the 2026-07-23/24
runs on reward, on mean-minus-std, and on full-horizon survival.  Six million
PPO steps produced something worse than ``np.zeros(21)`` and nothing in the
pipeline noticed, because no run was ever compared against this floor.

Run it before and after any plant or task change: the baseline re-anchors
automatically, so it stays a valid comparison across model revisions.  A
trained stage-1 policy that does not beat it on all three of reward,
mean-minus-std, and full-horizon share has not learned to balance.

It is also how the stage-1 reset noise gets calibrated.  Too little and the
plant survives on its own, leaving nothing for a policy to earn (T-Rex at
``reset_noise_scale = 0.05`` scores 93% full-horizon); too much and the task
stops being learnable.

Usage (``--model-id`` repeats; with none, the four anatomical models below)::

    python environments/shared/scripts/zero_action_baseline.py --model-id tyrannosaurus_rex anatomical
    python environments/shared/scripts/zero_action_baseline.py --stage 2 \
        --model-id tyrannosaurus_rex anatomical --model-id velociraptor_mongoliensis anatomical
    python environments/shared/scripts/zero_action_baseline.py --model-id tyrannosaurus_rex anatomical --sweep-noise
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import build_env, load_stage_config
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.gate_schema import STANCE_GATE_KINDS
from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND
from environments.shared.curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.result_bundle import read_bundle_status
from environments.shared.species_names import add_model_id_argument, model_display_name, parsed_model_ids

#: Legacy stage numbers this script accepts; files resolve via the manifest.
KNOWN_STAGES = frozenset({1, 2, 3})

#: The models scored when no ``--model-id`` is given: the panel this tool has
#: always measured, written out as models (decision D-D33) so its default
#: output does not change.
DEFAULT_MODEL_IDS = (
    ModelId("tyrannosaurus_rex", "anatomical"),
    ModelId("velociraptor_mongoliensis", "anatomical"),
    ModelId("brachiosaurus_altithorax", "anatomical"),
    ModelId("dibothrosuchus_elaphros", "anatomical"),
)

#: The baseline record: one per model under ``<log_base>/<species>/<embodiment>/zero_action_baselines/``
#: and a copy in the run directory.  v2 (decision D-D33) nests the results by
#: species, then embodiment (``results[<species>][<embodiment>]``), each
#: carrying both identity fields.
ZERO_ACTION_BASELINE_SCHEMA = "mesozoic.zero-action-baseline/v2"

NOISE_SWEEP = (0.01, 0.05, 0.10, 0.15, 0.20)


def score(env, episodes: int, seed: int, noise: float | None = None) -> dict:
    """Roll out the zero action for *episodes* and summarise the outcome."""
    if noise is not None:
        env.reset_noise_scale = noise
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    rewards: list[float] = []
    lengths: list[int] = []
    terminations: dict[str, int] = {}

    for index in range(episodes):
        env.reset(seed=seed + index)
        total, steps = 0.0, 0
        while True:
            _, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                break
        rewards.append(total)
        lengths.append(steps)

    rewards_arr = np.asarray(rewards)
    lengths_arr = np.asarray(lengths)
    horizon = env.max_episode_steps
    standing = lengths_arr >= horizon
    return {
        "reward_mean": float(rewards_arr.mean()),
        "reward_std": float(rewards_arr.std()),
        # The gate a trained policy must clear: a fat failure tail shows up here
        # while it hides in the mean.
        "reward_mean_minus_std": float(rewards_arr.mean() - rewards_arr.std()),
        # Conditioned on the statue not falling over.  This is the number a
        # stage-1 gate has to beat to mean anything, and it is much higher than
        # the unconditional mean: the reward pays per step, so the episodes the
        # do-nothing policy survives are worth far more than its average.  A
        # gate set above ``reward_mean`` but below this one is cleared by a
        # policy that has learned only "do not fall".
        #
        # ``None``, not NaN, when the statue never reaches the horizon: this
        # dict is serialised straight into the baseline records the notebook
        # writes to Drive, and NaN is not valid JSON.
        "reward_mean_standing": float(rewards_arr[standing].mean()) if standing.any() else None,
        "reward_std_standing": float(rewards_arr[standing].std()) if standing.any() else None,
        "n_standing": int(standing.sum()),
        "length_mean": float(lengths_arr.mean()),
        "full_horizon_share": float(standing.mean()),
        "terminations": terminations,
        "horizon": horizon,
        "episodes": int(rewards_arr.size),
        "reset_noise_scale": float(env.reset_noise_scale),
    }


def gate_margin(gate: float | None, floor: float | None) -> float | None:
    """Return the gate's margin over a measured floor when both exist."""
    if gate is None or floor is None:
        return None
    return float(gate - floor)


def _standing_text(result: dict) -> str:
    """Format the standing score, which is unavailable when nothing stood."""
    if result["n_standing"] == 0:
        return "-"
    return f"{result['reward_mean_standing']:.1f}"


def report(model_id: ModelId, stage: int, episodes: int, seed: int, sweep: bool) -> None:
    env = build_env(model_id, stage)
    try:
        label = f"{model_id} stage {stage}"
        if sweep:
            print(f"\n{label}: zero-action baseline vs reset_noise_scale ({episodes} episodes)")
            print(f"  {'noise':>7}{'reward':>12}{'mean-std':>11}{'standing':>11}{'ep length':>11}{'full-horizon':>14}")
            for noise in NOISE_SWEEP:
                result = score(env, episodes, seed, noise=noise)
                print(
                    f"  {noise:>7}{result['reward_mean']:>12.1f}{result['reward_mean_minus_std']:>11.1f}"
                    f"{_standing_text(result):>11}"
                    f"{result['length_mean']:>11.1f}{result['full_horizon_share']:>13.0%}"
                )
            return

        result = score(env, episodes, seed)
        print(f"\n{label}: zero-action baseline ({episodes} episodes, reset_noise={env.reset_noise_scale})")
        print(f"  reward             {result['reward_mean']:.2f} +/- {result['reward_std']:.2f}")
        print(f"  reward mean-std    {result['reward_mean_minus_std']:.2f}")
        standing = (
            "never reached the horizon"
            if result["n_standing"] == 0
            else f"{result['reward_mean_standing']:.2f} +/- {result['reward_std_standing']:.2f}"
        )
        print(
            f"  reward standing    {standing}"
            f"   ({result['n_standing']} of {result['episodes']} episodes reached the horizon)"
        )
        print(f"  episode length     {result['length_mean']:.1f} of {result['horizon']}")
        print(f"  full-horizon share {result['full_horizon_share']:.0%}")
        print(f"  terminations       {result['terminations']}")
        print("  a trained policy must beat all three of reward, mean-std, and full-horizon share")
        print("  and 'reward standing' is what it must beat to have learned more than 'do not fall'")
    finally:
        env.close()


def statue_stance_verdict(model_id: ModelId, stage: int, stage_cfg: dict, *, episodes: int, seed: int) -> dict:
    """The stance gate's verdict on the statue, for a stage that declares a stance kind.

    A stance gate is the opposite of a reward gate here: the statue is the
    quality ceiling a stance policy must MATCH, not a floor to beat
    (``curriculum/stance_gate.py``), so the check is that the statue PASSES
    the gate as configured -- the acceptance check every stance TOML needs,
    and for ``stance_quality/v2`` the >= 37/40-clean check on its declared
    bars.  The verdict is ``reporting.stance_report.build_stance_gate_report``
    with ``zero_action=True`` on the same seeds, so it is the one the
    pipeline would reach: the floor-truth panel for v2, the touch-duty panel
    for v1.

    A stage that declares the six-heading certificate (``stance_quality_heading/v1``,
    D-D32) is judged at its 0 degree cell only -- v2's decision on the statue
    under the same keys and bars, the maintainer's decision of 2026-10-11:
    the plant's heading invariance is covered by the tests (every plant
    declares the elliptic friction cone, and the turned statue repeats the
    unturned one to solver tolerance), so the notebook session does not pay
    the five other headings (about 6 minutes on the T. rex, 10 on the
    compsognathus); the full grid runs at certification.  The record names
    the cell it scored (``headings_deg``).
    """
    from environments.shared.reporting.stance_report import build_stance_gate_report

    kind = stage_cfg["curriculum_kwargs"].get("gate_kind")
    scored_cfg = stage_cfg
    if kind == STANCE_HEADING_GATE_KIND:
        # The certificate's 0 deg cell: the same block, scored by the v2 code it applies at each heading.
        scored_cfg = {
            **stage_cfg,
            "curriculum_kwargs": {**stage_cfg["curriculum_kwargs"], "gate_kind": STANCE_GATE_V2_KIND},
        }
    report = build_stance_gate_report(
        model_id, stage, stage_config=scored_cfg, zero_action=True, episodes=episodes, seed=seed
    )
    record = {
        "gate_kind": kind,
        "scored_gate_kind": report.get("scored_gate_kind"),
        "passed": bool(report["passed"]),
        "failures": list(report["failures"]),
    }
    if kind == STANCE_HEADING_GATE_KIND:
        record["headings_deg"] = [0.0]
    if "result" in report:
        record.update(
            {
                "n_clean": report["result"]["n_clean"],
                "n_episodes": report["result"]["n_episodes"],
                "clean_lcb": report["result"]["clean_lcb"],
            }
        )
    return record


def preflight(model_ids, *, stage: int, episodes: int, seed: int, model_id: ModelId, log_base, run_dir) -> None:
    """The notebook's pre-flight: judge each model's stage gate against its zero-action floor and print the table.

    Saves one record per model of *model_ids* under ``<log_base>/<species>/<embodiment>/zero_action_baselines/``
    and the table beside the record of *model_id* (the model the notebook trains), whose record is also copied
    into *run_dir* (what ``curriculum.baseline_watch`` reads) unless that run's bundle is complete.

    A stage that declares a stance kind (``gate_schema.STANCE_GATE_KINDS``) is judged by the stance gate's own
    verdict on the statue (:func:`statue_stance_verdict`), recorded under ``stance_gate``, rather than by its
    reward RAIL -- which the statue clears by design, so the old reading printed "FAILS — a statue clears this
    gate" for every stance stage while saying nothing about the stance criteria.
    """
    results: dict[ModelId, dict] = {}
    for name in model_ids:
        env = build_env(name, stage)
        try:
            result = score(env, episodes, seed)
        finally:
            env.close()

        stage_cfg = load_stage_config(name, stage)
        gate = stage_cfg["curriculum_kwargs"].get("min_avg_reward")
        stance_gate = None
        if stage_cfg["curriculum_kwargs"].get("gate_kind") in STANCE_GATE_KINDS:
            stance_gate = statue_stance_verdict(name, stage, stage_cfg, episodes=episodes, seed=seed)

        # A gate only means something if it sits above the floor a statue reaches.
        if stance_gate is not None:
            # ...except a stance gate, which the statue must PASS: it is the
            # quality ceiling, and a gate that refuses it is miscalibrated.
            verdict = (
                "OK — the statue passes the stance gate"
                if stance_gate["passed"]
                else "CHECK GATE — the stance gate refuses the statue"
            )
        elif gate is None:
            verdict = "NO GATE"
        elif gate <= result["reward_mean"]:
            verdict = "FAILS — a statue clears this gate"
        elif result["n_standing"] == 0:
            # The statue never reaches the horizon, so there is no standing floor to
            # compare against.  That is its own problem: a plant that falls over under
            # its own home controller is not ready for a balance stage.
            verdict = "CHECK PLANT — the statue never survives a full episode"
        elif gate <= result["reward_mean_standing"]:
            verdict = "WEAK — binds only against a falling statue"
        else:
            verdict = "OK"

        result.update(
            {
                **name.record(),
                "display_name": model_display_name(name),
                "stage": stage,
                "min_avg_reward": gate,
                "verdict": verdict,
                "margin_over_mean": gate_margin(gate, result["reward_mean"]),
                "margin_over_standing": gate_margin(gate, result["reward_mean_standing"]),
                **({} if stance_gate is None else {"stance_gate": stance_gate}),
                "env_kwargs": stage_cfg["env_kwargs"],
                "plant_identity": current_plant_identity(name).to_dict(),
            }
        )
        results[name] = result

    # ---- table ----------------------------------------------------------------
    header = f"{'model':<42}{'reward':>10}{'mean-std':>10}{'standing':>10}{'full-hz':>9}{'gate':>9}  verdict"
    lines = [
        f"zero-action baseline — stage {stage}, {episodes} episodes, seed {seed}",
        "",
        header,
        "-" * len(header),
    ]
    for name, r in results.items():
        gate_text = "—" if r["min_avg_reward"] is None else f"{r['min_avg_reward']:.0f}"
        standing_text = "—" if r["n_standing"] == 0 else f"{r['reward_mean_standing']:.1f}"
        lines.append(
            f"{model_display_name(name):<42}{r['reward_mean']:>10.1f}{r['reward_mean_minus_std']:>10.1f}"
            f"{standing_text:>10}{r['full_horizon_share']:>8.0%}"
            f"{gate_text:>9}  {r['verdict']}"
        )
    refused = [
        (name, r["stance_gate"]) for name, r in results.items() if r.get("stance_gate", {}).get("passed") is False
    ]
    for name, stance_gate in refused:
        lines += ["", f"{model_display_name(name)}: the {stance_gate['gate_kind']} gate refuses the statue:"]
        lines += [f"  - {failure}" for failure in stance_gate["failures"]]
    lines += [
        "",
        "A trained stage-1 policy must beat 'reward', 'mean-std' AND 'full-hz' to have",
        "learned to balance at all, and 'standing' to have learned more than 'do not fall'.",
    ]
    if any("stance_gate" in r for r in results.values()):
        lines.append(
            "A stance-gated stage is the exception: the statue is its quality ceiling and must PASS the stance gate."
        )
    report_text = "\n".join(lines)
    print(report_text)

    # ---- save to the log directory --------------------------------------------
    if log_base is None:
        print("\nLOG_BASE not defined — run the storage-configuration cell to save results.")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    captured_at = datetime.now().isoformat()

    def payload_for(names):
        nested: dict[str, dict[str, dict]] = {}
        for name in names:
            nested.setdefault(name.species, {})[name.embodiment] = results[name]
        return {
            "schema": ZERO_ACTION_BASELINE_SCHEMA,
            "captured_at": captured_at,
            "stage": stage,
            "episodes": episodes,
            "seed": seed,
            "results": nested,
        }

    for name in results:
        model_dir = name.path(Path(log_base)) / "zero_action_baselines"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{stamp}.json").write_text(
            json.dumps(payload_for([name]), indent=2, sort_keys=True, allow_nan=False)
        )
        print(f"\nSaved: {model_dir / (stamp + '.json')}")

    # The human-readable table covers every model measured, so it belongs with
    # the one this notebook is training.
    table_dir = model_id.path(Path(log_base)) / "zero_action_baselines"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / f"{stamp}.txt").write_text(report_text + "\n")

    # Drop a copy in the run directory too, so this run carries the calibration
    # its own gate was judged against — except in a run whose bundle is complete:
    # that bundle is immutable and keeps the copy it was sealed with (a rewrite,
    # with its new captured_at, would stop the bundle verifying).
    if run_dir is not None and model_id in results:
        if read_bundle_status(run_dir) == "complete":
            print(f"Not copied to run: {run_dir} holds a complete bundle, which keeps the copy it was sealed with")
        else:
            (Path(run_dir) / "zero_action_baseline.json").write_text(
                json.dumps(payload_for([model_id]), indent=2, sort_keys=True, allow_nan=False)
            )
            print(f"Copied to run: {Path(run_dir) / 'zero_action_baseline.json'}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_id_argument(parser, default=DEFAULT_MODEL_IDS, what="score")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=sorted(KNOWN_STAGES),
        help="legacy stage number, applied to every model (default 1)",
    )
    parser.add_argument("--episodes", type=int, default=30, help="episodes per measurement (default 30)")
    parser.add_argument(
        "--seed",
        type=int,
        default=PUBLICATION_SEED_START,
        help=f"first evaluation seed (default {PUBLICATION_SEED_START})",
    )
    parser.add_argument("--sweep-noise", action="store_true", help="sweep reset_noise_scale instead of one point")
    args = parser.parse_args(argv)

    for model_id in parsed_model_ids(parser, args):
        report(model_id, args.stage, args.episodes, args.seed, args.sweep_noise)


if __name__ == "__main__":
    main(sys.argv[1:])
