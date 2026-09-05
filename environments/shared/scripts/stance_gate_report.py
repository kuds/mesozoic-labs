"""Score a saved checkpoint against its stage's ``stance_quality/v1`` gate.

Answers one question: **would this policy advance?** It rolls the checkpoint
on the stage's own configured environment, reduces the episodes with the same
:mod:`~environments.shared.curriculum.stance_gate` code the training loop runs,
and prints the verdict criterion by criterion.

Why this exists as a script rather than a notebook cell: the gate's verdict is
the thing runs are judged on, and a verdict computed by hand-rolled
reimplementation is not the same verdict.  Everything here funnels through
``summarize_stance_panel`` and ``evaluate_stance_gate``, so what this prints is
what the curriculum would decide.

It also covers the case the training loop cannot: **an already-finished run**.
Run ``20260801_203206`` completed stage 1 under the old ``reward_and_length/v1``
gate and advanced on reward 2313.2 and length 1000 -- both of which a
zero-action statue clears.  Whether it would have passed the stance gate is a
question only a rollout can answer, and the checkpoint is a 4 MB binary that
lives next to the run, not in the repository.

Usage::

    python environments/shared/scripts/stance_gate_report.py trex \\
        --model  path/to/robust_best_model.zip \\
        --vecnorm path/to/robust_best_model_vecnorm.pkl

    # the do-nothing reference, for the same panel and seeds
    python environments/shared/scripts/stance_gate_report.py trex --zero-action

    # probe: does the pose this policy holds need feedback, or only a constant?
    python environments/shared/scripts/stance_gate_report.py trex \\
        --model path/to/robust_best_model.zip \\
        --vecnorm path/to/robust_best_model_vecnorm.pkl \\
        --hold-constant

Thresholds come from the stage TOML, so the report re-anchors automatically
when the gate is retuned.  ``--episodes`` defaults to the stage's own
``min_eval_episodes``, because that is the panel size the bound's power is
specified at; overriding it downward makes the bound weaker than the gate
claims, and the report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.constants import PUBLICATION_SEED_START  # noqa: E402
from environments.shared.reporting.stance_report import (  # noqa: E402
    _IMPULSE_SPEEDS,
    REPORT_SCHEMA,
    STANCE_PANEL_FIELDNAMES,
    ConstantHold,
    RootImpulse,
    StanceGateReportError,
    actuator_indices_matching,
    build_stance_gate_report,
    constant_hold_actions,
    constant_hold_release_variants,
    constant_hold_released,
    constant_hold_variants,
    impulse_variants,
    probe_stem,
    render_constant_hold_ablation,
    render_constant_hold_probe,
    render_impulse_probe,
    render_stance_gate_report,
    run_panel,
    saturated_actuator_indices,
    write_action_filter_sweep,
    write_constant_hold_ablation,
    write_constant_hold_probe,
    write_impulse_probe,
    write_stance_gate_report,
    write_stance_panel_evidence,
)

#: The implementation lives in :mod:`environments.shared.reporting.stance_report`
#: (library code, so the artifact path can call it without importing a
#: script).  Re-exported so ``from environments.shared.scripts.stance_gate_report
#: import ...`` keeps working for callers that address the report by its
#: script path.
__all__ = [
    "REPORT_SCHEMA",
    "STANCE_PANEL_FIELDNAMES",
    "ConstantHold",
    "RootImpulse",
    "StanceGateReportError",
    "actuator_indices_matching",
    "build_stance_gate_report",
    "constant_hold_actions",
    "constant_hold_release_variants",
    "constant_hold_released",
    "constant_hold_variants",
    "impulse_variants",
    "probe_stem",
    "render_constant_hold_ablation",
    "render_constant_hold_probe",
    "render_impulse_probe",
    "render_stance_gate_report",
    "run_panel",
    "saturated_actuator_indices",
    "write_action_filter_sweep",
    "write_constant_hold_ablation",
    "write_constant_hold_probe",
    "write_impulse_probe",
    "write_stance_gate_report",
    "write_stance_panel_evidence",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species", help="Species id, e.g. trex")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--model", help="Path to a saved SB3 checkpoint (.zip)")
    parser.add_argument("--vecnorm", help="Path to the checkpoint's VecNormalize .pkl")
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help="Score the do-nothing policy instead of a checkpoint (the reference the gate is calibrated against)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Defaults to the stage's min_eval_episodes")
    parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="Score a checkpoint that predates the plant contract. The verdict is then about a plant "
        "that may differ from the one in the tree, and the report records plant_validated=false.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PUBLICATION_SEED_START,
        help="First evaluation seed (default: publication seed)",
    )
    parser.add_argument("--config", help="Explicit stage TOML path")
    parser.add_argument("--out-dir", help="Also write stance_gate_report.{txt,json} to this directory")
    parser.add_argument(
        "--filter-actions",
        type=float,
        metavar="HZ",
        help="Low-pass the policy's actions at HZ before they reach the plant, and score THAT. "
        "A probe for whether a policy's high-frequency action content is load-bearing: if it "
        "still stands, the tremor was waste and belongs on the action path; if it falls, the "
        "tremor is closed-loop stabilisation and the fix belongs elsewhere (issue #489).",
    )
    parser.add_argument(
        "--hold-constant",
        action="store_true",
        help="Replace the policy's action with the constant it commands ON AVERAGE partway through "
        "each episode, and score THAT, over a standard variant set with two controls. Separates "
        "'the pose needs feedback' from 'the tremor is waste the action penalties missed' -- the "
        "one question the low-pass probe cannot answer, because a filtered policy still responds.",
    )
    parser.add_argument(
        "--hold-from-report",
        metavar="JSON",
        help="Take the constant from an existing stance_gate_report.json instead of measuring it. "
        "Skips the measurement panel when you already have one for this checkpoint.",
    )
    parser.add_argument(
        "--hold-episodes",
        type=int,
        default=10,
        help="Episodes per hold variant (default 10). The probe certifies nothing and the effect "
        "it measures is a fall or a full horizon, so it does not need the gate's panel size.",
    )
    parser.add_argument(
        "--hold-ramp-steps",
        type=int,
        default=50,
        help="Steps to blend into the constant in the ramped variants (default 50).",
    )
    parser.add_argument(
        "--hold-release-ablation",
        action="store_true",
        help="Instead of the standard hold variants, ablate the held pose one actuator group at a "
        "time -- releasing each group back to the home control, and holding each group alone. "
        "Answers WHICH joints make the pose unholdable, by testing necessity and sufficiency "
        "separately. Implies --hold-constant.",
    )
    parser.add_argument(
        "--impulse-probe",
        action="store_true",
        help="Shove the animal mid-episode and see whether it recovers, against the zero-action "
        "statue as the control. Stage 1 declares no disturbance, so nothing in a training "
        "artifact can say whether a policy learned to CORRECT or only to stand still; this can. "
        "The statue cannot respond at all, so only the policy's margin over it is active control.",
    )
    parser.add_argument(
        "--impulse-speeds",
        default=",".join(str(speed) for speed in _IMPULSE_SPEEDS),
        help=f"Comma-separated impulse magnitudes in m/s (default {','.join(str(s) for s in _IMPULSE_SPEEDS)}). "
        "Each is run in both lateral directions.",
    )
    parser.add_argument(
        "--impulse-step",
        type=int,
        default=None,
        help="Step at which the impulse lands (default: the stage's settle_steps, so the policy "
        "has established its own state first).",
    )
    parser.add_argument(
        "--impulse-episodes",
        type=int,
        default=8,
        help="Episodes per impulse row (default 8).",
    )
    args = parser.parse_args()
    args.hold_constant = args.hold_constant or args.hold_release_ablation

    if not args.zero_action and not args.model:
        parser.error("pass --model, or --zero-action for the do-nothing reference")
    if args.hold_constant and args.filter_actions is not None:
        parser.error("--hold-constant and --filter-actions are different probes; run them separately")
    if args.hold_episodes < 1:
        parser.error(f"--hold-episodes must be at least 1, got {args.hold_episodes}")
    if args.impulse_episodes < 1:
        parser.error(f"--impulse-episodes must be at least 1, got {args.impulse_episodes}")
    if args.impulse_probe and args.zero_action:
        # The statue is this probe's CONTROL and is rolled automatically; making
        # it the subject too would compare it against itself and report a margin
        # of zero as if it meant something.
        parser.error("--impulse-probe rolls the statue as its own control; pass --model, not --zero-action")
    try:
        impulse_speeds = [float(part) for part in args.impulse_speeds.split(",") if part.strip()]
    except ValueError:
        parser.error(f"--impulse-speeds must be comma-separated numbers, got {args.impulse_speeds!r}")
    if args.impulse_probe and not any(speed > 0 for speed in impulse_speeds):
        parser.error("--impulse-speeds needs at least one positive magnitude")
    if args.hold_ramp_steps < 0:
        parser.error(f"--hold-ramp-steps cannot be negative, got {args.hold_ramp_steps}")
    if args.filter_actions is not None and args.filter_actions <= 0:
        parser.error(f"--filter-actions must be positive, got {args.filter_actions}")
    # `--episodes 0` used to fall through `episodes or min_eval_episodes` to the
    # stage default AND skip the under-powered-panel warning below, so it
    # silently produced a full-size panel while reading as an override.
    if args.episodes is not None and args.episodes < 1:
        parser.error(f"--episodes must be at least 1, got {args.episodes}")

    stage_config = load_stage_config(args.species, args.stage, config_path=args.config)
    try:
        report = build_stance_gate_report(
            args.species,
            args.stage,
            stage_config=stage_config,
            model_path=args.model,
            vecnorm_path=args.vecnorm,
            zero_action=args.zero_action,
            episodes=args.episodes,
            seed=args.seed,
            allow_legacy_plant=args.allow_legacy_plant,
            filter_actions_hz=args.filter_actions,
        )
    except StanceGateReportError as exc:
        # The CLI boundary is where a diagnosable failure becomes an exit
        # status; inside the library it stays an ordinary exception.
        raise SystemExit(str(exc)) from exc
    print()
    print(render_stance_gate_report(report))

    if args.out_dir:
        for path in write_stance_gate_report(args.out_dir, report).values():
            print(f"written: {path}")

    if args.hold_constant:
        try:
            hold = (
                constant_hold_actions(json.loads(Path(args.hold_from_report).read_text(encoding="utf-8")))
                if args.hold_from_report
                else constant_hold_actions(report)
            )
        except StanceGateReportError as exc:
            raise SystemExit(str(exc)) from exc
        source_report = (
            json.loads(Path(args.hold_from_report).read_text(encoding="utf-8")) if args.hold_from_report else report
        )
        variants = (
            constant_hold_release_variants(
                hold,
                source_report,
                horizon=report["horizon"],
                ramp_steps=args.hold_ramp_steps,
            )
            if args.hold_release_ablation
            else constant_hold_variants(
                hold,
                settle_steps=report["settle_steps"],
                horizon=report["horizon"],
                ramp_steps=args.hold_ramp_steps,
            )
        )
        probes = [
            build_stance_gate_report(
                args.species,
                args.stage,
                stage_config=stage_config,
                model_path=args.model,
                vecnorm_path=args.vecnorm,
                zero_action=args.zero_action,
                episodes=args.hold_episodes,
                seed=args.seed,
                allow_legacy_plant=args.allow_legacy_plant,
                hold_constant=variant,
            )
            for variant in variants
        ]
        render = render_constant_hold_ablation if args.hold_release_ablation else render_constant_hold_probe
        write = write_constant_hold_ablation if args.hold_release_ablation else write_constant_hold_probe
        text, _ = render(probes, probe_episodes=args.hold_episodes)
        print()
        print(text, end="")
        if args.out_dir:
            for path in write(args.out_dir, probes, probe_episodes=args.hold_episodes).values():
                print(f"written: {path}")

    if args.impulse_probe:
        step = report["settle_steps"] if args.impulse_step is None else args.impulse_step
        # Its own name: `variants` above holds ConstantHold entries, and
        # reusing it for RootImpulse entries conflates the two probe types.
        impulse_sweep = impulse_variants(impulse_speeds, step=step)

        def _sweep(zero_action: bool) -> list[dict[str, Any]]:
            return [
                build_stance_gate_report(
                    args.species,
                    args.stage,
                    stage_config=stage_config,
                    model_path=None if zero_action else args.model,
                    vecnorm_path=None if zero_action else args.vecnorm,
                    zero_action=zero_action,
                    episodes=args.impulse_episodes,
                    seed=args.seed,
                    allow_legacy_plant=args.allow_legacy_plant,
                    impulse=variant,
                )
                for variant in impulse_sweep
            ]

        # Rolled once and reused: each sweep is a few thousand simulated steps,
        # and calling the helper twice per output would silently double the cost
        # of the probe for nothing.
        policy_sweep = _sweep(zero_action=False)
        statue_sweep = _sweep(zero_action=True)
        text, _ = render_impulse_probe(policy_sweep, statue_sweep, probe_episodes=args.impulse_episodes)
        print()
        print(text, end="")
        if args.out_dir:
            written = write_impulse_probe(
                args.out_dir, policy_sweep, statue_sweep, probe_episodes=args.impulse_episodes
            )
            for path in written.values():
                print(f"written: {path}")

    declared = report["thresholds"]["min_eval_episodes"]
    if args.episodes is not None and args.episodes != declared:
        print()
        print(
            f"WARNING: --episodes {args.episodes} differs from the stage's min_eval_episodes "
            f"{declared}. The bound's power is specified at the latter; this panel does not "
            "certify what the gate claims."
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
