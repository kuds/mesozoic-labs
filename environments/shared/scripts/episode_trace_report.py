"""Trace single episodes of a checkpoint step by step: per actuator, what the policy asks for and what the plant does.

Report only.  For each episode seed it rolls the checkpoint on its stage's own
environment exactly as the stance judge does (the same reset seeds, frozen
VecNormalize statistics, ``predict(deterministic=True)`` and its clip), and
records at every step and for every actuator the raw Gaussian mean before
the clip, the command sent, ``data.ctrl``, the driven joint's position and
velocity, and the actuator force with its fraction of the force limit.  It
writes one CSV per episode and ``episode_trace.json``, a summary per episode
and per actuator over the stance gate's post-settle window: the window means
of the command and the raw mean, the shares of window steps at
``|command| >= 0.99`` (the gate's ``SATURATION_ABS``) and ``>= 0.9`` and with
``|raw mean| > 1``, the raw mean's extremes, the first step at the rail, and
the episode's return and length.  The library half is
:mod:`environments.shared.reporting.episode_trace`.

The default rollout is the judge's: the trace of ``--seed 3042`` is the first
episode of the stance panel.  Two flags make it a probe instead, and the
summary records both:

* ``--stochastic SEED`` samples the command from the policy's own Gaussian,
  ``clip(mean + exp(log_std) z)``, ``z`` from ``numpy.random.default_rng(SEED + i)``
  in episode ``i``: behaviour under the policy's own exploration noise;
* ``--hold NAME=VALUE`` (repeatable) replaces one actuator's command by a
  constant, e.g. ``--hold tail_3_pitch_act=0``: the counterfactual, while the
  raw mean still says what the policy would have commanded.

The output goes to ``--out-dir`` (default ``episode_trace_<checkpoint stem>``
in the current directory), never into the run: a result bundle's manifest
hashes every file under the run directory, so an ``--out-dir`` inside the
run directory of the checkpoint or its statistics (the nearest ancestor
holding ``provenance.json``) is refused.

Usage::

    python environments/shared/scripts/episode_trace_report.py \\
        --species tyrannosaurus_rex --embodiment anatomical \\
        --model path/to/run/01_stance/models/stage1_2097152_steps.zip \\
        --vecnorm path/to/run/01_stance/models/stage1_vecnormalize_2097152_steps.pkl \\
        --seed 3042 --episodes 3 --actuators tail --out-dir scratch/trace_s42_2.1M

    # the counterfactual: the same episodes with tail_3 held level
    python environments/shared/scripts/episode_trace_report.py \\
        --species tyrannosaurus_rex --embodiment anatomical \\
        --model ... --vecnorm ... --seeds 3042,3059 \\
        --hold tail_3_pitch_act=0 --out-dir scratch/trace_s42_2.1M_hold0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.constants import PUBLICATION_SEED_START  # noqa: E402
from environments.shared.species_names import add_model_arguments, parsed_model_id  # noqa: E402


def _seeds(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[int]:
    """The episode seeds: ``--seeds`` as given, else ``--episodes`` consecutive seeds from ``--seed``."""
    if args.seeds is not None:
        try:
            seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
        except ValueError:
            parser.error(f"--seeds must be comma-separated whole numbers, got {args.seeds!r}")
        if not seeds:
            parser.error("--seeds names no seed")
        if len(set(seeds)) != len(seeds):
            parser.error(f"--seeds repeats a seed: {args.seeds!r}")
        return seeds
    if args.episodes < 1:
        parser.error(f"--episodes must be at least 1, got {args.episodes}")
    return [args.seed + index for index in range(args.episodes)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--config", help="Explicit stage TOML path")
    parser.add_argument("--model", required=True, help="Path to a saved SB3 checkpoint (.zip)")
    parser.add_argument(
        "--vecnorm", help="Path to the checkpoint's VecNormalize .pkl (required unless --allow-unnormalized)"
    )
    parser.add_argument(
        "--allow-unnormalized",
        action="store_true",
        help="trace without VecNormalize statistics (the trace then describes a different policy)",
    )
    parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="trace a checkpoint that predates the plant contract; the summary records checkpoint_plant_validated=false",
    )
    parser.add_argument(
        "--seed", type=int, default=PUBLICATION_SEED_START, help="First episode seed (default: publication seed)"
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes from --seed (default 1)")
    parser.add_argument("--seeds", help="Comma-separated episode seeds, instead of --seed and --episodes")
    parser.add_argument(
        "--actuators",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="trace only the actuators whose name contains SUBSTRING (repeatable; default every actuator)",
    )
    parser.add_argument(
        "--stochastic",
        type=int,
        metavar="SEED",
        help="sample the command from the policy's own Gaussian, numpy default_rng(SEED + episode index): a probe",
    )
    parser.add_argument(
        "--hold",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="replace actuator NAME's command by VALUE (action units) on every step (repeatable): a probe",
    )
    parser.add_argument("--out-dir", type=Path, help="Where to write (default: episode_trace_<checkpoint stem> here)")
    args = parser.parse_args(argv)
    model_id = parsed_model_id(parser, args)
    seeds = _seeds(parser, args)

    from environments.shared.config import load_stage_config
    from environments.shared.reporting.episode_trace import (
        ActuatorHold,
        EpisodeTraceError,
        build_episode_trace,
        refuse_run_directory_output,
        render_episode_trace,
        write_episode_trace,
    )

    holds = []
    for text in args.hold:
        try:
            holds.append(ActuatorHold.parse(text))
        except ValueError as exc:
            parser.error(f"--hold: {exc}")
    out_dir = args.out_dir if args.out_dir is not None else Path.cwd() / f"episode_trace_{Path(args.model).stem}"
    try:
        # Before anything is loaded or rolled: a refused destination costs nothing.
        refuse_run_directory_output(out_dir, [args.model, args.vecnorm])
        stage_config = load_stage_config(model_id, args.stage, config_path=args.config)
        report, traces = build_episode_trace(
            model_id,
            args.stage,
            stage_config=stage_config,
            model_path=args.model,
            vecnorm_path=args.vecnorm,
            seeds=seeds,
            actuators=args.actuators,
            holds=holds,
            stochastic_seed=args.stochastic,
            allow_unnormalized=args.allow_unnormalized,
            allow_legacy_plant=args.allow_legacy_plant,
        )
    except EpisodeTraceError as exc:
        # The CLI boundary is where a diagnosable failure becomes an exit status.
        raise SystemExit(str(exc)) from exc
    print(render_episode_trace(report), end="")
    for path in write_episode_trace(out_dir, report, traces).values():
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
