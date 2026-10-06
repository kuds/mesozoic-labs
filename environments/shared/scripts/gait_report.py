"""Measure a checkpoint's physical gait without rewriting its original run.

python -m environments.shared.scripts.gait_report trex --stage locomotion \
    --model RUN/models/robust_best_model.zip --vecnorm RUN/models/robust_best_model_vecnorm.pkl \
    --out-dir FRESH_DIRECTORY --episodes 10 --seed 9000

Every episode saves a substep trace. Existing reward-gated stages produce
development reports only, by default 10 episodes from the development block
(seed 9000); a report-only panel never rolls the reserved certification block
3042-3081. Enforcement requires an explicitly configured locomotion_gait/v2
with a pinned protocol and fixed certification panel, which is the default
panel for such a config. Exit 2 is a usage error (including an invalid
number or protocol option), 3 an evaluation refusal; an invalid argument
never touches an existing report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from environments.shared.config import load_stage_config
from environments.shared.constants import PUBLICATION_PANEL_EPISODES, PUBLICATION_SEED_START
from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND
from environments.shared.file_io import read_json_object
from environments.shared.gait.identity import (
    DEFAULT_DIRECTION_XY,
    DEFAULT_SETTLE_S,
    normalized_direction,
    protocol_sha256,
    stage_measurement_protocol,
)
from environments.shared.gait.report import stage_panel, write_gait_report
from environments.shared.gait.types import GaitProtocol
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import load_stage_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species")
    parser.add_argument("--stage", default="locomotion")
    controller = parser.add_mutually_exclusive_group(required=True)
    controller.add_argument("--model")
    controller.add_argument("--zero-action", action="store_true")
    controller.add_argument(
        "--protocol-only", action="store_true", help="Print the planned hash without rolling episodes"
    )
    parser.add_argument("--vecnorm")
    parser.add_argument("--config", help="Explicit stage TOML")
    parser.add_argument("--env-json", help="Frozen task env kwargs, e.g. downloaded replay_kwargs.json")
    parser.add_argument("--protocol-json", help="Explicit GaitProtocol option object; defaults are provisional")
    parser.add_argument("--episodes", type=int, help="Panel size (default: the stage's declared panel, else 10)")
    parser.add_argument("--seed", type=int, help="First panel seed (default: the declared panel, else 9000)")
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--direction", type=float, nargs=2, default=DEFAULT_DIRECTION_XY, metavar=("X", "Y"))
    parser.add_argument("--allow-legacy-plant", action="store_true", help="Legacy diagnostic only; can never certify")
    parser.add_argument("--out-dir", help="Fresh evaluation directory, outside a completed source bundle")
    args = parser.parse_args(argv)
    if not args.protocol_only and not args.out_dir:
        parser.error("--out-dir is required for a gait evaluation")
    # Pure arguments are usage errors (exit 2), checked before anything is
    # loaded or written: a typo never replaces an existing report.
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be a positive integer")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a nonnegative integer")
    if not math.isfinite(args.settle_s) or args.settle_s < 0:
        parser.error("--settle-s must be finite and nonnegative")
    try:
        direction = normalized_direction(args.direction)
    except ValueError:
        parser.error("--direction must be a finite nonzero 2-vector")
    protocol = GaitProtocol()
    if args.protocol_json:
        try:
            options = read_json_object(args.protocol_json)
        except (OSError, ValueError) as error:
            parser.error(f"--protocol-json cannot be read: {error}")
        if options is None:
            parser.error("--protocol-json must contain a detector options object")
        try:
            protocol = GaitProtocol(**options)
        except (TypeError, ValueError) as error:
            parser.error(f"--protocol-json is not a valid detector option object: {error}")
    try:
        species_cfg = get_species_config(args.species)
        stage = load_stage_manifest(species_cfg.species).resolve(args.stage).reference
        config = load_stage_config(species_cfg.species, stage, config_path=args.config)
        config["_gait_stage"] = stage
        if args.env_json:
            value = read_json_object(args.env_json)
            if value is None:
                raise ValueError("--env-json must contain an environment kwargs object")
            config["env_kwargs"] = value
    except Exception as error:
        print(f"Gait evaluation refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    curriculum = config.get("curriculum_kwargs", {})
    default_episodes, default_seed = stage_panel(curriculum)
    episodes = args.episodes if args.episodes is not None else default_episodes
    seed = args.seed if args.seed is not None else default_seed
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        parser.error("--episodes is required: the stage config declares no usable panel size")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        parser.error("--seed is required: the stage config declares no usable panel seed start")
    report_only = curriculum.get("gate_kind") != GAIT_GATE_KIND
    if (
        report_only
        and not args.protocol_only
        and seed < PUBLICATION_SEED_START + PUBLICATION_PANEL_EPISODES
        and seed + episodes > PUBLICATION_SEED_START
    ):
        parser.error(
            f"a report-only panel never rolls the reserved certification block {PUBLICATION_SEED_START}-"
            f"{PUBLICATION_SEED_START + PUBLICATION_PANEL_EPISODES - 1}; choose a disjoint --seed"
        )
    try:
        if args.protocol_only:
            payload = stage_measurement_protocol(
                species_cfg,
                config,
                episodes=episodes,
                seed_start=seed,
                protocol=protocol,
                settle_s=args.settle_s,
                direction_xy=direction,
            )
            print(
                json.dumps(
                    {"measurement_protocol_sha256": protocol_sha256(payload), "measurement_protocol": payload}, indent=2
                )
            )
            return 0
        report = write_gait_report(
            species_cfg,
            config,
            args.model,
            args.vecnorm,
            args.out_dir,
            episodes=episodes,
            seed=seed,
            protocol=protocol,
            settle_s=args.settle_s,
            direction_xy=tuple(args.direction),
            allow_legacy_plant=args.allow_legacy_plant,
        )
    except Exception as error:
        print(f"Gait evaluation refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "report_only": report["report_only"],
                "certified": report["certified"],
                "statistics": report["statistics"],
                "measurement_protocol_sha256": report["measurement_protocol_sha256"],
                "failure_counts": report["episode_failure_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
