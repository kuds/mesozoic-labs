"""Measure a checkpoint's physical gait without rewriting its original run.

python -m environments.shared.scripts.gait_report trex --stage locomotion \
    --model RUN/models/robust_best_model.zip --vecnorm RUN/models/robust_best_model_vecnorm.pkl \
    --out-dir FRESH_DIRECTORY --episodes 40 --seed 104042

Every episode saves a substep trace. Existing reward-gated stages produce
development reports only. Enforcement requires an explicitly configured
locomotion_gait/v2 with a pinned protocol and fixed certification panel.
"""

from __future__ import annotations

import argparse
import json
import sys

from environments.shared.config import load_stage_config
from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.file_io import read_json_object
from environments.shared.gait.identity import measurement_protocol, protocol_sha256
from environments.shared.gait.report import write_gait_report
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
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--direction", type=float, nargs=2, default=(1.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--allow-legacy-plant", action="store_true", help="Legacy diagnostic only; can never certify")
    parser.add_argument("--out-dir", help="Fresh evaluation directory, outside a completed source bundle")
    args = parser.parse_args(argv)
    if not args.protocol_only and not args.out_dir:
        parser.error("--out-dir is required for a gait evaluation")
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
        protocol = GaitProtocol()
        if args.protocol_json:
            options = read_json_object(args.protocol_json)
            if options is None:
                raise ValueError("--protocol-json must contain a detector options object")
            protocol = GaitProtocol(**options)
        curriculum = config.get("curriculum_kwargs", {})
        episodes = args.episodes if args.episodes is not None else int(curriculum.get("min_eval_episodes", 40))
        seed = (
            args.seed if args.seed is not None else int(curriculum.get("gait_panel_seed_start", PUBLICATION_SEED_START))
        )
        if args.protocol_only:
            import numpy as np

            from environments.shared.plant_contract import current_plant_identity, validate_environment_plant

            if episodes < 1 or seed < 0 or not np.isfinite(args.settle_s) or args.settle_s < 0:
                raise ValueError("protocol requires positive episodes and nonnegative seed/settle time")
            direction = np.asarray(args.direction, dtype=float)
            norm = float(np.linalg.norm(direction))
            if not np.all(np.isfinite(direction)) or norm <= 0:
                raise ValueError("protocol direction must be finite and nonzero")
            direction /= norm
            env = species_cfg.env_class(**config.get("env_kwargs", {}))
            try:
                validate_environment_plant(env, current_plant_identity(species_cfg.species), artifact="gait protocol")
                horizon = int(config.get("env_kwargs", {}).get("max_episode_steps", 1000))
                if horizon * float(env.dt) <= args.settle_s:
                    raise ValueError("protocol must leave a positive analysis window")
                payload = measurement_protocol(
                    species_cfg.species,
                    protocol,
                    settle_s=args.settle_s,
                    direction_xy=(float(direction[0]), float(direction[1])),
                    horizon=horizon,
                    physics_dt_s=float(env.model.opt.timestep),
                    control_dt_s=float(env.dt),
                    episodes=episodes,
                    seed_start=seed,
                )
            finally:
                env.close()
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
