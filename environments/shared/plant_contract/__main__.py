"""CLI: generate, print, or verify the layered MuJoCo plant manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import constants
from .errors import PlantContractError
from .manifest import (
    build_plant_manifest,
    check_plant_manifest,
    render_plant_manifest,
    write_plant_manifest,
)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify the layered MuJoCo plant manifest")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify the committed manifest")
    action.add_argument("--write", action="store_true", help="regenerate after deliberate revision bumps")
    action.add_argument("--show", action="store_true", help="print the currently computed manifest")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="with --check, enforce revision monotonicity against a base manifest",
    )
    args = parser.parse_args()
    if args.baseline is not None and not args.check:
        parser.error("--baseline requires --check")
    try:
        if args.write:
            path = write_plant_manifest()
            print(f"Wrote {path.relative_to(constants.REPOSITORY_ROOT)}")
        elif args.check:
            check_plant_manifest(baseline_path=args.baseline)
            print("Plant manifest is current")
        else:
            print(render_plant_manifest(build_plant_manifest()))
    except PlantContractError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
