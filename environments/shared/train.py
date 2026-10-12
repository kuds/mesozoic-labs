#!/usr/bin/env python3
"""Unified training entry point for every model.

Usage:
    python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical train --stage 1
    python -m environments.shared.train --species tyrannosaurus_rex --embodiment anatomical curriculum --n-envs 4
    python -m environments.shared.train --species brachiosaurus_altithorax --embodiment anatomical train --stage 2

The one training command line: there is no per-species wrapper script.
``--species`` and ``--embodiment`` are both required and take the
exact manifest IDs (decision D-D33): no alias, display name or default
embodiment.  Everything else on the command line is the command and its
options, parsed by :func:`environments.shared.cli.main`.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _main():
    # Only the manifest reader is imported before the model is known, so a
    # usage error fails fast without the training stack.
    from environments.shared.model_id import EMBODIMENTS
    from environments.shared.species_names import model_display_name, model_ids, require_model_id, species_ids

    models = model_ids()
    listing = "\n".join(
        f"  --species {m.species} --embodiment {m.embodiment}  ({model_display_name(m)})" for m in models
    )
    parser = argparse.ArgumentParser(
        prog="python -m environments.shared.train",
        usage=(
            "%(prog)s --species SPECIES --embodiment EMBODIMENT <command> [options]\n\n"
            f"Models:\n{listing}\n\n"
            "Commands and their options: %(prog)s --species SPECIES --embodiment EMBODIMENT --help"
        ),
        # --help and every option but these two belong to the command parser in cli.main.
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--species", required=True, choices=species_ids())
    parser.add_argument(
        "--embodiment",
        required=True,
        choices=[embodiment for embodiment in EMBODIMENTS if any(m.embodiment == embodiment for m in models)],
    )
    args, rest = parser.parse_known_args(sys.argv[1:])
    try:
        model_id = require_model_id(args.species, args.embodiment)
    except ValueError as exc:
        parser.error(str(exc))
    # cli.main parses sys.argv itself: hand it the command line without the model.
    sys.argv = [sys.argv[0], *rest]

    from environments.shared.species_registry import get_species_config
    from environments.shared.train_base import main

    main(get_species_config(model_id))


if __name__ == "__main__":
    _main()
