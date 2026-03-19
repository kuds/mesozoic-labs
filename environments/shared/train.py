#!/usr/bin/env python3
"""Unified training entry point for all species.

Usage:
    python -m environments.shared.train --species velociraptor train --stage 1 --timesteps 1000000
    python -m environments.shared.train --species trex curriculum --n-envs 4
    python -m environments.shared.train --species brachiosaurus train --stage 2

This replaces the per-species train_sb3.py scripts with a single entry point.
The per-species scripts still work for backward compatibility.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _main():
    # Parse --species before handing off to the shared main()
    if "--species" not in sys.argv:
        print("Usage: python -m environments.shared.train --species <name> <command> [options]")
        print("Species: velociraptor, trex, brachiosaurus")
        sys.exit(1)

    idx = sys.argv.index("--species")
    if idx + 1 >= len(sys.argv):
        print("Error: --species requires a value")
        sys.exit(1)

    species_name = sys.argv[idx + 1]
    # Remove --species and its value from argv so argparse in main() doesn't choke
    sys.argv = [sys.argv[0]] + sys.argv[1:idx] + sys.argv[idx + 2 :]

    from environments.shared.species_registry import get_species_config
    from environments.shared.train_base import main

    config = get_species_config(species_name)
    main(config)


if __name__ == "__main__":
    _main()
