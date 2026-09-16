"""Compatibility entry point for the original T-Rex behavior CLI.

New runs can use ``python -m environments.shared.train_behaviors`` for any species.
Historical ``--config`` recipes and command lines remain supported.
"""

from environments.shared.train_behaviors import (
    EpisodeManifestRecorder,
    main,
    read_recipe,
)

__all__ = ["EpisodeManifestRecorder", "main", "read_recipe"]

if __name__ == "__main__":
    main()
