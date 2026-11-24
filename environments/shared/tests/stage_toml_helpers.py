"""Edits the tests make to a stage TOML in a copy of a model's config directory.

Every committed walk (``locomotion.toml``) declares a live ``[commands]`` table and its
``[commands.stand_weights]`` sub-table (decision D-D62), so a test that runs a stage under another command
replaces that table rather than appending a second one, which TOML refuses.
"""

from __future__ import annotations

from pathlib import Path

from environments.shared.config import FORWARD_VELOCITY_RAMP_KEYS


def without_commands_table(text: str) -> list[str]:
    """*text*'s lines without its ``[commands]`` table and the sub-tables under it (``[commands.stand_weights]``)."""
    kept: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("["):
            inside = line.strip() == "[commands]" or line.startswith("[commands.")
        if not inside:
            kept.append(line)
    return kept


def replace_commands_table(path: Path, table: str) -> Path:
    """Rewrite *path* with its own ``[commands]`` (and sub-tables) replaced by *table*, added on a stage with none.

    The forward-velocity ramp keys a live stage refuses (``config.FORWARD_VELOCITY_RAMP_KEYS``) are dropped too, so
    the helper also turns a command-free stage copy live.
    """
    kept = [
        line for line in without_commands_table(path.read_text()) if not line.startswith(FORWARD_VELOCITY_RAMP_KEYS)
    ]
    path.write_text("\n".join(kept) + "\n\n" + table.strip() + "\n")
    return path


def command_free_walk(text: str) -> str:
    """*text*, a committed walk (``locomotion.toml``), as a command-free advancing node: the forward walk's shape.

    The ``[commands]`` tables go, and the ``command_tracking/v1`` block of ``[curriculum]`` becomes a
    ``reward_and_length/v1`` gate on ``min_avg_reward = 100.0`` (the forward walk's rail); every other key, the
    training-loop keys included, stays.  For the curriculum-mechanics tests, whose ladders need an in-training
    advancing gate on every node: the committed walk is judged after its stage, and a stage warm-started from a
    live walk may not ramp its cruise in (decision D-D62).
    """
    from environments.shared.curriculum.command_tracking_gate import COMMAND_TRACKING_THRESHOLD_KEYS
    from environments.shared.curriculum.gate_schema import GATE_KINDS

    dropped = COMMAND_TRACKING_THRESHOLD_KEYS - GATE_KINDS["reward_and_length/v1"]
    kept: list[str] = []
    table = None
    for line in without_commands_table(text):
        if line.startswith("["):
            table = line.strip()
        elif table == "[curriculum]" and "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key == "gate_kind":
                kept.extend(['gate_kind = "reward_and_length/v1"', "min_avg_reward = 100.0"])
                continue
            if key in dropped:
                continue
        kept.append(line)
    return "\n".join(kept) + "\n"
