"""Golden reset/step traces for the Phase C interface bump (BEHAVIOR_RECIPES_PLAN §4.6).

The Phase C command segment is appended to every species' observation
WITHOUT touching a single reset RNG draw (plan §4.6, decision D3: the
stance re-panel must be a near-reproduction, not a fresh roll of a
seed-sensitive certificate; decision D-C2: the compsognathus recovery
calibrations are restamped rather than re-measured because their
fixed-command nulls never read the observation and the seeded draw stream
is unchanged).  This module makes that claim executable: it captures, at
the pre-bump commit, what every seeded reset produces and what twenty
zero-action steps do afterwards, and the fixture it writes
(``fixtures/phase_c_reset_golden.json``) is replayed by the Phase C
interface tests on the current tree.

Two halves, with different strengths (amendment A4):

* ``reset`` — ``qpos`` and ``qvel`` right after ``reset(seed)``, every
  mocap target position, the push-schedule starts/directions when the
  stage schedules pushes, and the generator state (``rng_state``:
  ``np_random.bit_generator.state``) the reset leaves behind.  Quantized to
  6 decimals and compared EXACTLY: this is the contract (the seeded draw
  stream).  A reset draw added before the existing draws breaks it by
  design, and so does a draw added AFTER them -- the generator state moves.
* ``second_reset`` — the same record for an UNSEEDED ``reset()`` taken after
  the trajectory below, the way training resets every episode after the
  first.  Also exact: a discarded draw anywhere in ``reset()`` (one that
  leaves the seeded reset's state untouched but consumes the stream) shows
  up here, and stepping draws nothing, so the record does not depend on the
  solver.
* ``trajectory`` — ``qpos`` after each zero-action step.  Quantized to 6
  decimals and compared with ``assert_allclose(atol=1e-5)``, and only
  when the running ``mujoco.__version__`` equals the fixture's: stepping
  crosses the solver, so it is advisory across MuJoCo releases.

A third record, ``observation``, holds the reset observation as captured;
after the bump the current observation must equal it on the captured
prefix and be exactly zero on the appended command dims — the executable
form of "byte-inert except the trailing zeros".

Values are quantized like the plant contract's portable probe
(``policy_layer._portable_probe_values``), and the per-step ``qvel`` is
deliberately not stored so the fixture stays reviewable.

Capture (the tree must be at the pre-bump commit, nothing under
``environments/*/envs``, ``base_env.py``, ``obs_functions.py``,
``mjx_env.py`` or ``jax_setup.py`` modified)::

    PYTHONPATH=. python environments/shared/tests/reset_golden.py \\
        --out environments/shared/tests/fixtures/phase_c_reset_golden.json

The script imports the repository only through ``sys.path`` (PYTHONPATH or
the working directory), never through a hard-coded checkout path, so the
same file re-captures byte-for-byte from any pristine checkout of the
source commit.

A later PHYSICS revision of one species moves that species' observation and
trajectory halves by design (dynamics feed the reset observation through the
sensors) while its reset half must still hold.  Run
``test_reset_draw_stream_matches_the_pre_bump_golden`` on the new tree
first -- a pass proves the revision consumed no reset draw -- then re-take
only that species' captures, leaving every other capture byte-identical::

    PYTHONPATH=. python environments/shared/tests/reset_golden.py \\
        --recapture trex/stance trex/recovery \\
        --reason "trex physics r8 (plant_versions note 13)"

``recaptures`` then records, per re-taken capture, the plant it was taken
on and why; its observation is captured with the command segment already
appended.  A recapture's ``source_commit`` means something narrower than the
top-level one: it is the commit the revision was applied to (the working
tree's ``HEAD`` while the revision is still uncommitted, or
``--source-commit``), not a commit that reproduces the capture.  What binds
the capture is its ``physics_sha256``; it re-takes byte-for-byte from any
pristine checkout of the commit that adds the recapture.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FIXTURE_SCHEMA = "mesozoic.phase-c-reset-golden/v1"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "phase_c_reset_golden.json"
QUANTIZATION_DECIMALS = 6

#: The capture specification: (species, stage, seeds, steps).  Trex stance
#: seeds are the certified run's training seed and the first/second/last
#: publication-panel seeds; trex recovery exercises the push schedule; the
#: two compsognathus recovery captures back the calibration restamp.
DEFAULT_SPEC: tuple[tuple[str, str, tuple[int, ...], int], ...] = (
    ("trex", "stance", (42, 3042, 3043, 3081), 20),
    ("trex", "recovery", (3042,), 20),
    ("compsognathus", "recovery", (1042, 7042), 20),
    ("compsognathus_robot", "recovery", (1042,), 20),
)


def _quantize(values: Any) -> list[float]:
    """Round to :data:`QUANTIZATION_DECIMALS` as plain floats (JSON-portable)."""
    return [float(value) for value in np.round(np.asarray(values, dtype=np.float64), QUANTIZATION_DECIMALS).ravel()]


def _quantize_rows(values: Any) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return [_quantize(row) for row in array.reshape(array.shape[0], -1)]


def _reset_record(env: Any) -> dict[str, Any]:
    """The exact-contract view of the state ``reset()`` just produced."""
    record: dict[str, Any] = {
        "qpos": _quantize(env.data.qpos),
        "qvel": _quantize(env.data.qvel),
        "mocap_pos": _quantize_rows(env.data.mocap_pos) if env.model.nmocap else [],
    }
    starts = getattr(env, "_push_schedule_starts", None)
    if starts is not None:
        record["push_schedule_starts"] = [int(value) for value in np.asarray(starts).ravel()]
        record["push_schedule_directions"] = _quantize_rows(env._push_schedule_directions)
    # PCG64 state: plain ints and strings, JSON-portable, position-exact.
    record["rng_state"] = env.np_random.bit_generator.state
    return record


def capture_key(species: str, stage: str) -> str:
    return f"{species}/{stage}"


def capture(species: str, stage: str, seeds: Sequence[int], steps: int) -> dict[str, Any]:
    """Reset/step traces for one species stage, keyed by seed.

    Builds the stage environment exactly as training does
    (``config.build_env``), then for each seed records the reset state,
    the reset observation, ``steps`` zero-action steps, and the state an
    unseeded follow-up ``reset()`` produces.
    """
    from environments.shared.config import build_env

    env = build_env(species, stage)
    try:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            reset = _reset_record(env)
            trajectory = []
            for _ in range(int(steps)):
                env.step(action)
                trajectory.append(_quantize(env.data.qpos))
            env.reset()
            per_seed[str(int(seed))] = {
                "reset": reset,
                "observation": _quantize(observation),
                "trajectory": trajectory,
                "second_reset": _reset_record(env),
            }
    finally:
        env.close()
    return {"species": species, "stage": stage, "steps": int(steps), "seeds": per_seed}


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_fixture(
    spec: Sequence[tuple[str, str, Sequence[int], int]] = DEFAULT_SPEC,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """The complete fixture document for *spec*."""
    import mujoco

    captures = {
        capture_key(species, stage): capture(species, stage, seeds, steps) for species, stage, seeds, steps in spec
    }
    return {
        "schema": FIXTURE_SCHEMA,
        "source_commit": _source_commit() if source_commit is None else source_commit,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "quantization_decimals": QUANTIZATION_DECIMALS,
        "spec": [
            {"species": species, "stage": stage, "seeds": [int(seed) for seed in seeds], "steps": int(steps)}
            for species, stage, seeds, steps in spec
        ],
        "captures": captures,
    }


def recapture_fixture(
    fixture: dict[str, Any],
    keys: Sequence[str],
    *,
    reason: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """*fixture* with the captures named by *keys* re-taken on the current tree.

    Every other capture, and the top-level provenance of the original
    capture, stays byte-identical.  Each re-taken capture is recorded under
    ``recaptures`` with the species' plant identity at re-capture time, so a
    later physics move is reported as a stale recapture rather than as an
    unexplained trajectory mismatch.
    """
    import mujoco

    from environments.shared.plant_contract import current_plant_identity

    spec = {
        capture_key(species, stage): (species, stage, seeds, steps) for species, stage, seeds, steps in DEFAULT_SPEC
    }
    unknown = sorted(set(keys) - set(spec))
    if unknown:
        raise ValueError(f"no capture named {unknown}; the spec names {sorted(spec)}")
    updated = copy.deepcopy(fixture)
    recaptures = updated.setdefault("recaptures", {})
    for key in keys:
        species, stage, seeds, steps = spec[key]
        updated["captures"][key] = capture(species, stage, seeds, steps)
        plant = current_plant_identity(species)
        recaptures[key] = {
            "reason": reason,
            "source_commit": _source_commit() if source_commit is None else source_commit,
            "physics_revision": int(plant.physics_revision),
            "physics_sha256": plant.physics_sha256,
            "observation_includes_command_segment": True,
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        }
    return updated


_NUMERIC_ARRAY = re.compile(r"\[\s*((?:-?\d[\d.e+-]*,\s*)*-?\d[\d.e+-]*)\s*\]")


def render_fixture(fixture: dict[str, Any]) -> str:
    """Indented JSON with every flat numeric array on one line (reviewable diffs)."""
    rendered = json.dumps(fixture, indent=1, sort_keys=True)
    return _NUMERIC_ARRAY.sub(lambda match: "[" + re.sub(r",\s+", ", ", match.group(1)) + "]", rendered) + "\n"


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH, help="where to write the fixture JSON")
    parser.add_argument(
        "--source-commit",
        default=None,
        help="record this commit instead of `git rev-parse --short=7 HEAD` of the working directory",
    )
    parser.add_argument(
        "--recapture",
        nargs="+",
        metavar="SPECIES/STAGE",
        help="re-take only these captures in the existing --out fixture, leaving every other one untouched",
    )
    parser.add_argument("--reason", help="why the --recapture captures moved (recorded with them)")
    args = parser.parse_args(argv)
    if args.recapture:
        if not args.reason:
            parser.error("--recapture needs --reason")
        fixture = recapture_fixture(
            load_fixture(args.out), args.recapture, reason=args.reason, source_commit=args.source_commit
        )
    else:
        fixture = build_fixture(source_commit=args.source_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_fixture(fixture), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
