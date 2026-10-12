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
        --recapture tyrannosaurus_rex anatomical stance \\
        --recapture tyrannosaurus_rex anatomical recovery \\
        --reason "T-Rex physics r8 (plant_versions note 13)"

``recaptures`` then records, per re-taken capture, the plant it was taken
on and why; its observation is captured with the command segment already
appended.  A recapture's ``source_commit`` means something narrower than the
top-level one: it is the commit the revision was applied to (the working
tree's ``HEAD`` while the revision is still uncommitted, or
``--source-commit``), not a commit that reproduces the capture.  What binds
the capture is its ``physics_sha256``; it re-takes byte-for-byte from any
pristine checkout of the commit that adds the recapture.

An interface-only POLICY revision (a new observation layout on unchanged
physics, such as the heading-free observation of plant_versions note 18)
moves only the observation half: the reset, second-reset and trajectory
halves must re-take byte-identical, which is what shows the revision touched
neither the draw stream nor the dynamics.  Recapture the species' captures
the same way; each record also names the ``policy_interface_revision`` and
``observation_schema`` its observation was taken under, so the observation
half is bound to its layout as the trajectory half is to its physics.

Schema v2 (decision D-D33): every capture is named by its model, and the
fixture nests ``captures[species][embodiment][stage]`` (and ``recaptures``
the same way), each capture recording ``species``, ``embodiment`` and
``stage``; no joined key is written.  A v1 fixture named its captures by the
species IDs of its date (``"trex/stance"``).  ``--migrate-identity --reason
TEXT`` migrates a v1 fixture, and only a v1 fixture (a migrated one is left
untouched): it pairs the stored ``spec[i]`` with :data:`DEFAULT_SPEC` ``[i]``
(same stage, seeds and steps, so no old-to-new table is needed), re-takes
every capture on the current tree and refuses unless the ``reset``,
``second_reset`` and ``trajectory`` halves are byte-equal and the
observation reproduces (whole for a recapture, on the captured prefix with
zero command dims otherwise), then rewrites the keys and identity fields,
refreshes each recapture's ``policy_interface_revision`` (an identity move
is an interface revision, plant_versions note 19) and appends an
``identity_migrations`` record.  ``source_commit`` and every captured value
stay as recorded::

    PYTHONPATH=. python environments/shared/tests/reset_golden.py \\
        --migrate-identity --reason "plant_versions note 19 (D-D33)"
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

from environments.shared.model_id import ModelId

FIXTURE_SCHEMA = "mesozoic.phase-c-reset-golden/v2"
#: The one schema ``--migrate-identity`` migrates from (decision D-D33).
_PRE_D_D33_FIXTURE_SCHEMA = "mesozoic.phase-c-reset-golden/v1"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "phase_c_reset_golden.json"
QUANTIZATION_DECIMALS = 6

#: The capture specification: (model, stage, seeds, steps).  T-Rex stance
#: seeds are the certified run's training seed and the first/second/last
#: publication-panel seeds; T-Rex recovery exercises the push schedule; the
#: two Compsognathus recovery captures back the calibration restamp.
DEFAULT_SPEC: tuple[tuple[ModelId, str, tuple[int, ...], int], ...] = (
    (ModelId("tyrannosaurus_rex", "anatomical"), "stance", (42, 3042, 3043, 3081), 20),
    (ModelId("tyrannosaurus_rex", "anatomical"), "recovery", (3042,), 20),
    (ModelId("compsognathus_longipes", "anatomical"), "recovery", (1042, 7042), 20),
    (ModelId("compsognathus_longipes", "robot"), "recovery", (1042,), 20),
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


def _nested_set(records: dict[str, Any], model_id: ModelId, stage: str, value: Any) -> None:
    records.setdefault(model_id.species, {}).setdefault(model_id.embodiment, {})[stage] = value


def capture(model_id: ModelId, stage: str, seeds: Sequence[int], steps: int) -> dict[str, Any]:
    """Reset/step traces for one model stage, keyed by seed.

    Builds the stage environment exactly as training does
    (``config.build_env``), then for each seed records the reset state,
    the reset observation, ``steps`` zero-action steps, and the state an
    unseeded follow-up ``reset()`` produces.
    """
    from environments.shared.config import build_env

    env = build_env(model_id, stage)
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
    return {**model_id.record(), "stage": stage, "steps": int(steps), "seeds": per_seed}


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
    spec: Sequence[tuple[ModelId, str, Sequence[int], int]] = DEFAULT_SPEC,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """The complete fixture document for *spec*."""
    import mujoco

    captures: dict[str, Any] = {}
    for model_id, stage, seeds, steps in spec:
        _nested_set(captures, model_id, stage, capture(model_id, stage, seeds, steps))
    return {
        "schema": FIXTURE_SCHEMA,
        "source_commit": _source_commit() if source_commit is None else source_commit,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "quantization_decimals": QUANTIZATION_DECIMALS,
        "spec": [_spec_entry(model_id, stage, seeds, steps) for model_id, stage, seeds, steps in spec],
        "captures": captures,
    }


def _spec_entry(model_id: ModelId, stage: str, seeds: Sequence[int], steps: int) -> dict[str, Any]:
    return {**model_id.record(), "stage": stage, "seeds": [int(seed) for seed in seeds], "steps": int(steps)}


def recapture_fixture(
    fixture: dict[str, Any],
    keys: Sequence[tuple[ModelId, str]],
    *,
    reason: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """*fixture* with the captures named by *keys* (model, stage) re-taken on the current tree.

    Every other capture, and the top-level provenance of the original
    capture, stays byte-identical.  Each re-taken capture is recorded under
    ``recaptures`` with the model's plant identity at re-capture time (its
    physics, and the policy interface and observation schema its observation
    was taken under), so a later physics or interface move is reported as a
    stale recapture rather than as an unexplained mismatch.
    """
    import mujoco

    from environments.shared.plant_contract import current_plant_identity

    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"recapture needs a {FIXTURE_SCHEMA} fixture, not {fixture.get('schema')!r}")
    spec = {(model_id, stage): (seeds, steps) for model_id, stage, seeds, steps in DEFAULT_SPEC}
    unknown = sorted(f"{model_id} {stage}" for model_id, stage in set(keys) - set(spec))
    if unknown:
        raise ValueError(f"no capture named {unknown}; the spec names {sorted(f'{m} {s}' for m, s in spec)}")
    updated = copy.deepcopy(fixture)
    recaptures = updated.setdefault("recaptures", {})
    for model_id, stage in keys:
        seeds, steps = spec[(model_id, stage)]
        _nested_set(updated["captures"], model_id, stage, capture(model_id, stage, seeds, steps))
        plant = current_plant_identity(model_id)
        _nested_set(
            recaptures,
            model_id,
            stage,
            {
                "reason": reason,
                "source_commit": _source_commit() if source_commit is None else source_commit,
                "physics_revision": int(plant.physics_revision),
                "physics_sha256": plant.physics_sha256,
                "policy_interface_revision": int(plant.policy_interface_revision),
                "observation_schema": _observation_schema(model_id),
                "observation_includes_command_segment": True,
                "mujoco_version": mujoco.__version__,
                "numpy_version": np.__version__,
                "platform": platform.platform(),
            },
        )
    return updated


def _observation_schema(model_id: ModelId) -> str:
    """The observation schema configs/plant_versions.toml declares for *model_id*."""
    from environments.shared.plant_contract.versions import load_plant_versions

    _, versions = load_plant_versions()
    return str(versions[model_id].observation_schema)


def _command_width() -> int:
    from environments.shared.command_frame import COMMAND_WIDTH

    return int(COMMAND_WIDTH)


def _retake_refusals(stored: dict[str, Any], retaken: dict[str, Any], *, recaptured: bool) -> list[str]:
    """Why *retaken* is not the same capture as *stored* (empty when it is).

    The reset, second-reset and trajectory halves must be byte-equal (after
    quantization).  The observation must reproduce whole for a recapture
    (taken with the command segment) and, for a capture taken before the
    Phase C bump, on the captured prefix with zero command dims after it.
    """
    refusals: list[str] = []
    if int(stored.get("steps", -1)) != retaken["steps"] or set(stored.get("seeds", {})) != set(retaken["seeds"]):
        return ["the stored steps or seeds differ from the re-take's"]
    width = _command_width()
    for seed, record in stored["seeds"].items():
        current = retaken["seeds"][seed]
        for half in ("reset", "second_reset", "trajectory"):
            if record.get(half) != current[half]:
                refusals.append(f"seed {seed}: the {half} half moved")
        before, after = record.get("observation"), current["observation"]
        if recaptured:
            if before != after:
                refusals.append(f"seed {seed}: the recaptured observation moved")
        elif (
            not isinstance(before, list)
            or len(after) != len(before) + width
            or after[: len(before)] != before
            or any(value != 0.0 for value in after[len(before) :])
        ):
            refusals.append(f"seed {seed}: the observation is not the captured prefix plus {width} zero command dims")
    return refusals


def migrate_fixture_identity(
    fixture: dict[str, Any],
    *,
    reason: str,
    migrated_at_commit: str | None = None,
) -> dict[str, Any]:
    """*fixture* (schema v1) under the models' identities (decision D-D33); a v2 fixture is returned unchanged.

    The module docstring lists every guard.  Raises ``ValueError`` on every
    refusal.
    """
    import mujoco

    from environments.shared.plant_contract import current_plant_identity

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("the migration reason must be a non-empty string")
    if fixture.get("schema") == FIXTURE_SCHEMA:
        return fixture
    if fixture.get("schema") != _PRE_D_D33_FIXTURE_SCHEMA:
        raise ValueError(
            f"--migrate-identity migrates only {_PRE_D_D33_FIXTURE_SCHEMA!r}, not {fixture.get('schema')!r}"
        )
    stored_spec = fixture.get("spec")
    if not isinstance(stored_spec, list) or len(stored_spec) != len(DEFAULT_SPEC):
        raise ValueError(f"the stored spec does not pair with the {len(DEFAULT_SPEC)} entries of DEFAULT_SPEC")
    stored_captures = fixture.get("captures") or {}
    stored_recaptures = fixture.get("recaptures") or {}
    paired_keys: set[str] = set()
    captures: dict[str, Any] = {}
    recaptures: dict[str, Any] = {}
    refusals: list[str] = []
    for entry, (model_id, stage, seeds, steps) in zip(stored_spec, DEFAULT_SPEC):
        if (entry.get("stage"), entry.get("seeds"), entry.get("steps")) != (stage, list(seeds), steps):
            raise ValueError(f"stored spec entry {entry} does not pair with ({model_id}, {stage}, {seeds}, {steps})")
        if "embodiment" in entry:
            raise ValueError(f"stored spec entry {entry} already names an embodiment; refusing to guess")
        key = f"{entry['species']}/{stage}"
        stored = stored_captures.get(key)
        if not isinstance(stored, dict) or (stored.get("species"), stored.get("stage")) != (entry["species"], stage):
            raise ValueError(f"the stored fixture has no capture {key!r} for spec entry {entry}")
        paired_keys.add(key)
        recaptured = key in stored_recaptures
        retaken = capture(model_id, stage, seeds, steps)
        for refusal in _retake_refusals(stored, retaken, recaptured=recaptured):
            refusals.append(f"{key} as {model_id} {stage}: {refusal}")
        _nested_set(captures, model_id, stage, {**stored, **model_id.record()})
        if recaptured:
            record = dict(stored_recaptures[key])
            plant = current_plant_identity(model_id)
            if record.get("physics_sha256") != plant.physics_sha256:
                refusals.append(f"{key} was recaptured on physics {record.get('physics_sha256')!r}, not {model_id}'s")
            record["policy_interface_revision"] = int(plant.policy_interface_revision)
            _nested_set(recaptures, model_id, stage, record)
    unpaired = sorted(set(stored_captures) - paired_keys) + sorted(set(stored_recaptures) - paired_keys)
    if unpaired:
        refusals.append(f"stored captures or recaptures no spec entry pairs with: {unpaired}")
    if refusals:
        if mujoco.__version__ != fixture.get("mujoco_version"):
            refusals.append(f"running mujoco {mujoco.__version__} is not the fixture's {fixture.get('mujoco_version')}")
        raise ValueError("refusing to migrate the reset golden:\n  " + "\n  ".join(refusals))
    migrated = {key: value for key, value in fixture.items() if key not in {"schema", "spec", "captures", "recaptures"}}
    migrated["schema"] = FIXTURE_SCHEMA
    migrated["spec"] = [_spec_entry(model_id, stage, seeds, steps) for model_id, stage, seeds, steps in DEFAULT_SPEC]
    migrated["captures"] = captures
    if recaptures:
        migrated["recaptures"] = recaptures
    migrated["identity_migrations"] = [
        *fixture.get("identity_migrations", []),
        {
            "reason": reason,
            "migrated_at_commit": _source_commit() if migrated_at_commit is None else migrated_at_commit,
            "previous_schema": fixture["schema"],
            "previous_spec": copy.deepcopy(stored_spec),
        },
    ]
    return migrated


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
        nargs=3,
        action="append",
        metavar=("SPECIES", "EMBODIMENT", "STAGE"),
        help="re-take only this capture in the existing --out fixture, leaving every other one untouched; repeatable",
    )
    parser.add_argument(
        "--migrate-identity",
        action="store_true",
        help="migrate a schema-v1 --out fixture to the models' identities (decision D-D33); a no-op on a v2 one",
    )
    parser.add_argument("--reason", help="why the --recapture captures moved, or why --migrate-identity ran (recorded)")
    args = parser.parse_args(argv)
    if args.recapture and args.migrate_identity:
        parser.error("--recapture and --migrate-identity are separate runs")
    if args.migrate_identity:
        if not args.reason:
            parser.error("--migrate-identity needs --reason")
        original = load_fixture(args.out)
        try:
            fixture = migrate_fixture_identity(original, reason=args.reason, migrated_at_commit=args.source_commit)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        if fixture is original:
            print(f"{args.out} is already {FIXTURE_SCHEMA}; nothing to migrate")
            return 0
    elif args.recapture:
        if not args.reason:
            parser.error("--recapture needs --reason")
        try:
            keys = [(ModelId(species, embodiment), stage) for species, embodiment, stage in args.recapture]
        except ValueError as exc:
            parser.error(str(exc))
        fixture = recapture_fixture(load_fixture(args.out), keys, reason=args.reason, source_commit=args.source_commit)
    else:
        fixture = build_fixture(source_commit=args.source_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_fixture(fixture), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
