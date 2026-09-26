"""Hand-run harness: print every identity and digest a certified run depends on.

The acceptance check for a change that claims to move no digest (retiring a
backend, a refactor, a dependency bump).  Run it on the base and on the head
and ``diff`` the two outputs: any line that differs names the digest that
moved.  One tab-separated line per value, in a fixed order, with no
timestamps and no checkout paths, so two runs on the same tree are
byte-identical:

    plant.check_plant_manifest  OK
    plant     <species>  <identity key>  <value>
    policy    <species>  <payload key>   <digest>
    stage     <species>  <stage id>      <name>  <digest>
    recovery  <species>  <name>          <digest or OK>
    behavior  <species>  <recipe>        <name>  <digest>
    <label>   ERROR      <exception>     <first line of its message>

The sections are the plant identities and their policy-interface payload
(whole and per key), each stage's task fingerprint, gate, hyperparameter and
``stage_config.json`` digests for PPO and SAC, the recovery calibrations,
and every behavior recipe's identity with its source digests.  A value that
cannot be computed prints an ERROR line instead of stopping the run, so the
diff shows the failure.

Comparing two checkouts.  The digests are computed by whichever
``environments`` package Python imports, and ``configs/`` is read from
``--repo``; both must be the same checkout.  A base older than this file does
not have it, so run the head's copy BY FILE PATH, with ``PYTHONPATH`` set to
the checkout being measured (repo root, head checked out):

    git worktree add --detach /tmp/base <base commit>
    PYTHONPATH=/tmp/base python environments/shared/harnesses/digest_snapshot.py \\
        --repo /tmp/base --block-optional-backends > base.txt
    PYTHONPATH=. python environments/shared/harnesses/digest_snapshot.py \\
        --block-optional-backends > head.txt
    diff base.txt head.txt              # must print nothing
    git worktree remove /tmp/base

Do not run ``python -m environments.shared.harnesses.digest_snapshot --repo
<other checkout>``: ``-m`` imports this checkout's ``environments`` before the
arguments are read, so the plant and stage digests would come from here and
the recipe files from there.  The harness refuses any run in which
``environments.__file__`` does not resolve under ``--repo`` (default: the
working directory).  ``-m`` from the root of the checkout being measured,
without ``--repo``, is fine.

``--block-optional-backends`` makes every import of jax, jaxlib, flax, optax,
``mujoco.mjx``, ray, mjlab, hypertune and ``google.cloud`` raise
ImportError, so the run also proves the digests are computable on an
SB3-only install; it refuses to start if one of them is already imported.
``--skip-behaviors`` leaves out the behavior section, which builds one
environment per recipe and is most of the run time.

Exit status: 0 when every value was computed, 1 when any ERROR line was
printed (the rest of the snapshot is still complete), 2 when the invocation
is refused.  A one-line summary (line and error counts, elapsed time) goes to
stderr, so it never reaches the diff.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

#: The optional backends an SB3-only install does not have.
BLOCKED_BACKENDS = ("jax", "jaxlib", "flax", "optax", "mujoco.mjx", "ray", "mjlab", "hypertune", "google.cloud")

#: ``stage_config.json`` keys left out of ``stage_config_view_sha256``: the
#: line already names the species and stage, and the rest change per run.
UNPINNED_STAGE_CONFIG_KEYS = ("species", "stage", "library_version", "git_commit", "run", "gpu")


def _is_blocked(name: str) -> bool:
    return any(name == blocked or name.startswith(blocked + ".") for blocked in BLOCKED_BACKENDS)


class _BackendBlocker(importlib.abc.MetaPathFinder):
    """Make every import of an optional backend raise ImportError."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if _is_blocked(fullname):
            raise ImportError(f"blocked optional backend: {fullname}")
        return None


class _Snapshot:
    """Prints the snapshot lines and counts them and the ERROR lines."""

    def __init__(self, repo: Path, *, debug: bool = False) -> None:
        self.repo = repo
        self.debug = debug
        self.lines = 0
        self.errors = 0

    def emit(self, *parts: object) -> None:
        print("\t".join(str(part) for part in parts), flush=True)
        self.lines += 1

    def guard(self, label: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Return ``func(*args, **kwargs)``, or print an ERROR line and return None."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # The repo path is replaced so that the same failure on two
            # checkouts prints the same line.
            message = str(exc).replace(str(self.repo), "<repo>")
            self.emit(label, "ERROR", type(exc).__name__, message.splitlines()[0][:300] if message else "")
            self.errors += 1
            if self.debug:
                traceback.print_exc()
            return None


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plant_section(out: _Snapshot) -> dict[str, Any]:
    """Plant identities, and the policy-interface payload digest by key."""
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.plant_contract.manifest import check_plant_manifest
    from environments.shared.plant_contract.versions import _species_entries, load_plant_versions

    if out.guard("plant.check_plant_manifest", check_plant_manifest) is not None:
        out.emit("plant.check_plant_manifest", "OK")
    registry = out.guard("plant.registry", lambda: (load_plant_versions()[1], _species_entries()))
    if registry is None:
        return {}
    versions, entries = registry
    identities: dict[str, Any] = {}
    for species in sorted(entries):
        # The committed manifest must still describe the live plant ...
        if out.guard(f"plant.{species}.verify_generated", current_plant_identity, species) is not None:
            out.emit("plant", species, "verify_generated", "OK")
        # ... and the live identity prints either way, so a diff names the moved digest.
        identity = out.guard(f"plant.{species}.identity", current_plant_identity, species, verify_generated=False)
        if identity is None:
            continue
        identities[species] = identity
        for key, value in sorted(identity.to_dict().items()):
            out.emit("plant", species, key, value)
        out.guard(
            f"policy.{species}", _policy_lines, out, species, str(entries[species]["env_entrypoint"]), versions[species]
        )
    return identities


def _policy_lines(out: _Snapshot, species: str, entrypoint: str, version: Any) -> None:
    # Private helpers on purpose: the per-key digests name the part of the
    # payload that moved, which the identity's one digest cannot.
    from environments.shared.plant_contract.constants import POLICY_INTERFACE_SCHEMA
    from environments.shared.plant_contract.digests import _canonical_value, _semantic_digest
    from environments.shared.plant_contract.manifest import _load_environment
    from environments.shared.plant_contract.policy_layer import _policy_interface_payload

    env = _load_environment(entrypoint)(reset_noise_scale=0.0)
    try:
        payload = _policy_interface_payload(env.model, env, version, require_backend_parity=True)
    finally:
        env.close()
    out.emit("policy", species, "WHOLE", _semantic_digest(POLICY_INTERFACE_SCHEMA, payload))
    for key in sorted(payload):
        out.emit("policy", species, key, _semantic_digest("breakdown", _canonical_value(payload[key])))
    for group in ("interface_implementations", "jax_interface"):
        value = payload.get(group)
        if isinstance(value, dict):
            for sub in sorted(value):
                out.emit(
                    "policy", species, f"{group}.{sub}", _semantic_digest("breakdown", _canonical_value(value[sub]))
                )


def stage_section(out: _Snapshot, identities: dict[str, Any]) -> None:
    """Per stage: task fingerprint, gate, hyperparameters, stage_config.json body."""
    from environments.shared.config import SPECIES_NAMES, load_stage_config
    from environments.shared.stage_manifest import load_stage_manifest

    with tempfile.TemporaryDirectory(prefix="digest_snapshot_") as scratch:
        for species in SPECIES_NAMES:
            manifest = out.guard(f"stage.{species}.manifest", load_stage_manifest, species)
            if manifest is None:
                continue
            for entry in manifest.stages:
                config = out.guard(f"stage.{species}.{entry.id}.config", load_stage_config, species, entry.reference)
                if config is not None:
                    _stage_lines(out, species, entry, config, identities.get(species), Path(scratch))


def _stage_lines(
    out: _Snapshot, species: str, entry: Any, config: dict[str, Any], identity: Any, scratch: Path
) -> None:
    from environments.shared.config import hyperparameters_sha256
    from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
    from environments.shared.task_fingerprint import derive_stage_task_fingerprint

    ref = entry.reference
    label = f"{species}\t{entry.id}"
    prefix = f"stage.{species}.{entry.id}"
    out.emit("stage", label, "config_file", entry.config_file)
    if identity is not None:
        # The call train_base makes before it writes stage_config.json.
        fingerprint = out.guard(
            f"{prefix}.task",
            derive_stage_task_fingerprint,
            species=species,
            stage=ref,
            backend="stable-baselines3",
            env_kwargs=config.get("env_kwargs", {}),
            plant_identity=identity.to_dict(),
        )
        if fingerprint is not None:
            out.emit("stage", label, "task_sha256", fingerprint["task_sha256"])
    curriculum = config.get("curriculum_kwargs", {})
    gate = out.guard(f"{prefix}.gate", lambda: gate_config_sha256(gate_config_view(curriculum)))
    if gate is not None:
        out.emit("stage", label, "gate_sha256", gate)
    for algorithm in ("PPO", "SAC"):
        digest = out.guard(f"{prefix}.hyperparameters", hyperparameters_sha256, config, algorithm)
        if digest is not None:
            out.emit("stage", label, f"hyperparameters_sha256.{algorithm}", digest)
    for algorithm in ("PPO", "SAC"):
        stage_dir = scratch / species / str(entry.id) / algorithm
        digest = out.guard(f"{prefix}.view", _stage_config_view_sha256, species, ref, config, algorithm, stage_dir)
        if digest is not None:
            out.emit("stage", label, f"stage_config_view_sha256.{algorithm}", digest)


def _stage_config_view_sha256(species: str, stage: Any, config: dict[str, Any], algorithm: str, stage_dir: Path) -> str:
    """Digest of the stage_config.json a new stage records, minus UNPINNED_STAGE_CONFIG_KEYS.

    Written by the real writer, with the env class train_base passes, so a
    change to what a stage records shows here.
    """
    from environments.shared.config import save_stage_config
    from environments.shared.result_bundle.hashing import canonical_json_sha256
    from environments.shared.species_registry import get_species_config

    env_class = get_species_config(species).env_class
    path = save_stage_config(stage_dir, stage, config, algorithm, env_class=env_class, species=species)
    record = json.loads(path.read_text())
    return canonical_json_sha256({key: value for key, value in record.items() if key not in UNPINNED_STAGE_CONFIG_KEYS})


def recovery_section(out: _Snapshot) -> None:
    """Every committed recovery calibration: loads, and its file digest."""
    from environments.shared.recovery_calibration import load_recovery_calibration

    for path in sorted(Path("configs").glob("*/recovery_calibration.json")):
        species = path.parent.name
        if out.guard(f"recovery.{species}", load_recovery_calibration, species) is not None:
            out.emit("recovery", species, "load_recovery_calibration", "OK")
        out.emit("recovery", species, "file_sha256", "sha256:" + _file_sha(path))


def behavior_section(out: _Snapshot) -> None:
    """Every behavior recipe: its file digest, its identity, its source digests."""
    for recipe_path in sorted(Path("configs").glob("*/behaviors/*.toml")):
        species = recipe_path.parent.parent.name
        label = f"{species}\t{recipe_path.stem}"
        out.emit("behavior", label, "recipe_sha256", _file_sha(recipe_path))
        value = out.guard(f"behavior.{species}.{recipe_path.stem}", _behavior_identity, recipe_path, species)
        if value is not None:
            out.emit("behavior", label, "behavior_identity_sha256", _sha(value))
            sources = {**value.get("sources", {}), **value.get("sampler_sources", {})}
            for source, digest in sorted(sources.items()):
                out.emit("behavior", label, f"source:{source}", digest)


def _behavior_identity(recipe_path: Path, species: str) -> Any:
    """The identity the recipe's environment records, built as train_behaviors builds it."""
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    _, commands, terrain, kwargs = read_recipe(recipe_path, species)
    env = create_behavior_env(species, commands=commands, terrain=terrain, run_seed=0, **kwargs)
    try:
        return env.behavior_identity
    finally:
        env.close()


def _refuse(message: str) -> int:
    print(f"digest_snapshot: refused: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print every identity and digest a certified run depends on, one per line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--repo", default=".", help="the checkout to measure (default: the working directory)")
    parser.add_argument(
        "--block-optional-backends",
        action="store_true",
        help="make jax, flax, optax, mujoco.mjx, ray, mjlab, hypertune and google.cloud imports fail",
    )
    parser.add_argument("--skip-behaviors", action="store_true", help="leave out the behavior section (the slow part)")
    parser.add_argument("--debug", action="store_true", help="print the traceback of every ERROR line to stderr")
    args = parser.parse_args(argv)
    started = time.monotonic()

    if args.block_optional_backends:
        loaded = [name for name in BLOCKED_BACKENDS if any(m == name or m.startswith(name + ".") for m in sys.modules)]
        if loaded:
            return _refuse(f"--block-optional-backends, but already imported: {', '.join(loaded)}")
        sys.meta_path.insert(0, _BackendBlocker())

    repo = Path(args.repo).resolve()
    if not (repo / "environments" / "__init__.py").is_file():
        return _refuse(f"--repo {repo} is not a checkout (no environments/__init__.py)")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import environments

    # Under ``python -m``, or when another checkout is installed, the package
    # may already come from elsewhere: then the plant and stage digests would
    # not be --repo's.
    origin = environments.__file__
    if origin is None or Path(origin).resolve() != repo / "environments" / "__init__.py":
        return _refuse(
            f"environments is imported from {origin}, not from --repo {repo}; "
            f"run this file by path with PYTHONPATH={repo} (see --help)"
        )

    logging.disable(logging.CRITICAL)
    out = _Snapshot(repo, debug=args.debug)
    identities = plant_section(out)
    stage_section(out, identities)
    recovery_section(out)
    if not args.skip_behaviors:
        behavior_section(out)
    print(
        f"digest_snapshot: {out.lines} lines, {out.errors} errors, {time.monotonic() - started:.1f} s ({repo})",
        file=sys.stderr,
    )
    return 1 if out.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
