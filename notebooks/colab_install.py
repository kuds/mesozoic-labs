"""Install this checkout of mesozoic-labs into the running Python and verify the result.

``notebooks/sb3_training.ipynb`` runs it on Google Colab right after checking out ``REPO_REF``, so the
install steps always match the checked-out code:

    python notebooks/colab_install.py [--repo DIR] [--extras train,viz] [--constraints PATH] [--dry-run]

1. Uninstall the legacy packages Colab images may carry that break the install (only those present).
2. Install the project editable with its extras and the repository's ``constraints.txt`` in one resolver
   pass: ``uv pip install`` when uv is on PATH, else, or when uv fails (it can refuse an interpreter it
   treats as system-managed), ``python -m pip install``. Packages that already satisfy the requirements,
   such as Colab's own CUDA torch and numpy, are kept: the constraints pin neither.
3. Verify with ``packaging``: every exact pin of the constraints file that is installed matches, and every
   requirement of mesozoic-labs for the selected extras is satisfied. Any mismatch is listed and the
   script exits non-zero.

Standard library only, apart from ``packaging`` (pip's vendored copy when it is not installed), so it
runs on a bare interpreter. Its tests are environments/shared/tests/test_colab_install.py; CI's
``colab-install`` job runs it on a runtime set up like Colab's.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError:  # a bare interpreter: pip vendors packaging
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement  # type: ignore[no-redef,assignment]
    from pip._vendor.packaging.utils import canonicalize_name  # type: ignore[no-redef,assignment]
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore[no-redef,assignment]

PROJECT = "mesozoic-labs"
DEFAULT_EXTRAS = ("train", "viz")
#: Legacy gym (and its shim) segfaults under NumPy 2.x on some Colab images; dopamine-rl pins gym<=0.25.2,
#: so it raises a resolver conflict once gym is gone.
LEGACY_PACKAGES = ("gym", "gym-notices", "shimmy", "dopamine-rl")
SUMMARY_PACKAGES = ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy")
#: The setuptools floor of pyproject.toml's [build-system]: below it, or without wheel, pip builds in isolation.
SETUPTOOLS_FLOOR = Version("68")
_EXTRA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InstallError(RuntimeError):
    """A step failed; the message says which and why."""


def installed_version(name: str) -> str | None:
    """The installed version of distribution *name*, or None when it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def project_requirements() -> list[str] | None:
    """The installed project's requirement strings (each extra's with its marker); None when it has none."""
    return importlib.metadata.requires(PROJECT)


def read_constraints(path: Path) -> list[Requirement]:
    """The requirements of a constraints file: one per line; ``#`` comments and blank lines are skipped."""
    requirements = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line:
            continue
        if line.startswith("-"):
            raise InstallError(f"{path}:{number}: options are not supported in the constraints file: {line}")
        try:
            requirements.append(Requirement(line))
        except InvalidRequirement as error:
            raise InstallError(f"{path}:{number}: {error}") from error
    return requirements


def parse_extras(text: str) -> tuple[str, ...]:
    extras = tuple(part.strip() for part in text.split(",") if part.strip())
    bad = [extra for extra in extras if not _EXTRA_NAME.match(extra)]
    if bad:
        raise InstallError(f"invalid extra name(s): {', '.join(bad)}")
    return extras


def present_legacy_packages() -> list[str]:
    return [name for name in LEGACY_PACKAGES if installed_version(name) is not None]


def can_skip_build_isolation() -> bool:
    """True when this interpreter can build the project itself: setuptools>=68 and wheel are importable."""
    setuptools_version = installed_version("setuptools")
    if setuptools_version is None or not importable("setuptools") or not importable("wheel"):
        return False
    try:
        return Version(setuptools_version) >= SETUPTOOLS_FLOOR
    except InvalidVersion:
        return False


def install_commands(
    repo: Path, extras: Sequence[str], constraints: Path, *, uv: str | None, no_build_isolation: bool
) -> list[list[str]]:
    """The install commands to try in order: uv's (when uv is on PATH), then pip's."""
    target = f"{repo}[{','.join(extras)}]" if extras else str(repo)
    args = [*(["--no-build-isolation"] if no_build_isolation else []), "-e", target, "-c", str(constraints)]
    commands = [[sys.executable, "-m", "pip", "install", "-q", *args]]  # uv lists its changes; pip's list is long
    if uv:
        commands.insert(0, [uv, "pip", "install", "--python", sys.executable, *args])
    return commands


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def install(commands: Sequence[Sequence[str]], *, dry_run: bool) -> None:
    """Run the first command; on its failure, fall back to the next one."""
    for index, command in enumerate(commands):
        try:
            _run(command, dry_run=dry_run)
            return
        except (subprocess.CalledProcessError, OSError) as error:
            if index + 1 == len(commands):
                if "-q" in command:
                    # pip -q hides the resolver's explanation of a conflict: show it before failing.
                    print("Re-running pip without -q to show why it failed:", flush=True)
                    try:
                        _run([part for part in command if part != "-q"], dry_run=dry_run)
                        return
                    except (subprocess.CalledProcessError, OSError):
                        pass
                raise InstallError(f"{shlex.join(command)} failed: {error}") from error
            print(f"{Path(command[0]).name} failed ({error}); falling back to pip.", flush=True)


def _applies(requirement: Requirement, extras: Iterable[str]) -> bool:
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras))


def _mismatch(requirement: Requirement) -> str | None:
    """Why the installed environment does not satisfy *requirement*, or None when it does."""
    version = installed_version(requirement.name)
    if version is None:
        return f"{requirement.name} is not installed (required: {requirement})"
    if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
        return f"{requirement.name} {version} is installed; required: {requirement}"
    return None


def verification_errors(extras: Sequence[str], constraints: Sequence[Requirement]) -> list[str]:
    """Every installed constraint pin that does not match, and every unmet requirement of the selected extras."""
    errors = []
    for requirement in constraints:
        exact = [spec for spec in requirement.specifier if spec.operator in ("==", "===")]
        if exact and _applies(requirement, ()) and installed_version(requirement.name) is not None:
            problem = _mismatch(requirement)
            if problem:
                errors.append(f"constraint: {problem}")
    try:
        requires = project_requirements() or []
    except importlib.metadata.PackageNotFoundError:
        return [*errors, f"{PROJECT} is not installed"]
    selected = set(extras)
    declared = [Requirement(text) for text in requires]
    while True:  # an extra may name others through the project itself (all = train,viz,dev; dev = test)
        nested = {
            extra
            for requirement in declared
            if canonicalize_name(requirement.name) == PROJECT and _applies(requirement, selected)
            for extra in requirement.extras
        }
        if nested <= selected:
            break
        selected |= nested
    for requirement in declared:
        if canonicalize_name(requirement.name) != PROJECT and _applies(requirement, selected):
            problem = _mismatch(requirement)
            if problem:
                errors.append(f"{PROJECT}: {problem}")
    return errors


def version_summary() -> str:
    return ", ".join(f"{name}={installed_version(name) or 'not installed'}" for name in SUMMARY_PACKAGES)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    parser.add_argument(
        "--extras", default=",".join(DEFAULT_EXTRAS), help="comma-separated extras (default: %(default)s)"
    )
    parser.add_argument("--constraints", type=Path, help="constraints file (default: <repo>/constraints.txt)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    constraints_path = (args.constraints or repo / "constraints.txt").resolve()
    try:
        extras = parse_extras(args.extras)
        if not (repo / "pyproject.toml").is_file():
            raise InstallError(f"{repo} has no pyproject.toml; pass --repo with the repository root")
        if not constraints_path.is_file():
            raise InstallError(f"constraints file not found: {constraints_path}")
        constraints = read_constraints(constraints_path)

        legacy = present_legacy_packages()
        if legacy:
            print(f"Uninstalling legacy packages: {', '.join(legacy)}", flush=True)
            try:
                _run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *legacy], dry_run=args.dry_run)
            except subprocess.CalledProcessError as error:
                raise InstallError(f"uninstalling {', '.join(legacy)} failed: {error}") from error

        print(f"Installing {repo} with extras [{','.join(extras)}] under {constraints_path.name}", flush=True)
        commands = install_commands(
            repo, extras, constraints_path, uv=shutil.which("uv"), no_build_isolation=can_skip_build_isolation()
        )
        install(commands, dry_run=args.dry_run)
        if args.dry_run:
            print("Dry run: nothing installed, nothing verified.")
            return 0

        importlib.invalidate_caches()
        errors = verification_errors(extras, constraints)
        if errors:
            raise InstallError("verification failed:\n  " + "\n  ".join(errors))
    except InstallError as error:
        print(f"colab_install: {error}", file=sys.stderr)
        return 1
    print(f"Installed versions: {version_summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
