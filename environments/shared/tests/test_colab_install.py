"""``notebooks/colab_install.py``, the installer the SB3 notebook runs on Colab after its checkout.

The script is imported by path and run with its subprocess calls, ``uv`` lookup and installed versions faked:
the legacy purge touches only what is present, the install is one resolver pass (uv first, pip when uv is
missing or fails), ``--no-build-isolation`` only when the interpreter can build the project itself, the
verification reports every kind of mismatch, and ``--dry-run`` runs nothing. One test runs the real script with
``--dry-run`` against this repository, which needs no network and installs nothing. The real install, on a
runtime set up like Colab's (CPU torch and a legacy gym preinstalled), is CI's ``colab-install`` job: it
downloads hundreds of megabytes, too slow for the shared suite.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from environments.shared.paths import REPOSITORY_ROOT

SCRIPT = REPOSITORY_ROOT / "notebooks" / "colab_install.py"

#: The installed project's requirement strings, in importlib.metadata.requires' form.
PROJECT_REQUIRES = [
    "mujoco==3.10.0",
    "gymnasium>=0.29.0",
    "numpy>=1.24.0",
    'torch>=2.6; extra == "train"',
    'stable-baselines3>=2.9.0; extra == "train"',
    'mediapy>=1.1.0; extra == "viz"',
    'pytest>=7.0; extra == "test"',
    'mesozoic-labs[test]; extra == "dev"',
    'mypy==2.3.1; extra == "dev"',
    'mesozoic-labs[dev,train,viz]; extra == "all"',
]
#: An environment that satisfies PROJECT_REQUIRES for every extra and the repository's constraints.
SATISFIED = {
    "mujoco": "3.10.0",
    "gymnasium": "1.3.0",
    "numpy": "2.1.0",
    "torch": "2.8.0+cu126",
    "stable-baselines3": "2.9.0",
    "mediapy": "1.2.2",
    "pytest": "8.3.0",
    "mypy": "2.3.1",
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("colab_install_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ci() -> ModuleType:
    return _load_script()


class FakeRuntime:
    """Installed distributions, importable modules, uv on PATH and the commands run, for one test."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, module: ModuleType, installed: dict[str, str]) -> None:
        self.installed = dict(installed)
        self.importable = {"setuptools", "wheel"}
        self.uv: str | None = "/usr/local/bin/uv"
        self.commands: list[list[str]] = []
        self.failing: set[str] = set()  # command[0] basenames whose run fails
        self.on_install: dict[str, str] = {}  # what a successful install leaves installed
        self.project_requires: list[str] | None = list(PROJECT_REQUIRES)
        monkeypatch.setattr(module, "installed_version", self.installed.get)
        monkeypatch.setattr(module, "importable", lambda name: name in self.importable)
        monkeypatch.setattr(module, "project_requirements", self._requires)
        monkeypatch.setattr(shutil, "which", lambda name: self.uv if name == "uv" else None)
        monkeypatch.setattr(subprocess, "run", self._run)

    def _requires(self) -> list[str] | None:
        if self.project_requires is None:
            raise importlib.metadata.PackageNotFoundError("mesozoic-labs")
        return self.project_requires

    def _run(self, command: list[str], **kwargs: object) -> None:
        assert kwargs.get("check") is True
        self.commands.append(command)
        if Path(command[0]).name in self.failing:
            raise subprocess.CalledProcessError(2, command)
        if "uninstall" in command:
            for name in command[command.index("-q") + 1 :]:
                self.installed.pop(name, None)
        elif "install" in command:
            self.installed.update(self.on_install)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, ci: ModuleType) -> FakeRuntime:
    fake = FakeRuntime(
        monkeypatch,
        ci,
        {**SATISFIED, "setuptools": "78.1.0", "gym": "0.25.2", "dopamine-rl": "4.1.2", "gymnasium": "1.2.0"},
    )
    fake.on_install = {"gymnasium": "1.3.0"}
    return fake


def _uninstall(*names: str) -> list[str]:
    return [sys.executable, "-m", "pip", "uninstall", "-y", "-q", *names]


def _uv(*args: str) -> list[str]:
    return ["/usr/local/bin/uv", "pip", "install", "--python", sys.executable, *args]


def _pip(*args: str) -> list[str]:
    return [sys.executable, "-m", "pip", "install", "-q", *args]


def _install_args(extras: str = "train,viz", *, isolation: bool = False) -> list[str]:
    flag = [] if isolation else ["--no-build-isolation"]
    return [*flag, "-e", f"{REPOSITORY_ROOT}[{extras}]", "-c", str(REPOSITORY_ROOT / "constraints.txt")]


# --- constraints and arguments -------------------------------------------------------------------------------


def test_read_constraints_skips_comments_and_blank_lines(ci, tmp_path):
    path = tmp_path / "constraints.txt"
    path.write_text(
        "# header\n\n   # an indented comment\nmujoco==3.10.0  # the plant contract's\n\n"
        "gymnasium==1.3.0\nnumpy<3; python_version >= '3.11'\n"
    )
    requirements = ci.read_constraints(path)
    assert [str(requirement) for requirement in requirements] == [
        "mujoco==3.10.0",
        "gymnasium==1.3.0",
        'numpy<3; python_version >= "3.11"',
    ]


@pytest.mark.parametrize("line", ["-r other.txt", "--index-url https://example.invalid", "mujoco ==== 3"])
def test_read_constraints_rejects_options_and_malformed_lines(ci, tmp_path, line):
    path = tmp_path / "constraints.txt"
    path.write_text(f"mujoco==3.10.0\n{line}\n")
    with pytest.raises(ci.InstallError, match=r"constraints\.txt:2: "):
        ci.read_constraints(path)


def test_parse_extras(ci):
    assert ci.parse_extras(" train, viz ,,") == ("train", "viz")
    assert ci.parse_extras("") == ()
    with pytest.raises(ci.InstallError, match="invalid extra"):
        ci.parse_extras("train,viz]")


# --- the install -----------------------------------------------------------------------------------------------


def test_purges_only_present_legacy_packages_then_installs_in_one_uv_pass(ci, runtime, capsys):
    assert ci.main([]) == 0
    assert runtime.commands == [_uninstall("gym", "dopamine-rl"), _uv(*_install_args())]
    assert "gym" not in runtime.installed and runtime.installed["torch"] == "2.8.0+cu126"
    out = capsys.readouterr().out
    assert "Uninstalling legacy packages: gym, dopamine-rl" in out
    assert (
        "Installed versions: mujoco=3.10.0, gymnasium=1.3.0, stable-baselines3=2.9.0, torch=2.8.0+cu126, numpy=2.1.0"
        in out
    )


def test_nothing_to_purge_runs_no_uninstall(ci, runtime):
    del runtime.installed["gym"], runtime.installed["dopamine-rl"]
    assert ci.main(["--extras", "train"]) == 0
    assert runtime.commands == [_uv(*_install_args("train"))]


def test_a_failing_uv_falls_back_to_pip(ci, runtime, capsys):
    # uv can refuse Colab's interpreter as system-managed; pip installs there.
    runtime.failing.add("uv")
    assert ci.main([]) == 0
    assert runtime.commands[1:] == [_uv(*_install_args()), _pip(*_install_args())]
    assert "uv failed" in capsys.readouterr().out


def test_an_unrunnable_uv_falls_back_to_pip(ci, runtime, monkeypatch):
    def run(command, **kwargs):
        if command[0].endswith("uv"):
            raise FileNotFoundError(command[0])
        return runtime._run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    assert ci.main([]) == 0
    assert runtime.commands[-1] == _pip(*_install_args())


def test_without_uv_pip_installs(ci, runtime):
    runtime.uv = None
    assert ci.main([]) == 0
    assert runtime.commands == [_uninstall("gym", "dopamine-rl"), _pip(*_install_args())]


def test_a_failing_pip_exits_non_zero_and_says_which_command(ci, runtime, capsys):
    runtime.uv = None
    del runtime.installed["gym"], runtime.installed["dopamine-rl"]
    runtime.failing.add(Path(sys.executable).name)
    assert ci.main(["--extras", "train"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("colab_install: ") and "pip install" in err and "failed" in err


@pytest.mark.parametrize(
    "setuptools,importable,expected",
    [
        ("78.1.0", {"setuptools", "wheel"}, True),
        ("68.0.0", {"setuptools", "wheel"}, True),
        ("67.8.0", {"setuptools", "wheel"}, False),
        ("78.1.0", {"setuptools"}, False),
        (None, {"wheel"}, False),
        ("not-a-version", {"setuptools", "wheel"}, False),
    ],
)
def test_build_isolation_is_skipped_only_when_setuptools_68_and_wheel_are_importable(
    ci, monkeypatch, setuptools, importable, expected
):
    installed = {"setuptools": setuptools} if setuptools else {}
    monkeypatch.setattr(ci, "installed_version", installed.get)
    monkeypatch.setattr(ci, "importable", lambda name: name in importable)
    assert ci.can_skip_build_isolation() is expected


def test_an_interpreter_without_wheel_builds_in_isolation(ci, runtime):
    runtime.importable.discard("wheel")
    assert ci.main([]) == 0
    assert runtime.commands[-1] == _uv(*_install_args(isolation=True))


def test_dry_run_runs_nothing_and_verifies_nothing(ci, runtime, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("--dry-run ran a command"))
    runtime.project_requires = None  # verification would fail
    assert ci.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "$ " + " ".join(_uninstall("gym", "dopamine-rl")) in out
    assert "$ /usr/local/bin/uv pip install" in out
    assert "Dry run: nothing installed, nothing verified." in out
    assert runtime.installed["gym"] == "0.25.2"


def test_a_missing_constraints_file_or_repository_exits_non_zero(ci, runtime, tmp_path, capsys):
    assert ci.main(["--constraints", str(tmp_path / "absent.txt")]) == 1
    assert "constraints file not found" in capsys.readouterr().err
    assert ci.main(["--repo", str(tmp_path)]) == 1
    assert "has no pyproject.toml" in capsys.readouterr().err
    assert runtime.commands == []


# --- verification ----------------------------------------------------------------------------------------------


def _verify(ci, monkeypatch, installed, *, extras=("train", "viz"), requires=PROJECT_REQUIRES, constraints=None):
    def project_requirements():
        if requires is None:
            raise importlib.metadata.PackageNotFoundError("mesozoic-labs")
        return list(requires)

    monkeypatch.setattr(ci, "installed_version", installed.get)
    monkeypatch.setattr(ci, "project_requirements", project_requirements)
    if constraints is None:
        constraints = ci.read_constraints(REPOSITORY_ROOT / "constraints.txt")
    return ci.verification_errors(extras, constraints)


def test_verification_passes_on_a_satisfied_environment(ci, monkeypatch):
    assert _verify(ci, monkeypatch, SATISFIED) == []
    # A local build label satisfies a public pin (PEP 440), as Colab's +cuXXX torch does.
    assert _verify(ci, monkeypatch, {**SATISFIED, "mujoco": "3.10.0+local"}) == []


def test_verification_ignores_pins_and_extras_not_installed_or_selected(ci, monkeypatch):
    without_training = {
        name: version for name, version in SATISFIED.items() if name not in ("torch", "stable-baselines3")
    }
    # stable-baselines3's pin applies only once it is installed; the train extra is not selected.
    assert _verify(ci, monkeypatch, without_training, extras=("viz",)) == []


def test_verification_reports_every_mismatch(ci, monkeypatch):
    installed = {**SATISFIED, "gymnasium": "1.4.0", "stable-baselines3": "2.8.0", "mujoco": "3.9.0"}
    del installed["mediapy"]
    errors = _verify(ci, monkeypatch, installed)
    assert errors == [
        "constraint: mujoco 3.9.0 is installed; required: mujoco==3.10.0",
        "constraint: stable-baselines3 2.8.0 is installed; required: stable-baselines3==2.9.0",
        "constraint: gymnasium 1.4.0 is installed; required: gymnasium==1.3.0",
        "mesozoic-labs: mujoco 3.9.0 is installed; required: mujoco==3.10.0",
        'mesozoic-labs: stable-baselines3 2.8.0 is installed; required: stable-baselines3>=2.9.0; extra == "train"',
        'mesozoic-labs: mediapy is not installed (required: mediapy>=1.1.0; extra == "viz")',
    ]


def test_verification_follows_extras_named_through_the_project(ci, monkeypatch):
    # all = dev,train,viz; dev = test + mypy: each one's requirements count.
    installed = {**SATISFIED}
    del installed["pytest"]
    errors = _verify(ci, monkeypatch, installed, extras=("all",))
    assert errors == ['mesozoic-labs: pytest is not installed (required: pytest>=7.0; extra == "test")']


def test_verification_fails_when_the_project_is_not_installed(ci, monkeypatch):
    assert _verify(ci, monkeypatch, SATISFIED, requires=None) == ["mesozoic-labs is not installed"]


def test_a_failed_verification_exits_non_zero_listing_each_mismatch(ci, runtime, capsys):
    runtime.on_install = {}  # the install leaves gymnasium 1.2.0 behind
    runtime.installed["stable-baselines3"] = "2.8.0"
    assert ci.main([]) == 1
    err = capsys.readouterr().err
    assert err.startswith("colab_install: verification failed:\n")
    assert "gymnasium 1.2.0 is installed; required: gymnasium==1.3.0" in err
    assert "stable-baselines3 2.8.0 is installed; required: stable-baselines3==2.9.0" in err


# --- the real script ---------------------------------------------------------------------------------------------


def test_the_real_script_dry_runs_against_this_repository():
    """The script as Colab runs it, by path with its default arguments, in --dry-run: it parses the repository's
    constraints and prints one install of the project with its extras under them."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"], capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stderr
    install_lines = [line for line in result.stdout.splitlines() if line.startswith("$ ") and " install " in line]
    assert install_lines, result.stdout
    assert f"{REPOSITORY_ROOT}[train,viz]" in install_lines[0]
    assert f"-c {REPOSITORY_ROOT / 'constraints.txt'}" in install_lines[0]
    assert result.stdout.rstrip().endswith("Dry run: nothing installed, nothing verified.")
