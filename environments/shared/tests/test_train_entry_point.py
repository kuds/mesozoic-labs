"""Unit tests for the unified training entry point (``environments/shared/train.py``).

The module is a thin argv shim: it pulls ``--species <name>`` out of ``sys.argv``,
resolves the species config, and hands the remainder to
:func:`environments.shared.train_base.main`.  It deliberately defers those two
imports to call time so that ``--help``-style misuse fails fast without dragging
in the training stack.

That deferral is also what makes these tests cheap: patching ``main`` and
``get_species_config`` on their *owning modules* is enough, because the
function-level ``from ... import`` statements read the attributes fresh on every
call.  No stable-baselines3 required, and no training is ever started.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from environments.shared import train as train_entry


@pytest.fixture
def recorded_run(monkeypatch):
    """Replace the two names ``_main`` imports, recording what they receive.

    ``argv_at_main`` is captured inside the fake rather than after ``_main``
    returns, because ``_main`` mutates ``sys.argv`` in place and the assertion
    that matters is what the downstream argparse would have seen.
    """
    calls: dict = {}

    def fake_get_species_config(species):
        calls["species"] = species
        return f"config-for-{species}"

    def fake_main(config):
        calls["config"] = config
        calls["argv_at_main"] = list(sys.argv)

    monkeypatch.setattr("environments.shared.species_registry.get_species_config", fake_get_species_config)
    monkeypatch.setattr("environments.shared.train_base.main", fake_main)
    return calls


class TestSpeciesFlagValidation:
    """Misuse of --species must fail loudly instead of reaching train_base."""

    def test_missing_species_flag_exits_with_usage(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["train.py", "train", "--stage", "1"])

        with pytest.raises(SystemExit) as excinfo:
            train_entry._main()

        assert excinfo.value.code == 1
        stderr = capsys.readouterr().err
        assert "--species" in stderr
        # The usage text is the only place a user learns the valid names.
        for name in ("velociraptor", "trex", "brachiosaurus", "dibothrosuchus"):
            assert name in stderr

    def test_species_flag_without_a_value_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["train.py", "train", "--species"])

        with pytest.raises(SystemExit) as excinfo:
            train_entry._main()

        assert excinfo.value.code == 1
        assert "requires a value" in capsys.readouterr().err

    def test_validation_failure_does_not_reach_train_base(self, monkeypatch, recorded_run):
        monkeypatch.setattr(sys, "argv", ["train.py", "train"])

        with pytest.raises(SystemExit):
            train_entry._main()

        assert recorded_run == {}


class TestHandoffToTrainBase:
    """The species is resolved, and the flag pair is removed before hand-off."""

    def test_species_is_resolved_and_its_config_passed_to_main(self, monkeypatch, recorded_run):
        monkeypatch.setattr(sys, "argv", ["train.py", "--species", "velociraptor", "train", "--stage", "1"])

        train_entry._main()

        assert recorded_run["species"] == "velociraptor"
        assert recorded_run["config"] == "config-for-velociraptor"

    def test_species_pair_is_stripped_before_main_parses_argv(self, monkeypatch, recorded_run):
        monkeypatch.setattr(sys, "argv", ["train.py", "--species", "velociraptor", "train", "--stage", "1"])

        train_entry._main()

        assert recorded_run["argv_at_main"] == ["train.py", "train", "--stage", "1"]

    def test_species_pair_is_stripped_from_the_middle_of_argv(self, monkeypatch, recorded_run):
        monkeypatch.setattr(sys, "argv", ["train.py", "curriculum", "--species", "trex", "--n-envs", "4"])

        train_entry._main()

        assert recorded_run["argv_at_main"] == ["train.py", "curriculum", "--n-envs", "4"]

    def test_species_pair_is_stripped_from_the_end_of_argv(self, monkeypatch, recorded_run):
        monkeypatch.setattr(sys, "argv", ["train.py", "curriculum", "--species", "dibo"])

        train_entry._main()

        assert recorded_run["argv_at_main"] == ["train.py", "curriculum"]

    def test_program_name_survives_the_rewrite(self, monkeypatch, recorded_run):
        """argv[0] feeds argparse's usage line, so it must not be dropped."""
        monkeypatch.setattr(sys, "argv", ["/path/to/train.py", "--species", "trex", "train"])

        train_entry._main()

        assert recorded_run["argv_at_main"][0] == "/path/to/train.py"

    def test_an_unknown_species_propagates_the_registry_error(self, monkeypatch):
        """No blanket except: a typo'd species surfaces the registry's message."""
        monkeypatch.setattr(sys, "argv", ["train.py", "--species", "stegosaurus", "train"])

        with pytest.raises(ValueError, match="Unknown species"):
            train_entry._main()


class TestRepositoryRootBootstrap:
    """``_repo_root`` is computed from ``__file__`` by parent depth.

    Depth-counted paths silently change meaning when a module moves, so pin the
    resolved directory against landmarks rather than trusting the arithmetic.
    """

    def test_repo_root_resolves_to_the_repository(self):
        root = Path(train_entry._repo_root)

        assert (root / "pyproject.toml").is_file()
        assert (root / "environments" / "shared" / "train.py").is_file()

    def test_repo_root_is_on_sys_path(self):
        assert train_entry._repo_root in sys.path

    def test_repo_root_is_inserted_when_it_is_missing(self, monkeypatch):
        """The guarded insert is a no-op under pytest, where the root is already
        importable.  It earns its keep for ``python environments/shared/train.py``,
        so drive that branch by reloading with the root taken off ``sys.path``.
        """
        root = train_entry._repo_root
        monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != root])
        assert root not in sys.path

        importlib.reload(train_entry)

        assert sys.path[0] == root


class TestModuleExecution:
    """The ``__main__`` guard is the real entry point; exercise it as one."""

    def test_running_the_module_without_species_fails_with_usage(self):
        result = subprocess.run(
            [sys.executable, "-m", "environments.shared.train", "train"],
            capture_output=True,
            text=True,
            cwd=train_entry._repo_root,
        )

        assert result.returncode == 1
        assert "--species" in result.stderr
