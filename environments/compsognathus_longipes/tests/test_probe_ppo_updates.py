"""``scripts/probe_ppo_updates.py`` builds its baseline arm's stance file, taken from ``--baseline-ref``, with the
current environment.  Every stance file before ``3d3ba61`` sets the prey and target arguments and
``reset_height_noise_scale``, which went with the prey (decision D-D59), so such a baseline must be refused before
any arm trains, naming the keys and the last commit whose environment took them."""

from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("stable_baselines3")

from environments.compsognathus_longipes.scripts import probe_ppo_updates  # noqa: E402
from environments.shared.paths import REPOSITORY_ROOT  # noqa: E402

#: The ``[env]`` lines every Compsognathus stance file carried before ``3d3ba61`` (as at ``6c9982d``).
PRE_D_D59_ENV_LINES = (
    "reset_height_noise_scale = 0.0\n"
    "target_approach_weight = 0.0\n"
    "target_reach_bonus = 0.0\n"
    "prey_distance_range = [0.8, 1.2]\n"
    "prey_lateral_range = [0.0, 0.0]\n"
)


def _serve_stance_files(monkeypatch, extra_env_lines: str) -> None:
    """Answer the baseline's ``git show <ref>:<stance.toml>`` with the current file plus *extra_env_lines*."""

    def fake_check_output(command, **kwargs):
        assert command[:2] == ["git", "show"], command
        relative = command[2].split(":", 1)[1]
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert text.count("\n[env]\n") == 1, relative
        return text.replace("\n[env]\n", "\n[env]\n" + extra_env_lines, 1).encode("utf-8")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)


def test_a_baseline_with_the_removed_prey_keys_is_refused_naming_them_and_the_last_runnable_commit(
    tmp_path, monkeypatch
):
    _serve_stance_files(monkeypatch, PRE_D_D59_ENV_LINES)
    with pytest.raises(RuntimeError) as refused:
        probe_ppo_updates.require_a_runnable_baseline(tmp_path, "6c9982d")
    message = str(refused.value)
    for model_id in probe_ppo_updates.MODEL_IDS:
        assert str(model_id) in message
    for key in ("prey_distance_range", "prey_lateral_range", "reset_height_noise_scale", "target_approach_weight"):
        assert key in message
    assert "decision D-D59" in message
    assert f"last took them at commit {probe_ppo_updates.LAST_RUNNABLE_COMMIT}" in message
    assert probe_ppo_updates.LAST_RUNNABLE_COMMIT == "e3cccca"


def test_a_baseline_the_current_environment_builds_is_accepted(tmp_path, monkeypatch):
    _serve_stance_files(monkeypatch, "")
    probe_ppo_updates.require_a_runnable_baseline(tmp_path, "HEAD")
