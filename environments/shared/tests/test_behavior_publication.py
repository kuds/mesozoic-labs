"""Exercise behavior publication through the real immutable library."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.shared import behavior_certification as certification
from environments.shared.certified_library import copy_recommended, resolve_recommended
from environments.shared.result_bundle import sha256_file
from environments.shared.tests.test_behavior_certification import _identity, _report
from environments.shared.train_behaviors import _copy_explicit_certified_pair


@pytest.fixture
def publish(tmp_path, monkeypatch):
    identity = _identity()
    library = tmp_path / "library"
    calls = []
    monkeypatch.setattr(
        certification, "_panel_env", lambda *a: SimpleNamespace(terrain_families=("flat", "bumps"), close=lambda: None)
    )

    def evaluate(**kwargs):
        calls.append(kwargs)
        model = Path(kwargs["model_path"])
        data = json.loads(model.read_text())
        report = _report(per_family=kwargs["episodes"] // 2)
        for i, episode in enumerate(report["episodes"]):
            episode["episode_seed"] = kwargs["seed_start"] + i
            episode["tracking_fraction"] = data["tracking"]
        report["protocol"]["episode_seeds"] = [e["episode_seed"] for e in report["episodes"]]
        report["protocol"]["environment_run_seed"] = kwargs["run_seed"]
        report["model_sha256"] = sha256_file(model)
        report["normalization_sha256"] = sha256_file(Path(kwargs["normalization_path"]))
        report["training_lineage"] = {
            "seed": data["seed"],
            "parent_model_sha256": "sha256:" + "c" * 64,
            "parent_normalization_sha256": "sha256:" + data["parent_norm"] * 64,
            "stage_start_timesteps": 100,
            "num_timesteps": data["steps"],
            "stage_start_updates": 10,
            "optimizer_updates": data["updates"],
        }
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "evaluation_summary.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(certification, "evaluate_saved_panel", evaluate)

    def run(name, *, seed=1, tracking=0.7, episodes=50, steps=200, updates=11, parent_norm="d"):
        output = tmp_path / name
        output.mkdir()
        recipe = {"behavior": {"name": "difficult_terrain", "species": "trex"}}
        (output / "model.zip").write_text(
            json.dumps(dict(seed=seed, tracking=tracking, steps=steps, updates=updates, parent_norm=parent_norm))
        )
        (output / "vecnormalize.pkl").write_bytes(b"normalization")
        (output / "replay.mp4").write_bytes(b"video")
        (output / "terrain_heatmap.png").write_bytes(b"heatmap")
        (output / "bundle.json").write_text(
            json.dumps(
                {
                    "model_sha256": sha256_file(output / "model.zip").removeprefix("sha256:"),
                    "normalizer_sha256": sha256_file(output / "vecnormalize.pkl").removeprefix("sha256:"),
                    "training_recipe": recipe,
                }
            )
        )
        (output / "run.json").write_text(
            json.dumps(
                {
                    "certification_training_parent_sha256": "sha256:" + "c" * 64,
                    "certification_training_parent_normalization_sha256": "sha256:" + parent_norm * 64,
                }
            )
        )
        result = certification.certify_and_publish_behavior(
            output=output,
            library=library,
            recipe_path=output / "recipe.toml",
            species="trex",
            identity=identity,
            recipe=recipe,
            training_seed=seed,
            comparison_episodes=episodes,
        )
        return result, output

    run.calls = calls
    run.library = library
    run.key = certification.behavior_library_key("trex", "difficult_terrain", identity)
    return run


def test_default_50_paired_comparison_ties_and_material_improvement(publish, tmp_path):
    first, _ = publish("first")
    assert first["passed"] and first["publication"]["recommended"]
    assert [c["episodes"] for c in publish.calls] == [40, 50]
    second, _ = publish("second", seed=2)
    assert second["publication"]["recommended_version"] == first["publication"]["version"]
    third, _ = publish("third", seed=3, tracking=0.95)
    assert third["publication"]["recommended"]
    chosen = copy_recommended(publish.library, publish.key, tmp_path / "future")
    assert chosen["version"] == third["publication"]["version"]
    assert Path(chosen["directory"]).is_relative_to(tmp_path / "future")
    for name in ["replay.mp4", "terrain_heatmap.png", "certification/certificate.json", "comparison/candidate.json"]:
        assert (Path(chosen["directory"]) / name).is_file()
    explicit = _copy_explicit_certified_pair(
        Path(chosen["model"]), Path(chosen["normalizer"]), publish.library, tmp_path / "manual"
    )
    assert explicit["version"] == chosen["version"]
    assert Path(explicit["model"]).is_relative_to(tmp_path / "manual")


def test_changed_episode_count_rebenchmarks_copied_incumbent_on_matching_cases(publish):
    first, _ = publish("first")
    publish.calls.clear()
    result, output = publish("changed", seed=2, episodes=60)
    assert result["publication"]["recommended_version"] == first["publication"]["version"]
    assert [c["episodes"] for c in publish.calls] == [40, 60, 60]
    challenger, incumbent = publish.calls[-2:]
    assert challenger["seed_start"] == incumbent["seed_start"]
    assert challenger["run_seed"] == incumbent["run_seed"]
    assert Path(incumbent["model_path"]).is_relative_to(output / "certified_inputs")
    resolved = resolve_recommended(publish.library, publish.key)
    assert len(resolved["comparison"]["protocol"]["episode_seeds"]) == 60


@pytest.mark.parametrize("problem", ["failed_skill", "zero_training", "rollout_without_update"])
def test_failed_or_untrained_candidate_is_retained_without_recommendation(publish, problem):
    kwargs = {
        "failed_skill": {"tracking": 0.1},
        "zero_training": {"steps": 100},
        "rollout_without_update": {"updates": 10},
    }[problem]
    result, _ = publish("failed", **kwargs)
    assert not result["passed"]
    assert result["publication"]["status"] == "failed"
    assert Path(result["publication"]["directory"]).is_dir()
    assert len(publish.calls) == 1


def test_different_parent_normalization_cannot_count_as_same_training_recipe(publish):
    first, _ = publish("first", parent_norm="d")
    second, _ = publish("second", seed=2, parent_norm="e")
    assert first["publication"]["distinct_seeds"] == 1
    assert second["publication"]["distinct_seeds"] == 1
