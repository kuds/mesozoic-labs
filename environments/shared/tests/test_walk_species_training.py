"""Every model's committed walk trains, saves, reloads and is judged, under PPO and SAC (decision D-D62).

A short smoke run per model and algorithm, explicitly not a trained walker:

* **The stance parent** is built from scratch: 64 steps of the committed stance
  through ``train_base.train`` (its gate is never consulted), so it is
  command-free, as every stance is.
* **The walk** is the committed ``locomotion`` stage, warm-started from that
  parent through ``train_base.train`` under ``initialize_next_stage`` for 256
  steps: the entry zeroes the parent's command columns and proves the policy
  command-blind (``policy_loading.assert_command_blind``), the cruise ramp
  starts at its first fraction and the final checkpoint records its
  position, and the end of training chooses the handoff by the gated
  statistic (``models/handoff_choice.json``).
* **The reload**: the chosen pair loads through ``load_sb3_checkpoint`` with
  the command slice at identity, and now reads the command: the blindness
  proof refuses it, since training moved its command columns.
* **The judge**: the post-stage path (``generate_stage_artifacts``, the
  notebook's) rolls the ``command_tracking/v1`` walk panel on that pair and
  writes ``command_tracking_report.json`` and ``gate_verdict.json``.  The
  verdict is a FAIL, as it must be: a one-episode cruise family's success
  bound is at most LCB95(1, 1) = 0.025, under the 0.60 bar.

The test changes the budget, never the task: the algorithm blocks shrink to
fit a few hundred steps (PPO 64-step rollouts and one epoch; SAC learning from
step 64 on a small buffer), training runs no evaluation, and the walk panel's
minimums (``min_cruise_episodes``, ``min_stand_episodes``) are one episode
each.  The walk's recorded task digest is the committed one.  One model and
algorithm per test; at reduced depth the SB3 job runs the
compsognathus_longipes/robot pair only, the full depth every model
(consolidation PR-3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")

from environments.shared import policy_loading, train_base  # noqa: E402
from environments.shared.command_frame import command_identity_gap, command_slice  # noqa: E402
from environments.shared.command_tracking_panel import plan_command_panel, reset_command_family  # noqa: E402
from environments.shared.config import build_env, load_all_stages  # noqa: E402
from environments.shared.constants import PUBLICATION_SEED_START  # noqa: E402
from environments.shared.curriculum.checkpoints import (  # noqa: E402
    handoff_choice_candidates_problem,
    recorded_handoff_choice,
    select_handoff_checkpoint,
)
from environments.shared.curriculum.command_tracking_evidence import COMMAND_TRACKING_REPORT_JSON  # noqa: E402
from environments.shared.curriculum.command_tracking_gate import (  # noqa: E402
    COMMAND_TRACKING_GATE_KIND,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.cruise_ramp import (  # noqa: E402
    MODEL_CRUISE_RAMP_ATTRIBUTE,
    cruise_ramp_record,
    resume_ramp_steps,
)
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.policy_loading import ACTION_DELTA_ATOL, PolicyLoadError  # noqa: E402
from environments.shared.reporting import generate_stage_artifacts  # noqa: E402
from environments.shared.reporting.command_tracking_report import HANDOFF_SELECTION_SEED_START  # noqa: E402
from environments.shared.reporting.gates import command_tracking_report_refusals  # noqa: E402
from environments.shared.result_bundle import sha256_file  # noqa: E402
from environments.shared.species_names import model_ids  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest  # noqa: E402
from environments.shared.task_fingerprint import (  # noqa: E402
    MODEL_TASK_LINEAGE_ATTRIBUTE,
    read_checkpoint_task_fingerprint,
    stage_task_fingerprint,
)

MODEL_IDS = model_ids(backend="stable-baselines3")
ALGORITHMS = ("ppo", "sac")
STANCE_TIMESTEPS = 64
WALK_TIMESTEPS = 256
#: The run seed.  Its reset draw (make_env's ``seed + rank``, one worker) is a cruise episode, so the walk's
#: first updates see a live command (asserted below).
SEED = 5
#: The algorithm blocks shrunk to the budget: hyperparameters, outside the task digest.
SMALL_ALGORITHM: dict[str, dict[str, Any]] = {
    "ppo": {"n_steps": 64, "batch_size": 64, "n_epochs": 1},
    "sac": {"learning_starts": 64, "buffer_size": 2_000, "batch_size": 64, "train_freq": 16, "gradient_steps": 1},
}
#: The walk panel's minimums for this test only: one cruise and one stand episode.
TINY_PANEL = {"min_cruise_episodes": 1, "min_stand_episodes": 1}
#: No in-training evaluation or periodic save within the budget, and no post-training panels.
TRAIN_OPTIONS: dict[str, Any] = dict(
    n_envs=1,
    seed=SEED,
    eval_freq=1_000_000,
    save_freq=1_000_000,
    verbose=0,
    use_tensorboard=False,
    post_eval_episodes=0,
    report_metrics=False,
    save_on_interrupt=False,
)


def _configs(model_id: ModelId, algorithm: str) -> tuple[dict[Any, dict[str, Any]], Any, Any]:
    """The committed stage configs with the budget's algorithm blocks and the tiny walk panel."""
    manifest = load_stage_manifest(model_id)
    stance, walk = manifest.by_id("stance").reference, manifest.by_id("locomotion").reference
    configs = load_all_stages(model_id)
    for key in (stance, walk):
        configs[key][f"{algorithm}_kwargs"].update(SMALL_ALGORITHM[algorithm])
    configs[walk]["curriculum_kwargs"].update(TINY_PANEL)
    return configs, stance, walk


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_the_committed_walk_trains_from_a_stance_reloads_and_is_judged(
    model_id: ModelId, algorithm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    configs, stance, walk = _configs(model_id, algorithm)
    config = configs[walk]
    command = config["env_kwargs"]["command_config"]
    assert config["env_kwargs"]["command_mode"] == "heading_and_speed"
    assert config["curriculum_kwargs"]["gate_kind"] == COMMAND_TRACKING_GATE_KIND
    assert command.has_cruise_ramp and command.stand_weight_map(), "the committed walk: a ramp and a stand block"
    assert reset_command_family(command, SEED) == "cruise"
    species = get_species_config(model_id)
    committed = stage_task_fingerprint(model_id, "locomotion")

    # The stance parent: command-free, built from scratch.
    train_base.train(
        species,
        configs,
        stance,
        total_timesteps=STANCE_TIMESTEPS,
        algorithm=algorithm,
        output_dir=str(tmp_path / "stance"),
        **TRAIN_OPTIONS,
    )
    (parent,) = (tmp_path / "stance" / "models").glob("*_final.zip")
    parent_task = read_checkpoint_task_fingerprint(str(parent))
    assert parent_task is not None and "command" not in parent_task

    # The walk, warm-started from it; the entry's blindness proof is captured as it runs.
    probes: list[dict[str, Any]] = []
    prove = policy_loading.assert_command_blind

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        probes.append(prove(*args, **kwargs))
        return probes[-1]

    monkeypatch.setattr(policy_loading, "assert_command_blind", capture)
    stage_dir = tmp_path / "walk"
    train_base.train(
        species,
        configs,
        walk,
        total_timesteps=WALK_TIMESTEPS,
        algorithm=algorithm,
        output_dir=str(stage_dir),
        load_path=str(parent.with_suffix("")),
        task_load_mode="initialize_next_stage",
        **TRAIN_OPTIONS,
    )
    monkeypatch.undo()
    (entry,) = probes
    assert entry["prepared"], "the parent's command columns were zeroed"
    for key in ("max_action_delta", "max_value_delta", "max_probe_vector_action_delta", "max_probe_vector_value_delta"):
        assert entry[key] <= ACTION_DELTA_ATOL, key
    recorded = json.loads((stage_dir / "stage_config.json").read_text())
    assert recorded["task_fingerprint"]["task_sha256"] == committed["task_sha256"], "the committed walk's task"
    assert recorded["run"]["load_mode"] == "initialize_next_stage"
    assert recorded["run"]["parent_checkpoint_sha256"] == sha256_file(parent)

    # The handoff, chosen by the gated statistic on the selection panel.
    models = stage_dir / "models"
    choice, problem = recorded_handoff_choice(models)
    assert problem is None and choice is not None
    assert handoff_choice_candidates_problem(choice, models) is None
    assert choice["selection_seed_start"] == HANDOFF_SELECTION_SEED_START != PUBLICATION_SEED_START
    assert choice["gate_sha256"] == gate_config_sha256(gate_config_view(config["curriculum_kwargs"]))
    assert choice["candidates"][-1]["name"].endswith("_final")
    handoff = select_handoff_checkpoint(models)
    assert handoff is not None
    name, stem, vecnorm = handoff
    assert name == choice["selected"]

    # The reload: identity command slice, the entry's lineage, the ramp's position, and a policy that now reads
    # the command it was blind to at entry.
    model, normalizer, _ = policy_loading.load_sb3_checkpoint(
        f"{stem}.zip", vecnorm, lambda: build_env(model_id, "locomotion")
    )
    try:
        assert command_identity_gap(normalizer.obs_rms) == 0.0
        lineage = getattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE)
        assert lineage["mode"] == "initialize_next_stage"
        assert lineage["zeroed_command_parameters"] == entry["prepared"]
        assert lineage["parent_task_sha256"] == parent_task["task_sha256"]
        assert getattr(model, MODEL_CRUISE_RAMP_ATTRIBUTE) == cruise_ramp_record(
            command, steps_before_session=0, session_start_num_timesteps=0
        )
        assert resume_ramp_steps(f"{stem}.zip", command) == model.num_timesteps >= WALK_TIMESTEPS
        columns = command_slice(int(model.observation_space.shape[0]))
        first = model.policy.get_submodule(entry["prepared"][0].removesuffix(".weight"))
        assert torch.count_nonzero(first.weight[:, columns]) > 0, "training moved the zeroed command columns"
        with pytest.raises(PolicyLoadError, match="Command preparation changed parent"):
            prove(model, normalizer)
    finally:
        normalizer.close()

    # The post-stage judge: the walk panel on the chosen pair, a report bound to the stage, a FAIL verdict.
    results = generate_stage_artifacts(
        species, config, walk, algorithm, stage_dir, SEED, record_videos=False, generate_graphs=False
    )
    report = json.loads((stage_dir / COMMAND_TRACKING_REPORT_JSON).read_text())
    curriculum = config["curriculum_kwargs"]
    assert command_tracking_report_refusals(curriculum, report, stage=walk, stage_dir=stage_dir) == []
    assert report["plan"]["seeds"] == list(plan_command_panel(thresholds_from_curriculum(curriculum), command).seeds)
    assert report["plan"]["families"] == ["cruise", "stand"]
    verdict = json.loads((stage_dir / "gate_verdict.json").read_text())
    assert verdict["passed"] is False is results["gate_passed"] is report["passed"]
    assert verdict["gate_kind"] == COMMAND_TRACKING_GATE_KIND
    assert any("cruise family: success_lcb" in failure for failure in verdict["failures"]), verdict["failures"]
    assert verdict["checkpoint"] == f"models/{choice['selected']}.zip"
    assert verdict["checkpoint_sha256"] == choice["checkpoint_sha256"] == sha256_file(f"{stem}.zip")
    assert verdict["task_sha256"] == committed["task_sha256"]
    assert results["command_profile"] == "walk"
    assert (results["command_cruise_episodes"], results["command_stand_episodes"]) == (1, 1)
