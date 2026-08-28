"""JAX writers on the manifest stage layout (review 2026-08, F16).

The JAX path was the last writer still emitting the pre-manifest
generation: ``setup_output_dirs`` wrote ``stage{N}`` run directories and
``save_jax_stage_artifacts`` named the terminal checkpoint
``stage{stage}_final.pkl`` — which would have minted a
``stagerecovery_final.pkl`` the moment the JAX path gained semantic
stages.  These tests pin the writers to ``stage_dirname`` (NN_id
directories) and ``stage_label`` (file prefixes), and round-trip the new
layout through ``find_stage_dir`` — the helper the bundle readers
(``result_bundle/evidence.py``, ``result_bundle/audit.py``,
``reporting/bundles.py``) resolve stage directories with.
"""

import json

import pytest

from environments.shared.jax_setup import setup_output_dirs
from environments.shared.stage_manifest import find_stage_dir, stage_label

from .reporting_helpers import make_stage_result, plant_identity

_TIMESTAMP = "20260823_000000"


class TestJaxRunDirNaming:
    """setup_output_dirs must write the NN_id generation, never stage{N}."""

    def test_legacy_integer_stage_writes_position_prefixed_dirname(self, tmp_path):
        dirs = setup_output_dirs("velociraptor", 1, storage_root=tmp_path, timestamp=_TIMESTAMP)
        assert dirs["run_dir"] == tmp_path / "velociraptor" / "jax" / _TIMESTAMP
        assert dirs["stage_dir"] == dirs["run_dir"] / "01_stance"
        assert dirs["model_dir"] == dirs["stage_dir"] / "models"
        assert dirs["model_dir"].is_dir()

    def test_integer_reference_resolves_by_legacy_number_not_position(self, tmp_path):
        # trex locomotion is legacy stage 2 at manifest position 3.  A
        # "stage2" or "02_*" spelling here would be exactly the silent
        # renumbering the manifest exists to prevent.
        dirs = setup_output_dirs("trex", 2, storage_root=tmp_path, timestamp=_TIMESTAMP)
        assert dirs["stage_dir"].name == "03_locomotion"

    def test_semantic_stage_gets_its_id_dirname(self, tmp_path):
        dirs = setup_output_dirs("trex", "recovery", storage_root=tmp_path, timestamp=_TIMESTAMP)
        assert dirs["stage_dir"].name == "02_recovery"
        assert dirs["model_dir"].is_dir()


class TestFinalCheckpointNaming:
    """The terminal JAX checkpoint is stage_label-prefixed."""

    @pytest.fixture(autouse=True)
    def _clean_repository_state(self, monkeypatch):
        from environments.shared.result_bundle import provenance

        monkeypatch.setattr(
            provenance,
            "_repository_state",
            lambda _root: {
                "repository_url": "https://github.com/kuds/mesozoic-labs.git",
                "repository_commit": "a" * 40,
                "repository_dirty": False,
                "repository_patch_sha256": None,
            },
        )

    @staticmethod
    def _save(tmp_path, stage=1):
        from environments.shared.reporting import save_jax_stage_artifacts

        dirs = setup_output_dirs("velociraptor", stage, storage_root=tmp_path, timestamp=_TIMESTAMP)
        stage_config = {
            "name": f"Stage {stage}",
            "description": f"Curriculum stage {stage}",
            "env_kwargs": {"forward_vel_weight": float(stage)},
            "jax_kwargs": {"learning_rate": 3e-4},
            "curriculum_kwargs": {"min_avg_reward": 50.0},
        }
        stage_results = make_stage_result(
            stage=stage,
            model_path=str(dirs["model_dir"] / "best_model.pkl"),
            best_eval_reward=55.0,
            best_eval_timestep=50000,
            mean_distance_traveled=2.5,
        )
        paths = save_jax_stage_artifacts(
            species="velociraptor",
            stage=stage,
            stage_config=stage_config,
            stage_results=stage_results,
            stage_dir=dirs["stage_dir"],
            run_dir=dirs["run_dir"],
            eval_results=_FakeEvalResults(),
            params={"dense": [1.0, 2.0]},
            obs_rms=None,
            reward_cfg={"forward_vel_weight": 1.0},
            plant_identity=plant_identity(),
        )
        return paths, dirs

    def test_save_accepts_the_manifest_dirname_and_labels_the_final_pkl(self, tmp_path):
        paths, dirs = self._save(tmp_path)
        assert dirs["stage_dir"].name == "01_stance"
        assert paths["final_model"] == dirs["model_dir"] / f"{stage_label(1)}_final.pkl"
        # Identical to the historical name for legacy integers — old
        # consumers of int-stage JAX runs keep resolving.
        assert paths["final_model"].name == "stage1_final.pkl"
        assert paths["final_model"].exists()

    def test_final_pkl_name_goes_through_stage_label_not_a_literal(self):
        # save_jax_stage_artifacts still accepts only integer stages, so
        # the semantic case cannot be exercised end-to-end; pin the writer
        # to stage_label so widening the stage parameter cannot resurrect
        # the f"stage{stage}" literal.
        import inspect

        from environments.shared.reporting import stage_artifacts

        source = inspect.getsource(stage_artifacts.save_jax_stage_artifacts)
        assert 'f"{stage_label(stage)}_final.pkl"' in source
        assert 'f"stage{stage}_final.pkl"' not in source

    def test_stage_label_names_the_semantic_final_checkpoint(self):
        assert f"{stage_label('recovery')}_final.pkl" == "recovery_final.pkl"

    def test_saved_stage_resolves_through_the_bundle_reader_helper(self, tmp_path):
        paths, dirs = self._save(tmp_path)
        resolved = find_stage_dir(dirs["run_dir"], 1)
        assert resolved == dirs["stage_dir"]
        assert (resolved / "stage_result.json").exists()
        assert json.loads((resolved / "stage_result.json").read_text())["stage"] == 1


class TestReaderRoundtrip:
    """Artifacts written at the new paths are found by the existing readers."""

    def test_find_stage_dir_locates_int_stage_artifacts_in_the_new_layout(self, tmp_path):
        dirs = setup_output_dirs("velociraptor", 1, storage_root=tmp_path, timestamp=_TIMESTAMP)
        (dirs["stage_dir"] / "stage_result.json").write_text(json.dumps({"stage": 1}))
        resolved = find_stage_dir(dirs["run_dir"], 1)
        assert resolved == dirs["stage_dir"]
        assert json.loads((resolved / "stage_result.json").read_text()) == {"stage": 1}

    def test_find_stage_dir_locates_semantic_stage_artifacts(self, tmp_path):
        dirs = setup_output_dirs("trex", "recovery", storage_root=tmp_path, timestamp=_TIMESTAMP)
        fake = dirs["model_dir"] / f"{stage_label('recovery')}_final.pkl"
        fake.write_bytes(b"ckpt")
        resolved = find_stage_dir(dirs["run_dir"], "recovery")
        assert resolved == dirs["stage_dir"]
        assert (resolved / "models" / "recovery_final.pkl").read_bytes() == b"ckpt"


class _FakeEvalResults:
    """Minimal stand-in for jax_eval.EvalResults."""

    def __init__(self):
        self.rewards = [50.0, 55.0, 60.0]
        self.lengths = [200, 210, 220]
        self.forward_vels = [0.5, 0.6, 0.55]
        self.distances = [1.0, 1.2, 1.1]
        self.successes = [True, True, True]
        self.diag_tilt = [0.1, 0.2, 0.15]
        self.diag_fwd_vel = [0.5, 0.6, 0.55]
        self.diag_pelvis_h = [0.7, 0.72, 0.71]
        self.diag_energy = [0.01, 0.02, 0.015]
        self.diag_l_foot = [1.0, 0.0, 1.0]
        self.diag_r_foot = [0.0, 1.0, 0.0]
        self.diag_reward_components = {"forward": [0.3, 0.4, 0.35]}
        self.diag_reward_diagnostics = {"alive_gate": [0.7, 0.8, 0.9]}
