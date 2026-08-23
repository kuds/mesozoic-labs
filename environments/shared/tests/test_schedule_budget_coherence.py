"""Entropy-decay horizons must fit inside their stage budgets.

Incident (2026-08 review §3.4 finding 7; first-runs record
TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md §6.3): trex recovery.toml mirrored
stance's ``[ppo]`` verbatim as its warm-start default, so
``ent_coef_decay_timesteps = 7M`` — a derivation for stance's 11M budget —
shipped against a 5M stage.  The decay to ``ent_coef_end = 0.0`` therefore
never completed, and the ``20260821_142144`` pilot trained its entire budget
at ``ent_coef >= 0.0014``, above the 0.001 floor stance's own derivation
record calls stand-still-unlearnable — on the stage whose success event is
quiet stance between pushes.  Nothing tested ``decay <= budget``, so the
mirror-a-schedule-past-its-budget class could recur in any species' config.

The check walks every committed stage config of every species through the
stage manifest loader — the same resolution the trainers use — so a new
species or stage is covered the moment its config lands.
"""

from __future__ import annotations

from pathlib import Path

from environments.shared.config import load_all_stages

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_ROOT = REPO_ROOT / "configs"


def _species_dirs() -> list[str]:
    """Every species with committed stage configs, discovered from configs/."""
    return sorted(p.name for p in CONFIGS_ROOT.iterdir() if p.is_dir())


def test_ent_coef_decay_completes_within_every_committed_stage_budget() -> None:
    checked = 0
    for species in _species_dirs():
        for stage, cfg in load_all_stages(species).items():
            decay = cfg["ppo_kwargs"].get("ent_coef_decay_timesteps")
            budget = cfg["curriculum_kwargs"].get("timesteps")
            if decay is None or budget is None:
                continue
            checked += 1
            assert decay <= budget, (
                f"{species} stage {stage}: ent_coef_decay_timesteps={decay} exceeds "
                f"timesteps={budget}, so the entropy decay can never complete and the "
                "stage trains its whole budget above the intended ent_coef_end — the "
                "recovery-pilot bug this test exists to prevent (2026-08 review §3.4)."
            )
    # Never pass vacuously: every species sets both keys on at least one
    # stage today, so a zero count means the loader or the premise broke.
    assert checked >= 1, "no committed stage carries both ent_coef_decay_timesteps and timesteps"
