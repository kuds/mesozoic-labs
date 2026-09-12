"""Species registry for unified training entry point.

Maps species names to their SpeciesConfig, eliminating the need
for per-species training scripts with nearly identical boilerplate.

Usage:
    python -m environments.shared.train --species velociraptor train --stage 1
    python -m environments.shared.train --species trex curriculum
    python -m environments.shared.train --species brachiosaurus train --stage 2
    python -m environments.shared.train --species dibothrosuchus train --stage 1
"""

from environments.shared.species_names import resolve_species_id, species_display_name, species_display_names
from environments.shared.train_base import SpeciesConfig


def _make_raptor_config() -> SpeciesConfig:
    from environments.velociraptor.envs.raptor_env import RaptorEnv

    return SpeciesConfig(
        species="velociraptor",
        env_class=RaptorEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=strike",
        height_label="Pelvis height",
        stage3_section_label="Hunting",
        # The raptor env only emits strike_success (bite_success was
        # impossible and was removed).
        success_keys=["strike_success"],
    )


def _make_trex_config() -> SpeciesConfig:
    from environments.trex.envs.trex_env import TRexEnv

    return SpeciesConfig(
        species="trex",
        env_class=TRexEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=bite",
        height_label="Pelvis height",
        stage3_section_label="Hunting",
        success_keys=["bite_success", "strike_success"],
    )


def _make_brachio_config() -> SpeciesConfig:
    from environments.brachiosaurus.envs.brachio_env import BrachioEnv

    return SpeciesConfig(
        species="brachiosaurus",
        env_class=BrachioEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=food_reach",
        height_label="Torso height",
        stage3_section_label="Food Reaching",
        success_keys=["food_reached"],
    )


def _make_dibothrosuchus_config() -> SpeciesConfig:
    from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv

    return SpeciesConfig(
        species="dibothrosuchus",
        env_class=DibothrosuchusEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=snap",
        height_label="Trunk height",
        stage3_section_label="Hunting",
        success_keys=["snap_success"],
    )


def _make_compsognathus_config() -> SpeciesConfig:
    from environments.compsognathus.envs import CompsognathusEnv

    return SpeciesConfig(
        species="compsognathus",
        env_class=CompsognathusEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=target_reach",
        height_label="Pelvis height",
        stage3_section_label="Target Reaching",
        success_keys=["target_success"],
    )


def _make_compsognathus_robot_config() -> SpeciesConfig:
    from environments.compsognathus.envs import CompsognathusRobotEnv

    return SpeciesConfig(
        species="compsognathus_robot",
        env_class=CompsognathusRobotEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=target_reach",
        height_label="Pelvis height",
        stage3_section_label="Target Reaching",
        success_keys=["target_success"],
    )


# Lazy registry — factories are called only when the species is selected,
# so we don't import all env modules at startup.
SPECIES_FACTORIES = {
    "velociraptor": _make_raptor_config,
    "raptor": _make_raptor_config,  # alias
    "trex": _make_trex_config,
    "t-rex": _make_trex_config,  # alias
    "brachiosaurus": _make_brachio_config,
    "brachio": _make_brachio_config,  # alias
    "dibothrosuchus": _make_dibothrosuchus_config,
    "dibo": _make_dibothrosuchus_config,  # alias
    "compsognathus": _make_compsognathus_config,
    "compso": _make_compsognathus_config,
    "compsognathus_robot": _make_compsognathus_robot_config,
    "compso-robot": _make_compsognathus_robot_config,
}


def get_species_config(species: str) -> SpeciesConfig:
    """Look up a training config by full name, stable ID or legacy alias."""
    key = resolve_species_id(species)
    factory = SPECIES_FACTORIES.get(key)
    if factory is None:
        available = ", ".join(species_display_names().values())
        raise ValueError(
            f"{species_display_name(key)} is a model-only prototype: no registered "
            "Gymnasium environment or SB3 curriculum exists yet. "
            f"See environments/{key}/README.md for model validation. "
            f"Trainable species: {available}"
        )
    return factory()
