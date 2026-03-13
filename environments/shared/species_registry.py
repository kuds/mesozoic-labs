"""Species registry for unified training entry point.

Maps species names to their SpeciesConfig, eliminating the need
for per-species training scripts with nearly identical boilerplate.

Usage:
    python -m environments.shared.train --species velociraptor train --stage 1
    python -m environments.shared.train --species trex curriculum
    python -m environments.shared.train --species brachiosaurus train --stage 2
"""

from environments.shared.train_base import SpeciesConfig


def _make_raptor_config() -> SpeciesConfig:
    from environments.velociraptor.envs.raptor_env import RaptorEnv

    return SpeciesConfig(
        species="velociraptor",
        env_class=RaptorEnv,
        stage_descriptions="1=balance, 2=locomotion, 3=strike",
        height_label="Pelvis height",
        stage3_section_label="Hunting",
        success_keys=["strike_success", "bite_success"],
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


# Lazy registry — factories are called only when the species is selected,
# so we don't import all env modules at startup.
SPECIES_FACTORIES = {
    "velociraptor": _make_raptor_config,
    "raptor": _make_raptor_config,  # alias
    "trex": _make_trex_config,
    "t-rex": _make_trex_config,  # alias
    "brachiosaurus": _make_brachio_config,
    "brachio": _make_brachio_config,  # alias
}


def get_species_config(species: str) -> SpeciesConfig:
    """Look up and return the SpeciesConfig for the given species name."""
    key = species.lower().replace("_", "").replace("-", "")
    # Try exact match first, then normalized
    factory = SPECIES_FACTORIES.get(species.lower()) or SPECIES_FACTORIES.get(key)
    if factory is None:
        available = sorted(set(SPECIES_FACTORIES.keys()))
        raise ValueError(f"Unknown species '{species}'. Available: {available}")
    return factory()
