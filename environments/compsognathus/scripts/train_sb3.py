"""Train the anatomical variant; use shared.train for either named variant.

python -m environments.compsognathus.scripts.train_sb3 train --stage 1
python -m environments.shared.train --species compsognathus_robot train --stage 1
"""

from environments.shared.species_registry import get_species_config
from environments.shared.train_base import main

SPECIES_CONFIG = get_species_config("compsognathus")

if __name__ == "__main__":
    main(SPECIES_CONFIG)
