"""Print, as JSON, the facts of the tree under test that a synthetic run copy must carry (test helper; run via iso.py).

hyperparameters_sha256 of the locomotion stage, the plant identity, HEAD, the locomotion config's rollout and the
committed parent's critic-phase record (read from its JSON data member, nothing unpickled).
"""

import json
import subprocess
import zipfile
from pathlib import Path

from environments.shared.config import hyperparameters_sha256, load_stage_config
from environments.shared.plant_contract import current_plant_identity
from environments.shared.species_names import require_model_id

mid = require_model_id("tyrannosaurus_rex", "anatomical")
config = load_stage_config(mid, 2)
tree = Path.cwd()
with zipfile.ZipFile(tree / "pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final.zip") as z:
    parent_record = json.loads(z.read("data"))["mesozoic_critic_phase"]
print(json.dumps({
    "hp": hyperparameters_sha256(config, "ppo"),
    "plant": current_plant_identity(mid).to_dict(),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip(),
    "n_steps": config["ppo_kwargs"]["n_steps"],
    "ramp": config["curriculum_kwargs"].get("ramp_timesteps"),
    "warmup": config["curriculum_kwargs"].get("warmup_timesteps"),
    "curriculum": config["curriculum_kwargs"],
    "hyperparameters": config["ppo_kwargs"],
    "parent_record": parent_record,
}, default=str))
