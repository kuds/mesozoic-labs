"""Write the smoke lane's runner params (one file, used for the launch and the RESUME phase).

The values are pilot 4's Colab-step values as the setup lane typed them (prelaunch/setup/runs/s42/params.json), with
QUICK_TEST true and the smoke's RUN_LABEL (the readouts lane's params_quick_s42.json); path_map maps the Colab
steps' Drive folder onto this lane's read-only copy of seed 42's 8M pair (LANE/pairs). Paths are worked out from this
file's own location, so this script spells no absolute path.

    python -I make_params.py OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
DRIVE = "/content/drive/MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo"
STEM = f"{DRIVE}/pilot3_walk_s42/03_locomotion/models"

params = {
    "repo_ref": "claude/pilot-walk-r10-p4",
    "config": {"SPECIES": "tyrannosaurus_rex", "EMBODIMENT": "anatomical", "ALGORITHM": "ppo", "N_ENVS": 4,
               "SEED": 42, "QUICK_TEST": True, "USE_GOOGLE_DRIVE": False, "AUTO_DISCONNECT": False,
               "BEHAVIOR": "walk", "TRUNK_FROM": "", "RETRAIN_FROM": "", "RUN_LABEL": "walking pilot 4 smoke s42",
               "RUN_ID": "pilot4_walk_s42", "K3_PANEL": False, "CRITIC_PHASE": False},
    "manual": [
        'MANUAL_NODE = "locomotion"',
        f'MANUAL_LOAD_PATH = "{STEM}/stage2_8000000_steps"',
        f'MANUAL_VECNORM_PATH = "{STEM}/stage2_vecnormalize_8000000_steps.pkl"',
        'MANUAL_LOAD_MODE = "initialize_next_stage"',
        "MANUAL_TIMESTEPS = None",
    ],
    "path_map": [[DRIVE, str(LANE / "pairs")]],
    "guard_line": 'raise RuntimeError("Walking pilot 4: chain loop skipped on purpose. Now run the manual cell '
                  '(section 7) by itself -- unless you are resuming: then stop here.")',
    "resume_stage": "locomotion",
    "expect_head": "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8",
    "expect_hyperparameters_sha256": "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62",
}
Path(sys.argv[1]).write_text(json.dumps(params, indent=1) + "\n")
print(f"wrote {sys.argv[1]}")
