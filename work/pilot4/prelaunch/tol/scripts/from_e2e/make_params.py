"""Write the smoke runner's params for the e2e set-up path of each seed (QUICK_TEST False, the seed's own 8M pair
through the path map onto the lane's byte copies). Scratch helper of the e2e lane."""
import json
import sys
from pathlib import Path

lane = Path(sys.argv[1])
repo_ref = sys.argv[2]  # the pilot branch, read from the shared tree by the caller (tolerance lane)
drive = "/content/drive/MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo"
for seed in (42, 52, 60):
    models = f"{drive}/pilot3_walk_s{seed}/03_locomotion/models"
    p = {"repo_ref": repo_ref,
         "config": {"SPECIES": "tyrannosaurus_rex", "EMBODIMENT": "anatomical", "ALGORITHM": "ppo", "N_ENVS": 4,
                    "SEED": seed, "QUICK_TEST": False, "USE_GOOGLE_DRIVE": False, "AUTO_DISCONNECT": False,
                    "BEHAVIOR": "walk", "TRUNK_FROM": "", "RETRAIN_FROM": "", "RUN_LABEL": f"walking pilot 4 s{seed}",
                    "RUN_ID": f"pilot4_walk_s{seed}", "K3_PANEL": False, "CRITIC_PHASE": False},
         "manual": ['MANUAL_NODE = "locomotion"',
                    f'MANUAL_LOAD_PATH = "{models}/stage2_8000000_steps"',
                    f'MANUAL_VECNORM_PATH = "{models}/stage2_vecnormalize_8000000_steps.pkl"',
                    'MANUAL_LOAD_MODE = "initialize_next_stage"',
                    "MANUAL_TIMESTEPS = None"],
         "path_map": [[drive, str(lane / "tmp" / "pairs")]],
         "guard_line": 'raise RuntimeError("Walking pilot 4: chain loop skipped on purpose. Now run the manual cell '
                       '(section 7) by itself -- unless you are resuming: then stop here.")',
         "resume_stage": "locomotion",
         "expect_head": "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8",
         "expect_hyperparameters_sha256": "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62"}
    (lane / "setup" / f"params_s{seed}.json").write_text(json.dumps(p, indent=1) + "\n")
    print(seed, "written")
