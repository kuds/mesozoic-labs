"""e2e lane: stand-in run copies for the judge's --final-fallback path (s42), built from the stand-in run. NOTHING TRAINS.

    cd TREE && ISO_WT=TREE python -I TOOLS/iso.py make_final.py LANE

LANE/tampered/final_fallback/pilot4_walk_s42/03_locomotion/: the s42 stand-in's records and its 500,000 / 1,000,000 /
  1,500,000 pairs (byte copies), NO 2,000,000 pair, and a final pair: stage2_final.zip = the 2,000,000 zip loaded by
  the tree's load_sb3_model and saved again by SB3's model.save() with num_timesteps set to 2,015,232 (the
  uninterrupted end), stage2_final_vecnorm.pkl = the 2,000,000 sidecar byte for byte.
LANE/tampered/final_fallback_short/pilot4_walk_s42/03_locomotion/: the same, but stage2_final.zip is the 2,000,000 zip
  byte for byte (num_timesteps 2,000,000: not the uninterrupted end).
Writes LANE/tampered/make_final.json.
"""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

LANE = Path(sys.argv[1])
SRC = LANE / "runs" / "pilot4_walk_s42" / "03_locomotion"
OUT: dict = {}


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def base(case: str) -> Path:
    st = LANE / "tampered" / case / "pilot4_walk_s42" / "03_locomotion"
    (st / "models").mkdir(parents=True, exist_ok=False)
    for name in ("stage_config.json", "plant_identity.json", "task_fingerprint.json"):
        shutil.copy2(SRC / name, st / name)
    for steps in (500_000, 1_000_000, 1_500_000):
        for name in (f"stage2_{steps}_steps.zip", f"stage2_vecnormalize_{steps}_steps.pkl"):
            shutil.copy2(SRC / "models" / name, st / "models" / name)
    shutil.copy2(SRC / "models" / "stage2_vecnormalize_2000000_steps.pkl", st / "models" / "stage2_final_vecnorm.pkl")
    return st


import environments  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from environments.shared.policy_loading import load_sb3_model  # noqa: E402

OUT["environments_file"] = environments.__file__
st = base("final_fallback")
model = load_sb3_model(str(SRC / "models" / "stage2_2000000_steps.zip"), algorithm=PPO, device="cpu")
OUT["loaded_num_timesteps"] = int(model.num_timesteps)
model.num_timesteps = 2_015_232
model.save(str(st / "models" / "stage2_final.zip"))
with zipfile.ZipFile(st / "models" / "stage2_final.zip") as z:
    data = json.loads(z.read("data"))
OUT["final_fallback"] = {"final_num_timesteps": data.get("num_timesteps"), "lineage": data.get("mesozoic_task_lineage"),
                         "files": {str(p.relative_to(st)): sha(p) for p in sorted(st.rglob("*")) if p.is_file()}}
st2 = base("final_fallback_short")
shutil.copy2(SRC / "models" / "stage2_2000000_steps.zip", st2 / "models" / "stage2_final.zip")
OUT["final_fallback_short"] = {"files": {str(p.relative_to(st2)): sha(p) for p in sorted(st2.rglob("*")) if p.is_file()}}
(LANE / "tampered" / "make_final.json").write_text(json.dumps(OUT, indent=1) + "\n")
print(json.dumps({"final_num_timesteps": OUT["final_fallback"]["final_num_timesteps"],
                  "lineage_kept": bool(OUT["final_fallback"]["lineage"])}))
