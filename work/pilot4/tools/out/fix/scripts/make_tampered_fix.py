"""Fix lane: the tampered run copies for the fix stage's new checks (json and shutil only; nothing unpickled, no
repository import, nothing trained).

usage: python -I -B make_tampered_fix.py LANE

LANE/tampered/env_drift/pilot4_walk_s42/03_locomotion/: s42's stand-in records and its 2,000,000 pair (byte copies),
  stage_config.json's reward_weights.forward_vel_weight 2.0 -> 3.0, the recorded task digest left as it is (the
  validity review's case: an env weight that no named binding covers).
Writes LANE/tampered/make_tampered_fix.json.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

LANE = Path(sys.argv[1])
SRC = LANE / "runs" / "pilot4_walk_s42" / "03_locomotion"
out = {}


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


st = LANE / "tampered" / "env_drift" / "pilot4_walk_s42" / "03_locomotion"
(st / "models").mkdir(parents=True, exist_ok=False)
for name in ("stage_config.json", "plant_identity.json", "task_fingerprint.json"):
    shutil.copy2(SRC / name, st / name)
for name in ("stage2_2000000_steps.zip", "stage2_vecnormalize_2000000_steps.pkl"):
    shutil.copy2(SRC / "models" / name, st / "models" / name)
sc = json.loads((st / "stage_config.json").read_text())
before = sc["reward_weights"]["forward_vel_weight"]
sc["reward_weights"]["forward_vel_weight"] = 3.0
(st / "stage_config.json").write_text(json.dumps(sc, indent=2))
out["env_drift"] = {"reward_weights.forward_vel_weight": [before, 3.0],
                    "task_sha256_as_recorded": sc["task_fingerprint"]["task_sha256"],
                    "files": {str(p.relative_to(st)): sha(p) for p in sorted(st.rglob("*")) if p.is_file()}}
(LANE / "tampered" / "make_tampered_fix.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
