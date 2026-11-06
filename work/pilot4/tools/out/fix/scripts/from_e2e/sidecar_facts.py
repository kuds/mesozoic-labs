"""e2e lane: what the stand-in sidecars (saved by the set-up path's VecNormalize.save) hold against each parent's 8M
sidecar: obs_rms on columns 0-59 bit-equal, the command slice 60-62 (reseeded at the edge), ret_rms (fresh at the
edge). Reads pickles this lane wrote and pilot 3's verified run copies (read only); nothing trains.

    cd TREE && ISO_WT=TREE python -I TOOLS/iso.py sidecar_facts.py LANE OUT.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

import environments  # noqa: F401 - the tree's, so the pickles' classes resolve there

LANE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/runs/final")
res = {"environments_file": environments.__file__}
for s in (42, 52, 60):
    with open(LANE / "runs" / f"pilot4_walk_s{s}" / "03_locomotion" / "models" / "stage2_vecnormalize_2000000_steps.pkl",
              "rb") as f:
        new = pickle.load(f)
    with open(RUNS / f"pilot3_walk_s{s}" / "03_locomotion" / "models" / "stage2_vecnormalize_8000000_steps.pkl", "rb") as f:
        old = pickle.load(f)
    res[s] = {
        "obs_0_60_mean_bit_equal": bool(np.array_equal(new.obs_rms.mean[:60], old.obs_rms.mean[:60])),
        "obs_0_60_var_bit_equal": bool(np.array_equal(new.obs_rms.var[:60], old.obs_rms.var[:60])),
        "obs_60_63_mean": [new.obs_rms.mean[60:].tolist(), old.obs_rms.mean[60:].tolist()],
        "obs_60_63_var": [new.obs_rms.var[60:].tolist(), old.obs_rms.var[60:].tolist()],
        "obs_count": [float(new.obs_rms.count), float(old.obs_rms.count)],
        "ret_rms": [[float(np.ravel(new.ret_rms.mean)[0]), float(np.ravel(new.ret_rms.var)[0]), float(new.ret_rms.count)],
                    [float(np.ravel(old.ret_rms.mean)[0]), float(np.ravel(old.ret_rms.var)[0]), float(old.ret_rms.count)]],
        "training_flag": [bool(new.training), bool(old.training)],
    }
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
