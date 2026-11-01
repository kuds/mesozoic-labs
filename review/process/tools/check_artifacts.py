"""Process review: every model zip / sidecar the chain smoke wrote -- the sidecar class and width, the zip's observation
space, policy class, buffer class and critic-phase record; plus stage_config.json's switch.  usage: check_artifacts.py RUN_DIR"""

import json
import pickle
import sys
import zipfile
from pathlib import Path

run_dir = Path(sys.argv[1])
stage = run_dir / "01_stance"
for p in sorted((stage / "models").glob("*.pkl")):
    with open(p, "rb") as handle:
        vn = pickle.load(handle)
    print("SIDECAR", p.name, f"{type(vn).__module__}.{type(vn).__name__}", "obs_rms", tuple(vn.obs_rms.mean.shape),
          "own attrs", [a for a in ("critic_phase", "critic_observation_space") if a in vars(vn)])
for p in sorted((stage / "models").glob("*.zip")):
    data = json.loads(zipfile.ZipFile(p).read("data"))
    pk = data.get("policy_kwargs", {})
    print("ZIP", p.name, "obs_shape", data["observation_space"].get("_shape"),
          "policy_class", str(data.get("policy_class", {}).get(":serialized:", ""))[:0] or data.get("policy_class", {}).get("__module__", "?"),
          "rollout_buffer_class?", "rollout_buffer_class" in data,
          "critic_phase_record", (data.get("mesozoic_critic_phase") or {}).get("schema"),
          "policy_kwargs phase", {k: pk.get(k) for k in ("critic_phase_settle_steps", "critic_phase_gamma")})
sc = json.loads((stage / "stage_config.json").read_text())
print("STAGE_CONFIG critic_phase", sc.get("hyperparameters", {}).get("critic_phase"),
      "hyperparameters_sha256", sc.get("run", {}).get("hyperparameters_sha256"))
