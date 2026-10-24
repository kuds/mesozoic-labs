import json
from environments.shared.config import load_all_stages, load_stage_config
from environments.shared.cli import _resolve_stage_ref
from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds
from environments.shared.species_names import require_model_id
from environments.shared.plant_contract import current_plant_identity
mid = require_model_id("tyrannosaurus_rex", "anatomical")
cfgs = load_all_stages(mid)
st = _resolve_stage_ref("stance", cfgs, mid)
cfg = cfgs[st]
th = StanceV2Thresholds.from_curriculum(cfg["curriculum_kwargs"])
print(st, json.dumps(th.declared(), default=str))
print({k: v for k, v in cfg["env_kwargs"].items() if "lean" in k or "leg_home" in k or "noise" in k})
print({k: v for k, v in cfg["ppo_kwargs"].items() if "symm" in k})
print(current_plant_identity(mid).to_dict())
print({k: cfg["curriculum_kwargs"].get(k) for k in ("gate_kind","eval_freq","eval_episodes","checkpoint_freq","max_checkpoints","collapse_peak_floor_reference","timesteps")})
print(cfg.get("timesteps"), list(cfg.keys()))
