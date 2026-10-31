"""BS2 design input (read only): schedules at S and E, the parent's tail_3 normaliser and std of action 14 at S and E
from the F2 probe, and the seed check for BS2's proposed queue against every seed set named below.
Usage: PY -I -B parent_extras.py  (prints; no file written)."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
F2 = S1 / "foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
PROBE = F2 / "policy_probe.npz.copy1"
CONF = F2 / "stage_config.json.copy1"
S, E = 2097152, 3538944
PROBE_SHA = "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert sha(PROBE) == PROBE_SHA
hp = json.loads(CONF.read_text())["hyperparameters"]
total = 12_000_000          # BS criteria: _total_timesteps 12,000,000 (the cumulative-step schedules)
for t in (S, E):
    lr = hp["learning_rate"] + (hp["learning_rate_end"] - hp["learning_rate"]) * t / total
    ent = hp["ent_coef"] + (hp["ent_coef_end"] - hp["ent_coef"]) * min(1.0, t / hp["ent_coef_decay_timesteps"])
    print(f"step {t:>9,}: learning_rate {lr:.5e}  ent_coef {ent:.5f}  [computed from stage_config.json.copy1: "
          f"lr {hp['learning_rate']} -> {hp['learning_rate_end']} over {total:,}; ent {hp['ent_coef']} -> "
          f"{hp['ent_coef_end']} by {hp['ent_coef_decay_timesteps']:,}]")

with np.load(PROBE, allow_pickle=False) as z:
    u = z["update_timesteps"].astype(np.int64)
    om, ov = z["obs_rms_mean"], z["obs_rms_var"]
    std = z["std_per_actuator"]
    lr_p, ent_p = z["learning_rate"], z["ent_coef"]
for t in (S, E - 8192):
    i = int(np.flatnonzero(u == t)[0])
    print(f"probe row {i} (update_timesteps {t:,}): obs_rms mean[6] {float(om[i, 6]):+.6f} rad = "
          f"{math.degrees(float(om[i, 6])):+.3f} deg, SD {math.degrees(math.sqrt(float(ov[i, 6]))):.3f} deg; "
          f"std14 {float(std[i, 14]):.4f}; probe learning_rate {float(lr_p[i]):.5e} ent_coef {float(ent_p[i]):.5f}")

# ---- seeds
spec = importlib.util.spec_from_file_location("ecs_common_ro", S2 / "r10round/ecs/tools/ecs_common.py")
ecs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ecs)
used = {}


def add(name, seeds):
    for s in seeds:
        used.setdefault(int(s), set()).add(name)


for (start, arm), seeds in ecs.SEEDS.items():
    for s in seeds:
        add(f"ECS {start}_{arm}_{s} (derived)", ecs.derived_seeds(s))
bs_screen = json.loads((S2 / "r10round/bias_screen/SCREEN.json").read_text())
for arm, s in bs_screen["queue"]:
    add(f"BS {arm}_{s} (derived)", (s, s + 1, s + 2, s + 3, s + 1000))
for s in (9101, 9111, 9121, 9131, 9141):
    add("BS/ECS validation (derived)", (s, s + 1, s + 2, s + 3, s + 1000))
add("M1 end-trace / warning_margin 3042-3061", range(3042, 3062))
add("reader OWN 5001-5010 / early_warning", range(5001, 5011))
add("reader on-policy 9301-9305", range(9301, 9306))
add("early_bias probe 9101-9133 (env seeds seed+0..3)", range(9101, 9134))
add("verify_early_bias 7001-7216, 7501-7502", list(range(7001, 7217)) + [7501, 7502])
add("r10 training runs (seed 42 etc.; stage_config run.seed)", (42, 52, 60, 70, 80))
proposed = (2501, 2511, 2521, 2531)
suggested = (2201, 2211, 2221, 2231)
for label, q in (("PROPOSED", proposed), ("task's example", suggested)):
    print(f"\n{label} seeds {q}:")
    for s in q:
        d = (s, s + 1, s + 2, s + 3, s + 1000)
        hits = {x: sorted(used[x]) for x in d if x in used}
        print(f"  {s}: derived {d} -> collisions {hits if hits else 'none'}")
print(f"\nBS's own seeds against ECS (for the record): "
      f"{ {s: sorted(used[s]) for s in (2101, 2111, 2121, 2131) if len(used.get(s, ())) > 1} }")
