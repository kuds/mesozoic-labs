"""Move per-rollout JSONs written before the state-baseline split was added (no 'terms_gae_sb') to eb_old/."""
import json, os, shutil
H = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias"
os.makedirs(f"{H}/eb_old", exist_ok=True)
n = 0
for f in sorted(os.listdir(f"{H}/eb_out2")):
    p = f"{H}/eb_out2/{f}"
    try:
        d = json.load(open(p))
        ok = "terms_gae_sb" in d["split_per_epoch"]["adam_rollout_mean"]
    except Exception:
        ok = False
    if not ok:
        shutil.move(p, f"{H}/eb_old/{f}")
        n += 1
print("moved", n)
