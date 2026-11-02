"""Recompute run.hyperparameters_sha256 from a stage_config.json's own blocks (recipe: config.hyperparameters_sha256)."""
import hashlib
import json
import sys


def canon(v):
    return "sha256:" + hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


for p in sys.argv[1:]:
    sc = json.loads(open(p, "rb").read().decode("utf-8"))
    cur = sc.get("curriculum", {})
    view = {"algorithm": sc["algorithm"].upper(), "hyperparameters": sc.get("hyperparameters", {}),
            "shaping": {k: v for k, v in cur.items() if k.startswith(("warmup_", "ramp_"))}}
    print(p.rsplit("/", 3)[-3:], "recomputed", canon(view), "recorded", sc["run"]["hyperparameters_sha256"],
          "shaping keys", sorted(view["shaping"]))
