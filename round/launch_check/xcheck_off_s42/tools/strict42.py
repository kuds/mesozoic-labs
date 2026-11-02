"""Strict per-top-level-key canonical JSON comparison, REF (measured off, seed 42) vs RUN stage_config.json."""
import hashlib
import json
import sys


def canon(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


ref = json.load(open(sys.argv[1], encoding="utf-8"))
run = json.load(open(sys.argv[2], encoding="utf-8"))
for k in sorted(set(ref) | set(run)):
    a = canon(ref[k]) if k in ref else "<absent>"
    b = canon(run[k]) if k in run else "<absent>"
    print(f"{k}: {'EQUAL' if a == b else 'DIFFERS'}")
r1 = dict(run["run"]); r0 = dict(ref["run"])
r1.pop("label"); r0.pop("label")
print("run block minus label equal (canonical):", canon(r0) == canon(r1), "| keys:", sorted(r1))
print("RUN top-level key order:", list(run))
print("REF top-level key order:", list(ref))
