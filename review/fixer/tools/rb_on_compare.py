"""A5: the switch-ON first rollout (rb_on.py) against the switch-OFF one (rollbuf.py) on the same HEAD archive and seed:
the actor's 63-value observations, actions, rewards, log_probs and episode_starts bitwise; values, returns,
advantages bitwise or max |diff| stated; initial parameters: the OFF params digest is ROUND's pin (rb_compare.py).
usage: PY -I -B rb_on_compare.py RBDIR"""
import pickle, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1]); ok = True
for seed in (60, 42, 80):
    off = pickle.load(open(d / f"head_{seed}.pkl", "rb")); on = pickle.load(open(d / f"on_{seed}.pkl", "rb"))
    f = lambda r, k: np.frombuffer(r["buffer"][k], dtype=np.float32)
    o_off, o_on = f(off, "observations").reshape(-1, 63), f(on, "observations").reshape(-1, 66)
    res = {"obs[:63]": bool(np.array_equal(o_off, o_on[:, :63])), "rows": o_off.shape[0]}
    for k in ("actions", "rewards", "log_probs", "episode_starts"):
        res[k] = bool(np.array_equal(f(off, k), f(on, k)))
    for k in ("values", "returns", "advantages"):
        a, b = f(off, k), f(on, k)
        res[k] = "bitwise" if np.array_equal(a, b) else f"max|diff| {float(np.max(np.abs(a - b))):.3e}"
    ph = o_on[:, 63:]
    res["phase_cols_in_[0,1]"] = bool(np.all((ph >= 0) & (ph <= 1)))
    res["class"] = on["class"]; res["updates"] = (off["n_updates"], on["n_updates"])
    good = all(res[k] is True for k in ("obs[:63]", "actions", "rewards", "log_probs", "episode_starts")) and \
        all(res[k] == "bitwise" or float(res[k].split()[-1]) <= 1e-6 for k in ("values", "returns", "advantages"))
    ok &= good
    print(f"seed {seed}: {res} -> {'A5 HOLDS' if good else 'A5 FAILS'}")
print("RESULT", "A5 HOLDS FOR SEEDS 60, 42, 80" if ok else "A5 DOES NOT HOLD")
