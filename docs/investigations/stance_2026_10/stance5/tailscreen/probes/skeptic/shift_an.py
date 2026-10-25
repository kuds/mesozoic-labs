import sys, json, numpy as np
d = json.load(open(sys.argv[1])); v = json.load(open(sys.argv[2]))
ref = v["runs"]["stoch_free"]
mine = d["free"]
print("free repro vs noisevalue stoch_free (first %d seeds): max|ret diff| %.3e, max|win diff| %.3e" % (len(mine),
      max(abs(a["ret"] - b["return"]) for a, b in zip(mine, ref)), max(abs(a["win"] - b["window_return"]) for a, b in zip(mine, ref))))
for k in ("shift+0.15", "shift-0.15"):
    s = d[k]; assert [x["seed"] for x in s] == [x["seed"] for x in ref[:len(s)]]
    for key, rk in (("win", "window_return"), ("ret", "return")):
        diff = np.array([a[key] - b[rk] for a, b in zip(s, ref)])
        rng = np.random.default_rng(7); bs = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
        print(f"{k} - free {key}: n={len(diff)} mean {diff.mean():+.2f} boot95 [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] wins {(diff>0).sum()}/{len(diff)}")
    print(f"  {k}: cmd_mean {np.mean([x['cmd_mean'] for x in s]):+.3f} frac99 {np.mean([x['frac99'] for x in s]):.4f} falls {sum(x['L']<1000 for x in s)}")
print(f"  free(ref): cmd_mean {np.mean([x['t3_cmd_mean'] for x in ref]):+.3f} frac99 {np.mean([x['t3_frac99'] for x in ref]):.4f}")
