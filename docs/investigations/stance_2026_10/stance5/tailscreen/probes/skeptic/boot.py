import sys, json, numpy as np
for run in ("s42", "s80"):
    d = json.load(open(f"{sys.argv[1]}/value_{run}.json"))
    print(run, "sigma_t3", round(d["sigma_t3"], 4), "t3_index", d["t3_index"], d["actuators"][14])
    for mode in ("stoch", "det"):
        f = d["runs"][f"{mode}_free"]
        for c in ("hold0", "hold09"):
            h = d["runs"][f"{mode}_{c}"]
            assert [x["seed"] for x in f] == [x["seed"] for x in h]
            for key in ("window_return", "return"):
                diff = np.array([b[key] - a[key] for a, b in zip(f, h)])
                rng = np.random.default_rng(12345)
                bs = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                print(f"  {mode} {c}-free {key}: n={len(diff)} mean {diff.mean():+.2f} boot95 [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] t-CI [{diff.mean()-2.02*se:+.2f},{diff.mean()+2.02*se:+.2f}] wins {int((diff>0).sum())}")
