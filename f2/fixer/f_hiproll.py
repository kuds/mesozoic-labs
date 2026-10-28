import json, sys, os, glob
P = sys.argv[1]
for run in ("s60","s42","s80"):
    means=[]; eps=[]; diffs=[]; present=True; nlab=0
    for f in sorted(glob.glob(f"{P}/stance6_r10LTSB_{run}/*/m1/m1_summary.json")):
        d = json.load(open(f)); hr = d["hip_roll"]; nlab+=1
        r = hr["4"]; l = hr["8"]
        present = present and r["present_every_step"] and l["present_every_step"]
        means += [r["mean_over_episodes"], l["mean_over_episodes"]]
        eps += r["window_mean_by_episode"] + l["window_mean_by_episode"]
        diffs.append(sum(a-b for a,b in zip(r["window_mean_by_episode"], l["window_mean_by_episode"]))/len(r["window_mean_by_episode"]))
    print(run, "labels", nlab, "means over episodes [%.4f, %.4f]" % (min(means), max(means)),
          "per-episode [%.4f, %.4f]" % (min(eps), max(eps)), "mean r-l [%.4f, %.4f]" % (min(diffs), max(diffs)), "present", present)
