"""r10-only: baselines and baseline-subtracted maxima for the probe statue_settled tail_3 signal (no r9 data)."""
import json, sys
import numpy as np
EW = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_warning/ew_events.json"
d = json.load(open(EW))
FOB = {"s42": 3670016, "s60": 8126464, "s80": None}
def trailing(x, k):
    x = np.asarray(x, float); c = np.cumsum(np.insert(x, 0, 0.0)); out = np.empty_like(x)
    for i in range(len(x)):
        j = max(0, i - k + 1); out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out
for run in ("s42", "s60", "s80"):
    pr = d[run]["probe"]; t = np.array(pr["t"]); ss = np.array(pr["bank_ss"]); bm = np.array(pr["bank_mean"])
    rows = np.array(pr["bank_rows"])
    tr = trailing(ss, 8)
    dt = np.diff(t)
    print(run, "rows", len(t), "t0", t[0], "dt median", np.median(dt), "last", t[-1])
    for W in (262144, 524288, 1048576):
        B = ss[t <= W].mean(); print(f"  baseline mean raw bank_ss t<={W}: {B:+.4f} (n {int((t<=W).sum())}); sd {ss[t<=W].std():.4f}")
    B = ss[t <= 1048576].mean()
    pre = t < FOB[run] if FOB[run] else np.ones_like(t, bool)
    print(f"  max trailing (pre-FOB or whole run) {tr[pre].max():+.4f}; max delta (trail - B1M) {(tr - B)[pre].max():+.4f}")
    print(f"  first row values bank_ss {ss[:4].round(4).tolist()}  trailing at 1.05M {tr[t<=1048576][-1]:+.4f}")
    print("  labels per column: n cols", rows.shape)
    for thr in (0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20):
        i = np.flatnonzero(((tr - B) > thr) & pre)
        print(f"   delta>{thr:.2f}: first {int(t[i[0]]) if len(i) else None}")
ecs80 = [k for k in d if k.startswith("s80_")]
mx = []
for k in ecs80:
    pr = d[k]["probe"]; t = np.array(pr["t"]); ss = np.array(pr["bank_ss"]); tr = trailing(ss, 8)
    B = ss[t <= 1048576].mean(); mx.append(((tr - B)[t > 4194304]).max())
print("ECS s80 after S max delta", np.round(sorted(mx), 4).tolist())
