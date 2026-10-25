"""Independent read of the five band runs' policy_probe.npz (tail_3 = action index 14, obs q index 6)."""
import sys, hashlib, numpy as np
F = sys.argv[1]
T3, Q3 = 14, 6
for run in ["s42", "s52", "s60", "s70", "s80"]:
    p = f"{F}/stance5_LTSB_{run}/A/policy_probe.npz"
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    z = np.load(p, allow_pickle=False)
    if run == "s42":
        for k in z.files: print("  key", k, z[k].shape, z[k].dtype)
    t = z["update_timesteps"].astype(np.int64)
    A = z["bank_action_now"].astype(np.float64); Af = z["bank_action_frozen"].astype(np.float64)
    a = A[:, :, T3]
    labels = [str(x) for x in z["bank_labels"]]; seeds = z["bank_seeds"]; steps = z["bank_steps"]; off = z["bank_ankle_offset"]
    raw = z["bank_raw_obs"]
    print(f"== {run} sha256[:16] {sha}: rows {len(t)}, t {t[0]:,}..{t[-1]:,}, diffs {set(np.diff(t).tolist())}, monotone {bool(np.all(np.diff(t)>0))}, frozen_ref {set(z['frozen_reference_timesteps'].tolist())}")
    print("   bank:", [(l, int(s), int(k), float(o)) for l, s, k, o in zip(labels, seeds, steps, off)])
    print("   bank q3 (obs[6]):", np.round(raw[:, Q3], 4).tolist(), " q3 vel obs[21+6]:", np.round(raw[:, 21 + Q3], 4).tolist())
    print("   bank_raw_obs dim", raw.shape, " identical bank across runs check sha:", hashlib.sha256(raw.tobytes()).hexdigest()[:12])
    mx = a.max(1); arg = a.argmax(1)
    def first(c):
        i = np.flatnonzero(c); return (int(t[i[0]]), int(i[0])) if i.size else None
    for th in (0.8, 0.9, 0.99, 1.0):
        print(f"   max-over-bank first > {th}: {first(mx > th)}; share of rows {np.mean(mx > th):.4f} ({int(np.sum(mx>th))} rows)")
    i = int(mx.argmax())
    print(f"   overall max {mx.max():.4f} at {int(t[i]):,} (row {labels[arg[i]]}/{int(seeds[arg[i]])}); min-over-bank min {a.min():.4f}")
    # which bank rows carry the >1 values
    over = a > 1.0
    if over.any():
        cnt = over.sum(0)
        print("   rows>1 per bank state:", {f"{labels[j]}/{int(seeds[j])}": int(cnt[j]) for j in range(len(labels)) if cnt[j]})
        idx = np.flatnonzero(mx > 1.0)
        print(f"   first/last update with max>1: {int(t[idx[0]]):,} / {int(t[idx[-1]]):,}; contiguous runs:", end=" ")
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        print([(int(t[r[0]]), int(t[r[-1]]), len(r)) for r in runs][:12], "n_runs", len(runs))
    # frozen-normalizer variant
    af = Af[:, :, T3]
    print(f"   frozen-normalizer: max-over-bank overall {np.nanmax(af):.4f}, first >1 {first(np.nanmax(af,1) > 1.0)}")
    # per-statue-row trajectory at selected steps
    for s in [3145728, 3268608, 3530752, 3538944, 4194304, 4325376, 4456448, 5242880, 8388608, 8429568]:
        k = int(np.searchsorted(t, s));
        if k < len(t) and t[k] == s:
            print(f"   {s:>10,}: " + " ".join(f"{v:+.3f}" for v in a[k]) + f" | max {a[k].max():+.3f} sigma {z['std_per_actuator'][k, T3]:.3f}")
