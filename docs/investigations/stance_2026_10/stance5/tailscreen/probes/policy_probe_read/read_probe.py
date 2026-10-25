"""Read policy_probe.npz of the band runs: tail_3_pitch_act's raw deterministic mean at the bank states over training."""
import sys, numpy as np
F = sys.argv[1]
T3 = 14  # tail_3_pitch_act, the 15th actuator in the r9 MJCF order
ALARM = {"s42": (3538944, 3670016), "s80": (4325376, 4456448)}
for run in ["s42", "s52", "s60", "s70", "s80"]:
    z = np.load(f"{F}/stance5_LTSB_{run}/A/policy_probe.npz", allow_pickle=False)
    t = z["update_timesteps"]; a = z["bank_action_now"][:, :, T3]; sd = z["std_per_actuator"][:, T3]
    labels = list(z["bank_labels"])
    mx = a.max(axis=1); mean = a.mean(axis=1)
    def first(cond):
        i = np.flatnonzero(cond); return int(t[i[0]]) if i.size else None
    print(f"== {run}: updates {len(t)} ({t[0]:,}..{t[-1]:,}); bank {labels}")
    print(f"   tail_3 raw mean over bank: max-over-bank first > 0.9 at {first(mx > 0.9)}, > 0.99 at {first(mx > 0.99)}, > 1.0 at {first(mx > 1.0)}; bank-mean first > 0.9 at {first(mean > 0.9)}")
    print(f"   share of updates with max-over-bank > 1.0: {np.mean(mx > 1.0):.3f}; > 0.99: {np.mean(mx > 0.99):.3f}; overall max {mx.max():.3f} at {int(t[mx.argmax()]):,}")
    for s in [1048576, 2097152, 3145728, 3670016, 4194304, 4456448, 5242880, 6291456, 8388608, 10485760, 12001280]:
        i = int(np.searchsorted(t, s)); i = min(i, len(t) - 1)
        print(f"   {int(t[i]):>11,}: bank tail_3 mean {mean[i]:+.3f} (min {a[i].min():+.3f}, max {a[i].max():+.3f}); sigma {sd[i]:.3f}")
    if run in ALARM: print(f"   ledger: first flagged eval {ALARM[run][0]:,}, alarm {ALARM[run][1]:,}")
