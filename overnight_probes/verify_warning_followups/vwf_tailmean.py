"""Verifier extra (read only; post hoc, exploratory): training-rollout mean tail_3 angle per 1,048,576-step interval,
recovered from each probe's running normaliser (obs_rms_mean[:, 6] and obs_rms_count, float64 count):
interval mean = (c2 m2 - c1 m1) / (c2 - c1).  An attribution aid for the r9 runs' unnamed over-bar episodes."""
import numpy as np
F2 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2"
TR = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch"
marks = [1_048_576 * k for k in range(0, 12)] + [12_001_280]
def series(p):
    z = np.load(p); t = z["update_timesteps"]; m = z["obs_rms_mean"][:, 6].astype(np.float64); c = z["obs_rms_count"].astype(np.float64)
    out = []
    for a, b in zip(marks[:-1], marks[1:]):
        i = np.flatnonzero(t <= a); j = np.flatnonzero(t <= b)[-1]
        if len(i):
            i = i[-1]; v = (c[j] * m[j] - c[i] * m[i]) / (c[j] - c[i])
        else:
            v = m[j]
        out.append(np.degrees(v))
    return out
print("interval end (M): " + " ".join(f"{b / 1e6:6.2f}" for b in marks[1:]))
for r, note in (("s42", "r10 tail FOB 3.67M"), ("s60", "r10 tail FOB 8.13M"), ("s80", "r10 no tail over-bar; PARK 11.80M")):
    print(f"r10 {r}: " + " ".join(f"{v:+6.2f}" for v in series(f"{F2}/stance6_r10LTSB_{r}/copies/01_stance/policy_probe.npz.copy1")) + f" | {note}")
for r, note in (("s42", "ANY alarm 3.67M (tail per satwatch5)"), ("s52", "3 over-bar eps, never flagged"), ("s60", "1 over-bar ep"),
                ("s70", "1 over-bar ep"), ("s80", "ANY alarm 4.46M (tail onset per tailscreen)")):
    print(f"r9  {r}: " + " ".join(f"{v:+6.2f}" for v in series(f"{TR}/stance5_LTSB_{r}/A/policy_probe.npz")) + f" | {note}")
