"""TEST 1, proportional timing: for each tail rise, the step at which each signal first completes 10 / 25 / 50 % of its
own change from a reference window to a plateau window (16-rollout or 64-episode moving means, sustained 16 rollouts /
64 episodes).  Direction-free: a posture signal that moves the other way still gets its own fraction.  Windows were
chosen from the tail signal alone (reference: standing, tail flat; plateau: the tail level just before or at onset).
Read only.  Output: prop_timing.txt (stdout).
"""
import numpy as np

import pt_common as C

CASES = [
    # label, kind, run key, reference window, plateau window
    ("r10 s42 rise (onset 3.67M)", "r10", "s42", (524_288, 1_048_576), (3_670_016, 4_194_304)),
    ("r10 s42 re-formed rail (2nd)", "r10", "s42", (6_619_136, 7_012_352), (7_602_176, 8_388_608)),
    ("r10 s60 main rise (drive 7.34M)", "r10", "s60", (5_767_168, 6_029_312), (7_340_032, 7_602_176)),
    ("r10 s60 final approach (ob 8.13M)", "r10", "s60", (7_208_960, 7_471_104), (8_126_464, 8_650_752)),
    ("r9 s42 rise (alarm 3.67M)", "r9", "stance5_LTSB_s42", (524_288, 1_048_576), (3_670_016, 4_456_448)),
    ("r9 s80 rise (alarm 4.46M)", "r9", "stance5_LTSB_s80", (524_288, 1_048_576), (2_883_584, 3_670_016)),
    ("r9 s52 first rise (no alarm)", "r9", "stance5_LTSB_s52", (524_288, 1_048_576), (2_621_440, 3_145_728)),
]


def frac_times(t, y, ref_w, pl_w, k):
    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    ref = np.nanmedian(y[(t >= ref_w[0]) & (t < ref_w[1])])
    pl = np.nanmedian(y[(t >= pl_w[0]) & (t < pl_w[1])])
    span = pl - ref
    f = (y - ref) / span if span != 0 else np.full_like(y, np.nan)
    aft = t >= ref_w[1]
    out = {q: C.first_sustained(t[aft], f[aft], q, True, k) for q in (0.10, 0.25, 0.50)}
    return ref, pl, out


def main():
    print("TEST 1 / proportional timing of tail and posture within each rise (prop_timing.py; read only)")
    print("Each row: reference -> plateau level, and the first step at which the moving mean completes 10 / 25 / 50 % "
          "of that change (sustained).  Probe signals: 16-rollout mean, 16 rollouts sustained; training-episode CoM: "
          "64-episode mean, 64 episodes sustained.\n")
    for lab, kind, run, rw, pw in CASES:
        if kind == "r10":
            P = C.probe_batch_means(C.f2_file(run, "policy_probe.npz"))
            bm = P["batch_mean"]
            tail = ("tail_3 mu", P["t"], C.smooth(P["rollout_mu_mean"][:, C.TAIL3].astype(float), 16), 16)
            T = C.train_episode_series(C.f2_file(run, "train_episodes.csv"))
        else:
            P = C.probe_batch_means(C.R9_PROBE[run])
            bm = P["batch_mean"]
            tail = ("tail a_imp", P["t"], C.smooth((bm[:, C.OBS["tail_3_pitch"]] + 0.0924) / 0.0863, 16), 16)
            T = None
            try:
                T = C.train_episode_series(f"{C.R9_RUNS[run]}/train_episodes.csv")
            except FileNotFoundError:
                pass
        sigs = [tail,
                ("probe pitch", P["t"], C.smooth(C.pitch_from_gx(bm[:, C.OBS["g_x"]]), 16), 16),
                ("probe neck", P["t"], C.smooth(bm[:, C.OBS["neck_pitch"]] * C.R2D, 16), 16),
                ("probe head", P["t"], C.smooth(bm[:, C.OBS["head_pitch"]] * C.R2D, 16), 16)]
        if T is not None:
            sigs.append(("train CoM mm", T["t"], C.smooth(T["com"] * 1000, 64), 64))
            sigs.append(("train pitch", T["t"], C.smooth(T["pitch"], 64), 64))
        print(f"  {lab}: reference [{rw[0]/1e6:.2f}, {rw[1]/1e6:.2f})M, plateau [{pw[0]/1e6:.2f}, {pw[1]/1e6:.2f})M")
        for name, t, y, k in sigs:
            ref, pl, out = frac_times(t, y, rw, pw, k)
            print(f"     {name:13s} {ref:+8.3f} -> {pl:+8.3f} (change {pl-ref:+7.3f}):  10% {C.fmt(out[0.10]):>10s}  "
                  f"25% {C.fmt(out[0.25]):>10s}  50% {C.fmt(out[0.50]):>10s}")
        print()


if __name__ == "__main__":
    main()
