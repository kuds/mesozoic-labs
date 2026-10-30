"""PHASE PROBE, POST HOC (not pre-registered): did the habit start to move to the neighbouring tail joint in BS2's
blinded runs?  Reads (read only, allow_pickle False) each BS2 run's 01_stance/policy_probe.npz training records:
rollout_mu_mean[:, a] (the rollout mean of the policy's Gaussian mean, per update) for a = 13 (tail_2_pitch_act),
14 (tail_3_pitch_act), 0 (neck_pitch_act), 11 (tail_1_pitch_act), and the deterministic bank command on the two
statue_settled rows (bank_action_now[:, (0, 5), a]).  Reports, per run, the value at the first update after S and the
mean over the last 8 updates before E, and the per-arm medians.  No simulation, no training.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen2/runs")
S, E = 2_097_152, 3_538_944
ACTS = {"tail_2": 13, "tail_3": 14, "neck": 0, "tail_1": 11}


def main():
    out, lines = {}, []
    for arm in ("control", "critic_blind_tail3"):
        for seed in (2501, 2511, 2521, 2531):
            name = f"{arm}_{seed}"
            f = RUNS / name / "01_stance" / "policy_probe.npz"
            with np.load(f, allow_pickle=False) as z:
                ut = z["update_timesteps"]
                mu = z["rollout_mu_mean"].astype(np.float64)
                bank = z["bank_action_now"].astype(np.float64)
            sel = np.flatnonzero((ut > S) & (ut < E))
            first, last8 = sel[:8], sel[-8:]
            rec = {"n_rows_window": int(len(sel))}
            for nm, a in ACTS.items():
                rec[nm] = {"rollout_mu_first8": float(mu[first, a].mean()), "rollout_mu_last8": float(mu[last8, a].mean()),
                           "bank_ss_first8": float(bank[first][:, [0, 5], a].mean()),
                           "bank_ss_last8": float(bank[last8][:, [0, 5], a].mean())}
            out[name] = rec
            lines.append(f"{name:<24} rows {len(sel)}  " + "  ".join(
                f"{nm} rollout mu {rec[nm]['rollout_mu_first8']:+.3f} -> {rec[nm]['rollout_mu_last8']:+.3f} "
                f"(bank ss {rec[nm]['bank_ss_first8']:+.3f} -> {rec[nm]['bank_ss_last8']:+.3f})" for nm in ACTS))
    lines.append("per-arm medians of the rise (last 8 minus first 8 updates, rollout mu):")
    for nm in ACTS:
        for arm in ("control", "critic_blind_tail3"):
            rises = [out[f"{arm}_{s}"][nm]["rollout_mu_last8"] - out[f"{arm}_{s}"][nm]["rollout_mu_first8"]
                     for s in (2501, 2511, 2521, 2531)]
            lines.append(f"  {nm:<7} {arm:<20} rises " + " ".join(f"{x:+.3f}" for x in rises)
                         + f"  median {np.median(rises):+.3f}")
    print("\n".join(lines))
    (HERE / "posthoc_bs2_probe.json").write_text(json.dumps(out, indent=1))
    (HERE / "posthoc_bs2_probe.txt").write_text("POST HOC (not pre-registered)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
