"""The policy probe's recorded tail_3 series (read only): bank mean / statue_settled deterministic mu14 under the
current normalisation, the training-row mean mu14, per 262,144-step block, for s42, s80, s60."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
import eb_common as E
for run in ("s42", "s80", "s60"):
    pr = E.probe(run)
    t = pr["update_timesteps"]; a = pr["bank_action_now"][:, :, 14]; r = pr["rollout_mu_mean"][:, 14]
    lab = pr["bank_labels"]
    ss = np.flatnonzero(lab == "statue_settled")
    print(f"== {run}: {len(t)} rows, first {t[0]} last {t[-1]}; bank labels {list(lab)}")
    print("  step       bank_mean  statue_settled  rollout_mu_mean  std14   ent_coef   lr")
    for N in range(262144, 9_000_000, 262144):
        i = np.flatnonzero(t == N)
        if len(i) == 0: continue
        i = int(i[0])
        print(f"  {N:>9d}  {a[i].mean():+.4f}    {a[i, ss].mean():+.4f}        {r[i]:+.4f}         {pr['std_per_actuator'][i,14]:.4f}  {pr['ent_coef'][i]:.5f}  {pr['learning_rate'][i]:.3e}")
