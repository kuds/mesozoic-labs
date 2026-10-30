"""The probe's tail_3 series at 8-update resolution, 0.79M-2.10M (read only): bank mean, statue_settled, training-row mean."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
import eb_common as E
P = {r: E.probe(r) for r in ("s42", "s80")}
print("block (8 updates = 65,536 steps) means; columns per run: bank_mean, statue_settled, rollout_mu_mean")
for N in range(786432, 2097152 + 1, 65536):
    row = [f"{N/1e6:5.3f}M"]
    for r, pr in P.items():
        t = pr["update_timesteps"]
        sel = (t > N - 65536) & (t <= N)
        a = pr["bank_action_now"][sel][:, :, 14]
        row.append(f"{r}: {a.mean():+.3f} {a[:, [0, 5]].mean():+.3f} {pr['rollout_mu_mean'][sel, 14].mean():+.3f}")
    print("  ".join(row))
