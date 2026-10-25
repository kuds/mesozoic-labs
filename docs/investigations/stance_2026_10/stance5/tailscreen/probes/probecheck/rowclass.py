import sys, numpy as np
F = sys.argv[1]
for run in ["s42", "s52", "s60", "s70", "s80"]:
    z = np.load(f"{F}/stance5_LTSB_{run}/A/policy_probe.npz", allow_pickle=False)
    t = z["update_timesteps"]; a = z["bank_action_now"][:, :, 14].astype(float); lab = np.array([str(x) for x in z["bank_labels"]])
    out = []
    for name, sel in [("statue rows", np.isin(lab, ["statue", "statue_settled"])), ("ankle_plus", lab == "ankle_plus_0.03"),
                      ("ankle_minus_0.03", lab == "ankle_minus_0.03"), ("ankle_minus_0.05", lab == "ankle_minus_0.05_at400")]:
        m = a[:, sel].max(1); i = int(m.argmax())
        f9 = np.flatnonzero(m > 0.9); f8 = np.flatnonzero(m > 0.8); f6 = np.flatnonzero(m > 0.6)
        out.append(f"{name}: max {m.max():.3f} at {int(t[i]):,}; first>0.6 {int(t[f6[0]]) if f6.size else None}; first>0.8 {int(t[f8[0]]) if f8.size else None}; first>0.9 {int(t[f9[0]]) if f9.size else None}")
    print(f"== {run}\n   " + "\n   ".join(out))
    # statue-row mean around ledger onset, 131072-step blocks
