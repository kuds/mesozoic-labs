import sys, numpy as np, hashlib
F=sys.argv[1]
for run in ["s42","s52","s60","s70","s80"]:
    z=np.load(f"{F}/stance5_LTSB_{run}/A/policy_probe.npz",allow_pickle=False)
    t=z["update_timesteps"]; A=z["bank_action_now"][:,:,14].astype(float)
    lab=[f"{l}/{s}" for l,s in zip(z["bank_labels"],z["bank_seeds"])]
    statue=[i for i,l in enumerate(lab) if l.startswith("statue")]
    mx=A.max(1); i=mx.argmax()
    out=[f"{run}: max {mx.max():.4f} @ {t[i]} row {lab[A[i].argmax()]}", f"statue max {A[:,statue].max():.4f} @ {t[A[:,statue].max(1).argmax()]}"]
    for th in (0.8,0.9,1.0):
        k=np.flatnonzero(mx>th); out.append(f">{th}: {(t[k[0]] if k.size else None)} n={k.size}")
    # does any action index other than 14 differ? check abs max of all bank actions >1 per actuator
    allmx=z["bank_action_now"].max(axis=(0,1))
    out.append("per-act max over all: "+" ".join(f"{v:.2f}" for v in allmx))
    k=np.searchsorted(t,4325376); out.append(f"@4325376 max {mx[k]:.3f}")
    k=np.searchsorted(t,4194304); out.append(f"@4194304 max {mx[k]:.3f} statue {A[k,statue].round(3).tolist()}")
    over=np.flatnonzero(mx>1.0)
    if over.size: out.append(f">1 span {t[over[0]]}..{t[over[-1]]}, rows: {sorted(set(lab[j] for j in np.argwhere(A>1.0)[:,1]))}")
    print(" | ".join(map(str,out)))
