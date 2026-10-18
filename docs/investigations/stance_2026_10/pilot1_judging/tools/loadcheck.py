import sys, pickle, json, zipfile, glob, os
import numpy as np
ALLOW = {
 ("numpy._core.multiarray","_reconstruct"),("numpy.core.multiarray","_reconstruct"),
 ("numpy._core.multiarray","scalar"),("numpy.core.multiarray","scalar"),
 ("numpy","ndarray"),("numpy","dtype"),
 ("gymnasium.spaces.box","Box"),
 ("stable_baselines3.common.running_mean_std","RunningMeanStd"),
 ("stable_baselines3.common.vec_env.vec_normalize","VecNormalize"),
}
class U(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) not in ALLOW: raise pickle.UnpicklingError(f"blocked {m}.{n}")
        return super().find_class(m, n)
root = sys.argv[1]; ok = True
for s in (42, 52, 60):
    d = f"{root}/RUN_s{s}/03_locomotion"
    for p in sorted(glob.glob(f"{d}/models/*.pkl")):
        try:
            with open(p, "rb") as fh: v = U(fh).load()
            o = v.obs_rms
            print(f"s{s} {os.path.basename(p)}: {type(v).__name__} obs_rms.count={o.count:.4f} mean.shape={o.mean.shape} finite={bool(np.isfinite(o.mean).all() and np.isfinite(o.var).all())} ret_rms.count={v.ret_rms.count:.4f} gamma={v.gamma} clip_obs={v.clip_obs}")
        except Exception as e:
            ok = False; print(f"s{s} {os.path.basename(p)}: LOAD FAIL {e!r}")
    j = json.load(open(f"{d}/stage_config.json")); print(f"s{s} stage_config.json: parses, top keys={list(j)[:8]}, run.duration_seconds={j.get('run',{}).get('duration_seconds')}")
    t = open(f"{d}/stage_summary.txt", encoding="utf-8").read(); print(f"s{s} stage_summary.txt: {len(t.splitlines())} lines, last: {t.strip().splitlines()[-1]!r}")
    z = zipfile.ZipFile(f"{d}/gate_progress.npz"); bad = z.testzip(); npz = np.load(f"{d}/gate_progress.npz", allow_pickle=False); print(f"s{s} gate_progress.npz: testzip={bad} arrays={len(npz.files)}")
print("ALL_OK" if ok else "FAILURES")
