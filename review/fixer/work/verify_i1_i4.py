"""Fixer: re-derive build-check I1 (float64 feeding path vs float32 env path) and design I4 (discount weights)."""
import numpy as np
eps = 1e-8
def phi(k):
    return (min(k, 200) / 200, 1.0 if k >= 200 else 0.0, 0.98 ** max(0, 200 - k))
kk = np.arange(0, 1001)
f64 = np.array([phi(int(k)) for k in kk])
s = np.sqrt(1.0 + eps)
env_path = (f64.astype(np.float32).astype(np.float64) / s).astype(np.float32)
f64_path = (f64 / s).astype(np.float32)
cast_path = (f64.astype(np.float32).astype(np.float64) / s).astype(np.float32)  # the cast fix, = env path
diff = np.any(f64_path != f64.astype(np.float32), axis=1)
print("I1 env_path == float32(phi) for all k 0..1000:", bool(np.array_equal(env_path, f64.astype(np.float32))))
print("I1 float64 path: values differing", int((f64_path != f64.astype(np.float32)).sum()), "at k", kk[diff].tolist())
print("I1 settled k exact on the float64 path:", bool(np.array_equal(f64_path[200:], f64.astype(np.float32)[200:])))
rel = np.abs(f64_path.astype(np.float64) - f64.astype(np.float32).astype(np.float64)) / np.maximum(f64, 1e-30)
print("I1 max relative difference: %.3g" % rel.max())
# I4: mean discount weight of the gated stream over k 50..199 (0.98^(200-k)), and at k = 0
w = np.array([0.98 ** (200 - k) for k in range(50, 200)])
print("I4 mean 0.98^(200-k) over k 50..199: %.4f" % w.mean())
print("I4 0.98^200 = %.4f" % 0.98 ** 200)
for gap in (0.17, 0.23):
    g = gap / (1 - w.mean())       # gap = g x (1 - mean weight): implied settled gated value g
    print("I4 WHY gap %.2f -> gated value g %.3f -> phase-zero-to-settled swing g x (1 - 0.98^200) %.3f" % (gap, g, g * (1 - 0.98 ** 200)))
