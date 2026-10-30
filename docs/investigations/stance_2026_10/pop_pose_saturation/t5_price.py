"""Track 5 + Track 4 (gradient side): price per-actuator action signals under the current stance.toml weights."""
import json
import numpy as np

N = 15
W = dict(energy=0.075, smooth=2.0, jerk=3.0, sat=0.5)
ALPHA = 0.38586954509503757  # 10 Hz first-order filter at dt = 0.01


def lp(x):
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = y[t - 1] + ALPHA * (x[t] - y[t - 1])
    return y


def price(raw):
    """Per-step mean cost contributions of ONE actuator (already divided by N) for a raw command sequence."""
    a = np.clip(raw, -1, 1)
    f = lp(a)
    d1 = np.diff(a)
    d2 = a[2:] - 2 * a[1:-1] + a[:-2]
    sat = np.clip((np.abs(a) - 0.9) / 0.1, 0, 1)
    out = dict(
        energy=W["energy"] * np.mean(f[50:] ** 2) / N,
        smooth=W["smooth"] * np.mean(d1[50:] ** 2) / (N * 4),
        jerk=W["jerk"] * np.mean(d2[50:] ** 2) / (N * 16),
        sat=W["sat"] * np.mean(sat[50:]) / N,
    )
    out["total"] = sum(out.values())
    out["filtered_ac_rms"] = float(np.std(f[50:]))
    out["raw_ac_rms"] = float(np.std(a[50:]))
    return {k: float(v) for k, v in out.items()}


T = 20000
t = np.arange(T)
res = {}
res["DC_0"] = price(np.zeros(T))
res["DC_+0.5"] = price(np.full(T, 0.5))
res["DC_+0.8"] = price(np.full(T, 0.8))
res["DC_+0.9"] = price(np.full(T, 0.9))
res["DC_+1"] = price(np.ones(T))
for period in (2, 3, 4, 6, 8, 20, 100):
    sq = np.where((t % period) < period / 2, 1.0, -1.0)
    res[f"square_pm1_{100/period:.1f}Hz"] = price(sq)
# #489 statistics: DC 0.8 + AC rms 0.277 at 22.6 Hz
res["489_DC0.8_AC0.277_22.6Hz"] = price(0.8 + 0.277 * np.sqrt(2) * np.sin(2 * np.pi * 22.6 * 0.01 * t))
res["489_DC0_AC0.277_22.6Hz"] = price(0.277 * np.sqrt(2) * np.sin(2 * np.pi * 22.6 * 0.01 * t))
res["sine_AC0.277_1.2Hz"] = price(0.277 * np.sqrt(2) * np.sin(2 * np.pi * 1.2 * 0.01 * t))
# filter gain vs frequency
res["filter_gain"] = {f"{fq}Hz": float(abs(ALPHA / (1 - (1 - ALPHA) * np.exp(-2j * np.pi * fq * 0.01)))) for fq in (1, 2.5, 5, 10, 12.5, 16.7, 20, 22.6, 25, 33.3, 50)}
res["filter_phase_lag_deg"] = {f"{fq}Hz": float(-np.degrees(np.angle(ALPHA / (1 - (1 - ALPHA) * np.exp(-2j * np.pi * fq * 0.01))))) for fq in (1, 1.25, 2.5, 5, 10, 20)}

# Exploration-noise pricing near the rail: i.i.d. Gaussian raw samples around mean mu
rng = np.random.default_rng(0)
expl = {}
for sigma in (0.14, 0.22):
    for mu in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.11, 1.2, 1.3, 1.5):
        raw = mu + sigma * rng.standard_normal(200000)
        p = price(raw)
        p["clip_share"] = float(np.mean(raw > 1.0))
        expl[f"sigma{sigma}_mu{mu}"] = p
res["exploration"] = expl
# derivative of expected per-step cost wrt mu (finite difference), sigma=0.14
grad = {}
for sigma in (0.14, 0.22):
    for mu in (0.5, 0.9, 1.0, 1.11, 1.3):
        h = 0.01
        z = np.random.default_rng(1).standard_normal(400000)
        c_plus = price(mu + h + sigma * z)["total"]
        c_minus = price(mu - h + sigma * z)["total"]
        grad[f"sigma{sigma}_mu{mu}"] = (c_plus - c_minus) / (2 * h)
res["dcost_dmu_per_step"] = grad
with open("scratchpad/t5_price.json", "w") as f:
    json.dump(res, f, indent=1)
for k, v in res.items():
    if k in ("exploration",):
        for kk, vv in v.items():
            print(f"  {kk:22s} total {vv['total']:.5f} (E {vv['energy']:.5f} S {vv['smooth']:.5f} J {vv['jerk']:.5f} Sat {vv['sat']:.5f}) clip {vv['clip_share']:.2f}")
    elif isinstance(v, dict) and "total" in v:
        print(f"{k:28s} total/step {v['total']:.5f} = E {v['energy']:.5f} S {v['smooth']:.5f} J {v['jerk']:.5f} Sat {v['sat']:.5f} | per 1000 steps {1000*v['total']:.1f} | filtAC {v['filtered_ac_rms']:.3f} rawAC {v['raw_ac_rms']:.3f}")
    else:
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
