"""Detector controls.  (1) Robot seeds 111, 143 (known creepers) and 141, 102 (resting) at 0.02 through scanlib, and
the same seeds through the copied creeplib.run/summary: the speed numbers must agree and the flags must split them.
(2) Seed 143 with scanlib's LoadRecorder and creeplib's SubstepTracer on the same episode: per-foot load agreement.
(3) The noise-free statue of both scanned models: its window numbers against the class's pinned reference."""
import json
import numpy as np
import scanlib as S
import creeplib as C
from environments.shared.gait.morphology import Morphology
from environments.shared.tests.com_stance_helpers import run_tracked_episode

out = open("controls.jsonl", "w")
print("(1) robot controls")
for seed in (111, 143, 141, 102):
    row, ep, metrics, _ = S.run("robot", seed, 0.02)
    ep2, series2, m2, probs2, _ = C.run(seed)
    s2 = C.summary(ep2, m2, probs2)
    out.write(json.dumps(row) + "\n"); out.flush()
    print(f"  seed {seed}: scan speed_end {1e3*row['com_speed_end_rms_mps']:.4f} mm/s, creeplib {1e3*s2['com_speed_end_rms_mps']:.4f} "
          f"(identical {row['com_speed_end_rms_mps'] == s2['com_speed_end_rms_mps']}); creeplib creeps {s2['creeps']}; "
          f"flags speed {row['flag_speed']} travel {row['flag_travel']} ({1e3*row['foot_travel_after1s_m']:.2f} mm) "
          f"oscill {row['flag_oscill']} (sd {row['load_sd_tail_bw_max']:.4f} BW, {row['osc_freq_hz']:.2f} Hz, share {row['osc_peak_share']:.2f}); "
          f"creep {row['creep']}; n_problems {row['n_problems']} vs {len(probs2)}; com travel window {row['com_travel_window_mm']:.2f} mm vs creeplib {s2['com_disp_mm']:.2f}; "
          f"unloaded tail {row['unloaded_tail_frac']}; {row['wall_s']} s")

print("(2) per-foot load: scanlib recorder vs creeplib SubstepTracer, robot seed 143")
env = C.make_env("base")
u = env.unwrapped
morph = Morphology.from_env(env, C.CLS.model_id)
rec = S.LoadRecorder(env, morph)
tr = C.SubstepTracer(env)
u._substep_probe_hook = lambda: (rec(), tr())
ep = run_tracked_episode(env, C.CLS.model_id, seed=143, morphology=morph)
env.close()
t, load, other, vz, ncon = rec.arrays()
A = tr.arrays()
labels = [f.label for f in morph.feet]
# creeplib side 0 = left; map scanlib foot order by label
order = [0 if lab.startswith("l") else 1 for lab in labels]
diff = max(float(np.abs(load[:, k] - A["fn"][:, order[k]]).max()) for k in range(len(labels)))
print(f"  feet {labels}; substeps {len(t)}; max |load diff| {diff:.3e} N; other-floor max {other.max():.3e} N; root vz diff {np.abs(vz - A['rootvel'][:, 2]).max():.3e}")

print("(3) statues")
for model in ("compy_anat", "velo"):
    row, ep, metrics, _ = S.run(model, 0, None)
    CLS = S.CLASSES[model]
    devs = {k: metrics[k] - v for k, v in CLS.reference.items()}
    worst = max(devs, key=lambda k: abs(devs[k]) / (1e-3 if not k.endswith('_deg') else 0.1))
    out.write(json.dumps(row) + "\n"); out.flush()
    print(f"  {model} statue: stood {row['stood']}, n_problems {row['n_problems']}; largest |window value - pinned| on {worst}: {devs[worst]:.3e}; "
          f"speed_end {1e3*row['com_speed_end_rms_mps']:.4f} mm/s, foot travel after 1 s {1e3*row['foot_travel_after1s_m']:.4f} mm, "
          f"load sd tail {row['load_sd_tail_bw_max']:.2e} BW, root vz tail {row['root_vz_tail_rms_mmps']:.4f} mm/s; flags {row['flag_speed']} {row['flag_travel']} {row['flag_oscill']}; {row['wall_s']} s")
