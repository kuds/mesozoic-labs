"""Statue stance-quality baseline at a candidate Stage 1a operating point.

Companion to zero_action_baseline.py, which scores the do-nothing policy on
REWARD.  Stage 1a cannot gate on reward: summing the positive stage-1 weights
gives a ceiling of 3.35/step for T-Rex and the statue collects 97.0% of it with
energy and smoothness at exactly zero, so the statue IS the optimum and every
active policy scores below it.  See docs/STAGE1_SPLIT_PLAN.md 2.3.1.

What 1a can gate on is stance QUALITY, and these are the numbers a policy has to
MATCH rather than beat.  Usage:

    python -m environments.shared.scripts.stance_quality_baseline [noise] [episodes]
"""
import sys, warnings
import numpy as np, tomllib
warnings.filterwarnings('ignore')
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv
from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv

SPECIES = [(TRexEnv,'trex'),(RaptorEnv,'velociraptor'),(BrachioEnv,'brachiosaurus'),(DibothrosuchusEnv,'dibothrosuchus')]

def feet(info):
    keys = [k for k in info if k.endswith('_foot_contact')]
    return [float(info[k]) for k in sorted(keys)]

def run(cls, sp, noise, seeds):
    cfg = dict(tomllib.load(open(f'configs/{sp}/stage1_balance.toml','rb'))['env'])
    cfg['reset_noise_scale'] = noise
    env = cls(**cfg)
    z = np.zeros(env.action_space.shape, dtype=np.float32)
    full = 0; stand = []; duty = []; switch = []
    for s in seeds:
        env.reset(seed=int(s)); n = 0; R = 0.0; sup = []; 
        for _ in range(1000):
            _, r, term, trunc, info = env.step(z); n += 1; R += r
            f = feet(info)
            sup.append(sum(1 for v in f if v > 0.1))
            if term or trunc: break
        sup = np.array(sup); nf = len(feet(info))
        duty.append(((sup == 0).mean(), (sup == nf).mean()))
        allsup = (sup == nf).astype(int)
        switch.append(np.abs(np.diff(allsup)).mean() / env.dt if len(allsup) > 1 else 0.0)
        if n >= 1000: full += 1; stand.append(R)
    d = np.array(duty)
    return dict(full=full/len(seeds), unsup=d[:,0].mean(), allsup=d[:,1].mean(),
                switch=float(np.mean(switch)), stand=float(np.mean(stand)) if stand else float('nan'),
                stand_sd=float(np.std(stand)) if stand else float('nan'), n=len(seeds))

if __name__ == '__main__':
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    seeds = range(3042, 3042 + (int(sys.argv[2]) if len(sys.argv) > 2 else 40))
    print(f"statue stance quality at reset_noise={noise}, {len(list(seeds))} episodes")
    print(f"{'species':>15} {'full-hz':>8} {'all-feet':>9} {'unsup':>7} {'switch/s':>9} {'standing reward':>18}")
    for cls, sp in SPECIES:
        r = run(cls, sp, noise, seeds)
        sd = f"+/-{r['stand_sd']:.1f}" if r['stand_sd'] == r['stand_sd'] else ""
        print(f"{sp:>15} {r['full']:>7.0%} {r['allsup']:>9.3f} {r['unsup']:>7.3f} {r['switch']:>9.2f} "
              f"{r['stand']:>12.1f} {sd:>5}")
