import json, os
P='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/panels'
d=json.load(open(f'{P}/stance6_r10LTSB_s42/4194304/m2/m2_summary.json'))
print('posture keys',list(d['posture_means_full_horizon'].keys()))
print('other keys',[k for k in d.keys()])
J=['4194304','5242880','6291456','8388608','10485760','final']
for r in ('s60','s42','s80'):
    for l in J:
        d=json.load(open(f'{P}/stance6_r10LTSB_{r}/{l}/m2/m2_summary.json'))
        pm=d['posture_means_full_horizon']
        print(r,l,d['n_full_horizon'],d['n_clean'],d['passed'],round(d['mean_reward'],1),d['fall_dirs'],
              ' '.join(f'{k}={v:.3f}' if isinstance(v,(int,float)) else f'{k}={v}' for k,v in pm.items()))
