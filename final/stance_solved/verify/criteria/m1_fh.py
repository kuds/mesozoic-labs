"""Read-only: per PHASE run and M1 label, the existing M1 summary's n_episodes, full_horizon_episodes, fall_flags count
and any-actuator-over-bar episode count (deterministic replays, seeds 3042-3061; not a v2 panel, not K1)."""
import json
B = '/home/user/work/pc_study/round/panels/stance7_r10LTSBpc_%s/%s/m1/m1_summary.json'
for s in ('s60', 's42', 's80'):
    for j in ('4194304', '5242880', '6291456', '8388608', '10485760', 'final'):
        d = json.load(open(B % (s, j)))
        ff = d.get('fall_flags'); ao = d.get('any_actuator_over_bar_episodes')
        print(s, j, 'n', d['n_episodes'], 'full_horizon', d['full_horizon_episodes'],
              'fall_flags', (len(ff) if isinstance(ff, (list, dict)) else ff),
              'any_over_bar', (len(ao) if isinstance(ao, (list, dict)) else ao),
              'max|cmd|', round(d['max_abs_command_any_step'], 4))
