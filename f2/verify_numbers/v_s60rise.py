import csv, numpy as np
ROOT='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2'
rows=list(csv.DictReader(open(f'{ROOT}/stance6_r10LTSB_s60/copies/01_stance/eval_episodes.csv.copy1',newline='')))
ev={}
for x in rows:
    if x['full_horizon']=='1': ev.setdefault(int(float(x['timesteps'])),[]).append(float(x['tail_3_pitch_act_command_mean']))
for t in sorted(ev):
    if 6000000<=t<=7500000: print('TEL',t,round(sum(ev[t])/len(ev[t]),3))
p=np.load(f'{ROOT}/stance6_r10LTSB_s60/copies/01_stance/policy_probe.npz.copy1',allow_pickle=False)
print(list(p.keys())[:30])
