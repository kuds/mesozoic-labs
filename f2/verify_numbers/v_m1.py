import json, os
P='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/panels'
labs=['1048576','2097152','3145728','3670016','4194304','5242880','6291456','7340032','8388608','9437184','10485760','11534336','final']
res={}
for r in ('s60','s42','s80'):
    res[r]={}
    for l in labs:
        f=f'{P}/stance6_r10LTSB_{r}/{l}/m1/m1_summary.json'
        if not os.path.exists(f): continue
        d=json.load(open(f))
        tf=d['tail_flags']; ff=d['fall_flags']
        fh=d['full_horizon_episodes']
        ob=d['any_actuator_over_bar_episodes']
        hr=d['hip_roll']
        res[r][l]=dict(c14=round(d['cmd_a']['14'],4),c11=round(d['cmd_a']['11'],4),c12=round(d['cmd_a']['12'],4),c13=round(d['cmd_a']['13'],4),
          ntf=len(tf), tfby=d['tail_flags_full_horizon_by_actuator_sign'], nff=len(ff), fh=fh, nep=d['n_episodes'],
          ob=[(e.get('actuator') if isinstance(e,dict) else e) for e in ob][:3], nob=len(ob),
          hr4=round(hr['4']['mean_over_episodes'],4), hr8=round(hr['8']['mean_over_episodes'],4),
          hrmin=round(min(min(hr['4']['window_mean_by_episode']),min(hr['8']['window_mean_by_episode'])),4),
          hrmax=round(max(max(hr['4']['window_mean_by_episode']),max(hr['8']['window_mean_by_episode'])),4),
          hrpres=hr['4']['present_every_step'] and hr['8']['present_every_step'],
          c0=round(d['cmd_all_actuators']['0'],4))
        print(r,l,json.dumps(res[r][l]))
json.dump(res,open('/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/verify_numbers/m1_table.json','w'),indent=1)
# MEAN_a and margins
for r in res:
    R1=res[r]['10485760']; R2=res[r]['final']
    for a in ('c11','c12','c13','c14'):
        print(r,a,'MEAN_a',round((R1[a]+R2[a])/2,4))
    J=['4194304','5242880','6291456','8388608','10485760','final']
    print(r,'J c14',[ (j,res[r][j]['c14'],round(res[r][j]['c14']-0.6,4)) for j in J])
