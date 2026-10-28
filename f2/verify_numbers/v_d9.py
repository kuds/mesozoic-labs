import json,glob,statistics as st
F='/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2'
for f in sorted(glob.glob(f'{F}/d9_d10/*/*/d*_summary.json')):
    d=json.load(open(f))
    w=d.get('window_return_vs_deterministic_m1')
    name=f.split('d9_d10/')[1]
    tf=d['tail_flags_full_horizon_by_actuator_sign']; fh=d['full_horizon_episodes']; c14=d['cmd_a']['14']
    if w is None:
        print(name,'flags',tf,'fh',fh,'c14',round(c14,3)); continue
    diffs=[e['window_return']-e['baseline_window_return'] for e in w['per_episode']]
    print(name,'flags',tf,'fh',fh,'c14',round(c14,3),'own mean',round(st.mean(diffs),2),'min',round(min(diffs),2),'max',round(max(diffs),2),'pos',sum(1 for x in diffs if x>0),'stats',json.dumps(w['stats'])[:300])
