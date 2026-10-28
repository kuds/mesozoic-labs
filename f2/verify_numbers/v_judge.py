import json
P='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/panels'
J=['4194304','5242880','6291456','8388608','10485760','final']
K3P={}
for r in ('s60','s42','s80'):
    for l in J:
        d=json.load(open(f'{P}/stance6_r10LTSB_{r}/{l}/m4/judge_{l}.json'))
        own=d['noise_matched_own']; jt=d['noise_matched_control_joint']
        def fall_info(x):
            out=[]
            for e in x['episodes_detail']:
                if not e.get('reached_horizon'):
                    keys=[k for k in e.keys() if 'fore_aft' in k and 'com' in k]
                    v={k:e[k] for k in keys}
                    out.append((e['seed'],e['length'],e['termination_reason'],{k:(round(vv,3) if isinstance(vv,float) else vv) for k,vv in v.items()}))
            return out
        hard=d['checks'].get('hard_failures') if isinstance(d.get('checks'),dict) else None
        print(r,l,'mode',d['mode'],'phys',d['checkpoint_physics_revision'],'git',d['git']['head'][:8],'relaxed',any(v for k,v in d['checks']['relaxed'].items() if k!='task_binding') , d['checks']['relaxed']['task_binding'],
              '| own',own['falls'],own['fall_seeds'],'sig',round(own['sigma_mean'],4),'| joint',jt['falls'],jt['fall_seeds'],jt['fall_lengths'],'sig',round(jt['sigma_mean'],4),
              'eqjudge',d['k3_panel_sigma']['equal_to_judge_vector'])
        fi=fall_info(jt)
        for x in fi: print('     fall',x)
        print('     ckeys',[k for k in d['checks'].keys()])
