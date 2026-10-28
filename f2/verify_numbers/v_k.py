import json
F='/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2'
for r in ('s60','s42','s80'):
    d=json.load(open(f'{F}/kscore_stance6_r10LTSB_{r}.json'))['m4']
    c=d['criteria']
    k2c=c['K2']['K2c']
    med=k2c['median_mm']
    print(r,'K1',c['K1']['pass'],c['K1']['clean'],'durable',c['K1']['durable'],{k:(v['clean'],round(v['lcb'],4),v['hop_or_fall']) for k,v in c['K1']['detail'].items()})
    print('   K2',c['K2']['pass'],'K2a',c['K2']['K2a']['counts'],'K2b',c['K2']['K2b']['counts'])
    print('   K2c med range',round(min(med.values()),1),round(max(med.values()),1),'range_mm',round(k2c['range_mm'],2),'statue',set(round(v,2) for v in k2c['statue_median_mm'].values()))
    print('   K3',c['K3'],'K4',c['K4'],'failed',c['failed'],'HOLDS',c['HOLDS'],'FA',c['FORE_AFT_HOLDS'])
    print('   k1 modes',json.dumps(d['k1_failure_modes']),'missing',d['missing'],'invalid',d['invalid'])
