import csv
F='/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/readouts'
D={}
for r in ('s60','s42','s80'):
    rows=list(csv.DictReader(open(f'{F}/std_per_update_{r}.csv')))
    D[r]=[{k:float(v) for k,v in row.items()} for row in rows]
    t=[x['update_timesteps'] for x in D[r]]
    acts=[k for k in rows[0] if k.startswith('std_')]
    # max rise over any 1,048,576-step window (128 updates of 8192)
    maxrise=0; arg=None
    for a in acts:
        v=[x[a] for x in D[r]]
        for i in range(len(v)):
            for j in (i+128,):
                if j<len(v) and v[j]-v[i]>maxrise: maxrise=v[j]-v[i]; arg=(a,t[i])
    # 262144 window rises
    mr2=0
    for a in acts:
        v=[x[a] for x in D[r]]
        for i in range(len(v)-32):
            mr2=max(mr2,v[i+32]-v[i])
    # first below 0.08
    fb={}
    for a in acts:
        for x in D[r]:
            if x[a]<0.08: fb[a]=int(x['update_timesteps']); break
    mins={a:min(x[a] for x in D[r]) for a in acts}
    print(r,'n',len(D[r]),'max rise over 1M',round(maxrise,5),arg,'max rise over 262k',round(mr2,5))
    print('   first<0.08',{k.replace('std_',''):v for k,v in fb.items()})
    print('   final',{k.replace('std_','')[:10]:round(D[r][-1][k],4) for k in acts}, 'last t',D[r][-1]['update_timesteps'])
    print('   below 0.01',{k.replace('std_',''):[int(x['update_timesteps']) for x in D[r] if x[k]<0.01][:1] for k in acts if mins[k]<0.01})
# s42 tail_3 lowest from onset
idx={int(x['update_timesteps']):i for i,x in enumerate(D['s42'])}
bad=[]
for i,x in enumerate(D['s42']):
    t=int(x['update_timesteps'])
    if t>=3932160:
        if not (x['std_tail_3_pitch_act']<D['s60'][i]['std_tail_3_pitch_act'] and x['std_tail_3_pitch_act']<D['s80'][i]['std_tail_3_pitch_act']): bad.append(t)
print('s42 tail3 not lowest after 3932160 at',len(bad),'updates',bad[:5])
for t in (3932160,5505024):
    i=idx[t]; print(t,{r:round(D[r][i]['std_tail_3_pitch_act'],4) for r in D},{r:round(D[r][i]['std_neck_pitch_act'],4) for r in D})
