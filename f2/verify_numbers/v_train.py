import csv, collections
ROOT='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2'
RO='/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/readouts'
W=[(0,4194304),(4194304,8388608),(8388608,12001281)]
for r in ('s60','s42','s80'):
    rows=list(csv.DictReader(open(f'{ROOT}/stance6_r10LTSB_{r}/copies/01_stance/train_episodes.csv.copy1',newline='')))
    if r=='s60': print('cols',list(rows[0].keys()))
    out=[]
    for a,b in W:
        s=[x for x in rows if a<=int(float(x['timesteps']))<b]
        nt=[x for x in s if x['cause']!='truncated']
        out.append((len(nt),len(s),dict(collections.Counter(x['cause'] for x in nt))))
    print(r,out)
rows=list(csv.DictReader(open(f'{RO}/std_per_update_s60.csv')))
for x in rows:
    t=float(x['update_timesteps'])
    if t>=5990000 and t<=6030000: print('ent',int(t),x['ent_coef'])
