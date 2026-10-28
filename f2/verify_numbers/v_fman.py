import csv, collections
M='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/manifest.csv'
rows=list(csv.DictReader(open(M,newline='')))
print('rows',len(rows))
c=collections.Counter((r['fetch'],r['run_id'],r['status']) for r in rows)
for k,v in sorted(c.items()): print(k,v)
for fe in ('F0','F2'):
    sub=[r for r in rows if r['fetch']==fe]
    print(fe,'utc range',min(r['utc'] for r in sub),max(r['utc'] for r in sub))
    for rid in sorted(set(r['run_id'] for r in sub)):
        s=[r for r in sub if r['run_id']==rid]
        cp=[r for r in s if r['status']=='COPY']
        paths=set(r['path'] for r in cp)
        b=sum(int(r['decoded_size'] or 0) for r in cp)
        print('  ',rid,'rows',len(s),'copies',len(cp),'paths',len(paths),'bytes',b,'listings',sum(1 for r in s if r['status']=='LISTING'),'errors',[ (r['path'],r['transfer'],r['note'][:80]) for r in s if r['status'] not in ('COPY','LISTING')])
    cp=[r for r in sub if r['status']=='COPY']
    print('  total copies',len(cp),'distinct (run,path)',len(set((r['run_id'],r['path']) for r in cp)),'bytes',sum(int(r['decoded_size'] or 0) for r in cp),'listings',sum(1 for r in sub if r['status']=='LISTING'))
    # paths with 2 copies
    cc=collections.Counter((r['run_id'],r['path']) for r in cp)
    print('  multi-copy paths',sorted(set(p for (rid,p),n in cc.items() if n>=2)), collections.Counter(n for n in cc.values()))
