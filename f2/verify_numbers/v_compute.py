import csv, datetime as dt, glob
F='/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2'
def p(s): return dt.datetime.strptime(s,'%Y-%m-%dT%H:%M:%SZ')
tot=0; n=0
for f in glob.glob(f'{F}/logs/measure/summary_slot*.tsv'):
    for row in csv.reader(open(f),delimiter='\t'):
        if len(row)>=7 and row[0].startswith('stance6'):
            m=(p(row[6])-p(row[5])).total_seconds()/60; tot+=m; n+=1
print('C7 jobs',n,'job-min',round(tot,1))
T=0; N=0; Tm=0; per={}
for r in ('s60','s42','s80'):
    c=0; mins=0; mins_ts=0
    for row in csv.reader(open(f'{F}/logs/report_only/stance6_r10LTSB_{r}/summary.tsv'),delimiter='\t'):
        if len(row)>=9 and row[5]=='ok':
            c+=1; mins+=float(row[8]); mins_ts+=(p(row[7])-p(row[6])).total_seconds()/60
    per[r]=(c,round(mins,1),round(mins_ts,1)); N+=c; T+=mins; Tm+=mins_ts
print('report-only',per,'jobs',N,'minutes col',round(T,1),'from timestamps',round(Tm,1))
print('total CPU-h',round((tot+T)/60,2), round((tot+Tm)/60,2))
