import json, os
from decimal import Decimal, ROUND_HALF_UP
P='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/panels'
T={'1048576':('+0.013','-0.035','-0.021'),'2097152':('+0.011','+0.151','+0.018'),'3145728':('+0.011','+0.524','+0.078'),
 '3670016':(None,'+0.931',None),'4194304':('+0.067','+0.989','+0.020'),'5242880':('+0.144','+0.884','-0.020'),
 '6291456':('+0.273','+0.355','+0.188'),'7340032':('+0.641','+0.648','+0.126'),'8388608':('+0.880','+0.941','+0.116'),
 '9437184':('+0.862','+0.803','+0.217'),'10485760':('+0.727','+0.936','+0.423'),'11534336':('+0.645','+0.866','+0.550'),'final':('+0.557','+0.893','+0.588')}
bad=[];n=0
for l,vals in T.items():
    for r,rv in zip(('s60','s42','s80'),vals):
        if rv is None: continue
        d=json.load(open(f'{P}/stance6_r10LTSB_{r}/{l}/m1/m1_summary.json')); e=d['cmd_a']['14']; n+=1
        rd=Decimal(repr(e)).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)
        if rd!=Decimal(rv): bad.append((r,l,rv,e))
print(bad,'checked',n)
