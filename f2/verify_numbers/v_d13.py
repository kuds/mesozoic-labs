import json
from decimal import Decimal, ROUND_HALF_UP
F='/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2'
d=json.load(open(f'{F}/readouts/READOUTS.json'))
# REVIEW D-13 table values (W1,W2,W3) per run
REV={
 'head_tip_height_mean_m':{'s60':('1.005','1.066','1.090'),'s42':('0.943','0.963','1.003'),'s80':('1.061','1.305','1.134')},
 'neck_pitch_mean_deg':{'s60':('9.66','5.74','4.89'),'s42':('13.54','14.54','12.99'),'s80':('6.59','-13.12','8.22')},
 'head_pitch_mean_deg':{'s60':('10.65','8.92','6.68'),'s42':('12.60','7.76','4.16'),'s80':('8.91','7.88','-1.69')},
 'tail_3_pitch_act_command_mean':{'s60':('0.022','0.395','0.788'),'s42':('0.281','0.619','0.888'),'s80':('0.024','0.073','0.363')},
 'neck_pitch_act_command_mean':{'s60':('-0.086','-0.213','-0.242'),'s42':('0.031','0.040','-0.022'),'s80':('-0.181','-0.788','-0.154')},
 'head_pitch_act_command_mean':{'s60':('-0.094','-0.177','-0.272'),'s42':('-0.002','-0.202','-0.359'),'s80':('-0.166','-0.223','-0.601')},
 'tail_tip_chord_mean_deg':{'s60':('0.08','2.15','3.63'),'s42':('0.65','1.62','2.95'),'s80':('0.00','-0.52','0.78')},
 'neck_head_work_abs_j':{'s60':('0.85','1.59','3.61'),'s42':('0.62','2.26','4.87'),'s80':('2.46','7.42','7.80')},
 'root_height_mean_m':{'s60':('0.926',)*3,'s42':('0.926',)*3,'s80':('0.926',)*3},
}
bad=[]
for col,runs in REV.items():
    for r,vals in runs.items():
        w=d['runs'][f'stance6_r10LTSB_{r}']['D13']['windows']
        ex=[w[k][col] for k in w]
        for i,(rv,e) in enumerate(zip(vals,ex)):
            q=Decimal(rv).as_tuple().exponent
            rounded=Decimal(str(e)).quantize(Decimal(1).scaleb(q),rounding=ROUND_HALF_UP)
            if rounded!=Decimal(rv): bad.append((col,r,'W%d'%(i+1),'review',rv,'exact',e))
for b in bad: print(b)
print('checked',sum(len(v)*3 for v in REV.values()),'cells; mismatches',len(bad))
