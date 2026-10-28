import json, os, statistics as st
P='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/panels'
def load(r,l): return json.load(open(f'{P}/stance6_r10LTSB_{r}/{l}/m1/m1_summary.json'))
def stats(r,l,a):
    d=load(r,l); eps=d['episodes']
    raw=[e['actuators'][a]['raw_window_mean'] for e in eps]
    f99=[e['actuators'][a]['share_ge_0p99'] for e in eps]
    f90=[e['actuators'][a]['share_ge_0p9'] for e in eps]
    g1=[e['actuators'][a]['share_raw_gt_1'] for e in eps]
    ob=sum(1 for e in eps if e['actuators'][a]['over_bar'])
    return dict(raw_mean=round(st.mean(raw),3),raw_max=round(max(raw),3),f99=round(st.mean(f99),3),f90=round(st.mean(f90),3),gt1=round(st.mean(g1),3),overbar=ob)
print('s60 8388608 tail3',stats('s60','8388608','tail_3_pitch_act'))
print('s60 final tail3',stats('s60','final','tail_3_pitch_act'))
for l in ('3670016','4194304','8388608','10485760','final'):
    print('s42',l,'tail3',stats('s42',l,'tail_3_pitch_act'))
mx=0
for l in os.listdir(f'{P}/stance6_r10LTSB_s80'):
    for a in ('tail_1_pitch_act','tail_1_yaw_act','tail_2_pitch_act','tail_3_pitch_act'):
        s=stats('s80',l,a); mx=max(mx,s['f99'],s['f90'],s['gt1'])
print('s80 max tail frac99/frac90/gt1 over all labels',mx)
for l in ('5242880','6291456','7340032','8388608'):
    s=stats('s80',l,'neck_pitch_act'); d=load('s80',l)
    print('s80',l,'neck',s)
    # max saturation share for other actuators (non-neck)
    oth=max(max(e['actuators'][a]['share_ge_0p99'] for a in e['actuators'] if a!='neck_pitch_act') for e in d['episodes'])
    print('   others max share_ge_0p99',round(oth,4))
# non-tail over bar at any label s60/s42
for r in ('s60','s42'):
    found=[]
    for l in os.listdir(f'{P}/stance6_r10LTSB_{r}'):
        d=load(r,l)
        for e in d['episodes']:
            for a,v in e['actuators'].items():
                if v['over_bar'] and not a.startswith('tail'): found.append((l,a))
    print(r,'non-tail over-bar at M1',found)
