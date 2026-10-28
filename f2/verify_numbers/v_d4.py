import json
from decimal import Decimal, ROUND_HALF_UP
P='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/panels'
L={'4.19M':'4194304','5.24M':'5242880','6.29M':'6291456','8.39M':'8388608','10.49M':'10485760','final':'final'}
T='''s60 4.19M 40 4447.6 +0.66 +4.09 0.944 -33.0 +0.72 1.061 - -
s60 5.24M 39 4443.0 +1.22 +4.77 0.959 -26.9 +1.16 1.021 - -
s60 6.29M 40 4435.1 +1.63 +5.60 0.967 -33.7 +0.71 1.058 - -
s60 8.39M 34 4376.2 +4.26 +9.75 1.030 -30.4 +1.03 1.087 - -
s60 10.49M 38 4383.6 +3.69 +8.87 1.017 -28.6 +1.17 1.074 - -
s60 final 40 4396.0 +2.38 +7.09 0.986 -31.6 +0.94 1.115 - -
s42 4.19M 3 4349.1 +2.92 +8.35 1.001 -19.3 +1.87 0.924 +14.8 +10.7
s42 5.24M 30 4330.5 +1.89 +7.27 0.975 -37.0 +0.66 1.079 +6.0 +7.7
s42 6.29M 39 4412.4 +1.12 +5.13 0.958 -19.1 +1.54 0.901 +17.2 +11.2
s42 8.39M 4 4338.3 +2.47 +7.77 0.990 -17.3 +1.62 0.929 +21.5 +0.3
s42 10.49M 7 4351.4 +3.33 +8.67 1.011 -25.3 +1.75 1.024 +10.0 +5.4
s42 final 38 4353.1 +3.77 +9.01 1.019 -34.9 +1.17 1.155 -2.6 +9.1
s80 4.19M 39 4416.3 -0.40 +2.86 0.918 -34.4 +0.40 1.268 -6.5 +3.0
s80 5.24M 37 4418.0 -0.68 +2.45 0.911 -37.1 +0.28 1.308 -12.6 +6.7
s80 6.29M 5 4377.4 -0.27 +3.37 0.917 -42.0 -0.44 1.346 -17.4 +10.2
s80 8.39M 40 4426.4 -1.10 +2.21 0.899 -36.2 -0.14 1.014 +11.5 +9.6
s80 10.49M 40 4380.6 +0.91 +4.80 0.948 -36.0 +0.27 1.260 -0.9 -3.6
s80 final 40 4386.9 +2.12 +6.39 0.977 -30.2 +0.43 1.166 +6.5 -3.3'''
keys=['n_clean','mean_reward','chord_base_deg','chord_dist_deg','tip_z_m','com_fa_mm','pitch_deg','head_tip_z_m','neck_pitch_deg','head_pitch_deg']
bad=[];n=0
for line in T.splitlines():
    f=line.split(); r,l=f[0],L[f[1]]; vals=f[2:]
    d=json.load(open(f'{P}/stance6_r10LTSB_{r}/{l}/m2/m2_summary.json')); pm=d['posture_means_full_horizon']
    ex=[d['n_clean'],d['mean_reward']]+[pm[k] for k in keys[2:]]
    for k,rv,e in zip(keys,vals,ex):
        if rv=='-': continue
        n+=1
        q=Decimal(rv).as_tuple().exponent
        rd=Decimal(repr(float(e))).quantize(Decimal(1).scaleb(q),rounding=ROUND_HALF_UP)
        if rd!=Decimal(rv): bad.append((r,f[1],k,rv,e))
for b in bad: print(b)
print('checked',n,'mismatches',len(bad))
