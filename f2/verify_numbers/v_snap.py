import csv
ROOT='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2'
def f(x):
    try: return float(x)
    except: return float('nan')
for r,cut in (('s60',7602176),('s42',7602176),('s80',7733248)):
    rows=list(csv.DictReader(open(f'{ROOT}/stance6_r10LTSB_{r}/copies/01_stance/eval_episodes.csv.copy1',newline='')))
    pre=[x for x in rows if int(float(x['timesteps']))<=cut]
    pre1=[x for x in rows if int(float(x['timesteps']))<=3014656]
    ov=sum(1 for x in pre if f(x['saturated_actuator_fraction'])>0.10)
    ov1=sum(1 for x in pre1 if f(x['saturated_actuator_fraction'])>0.10)
    print(r,'rows<=cut',len(pre),'overbar',ov,'| snap1 rows',len(pre1),'overbar',ov1)
    # TEL tail_3 command per eval: mean over full-horizon episodes
    ev={}
    for x in rows:
        t=int(float(x['timesteps'])); ev.setdefault(t,[]).append(x)
    def m(t,col):
        v=[f(x[col]) for x in ev[t] if x['full_horizon'] in ('True','1','true','1.0')]
        return sum(v)/len(v) if v else None
    for t in (1310720,1441792,2883584,3014656,7471104,7602176):
        if t in ev:
            fh=[x for x in ev[t] if x['full_horizon'] in ('True','1','true','1.0')]
            n60=sum(1 for x in fh if abs(f(x['tail_3_pitch_act_command_mean']))>=0.60)
            print('   ',t,'tail3cmd mean(full-horizon)',round(m(t,'tail_3_pitch_act_command_mean'),4),'n|cmd|>=0.6',n60,'of',len(fh))
