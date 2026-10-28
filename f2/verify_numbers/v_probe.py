import numpy as np
ROOT='/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2'
W=[(0,4194304),(4194304,8388608),(8388608,12001281)]
for r in ('s60','s42','s80'):
    p=np.load(f'{ROOT}/stance6_r10LTSB_{r}/copies/01_stance/policy_probe.npz.copy1',allow_pickle=False)
    t=p['update_timesteps']; hi=p['rollout_mu_beyond_high_share']; lo=p['rollout_mu_beyond_low_share']; ent=p['ent_coef']
    print(r,'n',len(t),'shape',hi.shape,'ent first0 at',int(t[np.argmax(ent==0)]))
    h=hi[:,14]
    f05=t[np.argmax(h>=0.05)] if (h>=0.05).any() else None; f01=t[np.argmax(h>=0.01)] if (h>=0.01).any() else None
    print('   tail3 first>=0.05',f05,'first>=0.01',f01,'n>=0.05 after 7M',int(((h>=0.05)&(t>7e6)).sum()),'total',int((h>=0.05).sum()))
    print('   windows mean/max',[ (round(float(h[(t>=a)&(t<b)].mean()),4),round(float(h[(t>=a)&(t<b)].max()),3)) for a,b in W])
    if r=='s80':
        l=lo[:,0]
        print('   neck lo first>=0.05',t[np.argmax(l>=0.05)],'last>=0.05',t[np.where(l>=0.05)[0][-1]],'n after 7M',int(((l>=0.05)&(t>7e6)).sum()),'W2 mean/max',round(float(l[(t>=W[1][0])&(t<W[1][1])].mean()),4),round(float(l[(t>=W[1][0])&(t<W[1][1])].max()),3))
    hr=np.concatenate([hi[:,[4,8]],lo[:,[4,8]]],axis=1); print('   hip roll max share',float(hr.max()))
