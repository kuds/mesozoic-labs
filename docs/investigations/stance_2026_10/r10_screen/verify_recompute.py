import json, sys, os
from math import comb
J = sys.argv[1]
SEEDS = [42, 52, 60]; STEPS = [4194304, 5242880, 6291456]
def cp_lcb(k, n, alpha=0.05):
    # one-sided Clopper-Pearson lower bound via bisection on binomial tail
    if k == 0: return 0.0
    def tail(p):  # P(X >= k | p)
        return sum(comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo+hi)/2
        if tail(mid) < alpha: lo = mid
        else: hi = mid
    return (lo+hi)/2
out = {}
for s in SEEDS:
    for st in STEPS:
        d = json.load(open(f"{J}/s{s}/judge_{st}.json"))
        p = d['panel']; eps = p['episodes_detail']
        assert len(eps) == 40, len(eps)
        seeds = [e['seed'] for e in eps]
        assert seeds == list(range(3042, 3082)), seeds[:3]
        n_clean = sum(1 for e in eps if e['clean'])
        n_full = sum(1 for e in eps if e['length'] >= 1000)
        n_full_flag = sum(1 for e in eps if e['full_horizon'])
        n_reached = sum(1 for e in eps if e['reached_horizon'])
        hof = sum(1 for e in eps if e['hop_or_fall'])
        cop_fh = sum(1 for e in eps if e['full_horizon'] and e['max_foot_cop_fore_aft'] is not None and e['max_foot_cop_fore_aft'] > 0.80)
        cop_all = sum(1 for e in eps if e['max_foot_cop_fore_aft'] is not None and e['max_foot_cop_fore_aft'] > 0.80)
        cop_none = sum(1 for e in eps if e['max_foot_cop_fore_aft'] is None)
        roll_fh = sum(1 for e in eps if e['full_horizon'] and ((e['max_sole_tilt_deg'] or 0) > 2.0 or (e['max_sole_corner_lift_m'] or 0) > 0.004))
        roll_none = sum(1 for e in eps if e['full_horizon'] and (e['max_sole_tilt_deg'] is None or e['max_sole_corner_lift_m'] is None))
        roll_flag = sum(1 for e in eps if e['full_horizon'] and e['roll_rail'])
        # clean must imply no failing rails
        clean_check = sum(1 for e in eps if e['clean'] and (e['failing_rails'] or e['reasons']))
        nj = d['noise_matched_control_joint']; no = d['noise_matched_own']
        jf = sum(1 for e in nj['episodes_detail'] if not e['reached_horizon'])
        of = sum(1 for e in no['episodes_detail'] if not e['reached_horizon'])
        jf_len = sum(1 for e in nj['episodes_detail'] if e['length'] < 1000)
        of_len = sum(1 for e in no['episodes_detail'] if e['length'] < 1000)
        nseeds_j = [e['seed'] for e in nj['episodes_detail']]; nseeds_o = [e['seed'] for e in no['episodes_detail']]
        assert nseeds_j == list(range(3042, 3062)) and nseeds_o == nseeds_j
        lcb = cp_lcb(n_clean, 40)
        k1 = (p['verdict'] == 'PASS') and n_clean >= 37 and lcb >= 0.80 and n_full/40 >= 0.95 and hof <= 1
        mean_reward = sum(e['reward'] for e in eps)/40
        out[f"s{s}_{st}"] = dict(verdict=p['verdict'], n_clean=n_clean, rec_clean=p['n_clean'], lcb=lcb, rec_lcb=p['clean_lcb'],
            full=n_full, full_flag=n_full_flag, reached=n_reached, rec_full=p['n_full_horizon'], hof=hof, rec_hof=p['hop_or_fall_episodes'],
            cop_fh=cop_fh, cop_all=cop_all, cop_none=cop_none, rec_cop_fh=p['summary']['foot_cop_gt_bar_full_horizon'],
            roll_fh=roll_fh, roll_flag=roll_flag, roll_none=roll_none, rec_roll=p['summary']['roll_rail_full_horizon'],
            jf=jf, jf_len=jf_len, rec_jf=nj['falls'], of=of, of_len=of_len, rec_of=no['falls'], clean_check=clean_check,
            k1_seed=k1, mean_reward=mean_reward, rec_mean_reward=p['mean_reward'], statue=p['statue_mean_reward'],
            terms=p['terminations'], sigma_j=nj['sigma'], space_j=nj['space'], sigma_o=no['sigma'], space_o=no['space'],
            sigma_mean_j=nj['sigma_mean'], sigma_mean_o=no['sigma_mean'],
            )
for k, v in out.items():
    print(k, json.dumps(v))
# criteria
K1_seeds = [s for s in SEEDS if out[f"s{s}_6291456"]['k1_seed']]
K1 = len(K1_seeds) >= 2
K2_over = {k: v['cop_fh'] for k, v in out.items() if v['cop_fh'] > 3}
K2 = not K2_over
jp = sum(out[f"s{s}_6291456"]['jf'] for s in SEEDS); op = sum(out[f"s{s}_6291456"]['of'] for s in SEEDS)
K3 = jp <= 7 and op <= 3
rp = [out[f"s{s}_6291456"]['roll_fh'] for s in SEEDS]
K4 = sum(rp) <= 6 and all(r <= 3 for r in rp)
print('K1', K1, K1_seeds)
print('K2', K2, K2_over)
print('K3', K3, [out[f"s{s}_6291456"]['jf'] for s in SEEDS], jp, [out[f"s{s}_6291456"]['of'] for s in SEEDS], op)
print('K4', K4, rp, sum(rp))
json.dump(dict(per=out, K1=[K1, K1_seeds], K2=[K2, K2_over], K3=[K3, jp, op], K4=[K4, rp]), open(sys.argv[2], 'w'), indent=1, default=str)
