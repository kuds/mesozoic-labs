"""Read-only: per heading, the statue (zero action) panel's evidence and the policy panel's clean count, from the
existing M3 JSON of each PHASE final; also whether per-heading rewards differ across headings."""
import json, sys
base = '/home/user/work/pc_study/round/panels/stance7_r10LTSBpc_%s/final/m3/stance_heading_report.json'
for s in ('s60', 's42', 's80'):
    d = json.load(open(base % s))
    hs = d['headings']
    print('==', s, 'n headings', len(hs))
    rewards = []
    for h in hs:
        st = h.get('statue', {})
        ev = st.get('episode_evidence', [])
        keys = sorted(ev[0].keys()) if ev else []
        # count statue episodes that are 'clean' if a field exists
        clean_fields = [k for k in keys if 'clean' in k or 'qualif' in k or 'pass' in k]
        cnt = {k: sum(1 for e in ev if e.get(k) is True) for k in clean_fields}
        pol_clean = [k for k in h.keys() if 'clean' in k]
        print(' yaw', h.get('yaw_deg'), 'statue keys(clean-like)', clean_fields, cnt, 'statue ref', st.get('reference'),
              '| policy keys clean-like', {k: h[k] for k in pol_clean if not isinstance(h[k], (list, dict))})
        rewards.append(h.get('mean_reward', h.get('reward')))
    print(' per-heading policy mean_reward', rewards)
