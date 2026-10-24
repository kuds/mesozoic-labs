#!/bin/bash
O=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/rollouts
LOG=${1:-$O/logs/run_main.log}
grep '^{' $LOG | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l); v=d.get('verified'); print(d['run'],d['steps'],d['cond'],d['falls'],d['fall_seeds'],d['seconds'],d.get('MB'),('VERIFIED' if v['lengths_terms_equal'] and v['reward_max_abs_diff']<1e-6 and v['max_sole_cop_equal'] and v['max_sole_tilt_equal'] else 'MISMATCH') if v else '')
"
grep -i "error\|Traceback\|FAILED\|refused" $LOG | head
grep -c ALLDONE $LOG
df -h / | tail -1
