#!/bin/bash
# all 8 queue entries: run_bs.py --preflight-only (read only; its exit code recorded), then the real runner (expected
# refusal without APPROVED); each output captured under review/fixer/logs/preflight/
mkdir -p /home/user/work/phase_screen/review/fixer/logs/preflight
for seed in 2901 2911 2921 2931; do for arm in control critic_phase; do
  (cd /tmp && nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/phase_screen/tools/run_bs.py --arm $arm --seed $seed --preflight-only) > /home/user/work/phase_screen/review/fixer/logs/preflight/${arm}_$seed.out 2>&1
  rc=$?
  echo "$arm $seed exit $rc problems: $(tr -d '\n' < /home/user/work/phase_screen/review/fixer/logs/preflight/${arm}_$seed.out | sed 's/.*"problems": \[//; s/\].*//; s/  */ /g')"
done; done
(cd /tmp && nice -n 10 bash /home/user/work/phase_screen/tools/run_waves_bs.sh) > /home/user/work/phase_screen/review/fixer/logs/preflight/runner.out 2>&1
echo "runner exit $?: $(cat /home/user/work/phase_screen/review/fixer/logs/preflight/runner.out)"
