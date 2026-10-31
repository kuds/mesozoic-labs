#!/usr/bin/env bash
# PS build checker: the runner's refusals in test mode, inside PS/check_build/runner_root only (commands `false`), and
# the real root's refusal paths that write nothing before refusing (read the runner: every `die` before the helpers
# precedes the first write).  Output: PS/check_build/logs/runner_refusals.out (the caller's redirection).
set -u
CB=/home/user/work/phase_screen/check_build
PS=/home/user/work/phase_screen
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
PY=$S2/venv/bin/python
R=$CB/runner_root
rm -rf "$R"; mkdir -p "$R"
T=(env BS_WAVES_TEST=1 BS_ROOT="$R" BS_RUN_CMD=false BS_CHECK_CMD=false BS_READ_CMD=false BS_POLL_SECONDS=0.2
   BS_LOCK_WAIT_SECONDS=1)
run() {  # run <label> <expect> <cmd...>
  local label=$1 expect=$2; shift 2
  local out rc
  out=$("$@" 2>&1); rc=$?
  printf '%s: exit %s (expect %s) %s :: %s\n' "$label" "$rc" "$expect" "$([[ $rc == "$expect" ]] && echo OK || echo MISMATCH)" \
    "$(tr '\n' ' ' <<< "$out" | cut -c1-260)"
}
screen() { "$PY" -I -B -c "import json,sys; json.dump({'schema': 'bs-screen/1', 'intervention': sys.argv[1], 'queue': json.loads(sys.argv[2])}, open(sys.argv[3], 'w'))" "$@"; }
echo "date -u $(date -u +%FT%TZ)"
echo "--- test-mode root $R"
run "T1 no SCREEN.json" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh"
run "T2 no SCREEN.json --plan" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh" --plan
screen critic_phase '[["critic_phase",7001],["control",7001],["critic_phase",7011],["control",7011]]' "$R/SCREEN.json"
run "T3 valid SCREEN --plan" 0 "${T[@]}" bash "$PS/tools/run_waves_bs.sh" --plan
run "T4 valid SCREEN, no APPROVED" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh"
echo '{"tool_sha256": {"score_bs.py": "0000"}}' > "$R/APPROVED"
run "T5 APPROVED, no criteria.txt" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh"
echo text > "$R/criteria.txt"
run "T6 APPROVED with a wrong scorer pin" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh"
screen critic_blind_tail3 '[["critic_blind_tail3",7001],["control",7001]]' "$R/SCREEN.json"
run "T7 a BS2 arm as the intervention" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh" --plan
screen critic_phase '[["critic_phase",9401],["control",9401]]' "$R/SCREEN.json"
run "T8 a PS validation seed in SCREEN" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh" --plan
screen critic_phase '[["critic_phase",7001],["control",7001],["critic_phase",7011],["control",7011],["critic_phase",7021],["control",7021],["critic_phase",7031],["control",7031],["critic_phase",7041]]' "$R/SCREEN.json"
run "T9 nine continuations (> 8)" 2 "${T[@]}" bash "$PS/tools/run_waves_bs.sh" --plan
run "T10 test mode on the real root" 2 env BS_WAVES_TEST=1 BS_ROOT="$PS" BS_RUN_CMD=false BS_CHECK_CMD=false BS_READ_CMD=false bash "$PS/tools/run_waves_bs.sh" --plan
echo "test-mode root after: $(find "$R" -type f | sort | tr '\n' ' ')"
echo "--- the real root (no APPROVED: every path below refuses before any write)"
ls -la --time-style=full-iso "$PS" > "$CB/logs/ps_root_before_refusals.txt"
run "R1 real runner" 2 bash "$PS/tools/run_waves_bs.sh"
run "R2 real runner --plan" "0or2" bash "$PS/tools/run_waves_bs.sh" --plan
run "R3 run_bs continuation preflight control" 2 "$PY" -I -B "$PS/tools/run_bs.py" --arm control --seed 2601 --preflight-only
run "R4 run_bs continuation preflight critic_phase" 2 "$PY" -I -B "$PS/tools/run_bs.py" --arm critic_phase --seed 2601 --preflight-only
run "R5 run_bs a BS2 arm" 2 "$PY" -I -B "$PS/tools/run_bs.py" --arm critic_blind_tail3 --seed 2601 --preflight-only
ls -la --time-style=full-iso "$PS" > "$CB/logs/ps_root_after_refusals.txt"
if cmp -s "$CB/logs/ps_root_before_refusals.txt" "$CB/logs/ps_root_after_refusals.txt"; then
  echo "PS root listing unchanged by R1-R5"
else
  echo "PS root listing CHANGED by R1-R5:"; diff "$CB/logs/ps_root_before_refusals.txt" "$CB/logs/ps_root_after_refusals.txt"
fi
[[ -e "$PS/APPROVED" ]] && echo "PS/APPROVED EXISTS" || echo "PS/APPROVED absent"
