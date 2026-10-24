#!/bin/bash
# Stage the handoff tree for branch claude/handoff-stance5r: text evidence only (no checkpoints, no npz, no b64), files <= 5 MB,
# laid out under scratchpad/ exactly as in this session's scratchpad, plus patches of the report branch, SHA256SUMS and restore.sh.
set -euo pipefail
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
DST=${1:?usage: build_handoff.sh DEST_DIR}
mkdir -p "$DST/scratchpad"
cd "$S"
EXCL=( -name '*.zip' -o -name '*.pkl' -o -name '*.npz' -o -name '*.b64' -o -name '*.pt' -o -name '*.pth' -o -name '*.lock' -o -size +5M -o -name 'score_interim_*' )
list() {  # list files under the given paths, minus excluded names and pruned directories
  find "$@" \( -type d \( -name models -o -name fetch -o -name 'review_*' -o -name 'fix_r*' -o -name selftest -o -name tensorboard -o -name __pycache__ -o -name extract -o -name state_tmp -o -name validation -o -name panels \) -prune \) -o -type f ! \( "${EXCL[@]}" \) -print 2>/dev/null
}
{
  list foot/stance4/live handoff foot/stage_report/design.md foot/run_reports/regen_live.sh
  list foot/stance5/criteria.txt foot/stance5/criteria.sha256 foot/stance5/criteria_notes.md foot/stance5/judge foot/stance5/judged foot/stance5/scores foot/stance5/colab_steps.txt
  list foot/stance5/k3diag/report.md foot/stance5/k3diag/files foot/stance5/k3diag/rollouts/tables foot/stance5/k3diag/rollouts/scripts
  find foot/stance5/k3diag -maxdepth 2 -path '*verify_*' -type f \( -name '*.py' -o -name '*.jsonl' -o -name '*.json' -o -name '*.sh' -o -name '*.txt' \) -size -2M
  list foot/stance5/tailexp/report.md foot/stance5/tailexp/results.csv foot/stance5/tailexp/lambda.json foot/stance5/tailexp/train_telemetry.csv
  find foot/stance5/tailexp -maxdepth 1 -type f -name '*.py' -o -maxdepth 1 -type f -name '*.sh'
  list foot/stance5/replication
  for d in foot/stance5/runs/* foot/stance4/runs/*; do find "$d/01_stance" -maxdepth 1 -type f ! -name 'train_episodes.csv' ! \( "${EXCL[@]}" \) 2>/dev/null; done
  list foot/stance4/criteria.txt foot/stance4/judge foot/stance4/judged foot/stance4/scores
  list foot/stance3/judge foot/stance3/scores foot/stance3/criteria.txt foot/stance3/tools/b64_to_file.py foot/stance3/tools/iso.py
  list foot/symmetry/design/frozen_params.json foot/tonight/noise_robustness/work/tail_seeds.txt pilot_runs/tools_dl/decode_verify.py
} | sort -u > "$DST/FILELIST.txt"
while read -r f; do mkdir -p "$DST/scratchpad/$(dirname "$f")"; cp -p "$f" "$DST/scratchpad/$f"; done < "$DST/FILELIST.txt"
# The report branch's local commits, as patches (the branch itself is not pushed).
mkdir -p "$DST/patches/run-training-report"
git -C "$S/fix/run_report" format-patch -q -o "$DST/patches/run-training-report" 1b71f00b..HEAD
( cd "$DST" && find scratchpad patches -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
du -sh "$DST"; wc -l < "$DST/FILELIST.txt"
