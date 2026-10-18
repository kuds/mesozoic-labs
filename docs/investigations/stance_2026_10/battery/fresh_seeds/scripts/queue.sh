#!/bin/bash
F=scratchpad/foot/battery/fresh_seeds
printf '%s\n' "60 6291456 5042" "52 6291456 5042" "60 6291456 9042" "52 6291456 9042" \
              "60 robustbest 5042" "52 robustbest 5042" "60 robustbest 9042" "52 robustbest 9042" \
  | xargs -P 2 -L 1 bash $F/scripts/one_cert.sh
echo "QUEUEDONE $(date -Is)" >> $F/logs/queue.log
