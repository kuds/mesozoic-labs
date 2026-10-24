#!/bin/bash
# Regenerate the live replication pages (report only) into dated folders under S/foot/run_reports/live/<HHMM>/.
set -e
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
W=$S/fix/run_report
R5=$S/foot/stance5; RR=$R5/replication; R4=$S/foot/stance4
OUT=$S/foot/run_reports/live/$(date -u +%m%d_%H%M)
J="4194304 5242880 6291456 8388608 10485760 final"
export PYTHONPATH=$W PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="nice -n 19 python3 $W/environments/shared/scripts/plot_training_history.py"
mkdir -p $OUT
for s in 70 80; do
  $P $RR/runs/stance5_LTSB_s$s/01_stance --out $OUT/stance5r_s$s --mark-steps $J \
     --title "Tyrannosaurus Rex (Anatomical): stance5_LTSB_s$s, band LTSB replication (live)"
done
$P $RR/runs/stance5_LTSB_s70/01_stance $RR/runs/stance5_LTSB_s80/01_stance $R5/runs/stance5_LTSB_s52/01_stance $R5/runs/stance5_LTSB_s42/01_stance \
   --out $OUT/compare_replication_vs_stance5 --labels replication_s70 replication_s80 stance5_s52_holding stance5_s42_failing \
   --groups replication replication stance5 stance5 --mark-steps $J \
   --title "Band LTSB replication s70/s80 (live) against stance5 s52 (holding) and s42 (failing)"
$P $R5/runs/stance5_LTSB_s42/01_stance $R5/runs/stance5_LTSB_s52/01_stance $R5/runs/stance5_LTSB_s60/01_stance \
   $RR/runs/stance5_LTSB_s70/01_stance $RR/runs/stance5_LTSB_s80/01_stance \
   $R4/runs/stance4_LTS_s42/01_stance $R4/runs/stance4_LTS_s52/01_stance $R4/runs/stance4_LTS_s60/01_stance \
   --out $OUT/compare_band_vs_control --labels band_s42 band_s52 band_s60 band_s70 band_s80 control_s42 control_s52 control_s60 \
   --groups band band band band band control control control --mark-steps $J \
   --title "Band LTSB (stance5 s42/s52/s60 and replication s70/s80) against control LTS (stance4 s42/s52/s60)"
echo "$OUT"
