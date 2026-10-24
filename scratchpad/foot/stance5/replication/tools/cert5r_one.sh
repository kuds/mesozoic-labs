#!/bin/bash
# cert5r_one.sh RUN_ID: six-heading certificate (stance_quality_heading/v1) of a stance5 replication run's final checkpoint,
# report only (SP/foot/stance5/tools/cert5_one.sh with the runs s70/s80 and R = SP/foot/stance5/replication; S5 below
# names R).
# stance3's cert_one.sh with the stance5 paths and the band clone (claude/screen-trex-r9-armlts-copband at cf80ecb1); the
# runner (stance3/cert/scripts/run_cert_clone.py) uses the clone's own stance config unchanged.
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
S5=$S/foot/stance5/replication; RUN=$1; CL=$S/fix/scratch_copband   # R
case "$RUN" in stance5_LTSB_s70|stance5_LTSB_s80) ;; *) echo "RUN_ID $RUN is not a replication run"; exit 2;; esac
[ "$(git -C $CL rev-parse HEAD)" = cf80ecb14d1630ce3b4dcdea28a05835d143470b ] || { echo "clone HEAD is not cf80ecb1"; exit 2; }
[ -z "$(git -C $CL status --porcelain)" ] || { echo "clone not clean"; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
M=$S5/runs/$RUN/01_stance/models; O=$S5/cert/$RUN
[ -f $M/stage1_final.zip ] && [ -f $M/stage1_final_vecnorm.pkl ] || { echo "final pair not placed for $RUN"; exit 3; }
mkdir -p $O
[ -f $O/run_summary.json ] && { echo "done already: $O/run_summary.json"; exit 0; }
echo "start $(date -Is) cert $RUN" >> $S5/cert/cert.log
cd $S5 && ISO_WT=$CL nice -n 19 python3 $S/foot/stance3/tools/iso.py $S/foot/stance3/cert/scripts/run_cert_clone.py \
  --model $M/stage1_final.zip --vecnorm $M/stage1_final_vecnorm.pkl --out $O --clone $CL > $O/cert.log 2>&1
rc=$?; echo "end $(date -Is) cert $RUN rc=$rc" >> $S5/cert/cert.log; exit $rc
