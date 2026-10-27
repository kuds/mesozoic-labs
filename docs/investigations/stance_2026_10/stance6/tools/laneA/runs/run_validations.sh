#!/bin/sh
# Lane A's validations with the final tools, in order, at most 2 processes, nice -n 10, one thread each:
#   unit tests; V1 (M2 statue) with V3 (M3 statue); V1's comparison; V4 (M1 on the zero-action checkpoint) and its
#   check; V5 (the four runners refuse an r9 / an unrecorded physics_revision); the M4 r9 reproduction comparisons
#   (the judge files exist: compare only) and the scoring reproduction.
# usage: run_validations.sh OUT_ROOT COPY [BASIS]
#        COPY: a sealed measuring copy (required, fixer RA-6: measure_r10round.py build-copy --dest COPY ...; the unit
#        tests read it as R10M_COPY).  BASIS (criteria lean v5: no tool holds the branch HEAD): "--header-parent" for
#        a copy of the header's parent cf9402bc (before the header commit), or "--approved APPROVED" (default: the
#        approval record <ROUND>/APPROVED).  V1, V3, V4 and V5 run on the measuring copy of the header commit once it
#        is made (VALIDATIONS): build its copy (build-copy --approved ...), rebuild the statue checkpoints on it
#        (statue_ckpt.py) into OUT_ROOT/ckpt, and pass the copy.
A=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/lean/build/laneA
P=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
OUT=$1
COPY=${2:?usage: run_validations.sh NEW_OUT_ROOT COPY [BASIS] (a sealed measuring copy: measure_r10round.py build-copy --dest COPY)}
shift 2
BASIS="$*"
CK=${CKPT_ROOT:-$A/work/ckpt}
T=$A/tools
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
       MPLCONFIGDIR=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/caches/mpl
[ -n "$OUT" ] && [ ! -e "$OUT" ] || { echo "usage: $0 NEW_OUT_ROOT COPY"; exit 2; }
mkdir -p $OUT/logs
L=$OUT/logs/validations.log
note() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
cd $OUT/logs
note "lane A validations; copy $COPY; tools $(sha256sum $T/measure_r10round.py | cut -c1-16) (measure)"
$P -I -B $T/measure_r10round.py check-copy --copy $COPY $BASIS > $OUT/logs/copy_seal_before.json || { note "copy refused"; exit 2; }
(cd $T && R10M_COPY=$COPY nice -n 10 $P -I -B -m pytest -q -p no:cacheprovider tests/) > $OUT/logs/unit_tests.log 2>&1
note "unit tests rc=$? ($(tail -1 $OUT/logs/unit_tests.log))"
mkdir -p $OUT/V1 $OUT/V3 $OUT/V4
nice -n 10 $P -I -B $T/measure_r10round.py m2 --copy $COPY --statue --no-run-records --out $OUT/V1/m2_statue --label statue $BASIS > $OUT/logs/V1_m2_statue.log 2>&1 &
P1=$!
nice -n 10 $P -I -B $T/measure_r10round.py m3 --copy $COPY --zero-action --no-run-records --out $OUT/V3/m3_statue $BASIS > $OUT/logs/V3_m3_statue.log 2>&1 &
P3=$!
wait $P1; note "V1 M2 statue rc=$?"
$P -I -B $T/v1_check.py $OUT/V1/m2_statue $OUT/V1/v1_check.json > $OUT/logs/V1_check.log 2>&1; note "V1 check rc=$?"
nice -n 10 $P -I -B $T/measure_r10round.py m1 --copy $COPY --zip $CK/statue_r10/stage1_final.zip --pkl $CK/statue_r10/stage1_final_vecnorm.pkl --no-run-records --out $OUT/V4/m1_statue $BASIS > $OUT/logs/V4_m1_statue.log 2>&1
note "V4 M1 statue rc=$?"
$P -I -B $T/v4_check.py $OUT/V4/m1_statue $OUT/V4/v4_check.json > $OUT/logs/V4_check.log 2>&1; note "V4 check rc=$?"
$P -I -B $T/v5_check.py $COPY $CK $OUT/V5 $BASIS > $OUT/logs/V5_check.log 2>&1; note "V5 check rc=$?"
wait $P3; note "V3 M3 statue rc=$?"
$P -I -B -c "import json,sys; d=json.load(open('$OUT/V3/m3_statue/m3_summary.json')); ok=d['passed'] and all(h['n_clean']==40 and h['n_episodes']==40 for h in d['headings']) and len(d['headings'])==6 and d['physics_revision']==10; print(json.dumps({'V3_PASS': ok, 'n_clean_by_heading': {h['yaw_deg']: h['n_clean'] for h in d['headings']}})); sys.exit(0 if ok else 1)" > $OUT/logs/V3_check.log 2>&1
note "V3 check rc=$? $(cat $OUT/logs/V3_check.log)"
for spec in "stance5_LTSB_s70 final 70" "stance5_LTSB_s80 4194304 80"; do
  set -- $spec
  $P -I -B $T/m4_r9_check.py $1 $2 $3 $A/copies/r9_cf80ecb1 $A/work/r9runs/$1 $A/out/M4_r9/$1 > $OUT/logs/M4_r9_compare_$1_$2.log 2>&1
  note "M4 r9 compare $1 $2 rc=$?"
done
$P -I -B $T/kscore_r10ltsb.py --reproduce-repl --json $OUT/kscore_reproduce_repl.json > $OUT/logs/kscore_reproduce_repl.log 2>&1
note "M4 scoring reproduction rc=$?"
$P -I -B $T/measure_r10round.py check-copy --copy $COPY $BASIS > $OUT/logs/copy_seal_after.json; note "copy unchanged rc=$?"
note "done"
