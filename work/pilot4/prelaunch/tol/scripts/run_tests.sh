#!/bin/bash
# tolerance lane (copied from the fix lane, from the e2e lane): every pilot-4 test file against the shared tree tools/tree (READ ONLY), one pytest run per file
# (plus test_pilot4_score.py through iso.py with the tree, and its pure mode), with sha256s of every file to be
# pinned before and after, the tree's status before and after, and the cache checks. Nothing trains.
# Each step's own exit status goes to $F/rcs.txt (and a STATUS line on stdout).
SP=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)
W=$SP/work
T=$W/pilot4/tools
P3=$W/pilot3/tools
LANE=$SP/work/pilot4/prelaunch/tol
F=$LANE/tests_final
PY=$W/pilot3/branch/venv_full/bin/python
export TMPDIR=$LANE/tmp MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 OMP_NUM_THREADS=1
export PILOT4_TREE=$T/tree STRAIGHT_VAL=$LANE/val_final
unset PYTHONPATH
mkdir -p "$F" "$TMPDIR"
RCS=$F/rcs.txt
: > "$RCS"
st() { echo "$1 rc=$2" >> "$RCS"; echo "STATUS $1=$2"; }
cd "$LANE" || exit 9
(cd "$T" && find . -path ./out -prune -o -path ./tree -prune -o -type f -print | LC_ALL=C sort | sed 's|^\./||') > "$F/pinned_files.txt"; st list_files $?
(cd "$T" && xargs -d '\n' sha256sum < "$F/pinned_files.txt") > "$F/sha256_before.txt"; st sha256_before $?
date -u +%FT%TZ > "$F/started.txt"
touch "$TMPDIR/marker_tests_final"; st marker $?
echo "tree HEAD $(git -C "$T/tree" rev-parse HEAD) status entries (with ignored) $(git -C "$T/tree" status --porcelain --ignored | wc -l)" >> "$RCS"
echo "env: OMP_NUM_THREADS=$OMP_NUM_THREADS MUJOCO_GL=$MUJOCO_GL PILOT4_TREE=$PILOT4_TREE STRAIGHT_VAL=$STRAIGHT_VAL PYTHONPATH=${PYTHONPATH-unset}" >> "$RCS"
# test_pilot4_score.py: through iso.py with the tree (fixture recomputation through the frozen probe), then pure.
s=$(date +%s.%N)
ISO_WT="$T/tree" "$PY" -I "$T/iso.py" "$T/test_pilot4_score.py" "$T/tree" > "$F/test_pilot4_score.tree.log.txt" 2>&1; rc=$?
e=$(date +%s.%N); st test_pilot4_score_tree $rc; echo "test_pilot4_score_tree elapsed_s $(echo "$e - $s" | bc)" >> "$RCS"
s=$(date +%s.%N)
"$PY" -I -B "$T/test_pilot4_score.py" > "$F/test_pilot4_score.pure.log.txt" 2>&1; rc=$?
e=$(date +%s.%N); st test_pilot4_score_pure $rc; echo "test_pilot4_score_pure elapsed_s $(echo "$e - $s" | bc)" >> "$RCS"
for name in test_pilot4_judge test_pilot4_straight test_pilot4_habit test_pilot4_noise test_pilot4_video test_pilot4_vg \
            test_pilot4_mirror test_pilot4_mech test_nb_pilot_runner4; do
  rm -rf "$TMPDIR/pt_$name"
  "$PY" -I -B "$LANE/scripts/run_pytest.py" "$F/$name.run.json" -p no:cacheprovider -c "$LANE/scripts/pytest.ini" \
      --rootdir "$LANE" --confcutdir "$T/tests" --basetemp "$TMPDIR/pt_$name" -v -rfEs --durations=10 \
      --junitxml "$F/$name.xml" "$T/tests/$name.py" > "$F/$name.log.txt" 2>&1
  st "$name" $?
done
echo "tree status entries (with ignored) after: $(git -C "$T/tree" status --porcelain --ignored | wc -l)" >> "$RCS"
echo "tree HEAD after: $(git -C "$T/tree" rev-parse HEAD)" >> "$RCS"
echo "files under pilot 3's tools newer than the start: $(find "$P3" -newer "$TMPDIR/marker_tests_final" | wc -l)" >> "$RCS"
echo "__pycache__/.pytest_cache/.pyc under the pilot-4 tools folder outside out/: $(find "$T" -path "$T/out" -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name '*.pyc' \) -print | wc -l)" >> "$RCS"
(cd "$T" && xargs -d '\n' sha256sum < "$F/pinned_files.txt") > "$F/sha256_after.txt"; st sha256_after $?
cmp "$F/sha256_before.txt" "$F/sha256_after.txt"; st files_unchanged_during_run $?
date -u +%FT%TZ > "$F/ended.txt"
cat "$RCS"
