#!/usr/bin/env bash
# Tail_3 bound screen: the durable copy (criteria D8; BUILD_SPEC section 8).
#
#   copy_durable.sh {init | replicate <name> | panels} [--dry-run]
#
# Stages files into SCREEN/durable_stage/ (paths relative to SCREEN, plus patches/tail3-upper-bound.patch), refuses any
# file over 5 MB, and rewrites FILELIST.txt and SHA256SUMS over the whole stage.
#   init            criteria (final text + its sha256, v2, BUILD_SPEC, criteria.txt), APPROVED, APPROVAL_REQUEST.md,
#                   PINS.txt, calib/coef.json + calib.log, the scratch commits as a patch file (git format-patch
#                   PARENT..<pin or HEAD>), tools/ and their sha256, measure_copy seals, validate/v1.json +
#                   v1_panels.csv + V1 flags, tests_out/*, and the INPUTS needed to restore the screen if the
#                   container is lost: both start checkpoint pairs (inputs/fetch/<RUN>/A/, 4.03 MB zip + 5 KB pkl each),
#                   both runs' recipe-check stage_config.json (inputs/stage_config/<state>/) and RUNLOG
#                   (inputs/run_log.txt).
#   replicate NAME  runs/NAME/{run_meta.json,progress.csv,telemetry.npz}, the R1 (5111808) and R2 (final) checkpoint
#                   pairs, and every panel of NAME (episode_trace.json, flags.csv, csv_sha256.txt, trace_run.json),
#                   plus panels.csv, panels_crashed.csv and logs/NAME_rerun_compare.json as they stand, and RUNLOG.
#   panels          every panel's small files, panels.csv, scores/score.{json,txt}, LOST.txt.
#
# --dry-run (the only mode the build runs): stages and prints the git commands real mode would run.  Nothing is
# cloned, committed or pushed.
# Real mode refuses unless SCREEN/APPROVED exists, screen_common.DURABLE_BRANCH is pinned, and at least one
# runs/*/run_meta.json exists (training has started).  It works in SCREEN/durable_repo, a separate repository whose
# origin is the GitHub URL (never the scratch clone, a judge clone or the main checkout): orphan branch on first use,
# commit of the stage, push of that branch only.  The branch is never merged and never opened as a PR.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 LC_ALL=C

S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
SCREEN=$S2/tailround/screen
PY=$S2/venv/bin/python
STAGE=$SCREEN/durable_stage
REPO=$SCREEN/durable_repo
ORIGIN_URL=https://github.com/kuds/mesozoic-labs
MAX_BYTES=$((5 * 1024 * 1024))
R1=5111808

die() { echo "copy_durable: refused: $*" >&2; exit 2; }

common_state() {  # common_state <state> <key>: one screen_common.STATES value
  "$PY" -I -B -c "import importlib.util as u,sys;s=u.spec_from_file_location('c','$SCREEN/tools/screen_common.py');m=u.module_from_spec(s);s.loader.exec_module(m);print(m.STATES[sys.argv[1]][sys.argv[2]])" "$1" "$2"
}

common() {  # print one screen_common value (no torch, no environments)
  "$PY" -I -B -c "import importlib.util as u,sys,json;s=u.spec_from_file_location('c','$SCREEN/tools/screen_common.py');m=u.module_from_spec(s);s.loader.exec_module(m);v=getattr(m,sys.argv[1]);print(json.dumps(v) if isinstance(v,(tuple,list,dict)) else ('' if v is None else v))" "$1"
}

MODE=${1:-}
[[ -n "$MODE" ]] || die "usage: copy_durable.sh {init | replicate <name> | panels} [--dry-run]"
shift
NAME=""
if [[ "$MODE" == replicate ]]; then
  NAME=${1:-}
  [[ -n "$NAME" && "$NAME" != --* ]] || die "replicate needs a name"
  shift
fi
DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) die "unknown argument $a" ;;
  esac
done
case "$MODE" in init|replicate|panels) ;; *) die "unknown mode $MODE" ;; esac

CLONE=$(common CLONE)
PARENT=$(common PARENT)
PIN=$(common SCRATCH_COMMIT)
BRANCH=$(common DURABLE_BRANCH)
CHANGED=$(common CHANGED_FILES)

if [[ "$MODE" == replicate ]]; then
  "$PY" -I -B -c "import sys;n=sys.argv[1].split('_');sys.exit(0 if len(n)==3 and n[0] in('s42','s80') and n[1] in('control','b90','b60') and n[2] in('r101','r202','r303') else 1)" "$NAME" \
    || die "$NAME is not one of the 18 screen names"
fi

if [[ $DRY -eq 0 ]]; then
  [[ -f "$SCREEN/APPROVED" ]] || die "no SCREEN/APPROVED (real mode runs only after approval)"
  [[ -n "$BRANCH" ]] || die "screen_common.DURABLE_BRANCH is not pinned"
  [[ -n "$PIN" ]] || die "screen_common.SCRATCH_COMMIT is not pinned"
  compgen -G "$SCREEN/runs/*/run_meta.json" > /dev/null || die "no runs/*/run_meta.json: training has not started"
fi

MISSING=()
mkdir -p "$STAGE"

stage() {  # stage <path relative to SCREEN> [required]
  local rel=$1 req=${2:-}
  local src=$SCREEN/$rel
  if [[ ! -f "$src" ]]; then
    MISSING+=("$rel${req:+ (required)}")
    return 0
  fi
  local size
  size=$(stat -c %s "$src")
  (( size <= MAX_BYTES )) || die "$rel is $size bytes, over 5 MB"
  mkdir -p "$STAGE/$(dirname "$rel")"
  cp -p "$src" "$STAGE/$rel"
}

stage_ext() {  # stage_ext <absolute source outside SCREEN> <destination relative to the stage> [required]
  local src=$1 rel=$2 req=${3:-}
  if [[ ! -f "$src" ]]; then
    MISSING+=("$src${req:+ (required)}")
    return 0
  fi
  local size
  size=$(stat -c %s "$src")
  (( size <= MAX_BYTES )) || die "$src is $size bytes, over 5 MB"
  mkdir -p "$STAGE/$(dirname "$rel")"
  cp -p "$src" "$STAGE/$rel"
}

stage_inputs() {  # the start pairs, the recipe-check stage configs and RUNLOG (restorable by sha256)
  local st
  for st in s42 s80; do
    local zip pkl cfg
    zip=$(common_state "$st" zip)
    pkl=$(common_state "$st" pkl)
    cfg=$(common_state "$st" stage_config)
    stage_ext "$zip" "inputs/fetch/${zip#"$S2"/tailround/fetch/}" required
    stage_ext "$pkl" "inputs/fetch/${pkl#"$S2"/tailround/fetch/}" required
    stage_ext "$cfg" "inputs/stage_config/$st/stage_config.json" required
  done
  stage_ext "$(common RUNLOG)" inputs/run_log.txt required
}

stage_panel_dir() {  # stage_panel_dir <dir relative to SCREEN>
  local d=$1 f
  for f in episode_trace.json flags.csv csv_sha256.txt trace_run.json; do
    [[ -f "$SCREEN/$d/$f" ]] && stage "$d/$f"
  done
  return 0
}

make_patch() {
  [[ -d "$CLONE/.git" ]] || { MISSING+=("scratch clone $CLONE (required for the patch)"); return 0; }
  local tip=${PIN:-}
  if [[ -z "$tip" ]]; then
    tip=$(git --no-optional-locks -C "$CLONE" rev-parse HEAD)
    echo "note: SCRATCH_COMMIT is not pinned; the dry-run patch uses the clone's HEAD $tip"
  fi
  local files
  files=$(git --no-optional-locks -C "$CLONE" diff --name-only "$PARENT..$tip" | "$PY" -I -B -c "import sys,json;print(json.dumps(sorted(l.strip() for l in sys.stdin if l.strip())))")
  local want
  want=$("$PY" -I -B -c "import sys,json;print(json.dumps(sorted(json.loads(sys.argv[1]))))" "$CHANGED")
  if [[ "$files" != "$want" ]]; then
    [[ $DRY -eq 1 ]] || die "scratch commit changes $files, not $want"
    echo "WARNING: $PARENT..$tip changes $files, not $want"
  fi
  mkdir -p "$STAGE/patches"
  git --no-optional-locks -C "$CLONE" format-patch --stdout "$PARENT..$tip" > "$STAGE/patches/tail3-upper-bound.patch"
  local size
  size=$(stat -c %s "$STAGE/patches/tail3-upper-bound.patch")
  (( size <= MAX_BYTES )) || die "the patch is $size bytes, over 5 MB"
  (( size > 0 )) || MISSING+=("patches/tail3-upper-bound.patch is empty ($PARENT..$tip has no commit)")
  printf '%s\n' "$PARENT..$tip" > "$STAGE/patches/RANGE.txt"
}

case "$MODE" in
  init)
    stage criteria/criteria_tailscreen_final.txt required
    stage criteria/criteria_tailscreen_final.sha256 required
    stage criteria/criteria_tailscreen_v2.txt
    stage criteria/BUILD_SPEC.md
    stage criteria.txt required
    stage APPROVED required
    stage APPROVAL_REQUEST.md
    stage PINS.txt
    stage calib/coef.json required
    stage calib/calib.log
    for f in "$SCREEN"/tools/*.py "$SCREEN"/tools/*.sh "$SCREEN"/tools/tests/*.py; do
      [[ -f "$f" ]] && stage "${f#"$SCREEN"/}"
    done
    mkdir -p "$STAGE/tools"
    (cd "$SCREEN" && sha256sum tools/*.py tools/*.sh tools/tests/*.py 2>/dev/null) > "$STAGE/tools/TOOLS_SHA256.txt"
    stage measure_copy/OVERLAY_SHA256.txt required
    stage measure_copy/TREE_DIGEST.txt required
    stage validate/v1.json required
    stage validate/v1_panels.csv
    stage_panel_dir validate/V1_s42
    stage_panel_dir validate/V1_s80
    for f in build_tier.txt build_tier.json gate_tier.txt gate_pass.json; do stage "tests_out/$f"; done
    make_patch
    stage_inputs
    ;;
  replicate)
    stage "runs/$NAME/run_meta.json" required
    stage "runs/$NAME/progress.csv" required
    stage "runs/$NAME/telemetry.npz" required
    stage "runs/$NAME/stage1_${R1}_steps.zip" required
    stage "runs/$NAME/stage1_vecnormalize_${R1}_steps.pkl" required
    stage "runs/$NAME/stage1_final.zip" required
    stage "runs/$NAME/stage1_final_vecnorm.pkl" required
    if [[ -d "$SCREEN/panels/$NAME" ]]; then
      for d in "$SCREEN/panels/$NAME"/*/; do
        d=${d%/}
        [[ "$(basename "$d")" == *_crashed* ]] && continue
        stage_panel_dir "${d#"$SCREEN"/}"
      done
    fi
    stage panels.csv
    stage panels_crashed.csv
    stage "logs/${NAME}_rerun_compare.json"
    stage_ext "$(common RUNLOG)" inputs/run_log.txt
    ;;
  panels)
    if [[ -d "$SCREEN/panels" ]]; then
      for d in "$SCREEN"/panels/*/*/; do
        d=${d%/}
        [[ "$(basename "$d")" == *_crashed* ]] && continue
        stage_panel_dir "${d#"$SCREEN"/}"
      done
    fi
    stage panels.csv required
    stage scores/score.json
    stage scores/score.txt
    stage LOST.txt
    ;;
esac

# The manifest over the whole stage (everything staged so far by any mode).
cat > "$STAGE/DURABLE.md" <<EOF
# Tail_3 bound screen: durable copy (orphan branch, never merged, never a PR)

Files are SCREEN-relative (SCREEN = $SCREEN). patches/tail3-upper-bound.patch is the scratch commit
($(cat "$STAGE/patches/RANGE.txt" 2>/dev/null || echo "range not staged")) as git format-patch output; apply it with
\`git am\` on $PARENT. Verify with \`sha256sum -c SHA256SUMS\`. Written by tools/copy_durable.sh.
EOF
(
  cd "$STAGE"
  find . -type f ! -name FILELIST.txt ! -name SHA256SUMS -printf '%P\n' | sort > FILELIST.txt
  xargs -d '\n' -a FILELIST.txt sha256sum > SHA256SUMS
)
while IFS= read -r f; do
  size=$(stat -c %s "$STAGE/$f")
  (( size <= MAX_BYTES )) || die "staged $f is $size bytes, over 5 MB"
done < "$STAGE/FILELIST.txt"

n=$(wc -l < "$STAGE/FILELIST.txt")
bytes=$(du -sb "$STAGE" | cut -f1)
echo "staged: $n files, $bytes bytes in $STAGE (mode $MODE${NAME:+ $NAME})"
if (( ${#MISSING[@]} )); then
  echo "not staged (missing now):"
  printf '  %s\n' "${MISSING[@]}"
fi

BR=${BRANCH:-<DURABLE_BRANCH unpinned>}
MSG="tailscreen durable copy: $MODE${NAME:+ $NAME} (never merged)"
GIT_PLAN=(
  "test -d $REPO/.git || { git init $REPO && git -C $REPO remote add origin $ORIGIN_URL && git -C $REPO remote set-url --push origin $ORIGIN_URL; }"
  "if git -C $REPO ls-remote --exit-code --heads origin $BR; then git -C $REPO fetch origin $BR && git -C $REPO checkout -B $BR FETCH_HEAD; else git -C $REPO checkout --orphan $BR; fi"
  "find $REPO -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} + && cp -a $STAGE/. $REPO/"
  "git -C $REPO add -A"
  "git -C $REPO commit -m '$MSG'"
  "git -C $REPO push origin $BR"
)

if [[ $DRY -eq 1 ]]; then
  echo "DRY RUN: nothing cloned, committed or pushed. Real mode would run:"
  printf '  %s\n' "${GIT_PLAN[@]}"
  exit 0
fi

# ------------------------------------------------------------------------------------------- real mode (after approval)
(( ${#MISSING[@]} == 0 )) || { for m in "${MISSING[@]}"; do [[ "$m" == *"(required"* ]] && die "missing $m"; done; }
case "$REPO" in "$CLONE"*|/home/user/mesozoic-labs*|*/fix/*) die "durable repo path $REPO is not separate" ;; esac
if [[ ! -d "$REPO/.git" ]]; then
  git init -q "$REPO"
  git -C "$REPO" remote add origin "$ORIGIN_URL"
fi
[[ "$(git -C "$REPO" remote get-url origin)" == "$ORIGIN_URL" ]] || die "durable repo origin is not $ORIGIN_URL"
if [[ "$(git -C "$REPO" branch --show-current 2>/dev/null)" != "$BRANCH" ]]; then
  if git -C "$REPO" ls-remote --exit-code --heads origin "$BRANCH" > /dev/null 2>&1; then
    git -C "$REPO" fetch -q origin "$BRANCH"
    git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD
  else
    git -C "$REPO" checkout -q --orphan "$BRANCH"
  fi
fi
find "$REPO" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +   # the stage is cumulative: mirror it
cp -a "$STAGE/." "$REPO/"
git -C "$REPO" add -A
if git -C "$REPO" diff --cached --quiet; then
  echo "nothing new to commit"
else
  git -C "$REPO" commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X1VjQXgfkABgkidWb1QS9A"
fi
git -C "$REPO" push origin "$BRANCH"
