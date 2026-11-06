#!/bin/bash
# fix lane (copied from the e2e lane): the final state before pinning (read-only checks), written to logs/final_state.txt.
SP=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad
W=$SP/work
T=$W/pilot4/tools
LANE=$T/out/fix
F=$LANE/logs/final_state.txt
export GIT_OPTIONAL_LOCKS=0
{
  echo "date: $(date -u +%FT%TZ)"
  echo "tools/tree HEAD: $(git -C "$T/tree" rev-parse HEAD)"
  echo "tools/tree status entries (with ignored): $(git -C "$T/tree" status --porcelain --ignored | wc -l)"
  echo "tools/tree branch containing HEAD (local): $(git -C "$T/tree" branch --contains HEAD --format='%(refname:short)' 2>&1 | tr '\n' ' ')"
  echo "files under pilot 3's tools, run copies or judging newer than the start of the test run: $(find "$W/pilot3/tools" "$W/pilot3/runs/final" "$W/pilot3/judging" -newer "$LANE/tmp/marker_tests_final" | wc -l)"
  echo "files under the other pilot-4 lanes' folders newer than the start of the test run: $(find "$T/out" -path "$LANE" -prune -o -newer "$LANE/tmp/marker_tests_final" -type f -print | wc -l)"
  echo "__pycache__/.pytest_cache/.pyc under the pilot-4 tools folder outside out/: $(find "$T" -path "$T/out" -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name '*.pyc' \) -print | wc -l)"
  echo "__pycache__/.pyc anywhere under my lane: $(find "$LANE" \( -name __pycache__ -o -name '*.pyc' \) | wc -l)"
  echo "my clone removed: $(test -e "$LANE/tmp/clone" && echo no || echo yes)"
  echo "pinned tool/test/fixture/src files: sha256 now against tests_final/sha256_before.txt:"
  (cd "$T" && xargs -d '\n' sha256sum < "$LANE/tests_final/pinned_files.txt") | cmp - "$LANE/tests_final/sha256_before.txt" && echo "  equal (no tool or test changed since the test run)"
  echo "disk: $(df -h "$LANE" | tail -1)"
  echo "lane size: $(du -sh "$LANE" | cut -f1)"
} > "$F" 2>&1
cat "$F"
