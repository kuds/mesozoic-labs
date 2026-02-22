#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Install Python package with dev dependencies (linters + tests)
pip install -e ".[dev]"

# Install website dependencies
cd "$CLAUDE_PROJECT_DIR/website"
npm install
