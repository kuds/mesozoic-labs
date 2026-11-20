"""Standalone 50 cm T-Rex concept; not a registered training plant."""

from pathlib import Path

# The hardware study has a historical directory name, separate from runtime
# model identities. Keep its location owned by the standalone example.
PARTS_BUDGET_PATH = Path(__file__).resolve().parents[2] / "docs/hardware/studies/trex_50cm/parts_budget.json"
