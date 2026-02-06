import sys
from pathlib import Path

# Add repo root so package imports like `from environments.…` work
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
