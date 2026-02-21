import sys
from pathlib import Path

# Add repo root so package imports like `from environments.…` work
# even when running pytest without `pip install -e .`
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
