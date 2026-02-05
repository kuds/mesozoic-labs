import sys
from pathlib import Path

# Add project root so `from envs.raptor_env import RaptorEnv` works
sys.path.insert(0, str(Path(__file__).parent.parent))
