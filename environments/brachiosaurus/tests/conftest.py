import sys
from pathlib import Path

# Add project root so `from envs.brachio_env import BrachioEnv` works
sys.path.insert(0, str(Path(__file__).parent.parent))
