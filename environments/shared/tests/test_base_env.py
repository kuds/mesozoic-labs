import numpy as np

from environments.velociraptor.envs.raptor_env import RaptorEnv


def test_base_env_lifecycle():
    env = RaptorEnv(render_mode="rgb_array")

    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    # Test step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)

    # Test render
    frame = env.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)

    # Test close
    env.close()


def test_base_env_scale_action():
    env = RaptorEnv()

    # Test action scaling
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    scaled = env._scale_action(action)
    assert scaled.shape == action.shape

    env.close()
