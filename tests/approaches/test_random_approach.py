"""Tests for random_approach.py."""

from lbm_programming.approaches.random_approach import RandomActionsApproach
from lbm_programming.envs.button_env import ButtonAction, ButtonEnv


def test_random_approach():
    """Tests RandomApproach() in the button environment."""

    # Create the environment.
    env = ButtonEnv()

    # Create the approach.
    approach = RandomActionsApproach(env.action_space, seed=123)

    # Make sure the approach runs.
    obs, info = env.reset(seed=123)
    approach.reset(obs, info)

    for _ in range(10):
        action = approach.step()
        assert isinstance(action, ButtonAction)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        approach.update(obs, reward, done, info)
