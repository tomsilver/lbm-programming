"""Tests for button_env.py."""

from lbm_programming.envs.button_env import ButtonEnv, ButtonState, MoveButtonAction


def test_button_env():
    """Tests for ButtonEnv()."""
    env = ButtonEnv()
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, ButtonState)

    # Test moving.
    action = MoveButtonAction(1, 0)
    next_obs, _, _, _, _ = env.step(action)
    assert obs.robot_position[0] + 1 == next_obs.robot_position[0]
    assert obs.robot_position[1] == next_obs.robot_position[1]

    # Uncomment to save the image.
    # img = env.render()
    # import imageio.v2 as iio
    # iio.imsave("button_test_image.png", img)
