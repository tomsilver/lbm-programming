"""Tests for button_env.py."""

from lbm_programming.envs.button_env import ButtonAction, ButtonEnv, ButtonState


def test_button_env():
    """Tests for ButtonEnv()."""
    env = ButtonEnv()
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, ButtonState)
    img = env.render()
    # import ipdb; ipdb.set_trace()

    # Uncomment to save the image.
    # import imageio.v2 as iio
    # iio.imsave("button_test_image.png", img)

    # TODO: finish this test
