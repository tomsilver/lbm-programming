"""Simple 2D grid environment with pressable buttons."""

from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeVar

import gymnasium
import numpy as np

from lbm_programming.utils import render_avatar_grid

RenderFrame = TypeVar("RenderFrame")


@dataclass(frozen=True)
class ButtonState:
    """The state in a button environment."""

    robot_position: tuple[int, int]  # row, col
    button_positions: dict[str, tuple[int, int]]  # name to button positions
    button_statuses: dict[str, bool]  # name of button to status
    target_button: str  # name of the button that we want to press
    human_response: float | None  # used for querying the human
    world_dims: tuple[int, int]  # height and width

    def __post_init__(self) -> None:
        # Make sure positions are in bounds.
        for position in [self.robot_position] + sorted(self.button_positions.values()):
            assert 0 <= position[0] < self.world_dims[0]
            assert 0 <= position[1] < self.world_dims[1]
        # Make sure no two buttons are at the same position.
        assert len(self.button_positions) == len(set(self.button_positions.values()))
        # Make sure all buttons are determined.
        assert set(self.button_positions) == set(self.button_statuses)
        assert self.target_button in self.button_positions


class ButtonAction:
    """An action in the button environment."""


class ButtonEnv(gymnasium.Env[ButtonState, ButtonAction]):
    """Simple 2D grid environment with pressable buttons."""

    render_mode: str = "rgb_array"

    def __init__(self) -> None:
        self._current_state: ButtonState | None = None

    def step(
        self, action: ButtonAction
    ) -> tuple[ButtonState, SupportsFloat, bool, bool, dict[str, Any]]:
        import ipdb

        ipdb.set_trace()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ButtonState, dict[str, Any]]:

        # For now, use a constant initial state.
        self._current_state = ButtonState(
            robot_position=(0, 0),
            button_positions={"button1": (3, 4), "button2": (5, 7)},
            button_statuses={"button1": False, "button2": False},
            target_button="button2",
            human_response=None,
            world_dims=(10, 10),
        )

        # Return the current state.
        return self._get_obs(), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        # Create an avatar grid.
        assert self._current_state is not None
        avatar_grid = np.full(self._current_state.world_dims, None, dtype=object)
        # Add the robot.
        avatar_grid[self._current_state.robot_position] = "robot"
        # Add the buttons.
        for button_pos in self._current_state.button_positions.values():
            avatar_grid[button_pos] = "button"
        # Finish the image.
        return render_avatar_grid(avatar_grid)   # type: ignore

    def _get_obs(self) -> ButtonState:
        assert self._current_state is not None
        return self._current_state
