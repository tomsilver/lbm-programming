"""Simple 2D grid environment with pressable buttons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeVar

import gymnasium
import numpy as np
from prpl_utils.spaces import FunctionalSpace

from lbm_programming.utils import render_avatar_grid

RenderFrame = TypeVar("RenderFrame")


@dataclass(frozen=True)
class ButtonState:
    """The state in a button environment."""

    robot_position: tuple[int, int]  # row, col
    button_positions: dict[str, tuple[int, int]]  # name to button positions
    button_statuses: dict[str, bool]  # name of button to status
    button_values: dict[str, float]  # value of button to succeed in pressing
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

    def copy_with(self, robot_position: tuple[int, int]) -> ButtonState:
        """Create a new ButtonState with the input values changed."""
        return ButtonState(
            robot_position=robot_position,
            button_positions=self.button_positions,
            button_statuses=self.button_statuses,
            button_values=self.button_values,
            target_button=self.target_button,
            human_response=self.human_response,
            world_dims=self.world_dims,
        )


class ButtonAction:
    """An action in the button environment."""


@dataclass(frozen=True)
class PressButtonAction(ButtonAction):
    """Press the button in the button environment."""

    val: float

    def __post_init__(self) -> None:
        assert -100 <= self.val <= 100


@dataclass(frozen=True)
class QueryHumanButtonAction(ButtonAction):
    """Press the button in the button environment."""

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class MoveButtonAction(ButtonAction):
    """Move the robot by some limited amount."""

    dr: int  # should be -1, 0, 1
    dc: int  # should be -1, 0, 1

    def __post_init__(self) -> None:
        assert -1 <= self.dr <= 1
        assert -1 <= self.dc <= 1


class ButtonEnv(gymnasium.Env[ButtonState, ButtonAction]):
    """Simple 2D grid environment with pressable buttons."""

    render_mode: str = "rgb_array"

    def __init__(self) -> None:
        self._current_state: ButtonState | None = None

        # Define the observation and action spaces.
        self.observation_space = FunctionalSpace(
            contains_fn=lambda s: isinstance(s, ButtonState),
        )
        self.action_space = FunctionalSpace(
            contains_fn=lambda a: isinstance(a, ButtonAction),
            sample_fn=self._sample_action,
        )

    def step(
        self, action: ButtonAction
    ) -> tuple[ButtonState, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._current_state is not None, "Must call reset() first"
        # Handling moving the robot.
        if isinstance(action, MoveButtonAction):
            new_robot_position = (
                self._current_state.robot_position[0] + action.dr,
                self._current_state.robot_position[1] + action.dc,
            )
            # If the new robot position is out of bounds, stay in place.
            if not (
                0 <= new_robot_position[0] < self._current_state.world_dims[0]
                and 0 <= new_robot_position[1] < self._current_state.world_dims[1]
            ):
                new_robot_position = self._current_state.robot_position
            self._current_state = self._current_state.copy_with(
                robot_position=new_robot_position
            )
        # press that target button
        elif isinstance(action, PressButtonAction):
            true_val = self._current_state.button_values[
                self._current_state.target_button
            ]
            if action.val == true_val:
                self._current_state.button_statuses[
                    self._current_state.target_button
                ] = True

        # Ask human to give correct answer
        elif isinstance(action, QueryHumanButtonAction):
            self._current_state.button_statuses[self._current_state.target_button] = (
                True
            )

        else:
            raise NotImplementedError

        # Coming later.
        # terminate if all button pressed
        terminated = all(self._current_state.button_statuses.values())

        truncated = False

        # -1 if pressed, -100 if query human, 0 if finished
        reward = 0.0
        if not terminated:
            reward = reward - 1
            if isinstance(action, QueryHumanButtonAction):
                reward = reward - 99

        return self._get_obs(), reward, terminated, truncated, {}

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
            button_values={"button1": 1.0, "button2": 2.0},
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
        return render_avatar_grid(avatar_grid)  # type: ignore

    def _get_obs(self) -> ButtonState:
        assert self._current_state is not None
        return self._current_state

    def _sample_action(self, rng: np.random.Generator) -> ButtonAction:
        # Sample among the possible actions. For now, we only have moves.
        pick_which_action = int(rng.integers(1, 4))
        if pick_which_action == 1:
            dr = int(rng.integers(-1, 2))
            dc = int(rng.integers(-1, 2))
            return MoveButtonAction(dr, dc)
        if pick_which_action == 2:
            float_val = float(rng.uniform(-100, 100))
            return PressButtonAction(float_val)
        if pick_which_action == 3:
            return QueryHumanButtonAction()
        return ButtonAction()
