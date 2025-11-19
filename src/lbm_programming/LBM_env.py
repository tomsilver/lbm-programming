"""Implementes LBMenv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lbm_programming.envs.button_env import ButtonAction, ButtonEnv, ButtonState

# from prpl_utils.spaces import FunctionalSpace


@dataclass(frozen=True)
class LBMState:
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

    def copy_with(self, robot_position: tuple[int, int]) -> LBMState:
        """Create a new ButtonState with the input values changed."""
        return LBMState(
            robot_position=robot_position,
            button_positions=self.button_positions,
            button_statuses=self.button_statuses,
            target_button=self.target_button,
            human_response=self.human_response,
            world_dims=self.world_dims,
        )


class LBMPrimitive:
    """A primitive LBM."""


@dataclass(frozen=True)
class QueryLBM(LBMPrimitive):
    """Press the button in the button environment."""


@dataclass(frozen=True)
class QueryHuman(LBMPrimitive):
    """Press the button in the button environment."""


@dataclass(frozen=True)
class GetMotionPlan(LBMPrimitive):
    """Press the button in the button environment."""

    dr: int  # should be -1, 0, 1
    dc: int  # should be -1, 0, 1


# def QueryLBM(value)


class LBMEnv:
    """Enviroment for running the LBM."""

    robot_position: tuple[int, int]  # row, col
    button_positions: dict[str, tuple[int, int]]  # name to button positions
    button_statuses: dict[str, bool]  # name of button to status
    target_button: str  # name of the button that we want to press
    world_dims: tuple[int, int]  # height and width
    good_lbm_button_set: set[
        str
    ]  # are you a good button? needs preprocessing to function

    def __init__(self, buttonenv) -> None:
        self.buttonenv: ButtonEnv | None = buttonenv

    def get_random_oracle(self) -> list[ButtonAction] | Any:
        """Gets an action using the random method described in docs."""
        assert self.buttonenv is not None
        button_status_filter = {k: v for k, v in self.button_statuses.items() if v}
        val = int(np.random.randint(0, len(button_status_filter)))
        keys_list = list(button_status_filter.items())
        button_to_traverse = keys_list[val][0]
        position_to_traverse = self.button_positions[button_to_traverse]
        rand = np.random.randint(0, 3)
        if rand == 0:
            return self.buttonenv.get_motion_plan(
                position_to_traverse[0], position_to_traverse[1]
            )
        if rand == 1:
            return self.buttonenv.query_lbm()
        if rand == 2:
            return self.buttonenv.query_human()
        return []

    def get_action_oracle(self) -> list[ButtonAction]:
        """Gets an oracle using the random method described in docs."""
        assert self.buttonenv is not None
        if self.button_statuses[self.target_button] or self.target_button == "":
            button_status_filter = {k: v for k, v in self.button_statuses.items() if v}
            action: list[ButtonAction] = []
            action_flag = False
            for k, _ in button_status_filter.items():
                temp_action = self.buttonenv.get_motion_plan(
                    self.button_positions[k][0], self.button_positions[k][1]
                )
                if len(temp_action) < len(action) or not action_flag:
                    action = temp_action  # type: ignore
                    action_flag = True
            return action
        # If standing on button, check if this is a button that the LBM knows about.
        if self.target_button in self.good_lbm_button_set:
            action = [self.buttonenv.query_lbm()]
        else:
            action = [self.buttonenv.query_human()]
        return action

    def execute_action(self, action):
        """Executes the action."""
        assert self.buttonenv is not None
        obs, reward, terminated, _, _ = self.buttonenv.step(action)
        self.copy_state(obs)
        return reward, terminated

    def copy_state(self, current_state: ButtonState):
        """Copy the button state to the LMB."""
        self.robot_position = current_state.robot_position
        self.button_statuses = current_state.button_statuses
        self.button_positions = current_state.button_positions
        self.target_button = current_state.target_button
        self.world_dims = current_state.world_dims

    # TBD, make a good button set to enjoy
    def synthesize_good_button_set(self):
        """Create the good button set, how to do TBD."""
        return []

    def run_state(self, action_func):
        """Run the button state to the LMB."""
        terminate = True
        total_reward = 0
        while terminate:
            action_list = action_func()
            for action in action_list:
                reward, terminated = self.execute_action(action)
                total_reward += reward
                if terminated:
                    terminate = False
        return total_reward
