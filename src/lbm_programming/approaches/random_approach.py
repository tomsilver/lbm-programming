"""An approach that takes random actions."""

from typing import TypeVar

from gymnasium.spaces import Space
from prpl_utils.gym_agent import Agent

Obs = TypeVar("Obs")
Act = TypeVar("Act")


class RandomActionsApproach(Agent[Obs, Act]):
    """An approach that takes random actions."""

    def __init__(self, action_space: Space[Act], seed: int) -> None:
        self._action_space = action_space
        self._action_space.seed(seed)
        super().__init__(seed)

    def _get_action(self) -> Act:
        return self._action_space.sample()
