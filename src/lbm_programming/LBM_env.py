from src.lbm_programming.envs.button_env import ButtonEnv, ButtonAction, ButtonState
import gymnasium
import numpy as np
from prpl_utils.spaces import FunctionalSpace

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

#def QueryLBM(value)

class LBMEnv:

    robot_position: tuple[int, int]  # row, col
    button_positions: dict[str, tuple[int, int]]  # name to button positions
    button_statuses: dict[str, bool]  # name of button to status
    target_button: str  # name of the button that we want to press
    world_dims: tuple[int, int]  # height and width
    good_lbm_button_set: set[int] # are you a good button? needs preprocessing to function

    def __init__(self, buttonenv) -> None:
        self.buttonenv : ButtonEnv | None = None

    def get_random_oracle(self):
        button_status_filter = {k: v for k, v in self.button_statuses.iteritems() if v == True}
        val = np.random.randint(0, len(button_status_filter))
        keys_list = list(button_status_filter)
        button_to_traverse = keys_list[val]
        position_to_traverse = self.button_positions[button_to_traverse]
        rand = np.random.randint(0, 3)
        if rand == 0:
            return self.buttonenv.get_motion_plan(position_to_traverse)
        if rand == 1:
            return self.buttonenv.query_lbm()
        if rand == 2:
            return self.buttonenv.query_human()


    def get_action_oracle(self) -> ButtonAction:
        # If not yet standing on button, move to the closest button.
        if self.buttonenv.button_status[self.target_button] or self.target_button == "":
            button_status_filter = {k: v for k, v in self.button_statuses.iteritems() if v == True}
            action = []
            action_flag = False
            for k, v in button_status_filter.iteritems():
                temp_action = self.buttonenv.get_motion_plan(self.button_positions[k])
                if len(temp_action) < len(action) or not action_flag:
                    action = temp_action
                    action_flag = True
            return action
        # If standing on button, check if this is a button that the LBM knows about.
        if self.target_button in self.good_lbm_button_set:
            action = self.buttonenv.query_lbm()
        else:
            action = self.buttonenv.query_human()
        return action

    def copy_state(self, current_state : ButtonState):
        self.robot_position = current_state.robot_position
        self.button_statuses = current_state.button_statuses
        self.robot_position = current_state.robot_position
        self.robot_position = current_state.robot_position
        self.robot_position = current_state.robot_position

    def step(
        self, action: LBMPrimitive
    ) -> tuple[LBMPrimitive, SupportsFloat, bool, bool, dict[str, Any]]:
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
            if action.val >= true_val - 0.1 and action.val <= true_val + 0.1:
                self._current_state.button_statuses[
                    self._current_state.target_button
                ] = True

        # press that target button
        elif isinstance(action, QueryLBMAction):
            true_val = self._current_state.button_values[
                self._current_state.target_button
            ]
            if self._current_state.button_good_bad[self._current_state.target_button]:
                val = np.random.normal(action.val, 0.1)
            else:
                val = np.random.normal(action.val, 0.6)
            if action.val >= true_val - 0.1 and action.val <= true_val + 0.1:
                self._current_state.button_statuses[
                    self._current_state.target_button
                ] = True

        elif isinstance(action, GetMotionPlanAction):
            raise NotImplementedError


        # Ask human to give correct answer
        elif isinstance(action, QueryHumanButtonAction):
            #lol maybe make better sometimes
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
    ) -> tuple[LBMState, dict[str, Any]]:

        # For now, use a constant initial state.
        self._current_state = ButtonState(
            robot_position=(0, 0),
            button_positions={"button1": (3, 4), "button2": (5, 7)},
            button_statuses={"button1": False, "button2": False},
            button_values={"button1": 1.0, "button2": 2.0},
            button_good_bad={"button1": True, "button2": False},
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
