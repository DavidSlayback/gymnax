import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ....environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from functools import partial
from ..multistory_fourrooms import (
    MultistoryFourRooms,
    EnvState,
    upstairs,
    downstairs,
    numbered_four_rooms_map,
    rooms_map_to_multistory,
    coords_to_flat,
    flat_to_coords,
    SW,
    NE,
)


# @struct.dataclass
# class EnvState:
#     pos: chex.Array
#     goal: chex.Array
#     time: int


@struct.dataclass
class EnvParams:
    fail_prob: float = 1.0 / 3
    option: int = 0  # which policy are we rewarding?
    max_steps_in_episode: int = 500


def get_clockwise_and_counterclockwise_hallway(room_map: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Return a [n_halls, 2] set of hallway coordinates and a [2, y, x] set of indices into it

    Args:
        room_map: Single floor room map
    """
    n_rooms = n_hallways = len(np.unique(room_map[room_map >= 0])) # how many rooms
    hallways = np.zeros((n_rooms, 2), dtype=np.int32)  # yx coordinates of hallways
    hall_idx = np.full((2, *room_map.shape), -1, dtype=int)  # idx into "hallways" for each valid square
    for row in range(1, room_map.shape[0] - 1):
        for col in range(1, room_map.shape[1] - 1):
            if room_map[row, col] == -1: continue
            is_hallway = ((room_map[row - 1, col] == -1 and room_map[row + 1, col] == -1) or
                          (room_map[row, col - 1] == -1 and room_map[row, col + 1] == -1))
            c = np.array([row, col], dtype=np.int32)
            r = room_map[row, col]
            if is_hallway:  # Hallway goes to next hallway in either direction
                hallways[r] = c
                hall_idx[0, row, col] = (r + 1) % n_hallways
            else:  # Clockwise hallway is same "room"
                hall_idx[0, row, col] = r
            hall_idx[1, row, col] = (r - 1) % n_hallways
    return jnp.array(hallways), jnp.array(hall_idx)




class MultistoryFourRoomsPretrainer(MultistoryFourRooms):
    def __init__(
        self,
        num_floors: int = 3,
        obs_type: str = "vector_mdp",
        **kwargs,
    ):
        """Multistory FourRooms environment with agent position based goal

        Goals can be:
            0: Nearest hallway clockwise
            1: Nearest hallway counterclockwise
            2: NE (upstairs)
            3: SW (downstairs)

        Args:
            num_floors: number of stories
            obs_type: one of 'discrete', 'vector_mdp', 'visual', 'adjacent', 'grid', 'room'
        """
        super().__init__(num_floors, obs_type)
        self.hallways, self.goal_idxes = get_clockwise_and_counterclockwise_hallway(self.rooms_map[0])

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location

        # Reset agent first, goal depends on agent location. Agent can spawn in any empty location
        pos = reset_pos(key, self.available_agent_spawns)
        # Goal location is deterministic based on agent position and executing option
        goal = jnp.full_like(pos, pos[0])
        goal = goal.at[1:].set(self.hallways[self.goal_idxes[params.option, pos[1], pos[2]]])
        state = EnvState(pos, goal, 0)
        return self.get_obs(state), state

    @property
    def name(self) -> str:
        """Environment name."""
        return "MultistoryFourRooms-pretrainer"


def reset_pos(
    rng: chex.PRNGKey,
    available_pos: chex.Array,
) -> chex.Array:
    """Reset the position of the agent."""
    return jax.random.choice(rng, available_pos)
