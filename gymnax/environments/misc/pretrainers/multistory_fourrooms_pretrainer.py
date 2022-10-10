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


def get_clockwise_and_counterclockwise_hallway(
    room_map: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Return a [n_halls, 2] set of hallway coordinates and a [2, y, x] set of indices into it

    Args:
        room_map: Single floor room map
    """
    n_rooms = n_hallways = len(np.unique(room_map[room_map >= 0]))  # how many rooms
    hallways = np.zeros((n_rooms, 2), dtype=np.int32)  # yx coordinates of hallways
    hall_idx = np.full(
        (2, *room_map.shape), -1, dtype=int
    )  # idx into "hallways" for each valid square
    for row in range(1, room_map.shape[0] - 1):
        for col in range(1, room_map.shape[1] - 1):
            if room_map[row, col] == -1:
                continue
            is_hallway = (
                room_map[row - 1, col] == -1 and room_map[row + 1, col] == -1
            ) or (room_map[row, col - 1] == -1 and room_map[row, col + 1] == -1)
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
        self.hallways, self.goal_idxes = get_clockwise_and_counterclockwise_hallway(
            self.rooms_map[0]
        )

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
        goal = goal.at[1:].set(
            self.hallways[self.goal_idxes[params.option, pos[1], pos[2]]]
        )
        state = EnvState(pos, goal, 0)
        return self.get_obs(state), state

    @property
    def name(self) -> str:
        """Environment name."""
        return "MultistoryFourRooms-pretrainer"

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state info. Modified to not return goal"""
        if self.obs_fn == "vector_mdp":  # zyx positions of agent and goal
            return jnp.array([*state.pos, *state.goal])
        elif (
            self.obs_fn == "discrete_mdp"
        ):  # classic discrete observation, no goal (must assume fixed)
            return self.coord_to_state_map[state.pos[0], state.pos[1], state.pos[2]]
        elif (
            self.obs_fn == "adjacent"
        ):  # wall/empty/stairdown/stairup for adjacent squares
            adj = state.pos + self.directions  # [directions, 3]
            o = self.env_map[tuple(adj.T)]  # [directions,]
            return o
        elif (
            self.obs_fn == "visual"
        ):  # Visual obs. Note inversion of y and x coordinates
            agent_map = jnp.zeros(self.occupied_map.shape)
            agent_map = agent_map.at[state.pos[0], state.pos[2], state.pos[1]].set(1)
            obs_array = jnp.stack([self.occupied_map, agent_map], axis=-1)
            return obs_array
        elif (
            self.obs_fn == "room"
        ):  # room position of agent (clockwise hallway belongs to room), no goal (must assume fixed)
            return self.rooms_map[state.pos[0], state.pos[1], state.pos[2]]
        else:  # 3x3 grid centered on agent
            adj = state.pos + jnp.mgrid[:1, -1:2, -1:2]  # [3, 1, 3, 3]
            return jnp.squeeze(self.env_map[tuple(adj)])  # [3, 3]

    def observation_space(self, params: EnvParams) -> spaces.Space:
        """Observation space of the environment."""
        if self.obs_fn == "vector_mdp":
            return spaces.Box(
                jnp.min(self.coords), jnp.max(self.coords), (6,), jnp.int32
            )
        elif self.obs_fn == "discrete_mdp":
            spaces.Discrete(self.coord_to_state_map.max())
        elif self.obs_fn == "adjacent":
            return spaces.Box(0, 5, (4,), jnp.int32)
        elif self.obs_fn == "visual":
            return spaces.Box(0, 1, (13, 13, 2), jnp.float32)
        elif self.obs_fn == "room":
            return spaces.Discrete(self.rooms_map.max())
        else:
            return spaces.Box(0, 5, (3, 3), jnp.int32)


def reset_pos(
    rng: chex.PRNGKey,
    available_pos: chex.Array,
) -> chex.Array:
    """Reset the position of the agent."""
    return jax.random.choice(rng, available_pos)