import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from functools import partial


@struct.dataclass
class EnvState:
    pos: chex.Array
    goal: chex.Array
    time: int


@struct.dataclass
class EnvParams:
    fail_prob: float = 1.0 / 3
    resample_init_pos: bool = False
    resample_goal_pos: bool = False
    max_steps_in_episode: int = 500


four_rooms_map = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""

upstairs = NE = jnp.array([1, 11])
downstairs = SW = jnp.array([11, 1])


def string_to_bool_map(str_map: str) -> chex.Array:
    """Convert string map into boolean walking map."""
    bool_map = []
    for row in str_map.split("\n")[1:]:
        bool_map.append([r == " " for r in row])
    return jnp.array(bool_map)


def bool_map_to_multistory(map: chex.Array, num_floors: int = 1) -> chex.Array:
    """Convert boolean walking map into multistory layout"""
    int_map = map.astype(int)  # 0 wall, 1 empty, 2 stairs down, 3 stairs up
    ms = jnp.stack([int_map for _ in range(num_floors)], 0)  # z, y, x
    if num_floors > 1:
        ms = ms.at[1:, downstairs].set(2)
        ms = ms.at[:-1, upstairs].set(3)
    return ms


def coords_to_flat(coords: chex.Array, dims: chex.Array) -> int:
    """Convert (z), y, x coordinates to flat"""
    return jnp.ravel_multi_index(coords, dims, mode='clip', order='C')


def flat_to_coords(flat: int, dims: chex.Array) -> chex.Array:
    """Convert flat coordinates to (z), y, x"""
    return jnp.array(jnp.unravel_index(flat, dims))


class MultistoryFourRooms(environment.Environment):
    def __init__(
        self,
        num_floors: int = 3,
        obs_type: str = 'vector_mdp',
        goal_fixed: chex.Array = jnp.array([8, 9]),
        pos_fixed: chex.Array = jnp.array([4, 1]),
    ):
        """Multistory FourRooms environment

        Args:
            num_floors: number of stories
            obs_type: one of 'discrete', 'vector_mdp', 'visual', 'adjacent', 'grid'
            goal_fixed: y,x position (always top floor)
            pos_fixed: y,x position of agent (always bottom floor)
        """
        assert num_floors >= 1, "Must have at least 1 floor!"
        assert obs_type in ['vector_mdp', 'adjacent', 'grid', 'visual', '']
        super().__init__()
        self.env_map = bool_map_to_multistory(string_to_bool_map(four_rooms_map), num_floors)
        self.valid_map = (self.env_map == 0)  # Agent can go anywhere without a wall
        self.spawn_map = (self.env_map == 1)  # Agent cannot spawn on stairs
        self.occupied_map = (self.env_map == 0)
        self.coord_to_state_map = jnp.cumsum(self.occupied_map).reshape(self.occupied_map.shape)
        self.coords = jnp.stack(jnp.nonzero(self.spawn_map), axis=-1)
        self.directions = jnp.array([[0, -1, 0], [0, 0, 1], [0, 1, 0], [0, 0, -1]])
        self.upstairs = jnp.array([1, *(SW - NE)])  # Move from NE on floor 1 to SW on floor 2
        self.downstairs = jnp.array([-1, *(NE - SW)])  # Move from SW on floor 1 to NE on floor 0

        # Any open space in the map can be a goal for the agent
        self.available_goals = self.coords[self.coords[:, 0] == (num_floors - 1)]
        self.available_agent_spawns = self.coords[self.coords[:, 0] == 0]

        # Set fixed goal and position if we don't resample each time
        self.goal_fixed = jnp.array([num_floors - 1, *goal_fixed])
        self.pos_fixed = jnp.array([0, *pos_fixed])

        # Observation function
        self.obs_fn = obs_type

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        key_random, key_action = jax.random.split(key)
        # Sample whether to choose a random action
        choose_random = (
            jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
        )
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )

        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1], p[2]] > 0
        new_pos = jax.lax.select(in_map, p, state.pos)
        # Process stairs
        moved = jnp.any(new_pos != p, -1)  # Only use stairs if we had to move to this square
        map_val = self.env_map[new_pos[0], new_pos[1], new_pos[2]]
        go_down = (map_val == 2) & moved
        go_up = (map_val == 3) & moved
        new_pos = jnp.where(go_down, new_pos + self.downstairs, new_pos)
        new_pos = jnp.where(go_up, new_pos + self.upstairs, new_pos)
        # new_pos = new_pos.at[go_down].add(self.downstairs)
        # new_pos = new_pos.at[go_up].add(self.upstairs)
        reward = jnp.all(new_pos == state.goal, axis=-1).astype(float)

        # Update state dict and evaluate termination conditions
        state = EnvState(new_pos, state.goal, state.time + 1)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location
        rng_goal, rng_pos = jax.random.split(key, 2)
        # Only use resampled position if specified in EnvParams
        goal = jax.lax.select(
            params.resample_goal_pos, reset_goal(rng_goal, self.available_goals), self.goal_fixed
        )

        pos = jax.lax.select(params.resample_init_pos, reset_pos(rng_pos, self.available_agent_spawns, goal), self.pos_fixed)
        state = EnvState(pos, goal, 0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state info."""
        if self.obs_fn == 'vector_mdp':  # zyx positions of agent and goal
            return jnp.array([*state.pos, *state.goal])
        elif self.obs_fn == 'discrete_mdp':  # classic discrete observation, no goal (must assume fixed)
            return self.coord_to_state_map[state.pos[0], state.pos[1], state.pos[2]]
        elif self.obs_fn == 'adjacent':  # wall/empty/stairdown/stairup for adjacent squares
            adj = state.pos + self.directions  # [directions, 3]
            o = self.env_map[tuple(adj.T)]  # [directions,]
            return o.at[jnp.all(adj == state.goal, -1)].set(4)  # Fill in goal
        elif self.obs_fn == 'visual':  # Visual obs. Note inversion of y and x coordinates
            agent_map = jnp.zeros(self.occupied_map.shape)
            agent_map = agent_map.at[state.pos[0], state.pos[2], state.pos[1]].set(1)
            obs_array = jnp.stack([self.occupied_map, agent_map], axis=-1)
            return obs_array
        else:  # 3x3 grid centered on agent
            adj = state.pos + jnp.mgrid[:1, -1:2, -1:2]  # [3, 1, 3, 3]
            env_map_with_goal = self.env_map.at[state.goal].set(4)  # TODO: Inefficient
            return jnp.squeeze(env_map_with_goal[tuple(adj)])  # [3, 3]


    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        # Check if agent has found the goal
        done_goal = jnp.all(state.pos == state.goal)
        done = jnp.logical_or(done_goal, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "MultistoryFourRooms-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if self.obs_fn == 'vector_mdp': return spaces.Box(jnp.min(self.coords), jnp.max(self.coords), (6,), jnp.int32)
        elif self.obs_fn == 'discrete_mdp': spaces.Discrete(self.coord_to_state_map.max())
        elif self.obs_fn == 'adjacent': return spaces.Box(0, 5, (4,), jnp.int32)
        elif self.obs_fn == 'visual': return spaces.Box(0, 1, (13, 13, 2), jnp.float32)
        else: return spaces.Box(0, 5, (3,3), jnp.int32)


    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(
                    jnp.min(self.coords),
                    jnp.max(self.coords),
                    (3,),
                    jnp.int32,
                ),
                "goal": spaces.Box(
                    jnp.min(self.coords),
                    jnp.max(self.coords),
                    (3,),
                    jnp.int32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def render(self, state: EnvState, params: EnvParams):
        """Small utility for plotting the agent's state."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(self.occupied_map, cmap="Greys")
        ax.annotate(
            "A",
            fontsize=20,
            xy=(state.pos[2], state.pos[1]),
            xycoords="data",
            xytext=(state.pos[2] - 0.3, state.pos[1] + 0.25),
        )
        ax.annotate(
            "G",
            fontsize=20,
            xy=(state.goal[2], state.goal[1]),
            xycoords="data",
            xytext=(state.goal[2] - 0.3, state.goal[1] + 0.25),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


def reset_goal(
    rng: chex.PRNGKey, available_goals: chex.Array
) -> chex.Array:
    """Reset the goal state/position in the environment."""
    goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(
    rng: chex.PRNGKey, coords: chex.Array, goal: chex.Array
) -> chex.Array:
    """Reset the position of the agent."""
    pos_index = jax.random.randint(rng, (), 0, coords.shape[0] - 1)
    collision = jnp.all(coords[pos_index] == goal)
    pos_index = jax.lax.select(collision, coords.shape[0] - 1, pos_index)
    return coords[pos_index][:]
