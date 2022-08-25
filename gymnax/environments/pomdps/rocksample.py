# Parameterizable rocksample environment

import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

"""
States: Position of robot, {bad, good} status of rocks
Actions: up, down, left, right, sample (5 basic) + K sensing actions (check state of each of k rocks)
Transitions: Moves are deterministic. Can only exit by exit on right side of grid
Observations: Observe status of rock with some noise (varies exponentially with distance). Otherwise observe none
Rewards: Step penalty, reward for exit, 
"""

@struct.dataclass
class EnvState:
    pos: chex.Array  # Agent xy position
    rock_pos: chex.Array  # xy position of each rock
    rock_status: chex.Array  # bad/good status of each rock
    time: int


@struct.dataclass
class EnvParams:
    sensor_efficiency: float = 2.
    reward_bad_sample: float = -10
    reward_good_sample = 10
    reward_exit = 10
    reward_step = 0
    fixed_rock_positions: bool = False


def observe_rock(rng: chex.PRNGKey, agent_xy: chex.Array, rock_xy: chex.Array, rock_status: int, params: EnvParams) -> int:
    """Get noisy rock observation based on euclidean distance"""
    distance = jnp.linalg.norm(agent_xy - rock_xy, 2, -1)
    efficiency = params.sensor_efficiency ** (-distance / 20)
    prob_correct = 0.5 * (1 + efficiency)
    return jax.lax.select(jax.random.uniform(rng) <= prob_correct, rock_status, 1 - rock_status)


class RockSample(environment.Environment):
    def __init__(self,
                 map_size: Tuple[int, int] = (4, 4),
                 num_rocks: int = 4,
    ):
        """Create a rocksample environment of size map_size with k rocks

        Args:
            map_size:
            num_rocks:
        """
        super().__init__()
        self.map_size = jnp.array(map_size)
        self.spawn_coords = jnp.stack(jnp.ones(map_size).nonzero(), -1)  # [n_spaces, 2]
        self.k = num_rocks
        self.actions = ['up', 'right', 'down', 'left', 'sample'] + [f'sense_{k}' for k in range(self.k)]
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], *([0, 0] for _ in range(self.k + 1))])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[int, EnvState]:
        key_reset, key_obs = jax.random.split(key, 2)
        pos, *rock_pos = jax.random.choice(key_reset, self.spawn_coords, (self.k + 1,), False)
        rock_status = jax.random.randint(key_reset, (self.k,), 1, 3)
        s = EnvState(pos, jnp.stack(rock_pos, 0), rock_status, 0)
        return self.get_obs(key_obs, s, 0, params), s

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[int, EnvState, float, bool, dict]:
        p = state.pos + self.directions[action]  # Move
        d = jax.lax.select(p[0] >= self.map_size[0], True, False)  # Done if we exit right
        p = jax.lax.select((p >= self.map_size).any() or (p < 0).any(), state.pos, p)  # Invalid moves
        

    def get_obs(self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams) -> int:
        rock_idx = action - 5
        o = jax.lax.select(action >= 0,
                           observe_rock(rng, state.pos, state.rock_pos[rock_idx], state.rock_status[rock_idx], params),
                           0)
        return o

    @property
    def name(self) -> str:
        return f"RockSample-{self.map_size}-pomdp"


    def action_space(self, params: EnvParams):
        return spaces.Discrete(len(self.actions))

    def observation_space(self, params: EnvParams):
        return spaces.Discrete(3)  # none, bad, good

    def state_space(self, params: EnvParams):
        return spaces.Dict({
            "pos": spaces.Box(
                jnp.min(self.map_size),
                jnp.max(self.map_size),
                (2,),
                jnp.int32,
            ),
            "rock_pos": spaces.Box(
                jnp.min(self.map_size),
                jnp.max(self.map_size),
                (self.k, 2),
                jnp.int32,
            ),
            "rock_status": spaces.Box(
                0,
                1,
                (self.k,),
                jnp.bool_
            ),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })
