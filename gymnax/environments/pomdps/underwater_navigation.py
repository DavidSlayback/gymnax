# Underwater navigation domain from Kurniawati, Hsu, Lee 2008 (SARSOP)
from typing import Union
import functools

import jax
from enum import IntEnum
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from functools import partial

class CellState(IntEnum):
    empty = 0
    observable = 1
    start = 2
    rock = 3
    destination = 4


@struct.dataclass
class EnvState:
    pos: chex.Array  # Agent yx position
    time: int = 0


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 500

class AUVNavigation(environment.Environment):
    """Robot must navigate from to right border of environment, avoiding rocks"""
    def __init__(self, width: int = 12):
        midpoint = int((width - 1) // 2)
        self.yx = jnp.array([width - 1, width])  # y is always 1 less than x (last column is destination)
        self.domain = jnp.full(self.yx, CellState.empty, dtype=int)  # grid
        self.domain = self.domain.at[[0, -1], :].set(CellState.observable)  # Can localize at top and bottom
        # Possible start locations: anywhere on first column between edges, then middle 3, then 1
        self.domain = self.domain.at[1:-1, 0].set(CellState.start)
        self.domain = self.domain.at[jnp.arange(midpoint - 1, midpoint + 2), 1].set(CellState.start)
        self.domain = self.domain.at[midpoint, 2].set(CellState.start)
        # Rocks evenly spaced on 2nd-to-last column
        self.domain = self.domain.at[1:-1:2, -2].set(CellState.rock)
        # Destination at edge
        self.domain = self.domain.at[:, -1].set(CellState.destination)
        self.num_observations = (self.domain == CellState.observable).sum() + 1
        self.observation_map = (self.domain == CellState.observable).cumsum().reshape(self.yx) - 1
        self.observation_map = self.observation_map.at[1:-1, :].set(self.num_observations - 1)
        # Agent start states
        self.start_states = jnp.stack(jnp.where(self.domain == CellState.start), -1)
        # Agent moves
        self.directions = jnp.array([
            [-1, 0],  # N
            [-1, 1],  # NE
            [0, 1],  # E
            [1, 1],  # SE
            [1, 0],  # S
            [0, 0]  # Stay
        ], dtype=int)
        print("Done!")

    @property
    def default_params(self) -> EnvParams:
        """Default parameters"""
        return EnvParams()

    def name(self) -> str:
        """Environment name"""
        return "AUVNavigation-pomdp"

    @functools.lru_cache(1)
    def action_space(self, params: EnvParams):
        """N-NE-E-SE-S or stay"""
        return spaces.Discrete(6)

    @functools.lru_cache(1)
    def observation_space(self, params: EnvParams):
        """Discrete observation, either current location (if in localizable area) or null obs"""
        return spaces.Discrete(self.num_observations)

    def state_space(self, params: EnvParams):
        return spaces.Dict({
            "pos": spaces.Box(0, self.yx, (2,), jnp.int32),
            "time": spaces.Discrete(params.max_steps_in_episode)
        })

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check termination conditions"""
        done_steps = state.time >= params.max_steps_in_episode  # Overtime
        done_edge = state.pos[1] >= self.yx[1]
        done_rocks = self.domain[state.pos[0], state.pos[1]] == CellState.rock
        return jax.lax.select(done_steps | done_edge | done_rocks, True, False)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[int, EnvState]:
        """Sample agent position from start state"""
        pos = jax.random.choice(key, self.start_states, ())
        state = EnvState(pos)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> int:
        """Use precomputed observation map"""
        return self.observation_map[state.pos[0], state.pos[1]]

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[int, EnvState, float, bool, dict]:
        """Step agent in direction"""
        new_pos = state.pos + self.directions[action]
        new_state = state.replace(pos=new_pos, time=state.time + 1)
        at_dest = new_pos[1] >= self.yx[1]  # Reached goal
        hit_rock = self.domain[new_pos[0], new_pos[1]] == CellState.rock  # Hit rock
        # +1 for goal, -1 for rock, 0 otherwise
        r = jax.lax.select(at_dest, 1., 0.)
        r = jax.lax.select(hit_rock, -1., r)
        d = jax.lax.select((r != 0) | (state.time >= params.max_steps_in_episode), True, False)
        return self.get_obs(new_state), new_state, r, d, {}








