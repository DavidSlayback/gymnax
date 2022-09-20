# Homecare robot domain from Kurniawati, Hsu, Lee 2008 (SARSOP)

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
    human_pos: chex.Array  # Human yx position
    time: int = 0


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 500


class Homecare(environment.Environment):
    """4-level 7x20 underwater"""
    def __init__(self):
        """Homecare robot task (6x11)

        Robot must follow an elderly person around (who follows a predefined path but randomly pauses)
        Along the path, there's a bathroom, where he may stay for a while
        Person has a call button to call for robot. Stays on for some random time, then goes off
        Robot must arrive before call button goes off to receive reward
        Robot can observe person's position when they are close enough (should follow)
        Robot movement costs power, though. Must balance tracking with power cost

        """
        self.yx = jnp.array([11, 6])

        ...

    def name(self) -> str:
        """Environment name"""
        return 'AUVNavigation3D-pomdp'

    @functools.lru_cache(1)
    def action_space(self, params: EnvParams):
        """Can move any ordinal direction or stay"""
        return spaces.Discrete(9)

    def observation_space(self, params: EnvParams):
        """Can observe human's position if close enough"""
        return spaces.Box(0, self.yx, (2,), dtype=jnp.int32)

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return human position or 0s"""
        return jax.lax.select(jnp.linalg.norm(state.pos - state.human_pos, 2))


    def state_space(self, params: EnvParams):
        ...