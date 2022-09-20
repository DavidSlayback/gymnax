# Taxi environment from gym and

from typing import Union, List
import functools

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from enum import IntEnum
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from functools import partial

# Base map
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class CellState(IntEnum):
    empty = 0
    wall = 1
    passthrough = 2
    dest = 3

@struct.dataclass
class EnvState:
    row: int  # 5
    col: int  # 5
    pass_idx: int  # 4 + 1 for being in taxi
    dest_idx: int  # 4 desintations
    time: int = 0


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 500


def encode(row, col, pass_idx, dest_idx) -> int:
    """Encode state to integer"""
    return jnp.ravel_multi_index((row, col, pass_idx, dest_idx), (5, 5, 5, 4))


def decode(state: int) -> Tuple[int]:
    """Decode integer to (row, col, pass_idx, dest_idx)"""
    return jnp.unravel_index(state, (5, 5, 5, 4))


def initial_state(key: chex.PRNGKey) -> Tuple[int, int, int, int]:
    """Sample initial taxi environment state (row, col, pass_idx, dest_idx)"""
    # destination and passenger indices must be different
    row, col, pass_idx, dest_idx = jax.random.randint(key, (4,), 0, jnp.array((5, 5, 4, 3)))
    dest_idx = jax.lax.select(dest_idx >= pass_idx, (dest_idx + 1) % 4, dest_idx)  # Corresponds to np.arange() without pass_idx
    return row, col, pass_idx, dest_idx


class Taxi(environment.Environment):
    """Taxi environment"""
    def __init__(self, obs_type: str = 'discrete', reward_step: float = 0, reward_goal: float = 2, reward_bad_pickupdropoff: float = -1):
        self.obs_type = obs_type
        self.num_states = 500  # actually 400
        self.num_actions = 5  # NESW and pickup/dropoff
        map = np.asarray(MAP, dtype='c').astype(str)
        # Transition matrix




        ...

    def name(self) -> str:
        ...

