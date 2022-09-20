# 3D AUV task from Ong, Png, Hsu and Lee 2009 (Mixed observabilty)

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

class AUVNavigation3D(environment.Environment):
    """4-level 7x20 underwater"""
    def __init__(self):
        """3D underwater navigation

        Robot must navigate from rightmost portion of deepest level to some goal locations while avoiding rocks
        Rocks are present at all levels except surface. Episode terminates on hit (-1)
        Actions: Move to any adjacent square along orientation. Some probability of drifting horizontal
        Obs: Only knows horizontal position at surface, but surfacing costs
            Can acquire information on depth and orientation (24 orientations, 4 depths)
            x represents depth+orientation
            y represents horizontal



        """
        self.zyx = jnp.array([4, 7, 20])

        ...

    def name(self) -> str:
        """Environment name"""
        return 'AUVNavigation3D-pomdp'

    def action_space(self, params: EnvParams):
        ...

    def observation_space(self, params: EnvParams):
        ...

    def state_space(self, params: EnvParams):
        ...