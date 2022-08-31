from functools import lru_cache
from typing import Optional, Tuple, Union, List

import gym
import jax.random
from gym.core import ActType, ObsType, RenderFrame
import numpy as np
from .spaces import gymnax_space_to_gym_space
from .environment import Environment, EnvState, EnvParams

class GymnaxToGymWrapper(gym.Env):
    def __init__(self, env: Environment, params: Optional[EnvParams] = None, seed: Optional[int] = None):
        super().__init__()
        self._env = env
        self.env_params = params if params is not None else env.default_params
        self.metadata |= {
            'name': env.name,
            'render_modes': ['human', 'rgb_array'],
        }
        self._seed(seed)
        _, self.env_state = self._env.reset(self.rng, self.env_params)

    @property
    def action_space(self, params: Optional[EnvParams] = None):
        return gymnax_space_to_gym_space(self._env.action_space(params))

    @property
    def observation_space(self, params: Optional[EnvParams] = None):
        return gymnax_space_to_gym_space(self._env.observation_space(params))

    def _seed(self, seed: Optional[int] = None):
        seed = np.random.SeedSequence(seed).entropy  # Get initial seed if none was provided
        self.rng = jax.random.PRNGKey(seed)

    def step(
        self, action: ActType
    ) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        """Step environment, follow new step API"""
        self.rng, step_key = jax.random.split(self.rng)
        o, self.env_state, r, d, info = self._env.step(step_key, self.env_state, action, self.env_params)
        return o, r, d, d, info  # new step API

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Reset environment, update parameters and seed if provided"""
        if seed is not None: self._seed(seed)
        env_params = options.get('env_params', self.env_params)  # Allow changing environment parameters on reset
        self.rng, reset_key = jax.random.split(self.rng)
        o, self.env_state = self._env.reset(reset_key, env_params)
        return o

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """use underlying environment rendering if it exists, otherwise return None"""
        return getattr(self._env, 'render', lambda x: None)(mode)