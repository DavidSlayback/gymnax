import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    tiger: int  # 0 left, 1 right


@struct.dataclass
class EnvParams:
    listen_success_prob: float = 0.85  # probability of listening and hearing tiger at right location
    tiger_left_prob: float = 0.5  # Spawn probability for tiger
    max_steps_in_episode: int = 100
    reward_tiger: float = -100
    reward_not_tiger: float = 10
    reward_listen: float = -1


class Tiger(environment.Environment):
    def __init__(self):
        """Tiger problem"""
        super().__init__()
        self.actions = ['open_left', 'open_right', 'listen']

    @property
    def default_params(self):
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[int, EnvState, float, bool, dict]:
        """Listen or open door, suffer the consequences"""
        d = jax.lax.select(action < 2, True, False)  # Done if we opened a door
        r = jax.lax.select(action == state.tiger, params.reward_tiger, params.reward_not_tiger)  # Door reward
        r = jax.lax.select(action == 2, params.reward_listen, r)  # listen reward
        return self.get_obs(key, state, action, params), state, r, d, {}

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[int, EnvState]:
        """Redraw tiger position, first observation is as if we didn't listen"""
        rng_obs, rng_tiger = jax.random.split(key, 2)
        tiger_left = (jax.random.uniform(rng_tiger) > params.tiger_left_prob).astype(int)
        state = EnvState(tiger_left)
        return self.get_obs(key, state, 1, params), state

    def get_obs(self,
                key: chex.PRNGKey,
                state: EnvState,
                action: int,
                params: EnvParams) -> int:
        """If we listen, better chance of hearing tiger. Otherwise uniform"""
        prob_correct_obs = jax.lax.select(action == 2, params.listen_success_prob, 0.5)
        o = jax.lax.select(jax.random.uniform(key) <= prob_correct_obs, state.tiger, 1 - state.tiger)
        return o

    @property
    def name(self) -> str:
        return 'Tiger-pomdp'

    @property
    def num_actions(self) -> int:
        return 3

    def action_space(self, params: EnvParams):
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams):
        return spaces.Discrete(2)  # Hear tiger left or right

    def state_space(self, params: EnvParams):
        return spaces.Discrete(2)  # Tiger left or right











