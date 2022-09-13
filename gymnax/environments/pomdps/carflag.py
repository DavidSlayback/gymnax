"""Hai's CarFlag environment"""
import jax.random
import numpy as np
import jax.numpy as jnp
import chex
from ..environment import Environment

@chex.dataclass(frozen=True)
class EnvState:
    x: float  # where is agent
    x_vel: float  # where is agent moving?
    heaven: float  # where is heaven? hell is negative of this
    priest: float  # Where is priest
    time: int = 0

@chex.dataclass(frozen=True)
class EnvParams:
    max_pos: float = 1.1  # max x position, min_pos is negative
    max_speed: float = 0.07  # max car speed
    max_act: float = 1.  # max x action, min is negative this
    priest: float = 0.5  # Distance of priest from center
    priest_threshold: float = 0.2  # Distance within which we can see priest
    power: float = 0.0015  # Power of agent actions
    max_steps_in_episode: int = 160


class CarFlag(Environment):
    heaven_offset: float = 0.1
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Apply force"""
        action = jnp.clip(action, -params.max_act, params.max_act)  # Clip to range
        force = action * params.power  # Multiply by power
        new_x_vel = jnp.clip(state.x_vel + force, -params.max_speed, params.max_speed)
        new_x_pos = jnp.clip(state.x + new_x_vel, -params.max_pos, params.max_pos)
        new_state = state.replace(x=new_x_pos, x_vel=new_x_vel, time=state.time + 1)
        done, truncated = self.is_terminal(new_state, params)
        # +1 for heaven, -1 for hell, 0 otherwise
        reward = jax.lax.select(jnp.sign(state.x) == jnp.sign(state.heaven), 1., -1.)
        reward = jax.lax.select(done, reward, 0.)
        return self.get_obs(new_state, params), new_state, reward, done | truncated, {}


    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return x, x velocity, and heaven location (if priest in range)"""
        priest_in_range = jnp.abs(state.priest - state.x) <= params.priest_threshold
        return jnp.array([state.x, state.x_vel, jax.lax.select(priest_in_range, state.priest, 0.)])

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment. Priest and heaven are on negative or positive sides, agent starts with some noise"""
        rng_signs, rng_agent = jax.random.split(key)
        signs = jnp.where(jax.random.bernoulli(rng_signs, shape=(2,)), 1., -1.)
        heaven = signs[0] * (params.max_pos - self.heaven_offset)
        priest = signs[1] * params.priest
        x = jax.random.uniform(rng_agent, minval=-0.2, maxval=0.2)
        return EnvState(x, 0., heaven, priest)

    def is_terminal(self, state: EnvState, params: EnvParams) -> Tuple[bool, bool]:
        """Episode ends if we reach heaven/hell or run out of time"""
        done_steps = state.time >= params.max_steps_in_episode
        done_goal = jnp.abs(state.x) >= (params.max_pos - 0.1)
        return done_goal, done_steps
