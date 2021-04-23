import jax
import jax.numpy as jnp
from jax import lax

from gymnax.utils.frozen_dict import FrozenDict
from gymnax.environments import environment, spaces

from typing import Union, Tuple
import chex
Array = chex.Array
PRNGKey = chex.PRNGKey


class SimpleBandit(environment.Environment):
    """
    JAX Compatible version of DiscountingChain bsuite environment. Source:
    github.com/deepmind/bsuite/blob/master/bsuite/environments/bandit.py
    """
    def __init__(self, num_actions: int=11):
        super().__init__()
        # Default environment parameters
        self.env_params = FrozenDict({"num_actions": num_actions,
                                      "optimal_return": 1,
                                      "max_steps_in_episode": 100})

    def step(self, key: PRNGKey, state: dict, action: int
             ) -> Tuple[Array, dict, float, bool, dict]:
        """ Perform single timestep state transition. """
        reward = state["rewards"][action]
        state = {"rewards": state["rewards"],
                 "total_regret": (state["total_regret"] +
                                  self.env_params["optimal_return"] - reward),
                 "time": state["time"] + 1}

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state)
        state["terminal"] = done
        info = {"discount": self.discount(state)}
        return (lax.stop_gradient(self.get_obs(state)),
                lax.stop_gradient(state), reward, done, info)

    def reset(self, key: PRNGKey) -> Tuple[Array, dict]:
        """ Reset environment state by sampling initial position. """
        action_mask = jax.random.choice(key,
                        jnp.arange(self.env_params["num_actions"]),
                        shape=(self.env_params["num_actions"], ), replace=False)
        rewards = jnp.linspace(0, 1,
                               self.env_params["num_actions"])[action_mask]

        state = {"rewards": rewards,
                 "total_regret": 0,
                 "time": 0,
                 "terminal": 0}
        return self.get_obs(state), state

    def get_obs(self, state: dict) -> Array:
        """ Return observation from raw state trafo. """
        return jnp.ones(shape=(1, 1), dtype=jnp.float32)

    def is_terminal(self, state: dict) -> bool:
        """ Check whether state is terminal. """
        #done_steps = (state["time"] > self.env_params["max_steps_in_episode"])
        # Episode always terminates after single step - Do not reset though!
        return True

    @property
    def name(self) -> str:
        """ Environment name. """
        return "SimpleBandit-bsuite"

    @property
    def action_space(self):
        """ Action space of the environment. """
        return spaces.Discrete(self.env_params["num_actions"])

    @property
    def observation_space(self):
        """ Observation space of the environment. """
        return spaces.Box(1, 1, (1, 1))

    @property
    def state_space(self):
        """ State space of the environment. """
        return spaces.Dict(
            {"rewards": spaces.Box(0, 1, (self.env_params["num_actions"],)),
             "total_regret": spaces.Box(0,
                                        self.env_params["max_steps_in_episode"],
                                        ()),
             "time": spaces.Discrete(self.env_params["max_steps_in_episode"]),
             "terminal": spaces.Discrete(2)})