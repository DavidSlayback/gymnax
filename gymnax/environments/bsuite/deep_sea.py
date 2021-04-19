import jax
import jax.numpy as jnp
from jax import jit
from ...utils.frozen_dict import FrozenDict


# JAX Compatible version of DeepSea bsuite environment. Source:
# github.com/deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py

# Default environment parameters
params_deep_sea = FrozenDict({"size": 8,
                              "deterministic": True,
                              "sample_action_map": False,
                              "unscaled_move_cost": 0.01,
                              "randomize_actions": True,
                              "action_mapping": jnp.ones([8, 8])})


def step(rng_input, params, state, action):
    """ Perform single timestep state transition. """
    # Pull out randomness for easier testing
    rng_reward, rng_trans = jax.random.split(rng_input)
    rand_reward = jax.random.normal(rng_reward, shape=(1,))
    rand_trans_cond = (jax.random.uniform(rng_trans, shape=(1,), minval=0,
                                          maxval=1) > 1/params["size"])

    action_right = (action == action_mapping[state["row"], state["column"]])
    right_rand_cond = jnp.logical_or(rand_trans_cond, params["deterministic"])
    right_cond = jnp.logical_and(action_right, right_rand_cond)

    reward, state = step_reward(state, action_right, right_cond)
    state = step_transition(state, action_right, right_cond)
    done, state = step_terminal(state)
    return get_obs(state, params), state, reward, done, {}


def step_reward(state, action_right, right_cond):
    """ Get the reward for the selected action. """
    reward = 0.
    # Reward calculation.
    rew_cond = jnp.logical_and(state["column"] == params["size"] - 1,
                               action_right)
    reward += rew_cond
    state["denoised_return"] += rew_cond

    # Noisy rewards on the 'end' of chain.
    col_at_edge = jnp.logical_or(state["column"] == 0,
                                 state["column"] == params["size"] - 1)
    chain_end = jnp.logical_and(state["row"] == params["size"] - 1,
                                col_at_edge)
    det_chain_end = jnp.logical_and(chain_end, params["deterministic"])
    reward += rand_reward * det_chain_end
    reward -= right_cond * params["unscaled_move_cost"] / params["size"]
    return reward, state, right_cond


def step_transition(state, action_right, right_cond):
    """ Get the state transition for the selected action. """
    # Standard right path transition
    state["column"] = ((1 - right_cond) * state["column"] +
                       right_cond *
                       jnp.clip(state["column"] + 1, 0, params["size"] - 1))

    # You were on the right path and went wrong
    right_wrong_cond = jnp.logical_and(1 - action_right,
                                       state["row"] == state["column"])
    state["bad_episode"] = ((1 - right_wrong_cond) * state["bad_episode"]
                            + right_wrong_cond * 1)
    state["column"] = ((1 - action_right)
                       * jnp.clip(state["column"] - 1, 0, params["size"] - 1)
                       + action_right * state["column"])
    state["row"] = state["row"] + 1
    return state


def step_terminal(state):
    """ Check termination condition of state. """
    done = (state["row"] == params["size"])
    state["total_bad_episodes"] += done * state["bad_episode"]
    state["terminal"] = done
    return done, state


def reset(rng_input, params):
    """ Reset environment state. """
    optimal_no_cost = ((1 - params["deterministic"])
                        * (1 - 1 / params["size"]) ** (params["size"] - 1)
                        + params["deterministic"] * 1.)
    optimal_return = optimal_no_cost - params["unscaled_move_cost"]

    a_map_rand = jax.random.bernoulli(rng_input, 0.5,
                                      (params["size"], params["size"]))
    a_map_determ = jnp.ones([params["size"], params["size"]])

    new_a_map_cond = jnp.logical_and(1-params["deterministic"],
                                     params["sample_action_map"])
    old_a_map_cond = jnp.logical_and(1-params["deterministic"],
                                     1-params["sample_action_map"])
    action_mapping = (params["deterministic"] * a_map_determ
                      + new_a_map_cond * a_map_rand
                      + old_a_map_cond * params["action_mapping"])

    state = {"row": 0,
             "column": 0,
             "bad_episode": False,
             "total_bad_episodes": 0,
             "denoised_return": 0,
             "optimal_no_cost": optimal_no_cost,
             "optimal_return": optimal_return,
             "terminal": False,
             "action_mapping": action_mapping}
    return get_obs(state, params), state


def get_obs(state, params):
    """ Return observation from raw state trafo. """
    obs_end = jnp.zeros(shape=(params["size"], params["size"]),
                        dtype=jnp.float32)
    end_cond = (state["row"] >= params["size"])
    obs_upd = jax.ops.index_update(obs_end, jax.ops.index[state["row"],
                                                          state["column"]], 1.)
    return (1 - end_cond) * obs_end + end_cond * obs_upd


reset_deep_sea = jit(reset, static_argnums=(1,))
step_deep_sea = jit(step, static_argnums=(1,))