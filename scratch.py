import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.gym import GymnaxToGymWrapper, GymnaxToVectorGymWrapper

if __name__ == "__main__":
    B = 64  # batch size
    env, env_params = gymnax.make("MultistoryFourRooms-pretrain")
    e1 = GymnaxToGymWrapper(env)
    e2 = GymnaxToVectorGymWrapper(env, num_envs=64)
    # o1 = e1.reset(seed=2)
    # o1_2, r1, d1, d1_2, info = e1.step(e1.action_space.sample())
    o2 = e2.reset(seed=2)
    o2_2, r2, d2, d2_2, info = e2.step(e2.action_space.sample())
    rng = jax.random.PRNGKey(0)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    rng, *vmap_keys = jax.random.split(rng, B + 1)
    vmap_keys = jnp.stack(vmap_keys)
    obs, state = vmap_reset(vmap_keys, env_params)
    n_obs, n_state, reward, done, _ = vmap_step(
        vmap_keys, state, jnp.zeros(B, dtype=int), env_params
    )
    print(n_obs)
    # from gymnax.experimental.macro_actions.fourrooms import create_original_hallway_options
    # test = create_original_hallway_options()
