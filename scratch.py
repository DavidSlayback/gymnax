import gymnax
import jax
import jax.numpy as jnp

if __name__ == "__main__":
    B = 64  # batch size
    env, env_params = gymnax.make('Tiger-pomdp')
    rng = jax.random.PRNGKey(0)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    rng, *vmap_keys = jax.random.split(rng, B+1)
    vmap_keys = jnp.stack(vmap_keys)
    obs, state = vmap_reset(vmap_keys, env_params)
    n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, jnp.zeros(B, dtype=int), env_params)
    print(n_obs)
