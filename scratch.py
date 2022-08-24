import gymnax
import jax

if __name__ == "__main__":
    B = 64
    e, params = gymnax.make('MultistoryFourRooms-misc', num_floors=1)
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
    reset_rng = jax.vmap(e.reset, in_axes=(0, None))
    rng, *keys_reset = jax.random.split(rng, B + 1)
    keys_reset = jax.numpy.stack(keys_reset)
    # env_params = jax.tree_util.tree_map(lambda x: jax.numpy.broadcast_to(x, (B,)), params)
    # keys = jax.random.split(key_reset, 64)
    o, s = reset_rng(keys_reset, params)
    step_rng = jax.vmap(e.step, in_axes=(0, 0, 0, None))
    o, s, r, d, _ = e.step(key_step, s, e.action_space(params).sample(key_act), params)
    for _ in range(1000):
        o, s, r, d, _ = e.step(key_step, s, e.action_space(params).sample(key_act), params)
    print(3)