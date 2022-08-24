import gymnax
import jax

if __name__ == "__main__":
    e, params = gymnax.make('MultistoryFourRooms-misc')
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
    o, s = e.reset(key_reset, params)
    o, s, r, d, _ = e.step(key_step, s, e.action_space(params).sample(key_act), params)
    print(3)