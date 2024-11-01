import jax

# call rng function and return value with new rng
def rngcall(f, rng, *args, **kwargs):
    rng1, rng2 = jax.random.split(rng)
    return f(rng2, *args, **kwargs), rng1
