import jax
import jax.numpy as jnp
from jax import Array
from expfam.util.la import vdot
from expfam.util.tree import tree_map, tree_sub

DiagonalMvnNaturalParams = tuple[Array, Array]
DiagonalMvnMeanParams = tuple[Array, Array]
DiagonalMvnStandardParams = tuple[Array, Array]

def diagonal_mvn_natural_from_mean(mean_params):
    x, xx = mean_params
    j = -.5/(xx - jnp.square(x))
    h = -2*j*x
    return h, j

def diagonal_mvn_natural_from_standard(standard_params):
    m, v = standard_params
    j = -.5/v
    h = -2*j*m
    return h, j

def diagonal_mvn_mean_from_natural(natural_params):
    h, j = natural_params
    v = -.5/j
    mu = h*v
    return mu, v + jnp.square(mu)

def diagonal_mvn_standard_from_natural(natural_params):
    h, j = natural_params
    v = -.5/j
    mu = h*v
    return mu, v

def diagonal_mvn_stats(x):
    return x, jnp.square(x)

def diagonal_mvn_dot(a, b):
    return jnp.sum(a[0]*b[0], axis=-1) + jnp.sum(a[1]*b[1], axis=-1)

def diagonal_mvn_log_base_measure(x):
    return -.5*x.shape[-1]*jnp.log(2*jnp.pi)*jnp.ones_like(x[...,0])

def diagonal_mvn_log_partition(natural_params):
    h, j = natural_params
    v = -.5/j
    return .5*vdot(h, h*v) + .5*jnp.sum(jnp.log(v), -1)

def diagonal_mvn_log_prob(natural_params, x):
    return (diagonal_mvn_dot(natural_params, diagonal_mvn_stats(x))
        - diagonal_mvn_log_partition(natural_params) + diagonal_mvn_log_base_measure(x))

def diagonal_mvn_entropy(natural_params):
    h, j = natural_params
    D = h.shape[-1]
    return .5*D*(1 + jnp.log(2*jnp.pi)) - .5*jnp.sum(jnp.log(-2*j))

def diagonal_mvn_kl(natural_params_from, natural_params_to):
    return (diagonal_mvn_log_partition(natural_params_to) - diagonal_mvn_log_partition(natural_params_from)
        + diagonal_mvn_dot(
            tree_sub(natural_params_from, natural_params_to),
            diagonal_mvn_mean_from_natural(natural_params_from)))

def diagonal_mvn_sample(key, natural_params, shape=()):
    expand = lambda _: jnp.tile(_, shape + (1,)*_.ndim)
    mu, v = tree_map(expand, diagonal_mvn_standard_from_natural(natural_params))
    return mu + jax.random.normal(key, mu.shape)*jnp.sqrt(v)

def diagonal_mvn_mean(natural_params):
    return diagonal_mvn_mean_from_natural(natural_params)[0]

def diagonal_mvn_var(natural_params):
    return -.5/natural_params[1]

def diagonal_mvn_precision(natural_params):
    return -2*natural_params[1]