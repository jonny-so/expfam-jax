# multi-dimensional generalisation of categorical distribution. note that the operations
# in this file do NOT automatically batch over leading indices as the shape of parameters
# is used to infer the dimensionality of the distribution. 

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
from jax import Array

DiscreteNaturalParams = tuple[Array, Array]
DiscreteMeanParams = tuple[Array, Array]

def discrete_natural_from_mean(meanparams):
    return jnp.log(meanparams)

def discrete_mean_from_natural(natparams):
    return jnp.exp(discrete_normalize(natparams))

def discrete_stats(x, shape):
    return jnp.zeros(shape).at[tuple(x)].set(1.0)

def discrete_dot(natparams, stats):
    assert(natparams.shape == stats.shape)
    return jnp.sum(natparams*stats)

def discrete_log_base_measure(x):
    jnp.zeros_like(x, shape=())

def discrete_log_partition(natparams):
    return jax.scipy.special.logsumexp(natparams)

def discrete_log_prob(natparams, x):
    return discrete_normalize(natparams)[tuple(x)]

def discrete_entropy(natparams):
    p = discrete_mean_from_natural(natparams)
    return -jnp.sum(p*jnp.log(p))

def discrete_sample(rng, natparams, shape_prefix=()):
    x = jax.random.categorical(rng, natparams.reshape(-1), shape=shape_prefix)
    res = jnp.stack(jnp.unravel_index(x, natparams.shape), -1)
    return res

# marginalise out masked dimensions
def discrete_marginalise(natparams, mask):
    assert(mask.ndim == 1)
    assert(mask.shape[0] == natparams.ndim)
    alpha = jax.scipy.special.logsumexp(natparams, axis=np.arange(len(mask))[mask])
    _logZ = discrete_log_partition(alpha)
    return alpha - _logZ, _logZ

# condition on masked dimensions
def discrete_condition(natparams, mask, x):
    assert(x.ndim == 1)
    assert(x.shape[0] < natparams.ndim)
    axes = np.concatenate([np.arange(len(mask))[mask], np.arange(len(mask))[~mask]])
    res = jnp.transpose(natparams, axes)[tuple(x) if x.ndim > 0 else x]
    return res

def discrete_normalize(natparams):
    return natparams - jax.scipy.special.logsumexp(natparams)