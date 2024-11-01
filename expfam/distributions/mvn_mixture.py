# multivariate normal mixture model:
#   s ~ discrete
#   x ~ N(\mu_s, V_s)
# note the *joint* distribution over s and x is an exponential family.

import jax
import jax.numpy as jnp
from jax.random import split
from expfam.distributions.discrete import discrete_entropy
from expfam.distributions.mvn import *
from expfam.util.la import mvp, transpose

MvnMixtureNaturalParams = tuple[Array, Array, Array]
MvnMixtureMeanParams = tuple[Array, Array, Array]

def mvn_mixture_natural_from_mean(mean_params):
    p, px, pxx = mean_params
    x, xx = px/p[...,None], pxx/p[...,None,None]
    h, J = mvn_natural_from_mean((x, xx))
    gamma = jnp.log(p) - mvn_log_partition((h, J))
    return gamma, h, J

def mvn_mixture_mean_from_natural(natural_params):
    gamma, h, J = natural_params
    J = .5*(J + transpose(J))
    log_partition = mvn_log_partition((h, J))
    p = jnp.exp(gamma + log_partition - jax.scipy.special.logsumexp(gamma + log_partition))
    m = mvn_mean_from_natural((h, J))
    return p, p[...,None]*m[0], p[...,None,None]*m[1]

def mvn_mixture_standard_from_natural(natural_params):
    gamma, h, J = natural_params
    log_partition = mvn_log_partition((h, J))
    p = jnp.exp(gamma + log_partition - jax.scipy.special.logsumexp(gamma + log_partition))
    m, V = mvn_standard_from_natural((h, J))
    return p, m, V

def mvn_mixture_stats(sx, K):
    s, x = sx
    ds = jax.nn.one_hot(s, num_classes=K)
    x, xx = mvn_stats(x)
    dsx = ds[...,None]*x
    dsxx = ds[...,None,None]*xx
    return ds, dsx, dsxx

def mvn_mixture_dot(natural_params, stats):
    return (
        jnp.sum(natural_params[0]*stats[0], -1) + 
        jnp.sum(natural_params[1]*stats[1], (-1,-2)) + 
        jnp.sum(natural_params[2]*stats[2], (-1,-2,-3)))

def mvn_mixture_log_base_measure(sx):
    return mvn_log_base_measure(sx[1])

def mvn_mixture_log_partition(natural_params):
    gamma, h, J = natural_params
    log_partition = mvn_log_partition((h, J))
    assert(gamma.shape == log_partition.shape)
    return jax.scipy.special.logsumexp(gamma + log_partition)

def mvn_mixture_log_prob(natural_params, sx):
    K = natural_params[0].shape[-1]
    return (mvn_mixture_dot(natural_params, mvn_mixture_stats(sx, K))
        - mvn_mixture_log_partition(natural_params) + mvn_mixture_log_base_measure(sx))

def mvn_mixture_entropy(natural_params):
    h, J = natural_params[1:]
    p = mvn_mixture_mean_from_natural(natural_params)[0]
    entropy = jnp.sum(-p*jnp.log(p), -1) + jnp.sum(p*mvn_entropy((h, J)), -1)
    return entropy

def mvn_mixture_sample(key, natural_params, shape=()):
    expand = lambda _: jnp.tile(_, shape + (1,)*_.ndim)
    natural_params = tree_map(expand, natural_params)
    gamma, h, J = natural_params
    logp = gamma + mvn_log_partition((h, J))
    K = gamma.shape[-1]
    key_s, key_x = split(key)
    s = jax.random.categorical(key_s, logp)
    h = jnp.sum(h*jax.nn.one_hot(s, K)[...,None], -2)
    J = jnp.sum(J*jax.nn.one_hot(s, K)[...,None,None], -3)
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    x = jax.random.multivariate_normal(key_x, mu, V)
    return s, x

def mvn_mixture_symmetrize(natural_params):
    gamma, h, J = natural_params
    J = .5*(transpose(J) + J)
    return gamma, h, J
