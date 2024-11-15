import jax
import jax.numpy as jnp
from expfam.distributions.gamma import *
from jaxutil.tree import tree_sub


def inverse_gamma_natural_from_mean(mean_params):
    return _from_gamma_natural_params(
        gamma_natural_from_mean(
            _to_gamma_mean_params(mean_params)))

def inverse_gamma_natural_from_standard(standardparams):
    a, b = standardparams
    return -(a+1), -b

def inverse_gamma_mean_from_natural(natural_params):
    return _from_gamma_mean_params(
        gamma_mean_from_natural(
            _to_gamma_natural_params(natural_params)))

def inverse_gamma_standard_from_natural(natural_params):
    n1, n2 = natural_params
    return -(n1+1), -n2

def inverse_gamma_stats(x):
    return jnp.log(x), 1/x

def inverse_gamma_dot(x1, x2):
    return x1[0]*x2[0] + x1[1]*x2[1]

def inverse_gamma_log_base_measure(x):
    return jnp.ones_like(x)

def inverse_gamma_log_partition(natural_params):
    return gamma_log_partition(_to_gamma_natural_params(natural_params))

def inverse_gamma_log_prob(natural_params, x):
    a, b = inverse_gamma_standard_from_natural(natural_params)
    return a*jnp.log(b) - jax.scipy.special.gammaln(a) - (a+1)*jnp.log(x) - b/x

def inverse_gamma_entropy(natural_params):
    a, b = inverse_gamma_standard_from_natural(natural_params)
    return a + jnp.log(b) + jax.scipy.special.gammaln(a) - (1+a)*jax.scipy.special.digamma(a)

def inverse_gamma_sample(key, natural_params, shape=()):
    a, b = inverse_gamma_standard_from_natural(natural_params)
    return 1/(jax.random.gamma(key, a, (*shape, *a.shape))/b)

def inverse_gamma_kl(natural_params_from, natural_params_to):
    mean_params_from = inverse_gamma_mean_from_natural(natural_params_from)
    return (inverse_gamma_dot(tree_sub(natural_params_from, natural_params_to),mean_params_from)
            + inverse_gamma_log_partition(natural_params_to) - inverse_gamma_log_partition(natural_params_from))

def inverse_gamma_in_natural_domain(natural_params):
    return gamma_in_natural_domain(_to_gamma_natural_params(natural_params))

def inverse_gamma_in_mean_domain(mean_params):
    logx, x_inv = mean_params
    return (x_inv > 0) & (-logx < jnp.log(x_inv))

def inverse_gamma_mean(natural_params):
    a, b = inverse_gamma_standard_from_natural(natural_params)
    return b/(a - 1)

def inverse_gamma_var(natural_params):
    a, b = inverse_gamma_standard_from_natural(natural_params)
    return jnp.square(b)/(jnp.square(a-1)*(a-2))

def inverse_gamma_mode(natural_params):
    a, b = inverse_gamma_standard_from_natural(natural_params)
    return b/(a + 1)

# convert to/from the params of 1/x, which is gamma distributed

def _to_gamma_natural_params(natural_params):
    return gamma_natural_from_standard(
        inverse_gamma_standard_from_natural(natural_params))

def _from_gamma_natural_params(natural_params):
    return inverse_gamma_natural_from_standard(
        gamma_standard_from_natural(natural_params))

def _to_gamma_mean_params(mean_params):
    logx, x_inv = mean_params
    return -logx, x_inv

def _from_gamma_mean_params(mean_params):
    logx, x = mean_params
    return -logx, x