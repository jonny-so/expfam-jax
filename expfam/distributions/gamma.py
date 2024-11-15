import jax
import jax.numpy as jnp
from jax import vmap
from jaxopt import implicit_diff
from jaxutil.control_flow import bounded_while_loop
from jaxutil.tree import *

def gamma_natural_from_mean(meanparams):
    # handle batching over leading dimensions
    shape_prefix = meanparams[0].shape
    meanparams = tree_map(lambda _: _.reshape(-1, *_.shape[len(shape_prefix):]), meanparams)

    def _natural_params(meanparams):
        # f is concave, strictly increasing, tends to -inf as b -> 0 from above, and has
        # exactly one root when meanparams lie in the domain of valid mean parameters.
        f = lambda a, mp: jax.scipy.special.digamma(a) - jnp.log(a) + jnp.log(mp[1]) - mp[0]
        fprime = lambda a: jax.scipy.special.polygamma(1, a) - 1/a
        # find a starting point alpha0 > 0 such that f is below 0. note that some parameters
        # inside the mean domain can fail to converge for numerical reasons, hence the bound.
        alpha0 = bounded_while_loop(
            cond=lambda _: f(_, meanparams) >= 0,
            body=lambda _: .5*(_),
            init=2.0,
            maxiter=100)
        # define custom_root condition to get efficient reverse-mode derivatives from jaxopt.
        @implicit_diff.custom_root(f)
        def _newton_solve(alpha0, mp):
            return bounded_while_loop(
                cond=lambda _: jnp.abs(f(_, mp)) > 1e-12,
                body=lambda _: _ - f(_, mp)/fprime(_),
                init=alpha0,
                maxiter=100)
        # newton's method is guaranteed to converge to a root of f from our starting point.
        alpha = _newton_solve(jax.lax.stop_gradient(alpha0), meanparams)
        beta = alpha/meanparams[1]
        return gamma_natural_from_standard((alpha, beta))
    
    _null_natural_params = lambda _: tree_scale(_, jnp.nan)
    natural_params = vmap(lambda _: jax.lax.cond(gamma_in_mean_domain(_), _natural_params, _null_natural_params, _))(meanparams)
    return tree_map(lambda _: _.reshape(*shape_prefix, *_.shape[1:]), natural_params)

def gamma_natural_from_standard(standardparams):
    alpha, beta = standardparams
    return alpha-1, -beta

def gamma_natural_from_mean_var(m, v):
    b = m/v
    a = m*b
    return a-1, -b

def gamma_mean_from_natural(natural_params):
    n1, n2 = natural_params
    return jax.scipy.special.digamma(n1+1) - jnp.log(-n2), -(n1+1)/n2

def gamma_stats(x):
    return jnp.log(x), x

def gamma_dot(x1, x2):
    return x1[0]*x2[0] + x1[1]*x2[1]

def gamma_log_base_measure(x):
    return jnp.ones_like(x)

def gamma_log_partition(natural_params):
    n1, n2 = natural_params
    return jax.scipy.special.gammaln(n1+1) - (n1+1)*jnp.log(-n2)

def gamma_log_prob(natural_params, x):
    n1, n2 = natural_params
    a, b = n1+1, -n2
    return a*jnp.log(b) - jax.scipy.special.gammaln(a) + (a-1)*jnp.log(x) - b*x

def gamma_entropy(natural_params):
    a, b = gamma_standard_from_natural(natural_params)
    return a - jnp.log(b) + jax.scipy.special.gammaln(a) + (1-a)*jax.scipy.special.digamma(a)

def gamma_sample(key, natural_params, shape=()):
    a, b = gamma_standard_from_natural(natural_params)
    return jax.random.gamma(key, a, (*shape, *a.shape))/b

def gamma_kl(natural_params_from, natural_params_to):
    return (gamma_dot(tree_sub(natural_params_from, natural_params_to), gamma_mean_from_natural(natural_params_from))
            + gamma_log_partition(natural_params_to) - gamma_log_partition(natural_params_from))

def gamma_in_natural_domain(natural_params):
    n1, n2 = natural_params
    return (n1 > -1) & (n2 < 0)

def gamma_in_mean_domain(meanparams):
    logx, x = meanparams
    return (x > 0) & (logx < jnp.log(x))

def gamma_mean(natural_params):
    n1, n2 = natural_params
    return -(n1+1)/n2

def gamma_var(natural_params):
    n1, n2 = natural_params
    return (n1+1)/(n2**2)

def gamma_mode(natural_params):
    n1, n2 = natural_params
    return -n1/n2

def gamma_standard_from_natural(natural_params):
    n1, n2 = natural_params
    return n1+1, -n2

