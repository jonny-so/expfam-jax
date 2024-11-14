# normal-inverse-Wishart distribution
import jax
import jax.numpy as jnp
from expfam.util.control_flow import bounded_while_loop
from expfam.util.la import *
from expfam.util.tree import *
from jax import vmap, Array
from jax.lax import cond
from jax.scipy.special import digamma, multigammaln, polygamma
from jax.tree_util import tree_map
from jaxopt import implicit_diff

NiwNaturalParams = tuple[Array, Array, Array, Array]
NiwMeanParams = tuple[Array, Array, Array, Array]
NiwStandardParams = tuple[Array, Array, Array, Array]

def niw_natural_from_mean(mean_params):
    shape_prefix = mean_params[0].shape[:-2]
    mean_params = tree_map(lambda _: _.reshape(-1, *_.shape[len(shape_prefix):]), mean_params)

    def _niw_natural_from_mean(mean_params):
        D = mean_params[1].shape[-1]

        # f is concave, strictly increasing, tends to -inf as b -> (D-1) from above, and has
        # exactly one root when mean_params lie in the domain of valid mean parameters.
        def f(a, mp):
            E_Lam = -2*mp[0]
            E_logdetLam = 2*mp[3]
            return (jnp.sum(digamma(.5*(a[...,None] - jnp.arange(D))),-1) - D*jnp.log(.5*a)
            + jnp.linalg.slogdet(E_Lam)[1] - E_logdetLam)
        fprime = lambda _: .5*jnp.sum(polygamma(1,.5*(_[...,None] - jnp.arange(D))),-1) - D/_

        # find a starting point alpha0 > (D-1) such that f is below 0. note that some parameters
        # inside the mean domain can fail to converge for numerical reasons, hence the bound.
        alpha0 = bounded_while_loop(
            cond=lambda _: f(_, mean_params) >= 0,
            body=lambda _: .5*(_ + D - 1),
            init=D + 1.,
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
        alpha = _newton_solve(jax.lax.stop_gradient(alpha0), mean_params)

        E_Lam = -2*mean_params[0]
        E_h = mean_params[1]
        E_hTLaminvh = -2*mean_params[2]

        delta = jnp.linalg.solve(E_Lam, E_h)
        gamma = D/(E_hTLaminvh - vdot(delta, E_h))
        Psi = jnp.linalg.inv(E_Lam/alpha)
        return niw_natural_from_standard((Psi, delta, gamma, alpha))

    _null_natural_params = lambda _: tree_scale(_, jnp.nan)
    natural_params = vmap(lambda _: cond(niw_in_mean_domain(_), _niw_natural_from_mean, _null_natural_params, _))(mean_params)
    return tree_map(lambda _: _.reshape(*shape_prefix, *_.shape[1:]), natural_params)

def niw_natural_from_standard(standardparams):
    # following wikipedia notation
    Psi, delta, gamma, alpha = standardparams
    # following matt johnson's notation from https://github.com/mattjj/svae
    A = Psi + gamma[...,None,None] * outer(delta, delta)
    b = gamma[...,None] * delta
    kappa = gamma
    nu = alpha
    return A, b, kappa, nu

# based on implementation in https://github.com/mattjj/svae
def niw_mean_from_natural(natural_params):
    Psi, delta, gamma, alpha = niw_standard_from_natural(natural_params)
    D = delta.shape[-1]

    E_Lam = alpha[...,None,None] * symmetrize(jnp.linalg.inv(Psi)) + 1e-8 * jnp.eye(D)
    E_h = mvp(E_Lam, delta)
    E_hTLaminvh = D/gamma + vdot(delta, E_h)
    E_logdetLam = (jnp.sum(digamma(.5*(alpha[...,None] - jnp.arange(D))), -1)
        + D*jnp.log(2.) - jnp.linalg.slogdet(Psi)[1])

    return -.5*E_Lam, E_h, -.5*E_hTLaminvh, .5*E_logdetLam

def niw_standard_from_natural(natural_params):
    A, b, kappa, nu = natural_params
    alpha = nu
    gamma = kappa
    delta = b / gamma[...,None]
    Psi = A - gamma[...,None,None] * outer(delta, delta)
    return Psi, delta, gamma, alpha

def niw_stats(x):
    mu, V = x
    Lam = jnp.linalg.inv(V)
    h =  mvp(Lam, mu)
    hTLaminvh = vdot(mu, mvp(Lam, mu))
    logdetLam = jnp.linalg.slogdet(Lam)[1]
    return -.5*Lam, h, -.5*hTLaminvh, .5*logdetLam

def niw_dot(a, b):
    return jnp.sum(a[0]*b[0], (-1,-2)) + jnp.sum(a[1]*b[1], -1) + a[2]*b[2] + a[3]*b[3]

def niw_log_base_measure(x):
    mu, V = x
    D = mu.shape[-1]
    return -.5*(D+2)*jnp.linalg.slogdet(V)[1]

def niw_log_partition(natural_params):
    Psi, delta, gamma, alpha = niw_standard_from_natural(natural_params)
    D = delta.shape[-1]
    return (.5*alpha*D*jnp.log(2.) + .5*D*jnp.log(2*jnp.pi)
        + multigammaln(.5*alpha, D) - .5*D*jnp.log(gamma) - .5*alpha*jnp.linalg.slogdet(Psi)[1])

def niw_log_prob(natural_params, x):
    return niw_dot(natural_params, niw_stats(x)) + niw_log_base_measure(x) - niw_log_partition(natural_params)

def niw_entropy(natural_params):
    D = natural_params[1].shape[-1]
    mean_params = niw_mean_from_natural(natural_params)
    E_log_base_measure = (D+2)*mean_params[-1]
    return -niw_dot(natural_params, mean_params) + niw_log_partition(natural_params) - E_log_base_measure 

def niw_log_partition_dual(mean_params):
    natural_params = niw_natural_from_mean(mean_params)
    return niw_dot(natural_params, mean_params) - niw_log_partition(natural_params)

def niw_kl(natural_params1, natural_params2):
    return niw_log_partition(natural_params2) - niw_log_partition(natural_params1) \
        + niw_dot(tree_sub(natural_params1, natural_params2), niw_mean_from_natural(natural_params1))

def niw_in_mean_domain(mean_params):
    E_Lam = -2*mean_params[0]
    E_logdetLam = 2*mean_params[3]
    s, logdetE_lam = jnp.linalg.slogdet(E_Lam)
    return (s == 1) & (E_logdetLam < logdetE_lam)

def niw_in_natural_domain(natural_params):
    Psi, delta, gamma, alpha = niw_standard_from_natural(natural_params)
    return isposdefh(Psi) & (gamma > 0) & (alpha > delta.shape[-1] - 1)

def niw_mean(natural_params):
    Psi, delta, _, alpha = niw_standard_from_natural(natural_params)
    return delta, Psi / (alpha - delta.shape[-1] - 1)

def niw_symmetrize(_):
    return (.5*(transpose(_[0]) + _[0]), *_[1:])
