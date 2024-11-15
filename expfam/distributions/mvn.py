import jax
import jax.numpy as jnp
from jax import Array
from jaxutil.la import invcholp, outer, submatrix, vdot, mvp, transpose
from jaxutil.tree import tree_sub, tree_map

MvnNaturalParams = tuple[Array, Array]
MvnMeanParams = tuple[Array, Array]
MvnStandardParams = tuple[Array, Array]

def mvn_natural_from_mean(mean_params):
    x, xx = mean_params
    J = -.5*jnp.linalg.inv(xx - outer(x,x))
    h = -2*mvp(J, x)
    return h, J

def mvn_natural_from_standard(standard_params):
    mu, V = standard_params
    P = jnp.linalg.inv(V)
    return mvp(P, mu), -.5*P

def mvn_mean_from_natural(natural_params):
    h, J = natural_params
    J = .5*(J + transpose(J))
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return mu, V + outer(mu, mu)

def mvn_standard_from_natural(natural_params):
    h, J = natural_params
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return mu, V

def mvn_stats(x):
    return x, outer(x, x)

def mvn_dot(a, b):
    return jnp.sum(a[0]*b[0], axis=-1) + jnp.sum(a[1]*b[1], axis=(-1,-2))

def mvn_log_base_measure(x):
    return -.5*x.shape[-1]*jnp.log(2*jnp.pi) * jnp.ones_like(x[...,0])

def mvn_log_partition(natural_params, jitter=.0):
    h, J = natural_params
    D = h.shape[-1]
    J = .5*(J + transpose(J))
    L = jnp.linalg.cholesky(-2*J + jnp.eye(D)*jitter)
    v = jax.scipy.linalg.solve_triangular(L, h, lower=True)
    halflogdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)), -1)
    return .5*vdot(v, v) - halflogdet

def mvn_log_prob(natural_params, x):
    return mvn_dot(natural_params, mvn_stats(x)) - mvn_log_partition(natural_params) + mvn_log_base_measure(x)

def mvn_entropy(natural_params):
    h, J = natural_params
    D = h.shape[-1]
    return .5*D*(1 + jnp.log(2*jnp.pi)) - .5*jnp.linalg.slogdet(-2*J)[1]

def mvn_kl(natural_params_from, natural_params_to):
    return mvn_log_partition(natural_params_to) - mvn_log_partition(natural_params_from) \
        + mvn_dot(tree_sub(natural_params_from, natural_params_to), mvn_mean_from_natural(natural_params_from))

def mvn_sample(key, natural_params, shape=()):
    expand = lambda _: jnp.tile(_, shape + (1,)*_.ndim)
    natural_params = tree_map(expand, natural_params)
    h, J = natural_params
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return jax.random.multivariate_normal(key, mu, V)

def mvn_sample_mean_params(rng, mean_params):
    mu, V = mean_params[0], mean_params[1] - outer(mean_params[0], mean_params[0])
    return jax.random.multivariate_normal(rng, mu, V, method='eigh')

def mvn_mean(natural_params):
    return mvn_standard_from_natural(natural_params)[0]

def mvn_var(natural_params):
    return mvn_standard_from_natural(natural_params)[1]

# marginalise out masked dimensions
def mvn_marginalise(natural_params, mask):
    Jaa = submatrix(natural_params[1], mask, mask)
    Jab = submatrix(natural_params[1], mask, ~mask)
    Jbb = submatrix(natural_params[1], ~mask, ~mask)
    ha, hb = natural_params[0][mask], natural_params[0][~mask]
    L = jnp.linalg.cholesky(-2*Jaa)
    v = jax.scipy.linalg.solve_triangular(L, ha, lower=True)
    M = jax.scipy.linalg.solve_triangular(L, -2*Jab, lower=True)
    J = Jbb + .5*jnp.matmul(M.T, M)
    h = hb + 2*Jab.T.dot(jax.scipy.linalg.solve_triangular(L.T, v, lower=False))
    logZ = .5*vdot(v,v) - jnp.sum(jnp.log(jnp.diag(L)),-1)
    return (h, J), logZ

# condition on masked dimensions
def mvn_condition(natural_params, mask, x):
    Jab = submatrix(natural_params[1], mask, ~mask)
    Jbb = submatrix(natural_params[1], ~mask, ~mask)
    hb = natural_params[0][~mask]
    J = Jbb
    h = hb + 2*Jab.T.dot(x)
    return h, J

def mvn_condition_mean_params(mean_params, mask, x):
    Vall = mean_params[1] - outer(mean_params[0], mean_params[0])
    Vab = submatrix(Vall, mask, ~mask)
    Vaa = submatrix(Vall, mask, mask)
    Vbb = submatrix(Vall, ~mask, ~mask)
    mu_a = mean_params[0][mask]
    mu_b = mean_params[0][~mask]
    L = jnp.linalg.cholesky(Vaa)
    V = Vbb - jnp.matmul(transpose(Vab), invcholp(L, Vab))
    mu = mu_b + jnp.matmul(transpose(Vab), invcholp(L, x-mu_a))
    return mu, V + outer(mu, mu)

def mvn_embed(natural_params, mask):
    D = len(mask)
    h = jnp.zeros(D).at[mask].set(natural_params[0])
    J = jnp.zeros((D,D)).at[outer(mask,mask)].set(natural_params[1].reshape(-1))
    return h, J

def mvn_symmetrize(_):
    return _[0], .5*(transpose(_[1]) + _[1])
