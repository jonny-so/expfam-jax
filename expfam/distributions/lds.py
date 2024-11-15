# linear dynamical system:
#   z_0 ~ N(a_0; inv(Q0))
#   z_t ~ N(A z_{t-1} + a; inv(Q))

import jax.numpy as jnp
import numpy
from jax import vmap
from jax.lax import scan
from jax.random import split
from typing import NamedTuple

from expfam.distributions.mvn import *
from jaxutil.tree import *

LdsNaturalParams = tuple[MvnNaturalParams, MvnNaturalParams]
LdsMeanParams = tuple[MvnMeanParams, MvnMeanParams]

class LdsMarginals(NamedTuple):
    singletons: MvnNaturalParams
    pairwise: MvnNaturalParams

def lds_natural_from_standard(standardparams):
    A, a, a0, Q, Q0 = standardparams
    T = A.shape[0] + 1
    prior_natural_params = _prior_natural_params(a0, Q0)
    singleton_natural_params = tree_stack([prior_natural_params] + [tree_scale(prior_natural_params, .0)]*(T-1))
    pairwise_natural_params = vmap(_transition_natural_params)(A, a, Q)
    return singleton_natural_params, pairwise_natural_params

def lds_mean_from_natural(natural_params):
    return tuple(map(vmap(mvn_mean_from_natural), lds_marginals(natural_params)))

def lds_stats(z):
    stats_z = mvn_stats(z)
    stats_zz = mvn_stats(jnp.concatenate([z[..., :-1, :], z[..., 1:, :]], -1))
    return stats_z, stats_zz

def lds_dot(eta, stats):
    return jnp.sum(vmap(mvn_dot)(eta[0], stats[0]), 0) \
        + jnp.sum(vmap(mvn_dot)(eta[1], stats[1]), 0)

def lds_log_base_measure(z):
    T, D = z.shape[-2:]
    return -.5*T*D*jnp.log(2*jnp.pi) * jnp.ones_like(z[..., 0, 0])

def lds_log_partition(natural_params):
    eta_null = tree_map(jnp.zeros_like, tree_first(natural_params[0]))
    eta_fwd, log_partition = scan(_forward_step, init=(eta_null, .0),
        xs=(tree_drop_last(natural_params[0]), natural_params[1]))[0]
    return log_partition + mvn_log_partition(tree_add(eta_fwd, tree_last(natural_params[0])))

def lds_log_prob(natural_params, z):
    return lds_dot(natural_params, lds_stats(z)) - lds_log_partition(natural_params) + lds_log_base_measure(z)

def lds_entropy(natural_params):
    mean_params = lds_mean_from_natural(natural_params)
    E_z = mean_params[0][0]
    T, D = E_z.shape[-2:]
    E_log_base_measure = -.5*T*D*jnp.log(2*jnp.pi) * jnp.ones_like(E_z[..., 0, 0])
    return lds_log_partition(natural_params) - lds_dot(natural_params, mean_params) - E_log_base_measure

def lds_sample(key, natural_params):
    assert(natural_params[0][0].ndim == 2)
    q_z, q_zz = lds_marginals(natural_params)
    T, D = q_z[0].shape
    key = split(key, T)
    z_0 = mvn_sample(key[0], tree_first(q_z))
    def _step(z_in, _):
        k, q_zz_t = _
        q_z_out = mvn_condition(q_zz_t, _mask_first(D, 2*D), z_in)
        z_out = mvn_sample(k, q_z_out)
        return z_out, z_out
    z_t = scan(_step, z_0, (key[1:], q_zz))[1]
    return tree_prepend(z_t, z_0)

# return marginal mvn natural params for singleton and (neighbouring) pairwise cliques
def lds_marginals(natural_params):
    eta_null = tree_map(jnp.zeros_like, tree_first(natural_params[0]))
    eta_fwd = tree_prepend(
        scan(_forward_step, init=(eta_null, .0),
            xs=(tree_drop_last(natural_params[0]), natural_params[1]))[1],
        (eta_null, .0))[0]
    eta_bwd = tree_append(
        scan(_backward_step, init=(eta_null, .0),
             xs=(tree_drop_first(natural_params[0]), natural_params[1]), reverse=True)[1],
        (eta_null, .0))[0]
    q_z = tree_sum([eta_fwd, eta_bwd, natural_params[0]])
    q_zz = vmap(_join_natural_params)(
        tree_drop_last(tree_sum([eta_fwd, natural_params[0]])),
        tree_drop_first(tree_sum([eta_bwd, natural_params[0]])),
        natural_params[1])
    return LdsMarginals(q_z, q_zz)

_mask_first = lambda n,d: numpy.arange(d) < n
_mask_last = lambda n,d: numpy.arange(d) >= d-n

# singleton factor natural params N(z_0; b_0, inv(Q0))
def _prior_natural_params(a0, Q0):
    J = -.5*Q0
    h = Q0.dot(a0)
    return h, J

# pairwise factor natural params N(z_2; A z_1 + a; inv(Q)), with layout [z_1; z_2]
def _transition_natural_params(A, a, Q):
    QA = jnp.matmul(Q, A)
    Jaa, Jab, Jbb = -.5*jnp.matmul(A.T, QA), .5*QA.T, -.5*Q
    J = jnp.block([[Jaa, Jab], [Jab.T, Jbb]])
    h = jnp.concatenate([-QA.T.dot(a), Q.dot(a)])
    return h, J

def _join_natural_params(eta_a, eta_b, eta_ab):
    Da, Db = len(eta_a[0]), len(eta_b[0])
    eta_a = mvn_embed(eta_a, _mask_first(Da,Da+Db))
    eta_b = mvn_embed(eta_b, _mask_last(Db,Da+Db))
    return tree_sum([eta_a, eta_b, eta_ab])

def _forward_message(msg_in, eta_transition):
    mask = _mask_first(len(msg_in[0][0]), len(eta_transition[0]))
    eta_out, log_partition = mvn_marginalise(tree_add(mvn_embed(msg_in[0], mask), eta_transition), mask)
    return eta_out, log_partition + msg_in[1]

def _backward_message(msg_in, eta_transition):
    mask = _mask_last(len(msg_in[0][0]), len(eta_transition[0]))
    eta_out, log_partition = mvn_marginalise(tree_add(mvn_embed(msg_in[0], mask), eta_transition), mask)
    return eta_out, log_partition + msg_in[1]

def _forward_step(msg_in, eta_t):
    eta_in, scale_in = msg_in
    eta_z_t, eta_zz_t = eta_t
    msg_out = _forward_message((tree_add(eta_in, eta_z_t), scale_in), eta_zz_t)
    return msg_out, msg_out
    
def _backward_step(msg_in, eta_t):
    eta_in, scale_in = msg_in
    eta_z_t, eta_zz_t = eta_t
    msg_out = _backward_message((tree_add(eta_in, eta_z_t), scale_in), eta_zz_t)
    return msg_out, msg_out
