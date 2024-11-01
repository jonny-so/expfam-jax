# discrete dynamical system:
#   p(s_0 = i) = [a_0]_i
#   p(s_t = j | s_{t-1} = i) = A_{ij}

import jax
import jax.numpy as jnp
from typing import NamedTuple
from jax import vmap
from jax.lax import scan
from jax.random import split

from expfam.distributions.discrete import *
from expfam.util.tree import *

DdsNaturalParams = tuple[DiscreteNaturalParams, DiscreteNaturalParams]
DdsMeanParams = tuple[DiscreteMeanParams, DiscreteMeanParams]

class DdsMarginals(NamedTuple):
    singletons: DiscreteNaturalParams
    pairwise: DiscreteNaturalParams

# p(s0 = i) = a0[i]
# p(s_{t+1} = j | s_t = i) = A[t,i,j]
def dds_natural_from_standard(standardparams):
    A, a0 = standardparams
    T = A.shape[-3] + 1
    singleton_natural_params = jnp.pad(jnp.log(a0)[...,None,:], [(0,0)]*(a0.ndim-1) + [(0,T-1), (0,0)])
    pairwise_natural_params = jnp.log(A)
    return singleton_natural_params, pairwise_natural_params

def dds_mean_from_natural(natural_params):
    return tuple(map(vmap(discrete_mean_from_natural), dds_marginals(natural_params)))

def dds_stats(x, K):
    ds = jax.nn.one_hot(x, K) # [..., T, K]
    dss = ds[...,:-1,:,None] * ds[...,1:,None,:]
    return ds, dss

def dds_dot(natural_params, stats):
    return jnp.sum(vmap(discrete_dot)(natural_params[0], stats[0]), 0) \
        + jnp.sum(vmap(discrete_dot)(natural_params[1], stats[1]), 0)

def dds_log_base_measure(x):
    return jnp.zeros_like(x[...,0])

def dds_log_partition(natural_params):
    eta_null = tree_map(jnp.zeros_like, first(natural_params[0]))
    eta_fwd, log_partition = scan(_forward_step, init=(eta_null, .0),
        xs=(drop_last(natural_params[0]), natural_params[1]))[0]
    return log_partition + discrete_log_partition(eta_fwd + natural_params[0][-1])

def dds_log_prob(natural_params, x):
    assert natural_params[0].shape[:-1] == x.shape
    K = natural_params[0].shape[-1]
    return dds_dot(dds_stats(x, K), natural_params) - dds_log_partition(natural_params) + dds_log_base_measure(x)

def dds_entropy(natural_params):
    mean_params = dds_mean_from_natural(natural_params)
    E_log_base_measure = .0
    return -dds_dot(natural_params, mean_params) + dds_log_partition(natural_params) - E_log_base_measure

def dds_sample(key, natural_params):
    assert(natural_params[0].ndim == 2)
    q_s, q_ss = dds_marginals(natural_params)
    T = q_s.shape[0]
    key = split(key, T)
    s_0 = discrete_sample(key[0], q_s[...,0,:])
    def _step(s_in, _):
        k, q_ss_t = _
        q_s_out = discrete_condition(q_ss_t, _mask_first(), s_in)
        s_out = discrete_sample(k, q_s_out)
        return s_out, s_out    
    s_t = scan(_step, s_0, (key[1:], q_ss))[1]
    # discrete distribution has trailing singleton dimension in 1d case
    return tree_prepend(s_t, s_0).squeeze(-1)

def dds_kl(natural_params_from, natural_params_to):
    return (dds_dot(tree_sub(natural_params_from, natural_params_to), dds_mean_from_natural(natural_params_from))
            + dds_log_partition(natural_params_to) - dds_log_partition(natural_params_from))

def dds_marginals(natural_params):
    eta_null = jnp.zeros_like(first(natural_params[0]))
    eta_fwd = tree_prepend(
        scan(_forward_step, init=(eta_null, .0),
            xs=(tree_drop_last(natural_params[0]), natural_params[1]))[1],
        (eta_null, .0))[0]
    eta_bwd = tree_append(
        scan(_backward_step, init=(eta_null, .0),
             xs=(tree_drop_first(natural_params[0]), natural_params[1]), reverse=True)[1],
        (eta_null, .0))[0]
    q_s = eta_fwd + eta_bwd + natural_params[0]
    q_ss = vmap(_join_natural_params)(
        drop_last(eta_fwd + natural_params[0]),
        drop_first(eta_bwd + natural_params[0]),
        natural_params[1])
    return DdsMarginals(q_s, q_ss)

_mask_first = lambda: np.array([True, False])
_mask_last = lambda: np.array([False, True])

def _join_natural_params(eta_a, eta_b, eta_ab):
    return eta_a[:,None] + eta_b[None] + eta_ab

def _forward_message(msg_in, eta_transition):
    eta_out, log_partition = discrete_marginalise(msg_in[0][:,None] + eta_transition, _mask_first())
    return eta_out, log_partition + msg_in[1]

def _backward_message(msg_in, eta_transition):
    eta_out, log_partition = discrete_marginalise(msg_in[0][None] + eta_transition, _mask_last())
    return eta_out, log_partition + msg_in[1]

def _forward_step(msg_in, eta_t):
    eta_in, scale_in = msg_in
    eta_s_t, eta_ss_t = eta_t
    msg_out = _forward_message((tree_add(eta_in, eta_s_t), scale_in), eta_ss_t)
    return msg_out, msg_out
    
def _backward_step(msg_in, eta_t):
    eta_in, scale_in = msg_in
    eta_s_t, eta_ss_t = eta_t
    msg_out = _backward_message((tree_add(eta_in, eta_s_t), scale_in), eta_ss_t)
    return msg_out, msg_out
