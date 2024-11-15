import scipy.stats

from expfam.distributions.lds import *
from expfam.distributions.lds import _mask_first, _mask_last

from jaxutil.la import *
from jaxutil.random import rngcall
from jax import grad

jax.config.update("jax_enable_x64", True)

class TestLds():

    def _random_precision(rng, scale, shape):
        W = jax.random.normal(rng, shape)
        P = jnp.linalg.inv(mmp(W, transpose(W))*.01 + jnp.eye(shape[-1])*.1)
        P /= jnp.trace(P, axis1=-1, axis2=-2)[...,None,None] * P.shape[-1]
        return P*scale

    D = 2
    T = 100
    rng = jax.random.PRNGKey(0)
    A, rng = rngcall(jax.random.normal, rng, (T-1,D,D))
    a, rng = rngcall(jax.random.normal, rng, (T-1,D))
    a0, rng = rngcall(jax.random.normal, rng, (D,))
    Q, rng = rngcall(_random_precision, rng, 10., (T-1,D,D))
    Q0, rng = rngcall(_random_precision, rng, 10., (D,D))

    standard_params = A, a, a0, Q, Q0
    natural_params = lds_natural_from_standard(standard_params)
    obs_h, rng = rngcall(jax.random.normal, rng, (T,D))
    obs_J, rng = rngcall(_random_precision, rng, -.5*.1, (T,D,D))
    natural_params = tree_add(natural_params[0], (obs_h, obs_J)), natural_params[1]

    def test_meanparams_consistency(self):
        mu_z, mu_zz = lds_mean_from_natural(self.natural_params)
        print(mu_z[0].shape)
        print(self.natural_params[0][0].shape)
        T, D = mu_z[0].shape
        for t in range(T-1):
            # check means
            print(t)
            print(mu_z[0][t])
            print(mu_zz[0][t, _mask_first(D, 2*D)])
            assert jnp.allclose(mu_z[0][t], mu_zz[0][t, _mask_first(D, 2*D)])
            assert jnp.allclose(mu_z[0][t+1], mu_zz[0][t, _mask_last(D, 2*D)])
            # check covariances
            assert jnp.allclose(mu_z[1][t], submatrix(mu_zz[1][t], _mask_first(D, 2*D), _mask_first(D, 2*D)))
            assert jnp.allclose(mu_z[1][t+1], submatrix(mu_zz[1][t], _mask_last(D, 2*D), _mask_last(D, 2*D)))

    # the LDS representation is not minimal, so we check all duplicated natural_params
    def test_log_partition_meanparams_consistency(self):
        mu_z, mu_zz = lds_mean_from_natural(self.natural_params)
        g_z, g_zz = grad(lds_log_partition)(self.natural_params)
        T, D = mu_z[0].shape

        # check means
        for t in range(T):
            assert jnp.allclose(mu_z[0][t], g_z[0][t])
            if t > 0:
                assert jnp.allclose(mu_z[0][t], g_zz[0][t-1, _mask_last(D, 2*D)])
            if t < T - 1:
                assert jnp.allclose(mu_z[0][t], g_zz[0][t, _mask_first(D, 2*D)])

        # check covariances
        for t in range(T):
            assert jnp.allclose(mu_z[1][t], symmetrize(g_z[1][t]))
            if t < T - 1:
                assert jnp.allclose(mu_zz[1][t], symmetrize(g_zz[1][t]))

    def test_sample_meanparams_consistency(self):
        nsamples = 100000
        mu_z, mu_zz = lds_mean_from_natural(self.natural_params)
        z = vmap(lds_sample, (0, None))(split(self.rng, nsamples), self.natural_params)
        stats = lds_stats(z)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0][0] - mu_z[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0][1] - mu_z[1]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1][0] - mu_zz[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1][1] - mu_zz[1]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 100000
        z = vmap(lds_sample, (0, None))(split(self.rng, nsamples), self.natural_params)
        logprob = vmap(partial(lds_log_prob, self.natural_params))(z)
        entropy = lds_entropy(self.natural_params)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_marginals_entropy_consistency(self):
        marginals = lds_marginals(self.natural_params)
        singleton_entropies = vmap(mvn_entropy)(marginals[0])
        pairwise_entropies = vmap(mvn_entropy)(marginals[1])
        entropy = lds_entropy(self.natural_params)
        assert jnp.isclose(entropy, pairwise_entropies.sum() - singleton_entropies[1:-1].sum())