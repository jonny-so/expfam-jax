import scipy.stats
import numpy as np
import numpy.random as npr
from expfam.distributions.mvn import *
from jaxutil.random import rngcall
from functools import partial
from jax import grad, vmap

jax.config.update("jax_enable_x64", True)

class TestMvn():

    rng = jax.random.PRNGKey(0)
    D = 8
    h, rng = rngcall(jax.random.normal, rng, (D,))
    J, rng = rngcall(jax.random.normal, rng, (D,D))
    J = -.5*((J.T @ J) + jnp.eye(D)*1e-2)
    
    natural_params = h, J
    mean_params = mvn_mean_from_natural(natural_params)

    def test_natural_params_mean_params_consistency(self):
        natural_params2 = mvn_natural_from_mean(self.mean_params)
        assert jnp.allclose(self.natural_params[0], natural_params2[0])
        assert jnp.allclose(self.natural_params[1], natural_params2[1])

    def test_natural_params_standard_params_consistency(self):
        standard_params = mvn_standard_from_natural(self.natural_params)
        natural_params2 = mvn_natural_from_standard(standard_params)
        assert jnp.allclose(self.natural_params[0], natural_params2[0])
        assert jnp.allclose(self.natural_params[1], natural_params2[1])

    def test_log_partition_mean_params_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        mean_params2 = mvn_symmetrize(grad(mvn_log_partition)(self.natural_params))
        assert jnp.allclose(self.mean_params[0], mean_params2[0])
        assert jnp.allclose(self.mean_params[1], mean_params2[1])

    def test_sample_mean_params_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(mvn_sample, self.rng, self.natural_params, (nsamples,))
        stats = vmap(mvn_stats)(x)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0] - self.mean_params[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1] - self.mean_params[1]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        x, self.rng= rngcall(mvn_sample, self.rng, self.natural_params, (nsamples,))
        logprob = vmap(partial(mvn_log_prob, self.natural_params))(x)
        entropy = mvn_entropy(self.natural_params)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_marginalise(self):
        mask = npr.permutation(np.array([True]*3 + [False]*(self.D-3)))
        marginal_natural_params = mvn_marginalise(self.natural_params, mask)[0]
        marginal_mean_params2 = self.mean_params[0][~mask], submatrix(self.mean_params[1], ~mask, ~mask)
        marginal_natural_params2 = mvn_natural_from_mean(marginal_mean_params2)
        assert jnp.allclose(marginal_natural_params[0], marginal_natural_params2[0])
        assert jnp.allclose(marginal_natural_params[1], marginal_natural_params2[1])
