import scipy.stats
from expfam.distributions.diagonal_mvn import *
from jaxutil.random import rngcall
from functools import partial
from jax import grad, vmap

jax.config.update("jax_enable_x64", True)

class TestDiagonalMvn():

    rng = jax.random.PRNGKey(0)
    D = 8
    h, rng = rngcall(jax.random.normal, rng, (D,))
    logv, rng = rngcall(jax.random.normal, rng, (D,))
    j = -.5/jnp.exp(logv)
    
    natural_params = h, j
    mean_params = diagonal_mvn_mean_from_natural(natural_params)

    def test_natural_params_mean_params_consistency(self):
        natural_params2 = diagonal_mvn_natural_from_mean(self.mean_params)
        assert jnp.allclose(self.natural_params[0], natural_params2[0])
        assert jnp.allclose(self.natural_params[1], natural_params2[1])

    def test_log_partition_mean_params_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        mean_params2 = grad(diagonal_mvn_log_partition)(self.natural_params)
        assert jnp.allclose(self.mean_params[0], mean_params2[0])
        assert jnp.allclose(self.mean_params[1], mean_params2[1])

    def test_sample_mean_params_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(diagonal_mvn_sample, self.rng, self.natural_params, (nsamples,))
        stats = vmap(diagonal_mvn_stats)(x)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0] - self.mean_params[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1] - self.mean_params[1]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        x, self.rng= rngcall(diagonal_mvn_sample, self.rng, self.natural_params, (nsamples,))
        logprob = vmap(partial(diagonal_mvn_log_prob, self.natural_params))(x)
        entropy = diagonal_mvn_entropy(self.natural_params)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01
