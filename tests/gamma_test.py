import scipy.stats
from expfam.distributions.gamma import *
from jaxutil.random import rngcall
from functools import partial
from jax import grad, vmap

jax.config.update("jax_enable_x64", True)

class TestGamma():

    rng = jax.random.PRNGKey(0)
    def _init_standard_params(rng):
        a, rng = rngcall(jax.random.gamma, rng, 2.)
        b, rng = rngcall(jax.random.gamma, rng, 2.)
        return a, b
    
    standard_params = _init_standard_params(rng)
    natural_params = gamma_natural_from_standard(standard_params)
    mean_params = gamma_mean_from_natural(natural_params)

    def test_natural_params_mean_params_consistency(self):
        natural_params2 = gamma_natural_from_mean(self.mean_params)
        assert jnp.allclose(self.natural_params[0], natural_params2[0])
        assert jnp.allclose(self.natural_params[1], natural_params2[1])

    def test_log_partition_mean_params_consistency(self):
        mean_params2 = grad(gamma_log_partition)(self.natural_params)
        assert jnp.allclose(self.mean_params[0], mean_params2[0])
        assert jnp.allclose(self.mean_params[1], mean_params2[1])

    def test_sample_mean_params_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(gamma_sample, self.rng, self.natural_params, (nsamples,))
        stats = vmap(gamma_stats)(x)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0] - self.mean_params[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1] - self.mean_params[1]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(gamma_sample, self.rng, self.natural_params, (nsamples,))
        logprob = vmap(partial(gamma_log_prob, self.natural_params))(x)
        entropy = gamma_entropy(self.natural_params)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01
