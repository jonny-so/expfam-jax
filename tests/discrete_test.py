import scipy.stats
from expfam.distributions.discrete import *
from expfam.util.random import rngcall
from functools import partial
from jax import grad, vmap
from jax.random import split
from itertools import product

jax.config.update("jax_enable_x64", True)

class TestDiscrete():

    rng = jax.random.PRNGKey(0)
    D1 = 8
    D2 = 4
    natural_params, rng = rngcall(jax.random.normal, rng, (D1,D2))
    meanparams = discrete_mean_from_natural(natural_params)

    def test_meanparams(self):
        p = jnp.exp(self.natural_params) / jnp.sum(jnp.exp(self.natural_params))
        assert jnp.allclose(self.meanparams, p)

    def test_natural_params_meanparams_consistency(self):
        natural_params2 = discrete_natural_from_mean(self.meanparams)
        assert jnp.allclose(
            discrete_normalize(self.natural_params),
            discrete_normalize(natural_params2))

    def test_logZ_meanparams_consistency(self):
        meanparams2 = grad(discrete_log_partition)(self.natural_params)
        assert jnp.allclose(self.meanparams, meanparams2)

    def test_sample_meanparams_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(discrete_sample, self.rng, self.natural_params, (nsamples,))
        stats = vmap(discrete_stats, (0, None))(x, (self.D1, self.D2))
        assert jnp.all(scipy.stats.ttest_1samp((stats - self.meanparams), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(discrete_sample, self.rng, self.natural_params, (nsamples,))
        logprob = vmap(partial(discrete_log_prob, self.natural_params))(x)
        entropy = discrete_entropy(self.natural_params)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_marginalise(self):
        p = jnp.exp(self.natural_params) / jnp.sum(jnp.exp(self.natural_params))
        p1 = p.sum(1)
        p2 = p.sum(0)
        marginal1 = discrete_marginalise(self.natural_params, np.array([False, True]))[0]
        marginal2 = discrete_marginalise(self.natural_params, np.array([True, False]))[0]
        assert jnp.allclose(discrete_mean_from_natural(marginal1), p1)
        assert jnp.allclose(discrete_mean_from_natural(marginal2), p2)

    def test_condition(self):
        p = jnp.exp(self.natural_params) / jnp.sum(jnp.exp(self.natural_params))
        for i, j in product(range(self.D1), range(self.D2)):
            p1 = p[:,i] / jnp.sum(p[:,i])
            p2 = p[j] / jnp.sum(p[j])
            conditional1 = discrete_condition(self.natural_params, np.array([False, True]), jnp.array([i]))
            conditional2 = discrete_condition(self.natural_params, np.array([True, False]), jnp.array([j]))
            assert jnp.allclose(discrete_mean_from_natural(conditional1), p1)
            assert jnp.allclose(discrete_mean_from_natural(conditional2), p2)