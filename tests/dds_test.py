import jax
import scipy.stats
import jax.numpy as jnp
import jax.random as jr
from expfam.distributions.discrete import *
from expfam.distributions.dds import *
from jaxutil.random import rngcall
from jax import grad

jax.config.update("jax_enable_x64", True)

class TestDds():

    rng = jax.random.PRNGKey(0)

    K, T = 3, 100
    singleton_natparams, rng = rngcall(jr.normal, rng, (T,K))
    pairwise_natparams, rng = rngcall(jr.normal, rng, (T-1,K,K))
    natparams = singleton_natparams, pairwise_natparams

    def test_meanparams_consistency(self):
        mu_s, mu_ss = dds_mean_from_natural(self.natparams)

        # check all singleton marginals sum to 1
        assert jnp.allclose(mu_s.sum(1), 1.0)

        # check singleton / pairwise consistency
        assert jnp.allclose(mu_s[:-1], mu_ss.sum(2))
        assert jnp.allclose(mu_s[1:], mu_ss.sum(1))

    def test_marginals_meanparams_consistency(self):
        mu_s, mu_ss = dds_mean_from_natural(self.natparams)

        singleton_marginals, pairwise_marginals = dds_marginals(self.natparams)

        assert jnp.allclose(mu_s, vmap(discrete_mean_from_natural)(singleton_marginals))
        assert jnp.allclose(mu_ss, vmap(discrete_mean_from_natural)(pairwise_marginals))

    def test_log_partition_meanparams_consistency(self):
        mu_s, mu_ss = dds_mean_from_natural(self.natparams)
        g_s, g_ss = grad(dds_log_partition)(self.natparams)

        # note that the DDS representation is not minimal
        assert jnp.allclose(mu_s, g_s)
        assert jnp.allclose(mu_ss, g_ss)

    def test_sample_meanparams_consistency(self):
        nsamples = 100000
        mu_s = dds_mean_from_natural(self.natparams)[0]
        s = vmap(dds_sample, (0, None))(split(self.rng, nsamples), self.natparams)
        stats = tree_map(partial(jnp.mean, axis=0), dds_stats(s, self.K))
        assert jnp.abs(stats[0] - mu_s).mean() < .01

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 100000
        s = vmap(dds_sample, (0, None))(split(self.rng, nsamples), self.natparams)
        logprob = vmap(partial(dds_log_prob, self.natparams))(s)
        entropy = dds_entropy(self.natparams)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_marginals_entropy_consistency(self):
        marginals = dds_marginals(self.natparams)
        singleton_entropies = vmap(discrete_entropy)(marginals[0])
        pairwise_entropies = vmap(discrete_entropy)(marginals[1])
        entropy = dds_entropy(self.natparams)
        assert jnp.isclose(entropy, pairwise_entropies.sum() - singleton_entropies[1:-1].sum())