import scipy.stats
from expfam.distributions.mvn_mixture import *
from jaxutil.random import rngcall
from jaxutil.la import mmp
from functools import partial
from jax import grad, vmap
from jax.random import split

jax.config.update("jax_enable_x64", True)

class TestMvnMixture():

    rng = jax.random.PRNGKey(0)
    K = 3
    D = 8
    logp, rng = rngcall(jax.random.normal, rng, (K,))
    h, rng = rngcall(jax.random.normal, rng, (K,D,))
    J, rng = rngcall(jax.random.normal, rng, (K, D,D))
    J = -.5*((mmp(transpose(J), J)) + jnp.eye(D)*1e-2)
    logp -= jax.scipy.special.logsumexp(logp)
    gamma = logp - mvn_log_partition((h, J))
    
    natparams = gamma, h, J
    meanparams = mvn_mixture_mean_from_natural(natparams)

    def test_natparams_meanparams_consistency(self):
        natparams2 = mvn_mixture_natural_from_mean(self.meanparams)
        assert jnp.allclose(self.natparams[0], natparams2[0])
        assert jnp.allclose(self.natparams[1], natparams2[1])

    def test_log_partition_meanparams_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        meanparams2 = mvn_mixture_symmetrize(grad(mvn_mixture_log_partition)(self.natparams))
        assert jnp.allclose(self.meanparams[0], meanparams2[0])
        assert jnp.allclose(self.meanparams[1], meanparams2[1])

    def test_sample_meanparams_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(mvn_mixture_sample, self.rng, self.natparams, (nsamples,))
        stats = vmap(mvn_mixture_stats, (0, None))(x, self.K)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0] - self.meanparams[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1] - self.meanparams[1]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[2] - self.meanparams[2]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        x, self.rng = rngcall(mvn_mixture_sample, self.rng, self.natparams, (nsamples,))
        logprob = vmap(partial(mvn_mixture_log_prob, self.natparams))(x)
        entropy = mvn_mixture_entropy(self.natparams)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01
