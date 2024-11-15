import scipy.stats
from expfam.distributions.niw import *
from jaxutil.random import rngcall
from functools import partial
from jax import grad, vmap

jax.config.update("jax_enable_x64", True)

class TestNiw():

    rng = jax.random.PRNGKey(0)

    def _init_standardparams(rng, D):
        W, rng = rngcall(jax.random.normal, rng, (D,D))
        Psi = W @ W.T
        delta, rng = rngcall(jax.random.normal, rng, (D,))
        gamma, rng = rngcall(jax.random.gamma, rng, 2.0)
        alpha, rng = rngcall(jax.random.randint, rng, (), D-1, D+10)
        alpha = alpha.astype(jnp.float64)
        return Psi, delta, gamma, alpha    
    
    standard_params = _init_standardparams(rng, D=5)
    natural_params = niw_natural_from_standard(standard_params)
    mean_params = niw_mean_from_natural(natural_params)

    def test_natural_params_mean_params_consistency(self):
        natural_params2 = niw_natural_from_mean(self.mean_params)
        assert jnp.allclose(self.natural_params[0], natural_params2[0])
        assert jnp.allclose(self.natural_params[1], natural_params2[1])

    def test_log_partition_mean_params_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        mean_params = niw_symmetrize(grad(niw_log_partition)(self.natural_params))
        assert jnp.allclose(self.mean_params[0], mean_params[0])
        assert jnp.allclose(self.mean_params[1], mean_params[1])

    def test_log_partition_dual_nat_params_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        natural_params = grad(lambda _: niw_log_partition_dual(niw_symmetrize(_)))(self.mean_params)
        assert jnp.allclose(self.natural_params[0], natural_params[0])
        assert jnp.allclose(self.natural_params[1], natural_params[1])

    def test_entropy(self):
        Psi, delta, gamma, alpha = self.standard_params
        V = jnp.array(scipy.stats.invwishart(int(alpha), Psi, seed=0).rvs(size=10000))
        mu = jax.random.multivariate_normal(self.rng, delta, V/gamma)
        logprob = vmap(partial(niw_log_prob, self.natural_params))((mu, V))
        entropy = niw_entropy(self.natural_params)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01
