import jax.numpy as jnp
from jax import random
from numpyro.infer import SA, MCMC
from natural_models.guides.filters.FilterInterface import FilterInterface
try:
    from numpyro.contrib.nested_sampling import NestedSampler
except Exception as e:
    print(e)


class MomentMatchingFilter(FilterInterface):
    """
    Obtaining posterior mean and covariance using sampling methods and matching the moments
    """

    def __init__(self, likelihood, num_samples=1000, num_warmup=10000, adapt_state_size=100, num_chains=1,
                 samples_per_step=10, method="SA"):
        super(MomentMatchingFilter, self).__init__(likelihood)
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.adapt_state_size = adapt_state_size
        self.samples_per_step = samples_per_step  # Only for Nested Sampling
        self.num_chains = num_chains
        self.method = method

        if self.method == 'SA':
            kernel = SA(
                self._model, adapt_state_size=self.adapt_state_size
            )
            self.mcmc = MCMC(
                kernel,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                chain_method='vectorized',
                progress_bar=False
            )

        elif self.method == 'nested':
            self.num_chains = 1
            self.ns = NestedSampler(
                self._model,
                constructor_kwargs={
                    'samples_per_step': self.samples_per_step,
                    'num_parallel_samplers': self.num_chains
                }
            )
        else:
            raise NotImplementedError

    def call(self, params, rng_key, obs, mu, sigma):
        if self.method == 'SA':
            self.mcmc.run(rng_key, obs, params, mu, sigma)
            samples = self.mcmc.get_samples()
        elif self.method == 'nested':
            rng_key, _rng_key = random.split(rng_key)
            self.ns.run(_rng_key, obs, params, mu, sigma)
            rng_key, _rng_key = random.split(rng_key)
            samples = self.ns.get_samples(_rng_key, num_samples=self.num_samples)
        else:
            raise Exception(f"Unsupported method: {self.method}")
        mu_tt = samples['state'].mean(0)
        sigma_tt = jnp.cov(samples['state'], rowvar=False)
        return mu_tt, sigma_tt
