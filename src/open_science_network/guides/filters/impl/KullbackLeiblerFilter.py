import optax
from numpyro.optim import optax_to_numpyro
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer import SVI, Trace_ELBO
from open_science_network.guides.filters.FilterInterface import FilterInterface


class KullbackLeiblerFilter(FilterInterface):
    """
    Minimising the KL divergence between local joint p(y_t, x_t|y^{t-1}) and an approximate posterior q(x) in the form
    of a multivariate normal distribution.
    """

    def __init__(self, likelihood, num_iter=100, lr=1e-2, num_particles=10):
        super(KullbackLeiblerFilter, self).__init__(likelihood)
        self.num_iter = num_iter
        self.num_particles = num_particles
        self.optimizer = optax_to_numpyro(optax.chain(optax.adabelief(lr)))

        self._guide = AutoMultivariateNormal(self._model)
        self.svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO(num_particles=self.num_particles))

    def call(self, params, rng_key, obs, mu, sigma):
        result = self.svi.run(rng_key, self.num_iter, obs, params, mu, sigma, progress_bar=False)

        mu_tt = result.params['auto_loc']
        l = result.params['auto_scale_tril']
        sigma_tt = l @ l.T
        return mu_tt, sigma_tt
