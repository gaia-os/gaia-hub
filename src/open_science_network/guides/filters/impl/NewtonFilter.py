import jax.numpy as jnp
from numpyro.infer.util import log_density
from jax import lax, grad, hessian
from open_science_network.guides.filters.FilterInterface import FilterInterface


class NewtonFilter(FilterInterface):

    def __init__(self, likelihood, num_iter=10):
        super(NewtonFilter, self).__init__(likelihood)
        self.num_iter = num_iter

    def call(self, params, rng_key, obs, mu, sigma):

        def log_ll(x):
            log_d, _ = log_density(self._model, (obs, params, mu, sigma), {}, {'state': x})
            return log_d

        h = hessian(log_ll)
        g = grad(log_ll)

        def scan_fn(carry, t):
            _mu = carry
            mu_ = _mu - jnp.linalg.solve(h(_mu), g(_mu))
            return mu_, None

        mu_tt, _ = lax.scan(scan_fn, mu, jnp.arange(self.num_iter))
        h = h(mu_tt)
        sigma_tt = - sigma @ jnp.linalg.solve(sigma - h, h)

        return mu_tt, sigma_tt
