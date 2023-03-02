import abc
import numpyro.distributions as dist
from numpyro import sample, handlers


class FilterInterface(object):

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def _model(self, obs, params, mu, sigma):
        state = sample('state', dist.MultivariateNormal(loc=mu, covariance_matrix=sigma))
        with handlers.condition(data={self.likelihood.name: obs}):
            self.likelihood(params, state)

    @abc.abstractmethod
    def call(self, params, rng_key, obs, mu, sigma):
        ...

    def __call__(self, params, rng_key, obs, mu, sigma):
        return self.call(params, rng_key, obs, mu, sigma)
