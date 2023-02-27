from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Minimize
from numpyro.infer.autoguide import AutoLaplaceApproximation
from natural_models.guides.filters.FilterInterface import FilterInterface


class LaplaceKF(FilterInterface):
    """
    Laplace approxiamtion for valamn filter. NOTE: Cannot be used currently as Minimize method does not support reverse mode differentiation
    """

    def __init__(self, likelihood, num_iter=10):
        super(LaplaceKF, self).__init__(likelihood)
        self.optimizer = Minimize(method="BFGS")
        self.num_iter = num_iter
        self.num_particles = 1

        self._guide = AutoLaplaceApproximation(self._model)
        self.svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO(num_particles=self.num_particles))

    def call(self, params, rng_key, obs, mu, sigma):
        result = self.svi.run(rng_key, self.num_iter, obs, params, mu, sigma, progress_bar=False)
        transform = self._guide.get_transform(result.params)
        mu_tt = transform.loc
        l = transform.scale_tril
        sigma_tt = l @ l.T
        return mu_tt, sigma_tt
