import jax.numpy as jnp
from jax import lax, vmap, random
from functools import partial
import numpyro.distributions as dist
from numpyro import sample, handlers
from numpyro.contrib.control_flow import scan
from natural_models.guides.GuideInterface import GuideInterface
try:
    from numpyro.contrib.nested_sampling import NestedSampler
except Exception as e:
    print(e)


# Papers on different ways for doing KF with nonlinear likelihoods
# https://arxiv.org/pdf/1705.00722.pdf
class KalmanFilterGuide(GuideInterface):
    """
    Expectation maximisation over each state
    """

    def __init__(self, rng_key, forward_step=None):
        super().__init__(rng_key)
        self.forward_step = KalmanFilterGuide.linear_kf if forward_step is None else forward_step
        self.backward_step = KalmanFilterGuide.backward_step

    def call(self, params):
        time_horizon = params['T']
        n_lots = params['L']

        # auxiliary vars for conditioning on observations of state_report (sr) and action
        sr = sample(
            'aux_sr',
            dist.Normal(0., 1.).expand((time_horizon, n_lots, params['U'])).to_event(1).mask(False),
            infer={'is_auxiliary': True}
        )

        events = sample(
            'aux_event',
            dist.Categorical(jnp.zeros(params['E'])).expand([time_horizon, n_lots]).mask(False),
            infer={'is_auxiliary': True}
        )

        sl = jnp.arange(n_lots)  # aux slice vector

        # Mean and covariance of the prior
        loc0 = params['loc0']
        c0 = params['Sigma0']

        with handlers.block():
            bst = partial(self.backward_step, params)

            def forward_filtering(carry, inpt):
                obs, event, t = inpt
                params["ci"] = params["confidence_interval"][t]
                fst = partial(self.forward_step, params)

                loc_tt, sigma_tt = carry  # (t|t)
                loc_t1t = loc_tt + params['momentum'][sl, :, event]  # (t+1|t)
                sigma_t1t = sigma_tt + params['Q']  # (t+1|t)

                rng_keys = random.split(self.rng_key, n_lots)
                loc_t1t1, sigma_t1t1 = vmap(fst)(rng_keys, obs, loc_t1t, sigma_t1t)  # (t+1|t+1)
                return (loc_t1t1, sigma_t1t1), (loc_t1t1, sigma_t1t1)

            (last_loc, last_cov), (forward_loc, forward_cov) = \
                lax.scan(forward_filtering, (loc0, c0), (sr, events, jnp.arange(time_horizon)))

        forward_loc = jnp.concatenate([jnp.expand_dims(loc0, 0), forward_loc], 0)
        forward_cov = jnp.concatenate([jnp.expand_dims(c0, 0), forward_cov], 0)

        def backward_sampling(carry, xs):
            mu_tt1, sigma_tt1 = carry  # (t|t+1)

            mn_dist = dist.MultivariateNormal(loc=mu_tt1, covariance_matrix=sigma_tt1)
            x_k1 = sample('state', mn_dist)  # (k+1) sample

            mu_kk, sigma_kk, event = xs
            u_k = params['momentum'][sl, :, event]

            mu_kk1, sigma_kk1 = vmap(bst)(x_k1, mu_kk, sigma_kk, u_k)

            return (mu_kk1, sigma_kk1), None

        (mu_0, sigma_0), _ = scan(
            backward_sampling,
            (last_loc, last_cov),
            (forward_loc[:-1], forward_cov[:-1], events),
            reverse=True
        )

        sample('state_0', dist.MultivariateNormal(loc=mu_0, covariance_matrix=sigma_0))

    @staticmethod
    def linear_kf(params, rng_key, obs, mu, sigma):
        a = params['A']  # A.T
        r = params['R']
        b = params['b']
        y = obs - mu @ a - b
        h = sigma @ a  # (Σ A.T)
        s = a.T @ h + r
        k = jnp.linalg.solve(s, h)  # K.T transposed kalman gain
        mu_tt = mu + y @ k
        sigma_tt = sigma - h @ k
        return mu_tt, sigma_tt

    @staticmethod
    def backward_step(params, x_k1, mu_kk, sigma_kk, u_k):
        q = params['Q']
        mu_k1k = mu_kk + u_k
        z_k1 = x_k1 - mu_k1k

        s_k = sigma_kk + q
        k_k = jnp.linalg.solve(s_k, sigma_kk)

        mu_kk1 = mu_kk + k_k @ z_k1
        sigma_kk1 = sigma_kk - k_k @ sigma_kk

        return mu_kk1, sigma_kk1
