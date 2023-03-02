import jax.numpy as jnp
from jax.scipy.special import log_ndtr, xlogy
from jax import lax, random
from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints

from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    validate_sample
)


# from https://arxiv.org/abs/0911.2093
class SkewNormal(Distribution):
    arg_constraints = {"loc": constraints.real, "skew": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale", "skew"]

    def __init__(self, loc=0.0, skew=0.0, scale=1.0, *, validate_args=None):
        self.loc, self.skew, self.scale = promote_shapes(loc, skew, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(SkewNormal, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )
        self.delta = self.scale**2 * self.skew /(1 + self.skew**2 * self.scale ** 2)
        self.a = self.scale/(1 + self.skew**2 * self.scale ** 2)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps1 = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )

        eps2 = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )

        return self.loc + self.delta * eps1 + self.a * eps2

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        log_prob = -0.5 * value_scaled**2 - normalize_term
        log_prob += log_ndtr(self.skew * value_scaled)
        return log_prob

    @property
    def mean(self):
        return jnp.broadcast_to(
            self.loc + self.scale * self.delta * jnp.sqrt(2/jnp.pi),
            self.batch_shape
        )

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale**2 * (1 - 2 * self.delta**2/jnp.pi), self.batch_shape)


# LUB: Lambda upper bound distribution - Ekum, M. I., I. A. Adeleke, and E. E. E. Akarawak. "Lambda Upper Bound Distribution, Some Properties and Application." Benin Journal of Statistics 3.1 (2020): 12-40.
class LUB(Distribution):
    lam = 1.
    arg_constraints = {"concentration": constraints.positive, "lam": constraints.positive}
    support = constraints.interval(0., lam)

    def __init__(self, concentration=1.0, lam=1.0, *, validate_args=None):
        self.concentration, self.lam = promote_shapes(concentration, lam)
        self.minval = jnp.finfo(self.lam).tiny

        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(lam))
        super(LUB, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        minval = jnp.finfo(jnp.result_type(float)).tiny
        u = random.uniform(key, shape=sample_shape + self.batch_shape, minval=minval)
        log_sample = jnp.log1p(-u) / self.concentration
        return jnp.clip(self.lam * jnp.exp(log_sample), a_min=self.minval)

    @validate_sample
    def log_prob(self, value):
        lam = jnp.clip(self.lam, a_min=self.minval)
        normalize_term = self.concentration * jnp.log(lam) + jnp.log(self.concentration)
        return (self.concentration - 1) * jnp.log(value) + normalize_term

    @property
    def mean(self):
        return jnp.broadcast_to(
            self.concentration * self.lam /(self.concentration + 1),
            self.batch_shape
        )

    @property
    def variance(self):
        return jnp.broadcast_to(self.concentration * self.lam ** 2 / (self.concentration + 2) - self.mean ** 2, self.batch_shape)