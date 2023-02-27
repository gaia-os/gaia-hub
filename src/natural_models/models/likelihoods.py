from numpyro import deterministic
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro import sample
from natural_models.models.distributions import SkewNormal, LUB
from numpyro.distributions import LogNormal, Gamma
from jax import nn
import jax


class Standard:

    @staticmethod
    def poisson(params, state):
        rate = nn.softplus(state @ params['A'] + params['b'])
        sample('sr', dist.Poisson(rate).to_event(1))

    @staticmethod
    def normal(params, state):
        scale = params['scale']
        loc = state @ params['A'] + params['b']
        sample('sr', dist.Normal(loc, scale).to_event(1))

    @staticmethod
    def normal_softmax(params, state, axis=-1):
        scale = params['scale']
        loc = nn.softmax(state @ params['A'] + params['b'], axis=axis)
        sample('sr', dist.Normal(loc, scale).to_event(1))

    @staticmethod
    def normal_logsumexp(params, state, axis=-1):
        scale = params['scale']
        loc = nn.logsumexp(state @ params['A'] + params['b'], axis=axis)
        sample('sr', dist.Normal(loc, scale))

    @staticmethod
    def skew_normal(params, state):
        alpha = params['alpha']
        scale = params['scale']
        loc = state @ params['A'] + params['b']
        sample('sr', SkewNormal(loc, scale, alpha).to_event(1))


class NonLinearNormal:

    def __init__(self, f, name='sr'):
        self.f = f
        self.name = name

    def dist(self, params, state):
        loc, scale = self.f(params, state)
        return dist.Normal(loc, scale).to_event(1)

    def __call__(self, params, state):
        sample(self.name, self.dist(params, state))


class ThreeAgro:

    def __call__(self, params, state):
        n_veg_types = params['V']
        n_states = params['S']
        plant_biomass = state[..., :n_veg_types]
        soil_biomass = state[..., n_veg_types:n_states]
        confidence_interval = params["ci"]

        # Sample plant biomass carbon
        plant_biomass_carbon_mean = nn.softmax(plant_biomass, axis=-1) * params['plant_carbon_coefficients']

        # Sample soil biomass carbon
        soil_biomass_carbon_mean = nn.softmax(soil_biomass, axis=-1) * params['soil_carbon_coefficients']

        # Total plant biomass
        total_plant_biomass_mean = jnp.expand_dims(plant_biomass.sum(axis=-1), axis=-1)

        mean = [plant_biomass_carbon_mean, soil_biomass_carbon_mean, total_plant_biomass_mean]
        mean = jnp.concatenate(mean, axis=-1)
        std = ThreeAgro.std_from(confidence_interval)
        return mean, std

    @staticmethod
    def std_from(confidence_interval):
        return confidence_interval / 2  # TODO check if std = scale parameter and update if not


class ElementSix:

    def __call__(self, soil_biomass, plant_density, plant_height, yield_density, params, mask, data_level=0):
        soil_biomass_to_soc = params['soil_biomass_to_soc']
        plant_height_to_biomass = params['plant_height_to_biomass']
        plant_biomass_to_carbon = params['plant_biomass_to_carbon']

        plant_unit_biomass = jnp.clip(plant_height * plant_height_to_biomass, a_min=1e-16)
        biomass_density = plant_density * plant_unit_biomass
        deterministic('biomass_carbon_per_m2', biomass_density * plant_biomass_to_carbon)

        if data_level > 2:
            log_soc_coeff = jnp.log(soil_biomass_to_soc)
            sample('soil_organic_carbon', LogNormal(jnp.log(soil_biomass) + log_soc_coeff, params['soc_std']))

        if data_level > 1:
            count_prec = params['count_prec'].transpose()
            rng_key = jax.random.PRNGKey(0)  # TODO
            sample(
                'obs_plant_density',
                Gamma(jnp.clip(plant_density, a_min=1e-16) * count_prec, count_prec)
                .to_event(1)
                .mask(True if mask is None else mask['obs_plant_density']),
                rng_key=rng_key
            )

            height_prec = params['height_prec'].transpose()
            rng_key = jax.random.PRNGKey(0)  # TODO
            sample(
                'obs_plant_height',
                Gamma(jnp.clip(plant_height, a_min=1e-16) * height_prec, height_prec)
                .to_event(1)
                .mask(True if mask is None else mask['obs_plant_height']),
                rng_key=rng_key
            )

        if data_level > 0:
            rng_key = jax.random.PRNGKey(0)  # TODO
            sample(
                'obs_yield_density',
                LUB(params['harvest_concentration'], jnp.clip(yield_density, a_min=1e-16))
                .to_event(1)
                .mask(True if mask is None else mask['obs_yield_density']),
                rng_key=rng_key
            )

        lai = params['lai_beta'] * biomass_density.sum(-1) / 100 + params['lai_alpha']
        sample("lai", LogNormal(jnp.log(lai), params['lai_scale'] / 10).mask(True if mask is None else mask['lai']))
