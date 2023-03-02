from meteostat import Stations, Daily
from datetime import datetime
from open_science_network.ontology.v1.measurement.base.agriculture.PlantDensity import HempDensity
from open_science_network.ontology.v1.measurement.base.agriculture.PlantHeight import HempHeight
from open_science_network.ontology.v1.measurement.base.agriculture.Yield import HempYield
from open_science_network.ontology.v1.genetics.base.plant.Plant import PlantSpecies
from open_science_network.ontology.v1.management.base.agriculture.Harvest import HarvestCrops as HarvestCrops
from open_science_network.ontology.v1.management.base.agriculture.BioChar import UseBioChar as UseBioChar
from open_science_network.ontology.v1.management.base.agriculture.Planting import PlantingSeeds as PlantingSeeds
from open_science_network.ontology.v1.management.base.agriculture.Fertilizer import FertilizeSoil as FertilizeSoil
from open_science_network.ontology.v1.management.base.agriculture.AlfalfaCompost import UseAlfalfaCompost as UseAlfalfaCompost
from open_science_network.ontology.v1.management.base.agriculture.VermiCompost import UseVermiCompost as UseVermiCompost
from open_science_network.ontology.v1.management.base.agriculture.CowManure import UseCowManure as UseCowManure
from open_science_network.ontology.v1.management.base.agriculture.Mycorrhizae import UseMycorrhizae as UseMycorrhizae
from open_science_network.ontology.v1.management.base.agriculture.Irrigation import IrrigateCrops as IrrigateCrops
from open_science_network.ontology.v1.management.base.agriculture.Pesticide import SpreadPesticide as SpreadPesticide
from open_science_network.ontology.v1.management.base.agriculture.Pruning import PruneCrops as PruneCrops
from open_science_network.pydantic.Report import Report
from open_science_network.pydantic.Observation import Observation
from numpyro.infer.autoguide import AutoMultivariateNormal
from jax.tree_util import tree_map
import jax.tree_util as jtu
from jax.numpy import stack, pad, array
from datetime import timedelta
from open_science_network.agents.AgentInterface import AgentInterface
import jax.numpy as jnp
import numpy as np
from numpyro import sample, deterministic, handlers, plate
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import TransformReparam, LocScaleReparam
from numpyro.distributions import FoldedDistribution, Normal, LogNormal, Gamma, InverseGamma, Uniform, TruncatedNormal
from open_science_network.models.likelihoods import ElementSix as Likelihood
from abc import ABC


class AbstractRootsAndCultureAgent(AgentInterface, ABC):

    # The agent's species
    species = [
        AgentInterface.ontology_name(PlantSpecies, "Hemp"),
        AgentInterface.ontology_name(PlantSpecies, "Alfalfa")
    ]

    # The agent's actions
    actions = [
        AgentInterface.ontology_name(HarvestCrops, "Hemp"),
        AgentInterface.ontology_name(HarvestCrops, "Alfalfa"),
        AgentInterface.ontology_name(UseBioChar, "Yes"),
        AgentInterface.ontology_name(PlantingSeeds, "HempSeeds"),
        AgentInterface.ontology_name(PlantingSeeds, "AlfalfaSeeds"),
        AgentInterface.ontology_name(FertilizeSoil, "No"),
        AgentInterface.ontology_name(UseAlfalfaCompost, "Yes"),
        AgentInterface.ontology_name(UseVermiCompost, "Yes"),
        AgentInterface.ontology_name(UseCowManure, "Yes"),
        AgentInterface.ontology_name(UseMycorrhizae, "Yes"),
        AgentInterface.ontology_name(IrrigateCrops, "Yes"),
        AgentInterface.ontology_name(SpreadPesticide, "Yes"),
        AgentInterface.ontology_name(PruneCrops, "Yes")
    ]

    def __init__(self, agent_name, data):
        # Call parent constructor
        obs_to_site = {
            AgentInterface.ontology_name(HempYield, "Continuous"): "obs_yield_density",
            AgentInterface.ontology_name(HempHeight, "Continuous"): "obs_plant_height",
            AgentInterface.ontology_name(HempDensity, "Continuous"): "obs_plant_density"
        }
        super().__init__(agent_name, data, obs_to_site)

        self.n_species = 1  # TODO = len(self.data.project.strategies[0].species)

        self.time_horizon = 1
        self.n_lots = len(self.data.project.lots)

        self.min_val = jnp.finfo(float).eps

        self.likelihood = Likelihood()

        self.params = dict(
            seeds_planted=10.0,
            irrigation=1,
            rain=1,
            soil_biomass=.1,
            soc_std=.1,
            count_std=.05,
            height_std=.1,
            harvest_concentration=10.,
            survival_rate_noise=.1,
            effect_sizes=.1,
            satellite_soc=0.1,
            satellite_plant_biomass=100,
            satellite_plant_biomass_carbon=100,
            sentinel_bands=100
        )  # TODO some of these parameters are plant specific

        lot_area = jnp.array([100.] * self.n_lots)  # ha, update with calculation based on polygon coordinates
        self.area = jnp.expand_dims(lot_area * 100, -1)
        n_lng, n_lat = (10, 10)
        self.pixel_area = lot_area / (n_lng * n_lat)

        self.loc_max_plant_height = jnp.array([2.0] * self.n_species)
        self.plant_height_to_biomass = jnp.array([10.0] * self.n_species)
        self.seeds_planted_weekly = jnp.array([50000] * self.n_species)  # n, per species
        self.max_weekly_harvest = jnp.array([1000000] * self.n_species)  # kg, total

        self.irrigation_weekly = jnp.array([100] * self.n_lots)

        self.N_satellite_obs = n_lng * n_lat

        self.params['lot_area'] = lot_area
        self.params['soil_biomass_to_soc'] = jnp.array([0.45] * self.n_lots)
        self.params['plant_height_to_biomass'] = self.plant_height_to_biomass
        self.params['plant_biomass_to_carbon'] = jnp.array([0.45] * self.n_species)

        # collect names of all possible interventions
        self.interventions = {'base'}
        for strategy in self.data.project.strategies:
            self.interventions = self.interventions.union(set(strategy.interventions))
        self.name_integer = {k: v + 1 for v, k in enumerate(sorted(self.interventions))}

        # Get weather information
        self.weather = self.get_weather()

        # Store actions information
        self.action_names = [
            "planting-alfalfa", "harvest-alfalfa", "planting-hemp", "harvest-hemp", "irrigation",
            "fertilizer", "vermi-compost", "myco", "manure", "pesticide", "biochar", "pruning",
        ]
        self.n_actions = len(self.action_names)

        self.lots = list({lot.name for lot in self.data.lots})

        # Pre-process the default policies
        self.policy = stack([self.to_array(policy) for policy in data.policies], -2)

    def to_array(self, policy):
        """
        Convert the list of action names into the corresponding array of action indices
        :param policy: the list of action names
        :return: the array of action indices
        """

        # Replace action names by their corresponding indices
        policy = tree_map(lambda action: self.action_names.index(action) + 1 if action else action, policy)

        # Pad the actions to ensure they all have the same length
        return stack([pad(array(actions), (0, self.n_actions - len(actions))) for actions in policy])

    def get_weather(self):
        """
        Get weekly weather data for all locations
        :return: the weekly weather data
        """
        start = datetime(2021, 1, 1)  # TODO this needs to be loaded from the config file
        end = datetime(2021, 12, 31)  # TODO this needs to be loaded from the config file
        all_stations = Stations()
        weather = None
        for lot in self.data.project.lots:
            long, lat = jnp.array(lot.bounds.coordinates[0]).mean(-2)
            stations = all_stations.nearby(lat.item(), long.item())
            data = Daily(stations.fetch(1), start, end)
            data = data.normalize()
            data = data.aggregate('1W')
            data = data.interpolate()
            df = data.fetch()
            if weather is None:
                # Same (deterministic) weather for all locations, monthly (calendar year)
                weather = df.iloc[:-1].reset_index().to_dict(orient='list')
                for key in weather:
                    if key != 'time':
                        weather[key] = jnp.expand_dims(jnp.array(weather[key]), -1)
            else:
                tmp = df.iloc[:-1].reset_index().to_dict(orient='list')
                for key in tmp:
                    if key != 'time':
                        weather[key] = jnp.concatenate([weather[key], jnp.expand_dims(jnp.array(tmp[key]), -1)], -1)

        return weather

    def create_reports(self):
        """
        Create reports using the agent's model
        :return: a list of created reports
        """
        # Predict the observations' value(s)
        predictions = self.predict(model=self.model if self.conditioned_model is None else self.model, num_samples=1)

        # Create the reports
        reports = []
        locations = self.get_report_locations()
        start_date = self.data.project.start_date
        for t in range(1, self.data.T + 1):
            for lot_index in range(len(self.data.project.lots)):
                # Collect the observation from the predictions
                observations = []
                for obs_name in self.sample_sites():
                    observation = {
                        "name": obs_name,
                        "lot_name": self.data.lots[lot_index].name,
                        "value": predictions[obs_name][0][t - 1][lot_index]
                    }
                    observations.append(Observation(**observation))

                # Create the report
                report = Report(
                    datetime=start_date,
                    location=self.to_point(locations[lot_index, 55]),
                    project_name=self.name,
                    reporter='rob@element.six.com',
                    provenance="https://api.element.six.com/get_data?12345",
                    observations=observations
                )
                reports.append(report)
                start_date += timedelta(days=7)

        return reports

    def set_time_horizon(self, t):
        """
        Set the new time horizon
        :param t: the new time horizon
        """
        self.time_horizon = t

    def add_reports(self, reports):
        """
        Provide new reports to the model
        :param reports: the new reports
        """

        # Extract report as numpy array
        n_values = sum([len(column) for column in reports.iloc[0] if isinstance(column, list)]) / len(self.lots)
        np_report = np.zeros([len(reports.index) * len(self.lots), int(n_values)])
        for j, lot in enumerate(self.lots):
            k = 0
            for obs_name in self.sample_sites():
                for column in reports[obs_name]:
                    for value in column:
                        np_report[j][k] = value
                        k += 1
        np_report = jnp.expand_dims(np_report.transpose(), axis=(0, 2))
        np_report = np_report.mean(axis=-1).swapaxes(1, 2)

        # Merge new report to all reports
        self.reports = np_report if self.reports is None else jnp.concatenate((self.reports, np_report), axis=0)

        # Update time horizon and number of lots
        self.time_horizon = self.reports.shape[0]
        self.n_lots = self.reports.shape[1]

    def guide(self, *args, **kwargs):
        """
        Getter
        :param args: the guide's arguments
        :param kwargs: the guide's keyword arguments
        :return: the guide
        """
        return AutoMultivariateNormal(self.model)()

    def pre_process(self, data):
        """
        Preprocess the data to fit the model requirements
        :param data: the data
        :return: the pre-processed data
        """
        # Retrieve maximum number of interventions
        max_interventions = 1
        for policy in data.policies:
            mi = max([len(pi) for pi in policy])
            if mi + 1 > max_interventions:
                max_interventions = mi + 1

        # Pre-process policies
        policies = []
        for policy in data.policies:
            policies.append(self.policy_to_array(policy, max_interventions, self.name_integer))
        data.policies = jnp.stack(policies, -2)
        return data

    @staticmethod
    def policy_to_array(policy, max_num_interventions, name_integer):
        def name_map(key):
            return name_integer[key] if key is not None else key

        int_policy = jtu.tree_map(name_map, policy)
        arrays = []
        for pi in int_policy:
            arr = jnp.array(pi + [name_integer['base']])
            arr = jnp.pad(arr, (0, max_num_interventions - len(arr)), constant_values=name_integer['base'])
            arrays.append(arr)

        return jnp.stack(arrays)

    def global_parametric_model(self, data_level=3, mask=None, forecast=False, *args, **kwargs):

        length = len(self.data.policies)

        # Time-independent RVs
        effects = {
            'soil': {
                k: sample(f"soil_effect_{k}", Normal(v, self.params['effect_sizes']))
                for k, v in {"base": 0., "biochar": 0.3}.items()
            },
            'survival': {
                k: sample(f"survival_effect_{k}", Normal(v, self.params['effect_sizes']))
                for (k, v) in {"base": 0., "biochar": 0.1, "fertilizer": 0.05, "vermi-compost": 0.05}.items()
            },
            'growth': {
                k: sample(f"growth_effect_{k}", Normal(v, self.params['effect_sizes']))
                for k, v in {"base": 0., "biochar": 0.1, "fertilizer": 0.1, "myco": 0.05, "manure": 0.02}.items()
            },
            'nutrients': {
                # TODO: biochar and fertiliser are not directly nutrients, nitrogen is
                # TODO: do we want to model that?
                k: sample(f"nutrient_effect_{k}", Normal(v, self.params['effect_sizes']))
                for k, v in {"biochar": 0.4, "fertilizer": 0.1}.items()
            }
        }
        effects['nutrients']['base'] = 0.

        coefficients = {
            'survival': {
                'soil_biomass': sample("soil_biomass_to_ps", Normal(0.5, self.params['effect_sizes'])),
                'plant_density': sample("plant_density_to_ps", Normal(-0.5, self.params['effect_sizes'])) / 10000
            },
            'height': {
                'soil_biomass': sample("soil_biomass_to_ph", Normal(.5, self.params['effect_sizes'])),
                'plant_density': sample("plant_density_to_ph", Normal(-.5, self.params['effect_sizes'])) / 10000
            }
        }

        with plate('num_lots', self.n_lots):
            self.params['max_height'] = sample('max_plant_height', LogNormal(jnp.log(self.loc_max_plant_height), .1).to_event(1))

            if data_level > 1:
                self.params['count_prec'] = 10 * sample('count_prec', Gamma(10, 10))
                self.params['height_prec'] = 10 * sample('height_prec', Gamma(10, 10))

            self.params['lai_beta'] = sample('lai_beta', Gamma(2, 2))
            self.params['lai_alpha'] = sample('lai_alpha', Gamma(2, 2))
            self.params['lai_scale'] = jnp.sqrt(sample('lai_var', InverseGamma(2, 2)))

            # seeds per Are - from 25 seeds for CBD crop up to 3000 seeds for seed crop
            # n per Are per species
            seeds_planted_weekly = sample(
                'planted_seeds', Uniform(jnp.array([25] * self.n_species), jnp.array([3000] * self.n_species)).to_event(1))

            # maximum weekly harvest kg per are
            max_weekly_harvest = sample(
                'max_weekly_harvest', Uniform(jnp.array([10] * self.n_species), jnp.array([1000] * self.n_species)).to_event(1))
            self.params['max_weekly_harvest'] = max_weekly_harvest

            # weekly irrigation
            irrigation_weekly = sample('weekly_irrigation', Uniform(50., 150.))

            # time dependent variables
            with plate('num_weeks', length):
                plant = jnp.expand_dims(jnp.any(self.data.policies == self.name_integer['planting-hemp'], -1), -1)
                loc = seeds_planted_weekly * plant - 10 * (1 - plant)
                scale = self.params['seeds_planted'] * plant + (1 - plant) / 10

                with handlers.reparam(config={"_seeds": LocScaleReparam(0)}):
                    _seeds = sample("_seeds", Normal(loc, scale).to_event(1))

                irrigate = jnp.any(self.data.policies == self.name_integer['irrigation'], -1)
                loc = irrigation_weekly * irrigate - 10 * (1 - irrigate)
                scale = self.params['irrigation'] * irrigate + (1 - irrigate) / 10

                with handlers.reparam(config={"irrigation": LocScaleReparam(0)}):
                    irrigation = sample("irrigation", Normal(loc, scale))
                    irrigation = jnp.clip(irrigation, a_min=0.)

                if forecast:
                    tmin = self.weather['tmin']
                    tmax = self.weather['tmax']
                    loc = self.weather['tavg']
                    scale = (tmax - tmin) / 2
                    temp = sample('temp', TruncatedNormal(loc, scale, high=tmax, low=tmin))
                    prcp = sample('prcp', Gamma(jnp.clip(self.weather['prcp'], a_min=self.min_val) * 2, 2))
                    water_volume = irrigation + prcp
                else:
                    water_volume = irrigation + self.weather['prcp'][:length]
                    temp = self.weather['tavg'][:length]

        effects['water'] = deterministic('water_effect', self.water_to_rate(water_volume))
        effects['temp'] = deterministic('temp_effect', self.temp_to_rate(temp))
        effects['seeding'] = deterministic('seeds_planted_per_are', jnp.clip(_seeds, a_min=0.))

        return effects, coefficients, mask, data_level

    @staticmethod
    def sample_folded_normal(name, base_dist):
        with handlers.reparam(config={name: TransformReparam()}):
            return sample(name, FoldedDistribution(base_dist))

    def water_to_rate(self, volume, lower=-10, upper=360):
        opt = (lower + upper) / 2
        size = 1 / self.parabola(lower, upper, 1, opt)
        return jnp.clip(-1 + self.parabola(lower, upper, size, volume), a_min=-1., a_max=0.)

    def temp_to_rate(self, temperature, lower=-5, upper=49):
        opt = (lower + upper) / 2
        size = 1 / self.parabola(lower, upper, 1, opt)
        return jnp.clip(-1 + self.parabola(lower, upper, size, temperature), a_min=-1, a_max=0.)

    @staticmethod
    def parabola(lower, upper, size, x):
        return - size * (x - lower) * (x - upper)


class DeterministicRootsAndCultureAgent(AbstractRootsAndCultureAgent):

    def __init__(self, data):
        super().__init__("Roots-and-Culture.roots-indoor1.Deterministic", data)

    def model(self, *args, **kwargs):
        return self.deterministic_dynamics(*self.global_parametric_model(*args, **kwargs))

    def simulate_plant_dynamic(self):
        pass  # TODO implement the plant dynamic

    def deterministic_dynamics(self, effects, coefficients, mask, data_level):
        # Initial states
        soil_nutrients = jnp.ones(self.n_lots) * 0.1
        soil_biomass = jnp.ones(self.n_lots) * 0.1
        plant_density = jnp.zeros((self.n_lots, self.n_species))
        plant_height = jnp.zeros((self.n_lots, self.n_species))

        # TODO soil_temperature = jnp.ones(L) * 0.1
        # TODO microbial_biomass = jnp.ones(L) * 0.1
        # TODO soil_moisture = jnp.ones(L) * 0.1
        # TODO creating: _macrobial_biomass ? + _soil_biomass = _macrobial_biomass + _microbial_biomass

        init = (soil_nutrients, soil_biomass, plant_density, plant_height)
        # TODO , soil_temperature, microbial_biomass, soil_moisture

        # Time steps
        def step_fn(carry, xs):

            events, _water_effect, _temp_effect, num_seeds, _mask = xs
            _soil_nutrients, _soil_biomass, _plant_density, _plant_height = carry
            # TODO , _soil_temperature, _microbial_biomass, _soil_moisture

            # TODO modelling: _soil_temperature, _microbial_biomass, _soil_moisture
            # TODO _soil_moisture = ? # Sun reduce soil moisture,
            # Rain increase soil moisture,
            # Frost decreases soil permeability (decreasing moisture?),
            # Snow increase soil moisture (keep soil hotter and permeable)
            # TODO _soil_temperature = ? # Sun heat the soil
            # rain reduces soil temperature
            # frost reduces soil temperature
            # snow increase soil temperature (keep it close to zero when air temperature become negative)
            # TODO _microbial_biomass = ? # Soil temperature kill (10℃ > t > 35.6℃) or activate (10℃-35.6℃) microorganism
            # TODO _soil_nutrients = ? # organic matter + microbial biomass increases soil nutrient, and fertiliser increase soil nutrients

            # Compute the total effect of interventions on soil, plant survival, and plant growth
            mod_nutrients = jnp.stack([
                eff * jnp.any(events == self.name_integer[inter], -1) for (inter, eff) in effects['nutrients'].items()
            ], -1).sum(-1)

            mod_soil = jnp.stack([
                eff * jnp.any(events == self.name_integer[inter], -1) for (inter, eff) in effects['soil'].items()
            ], -1).sum(-1)

            mod_survival = jnp.stack([
                eff * jnp.any(events == self.name_integer[inter], -1) for (inter, eff) in effects['survival'].items()
            ], -1).sum(-1)

            mod_growth = jnp.stack([
                eff * jnp.any(events == self.name_integer[inter], -1) for (inter, eff) in effects['growth'].items()
            ], -1).sum(-1)

            with plate('num_lots', self.n_lots):
                # dynamics
                r = 0.99
                soil_nutrients = deterministic('soil_nutrients',
                                               jnp.clip(_soil_nutrients * r + mod_nutrients, a_min=0., a_max=1.))
                sn_eff1 = soil_nutrients - 1.
                sn_eff2 = jnp.clip(soil_nutrients - .05, a_min=0., a_max=1) - 1.

                # Compute new soil biomass
                eta_max = 0.1 + 0.9 * jnp.heaviside(_plant_density - 1 / self.area, 1.).mean(-1)
                soil_bio_rate = mod_soil + _water_effect
                kappa = (1 + soil_bio_rate) / 52
                r = jnp.clip(1 - jnp.exp(-kappa), a_min=0., a_max=1.)
                soil_biomass = deterministic('soil_biomass', _soil_biomass + r * (eta_max - _soil_biomass))

                # Ensure the soil_biomass' shape is correct
                assert soil_biomass.shape == (self.n_lots,)

                # Compute the plant dynamic
                self.simulate_plant_dynamic()

                # TODO is this plant specific
                # Compute the new plant height
                growth_rate = jnp.expand_dims(
                    _soil_biomass * coefficients['height']['soil_biomass'] + _water_effect + _temp_effect + sn_eff2 +
                    mod_growth, -1
                ) + _plant_density * coefficients['height']['plant_density']
                kappa = (1 + growth_rate) / 4
                r = jnp.clip(1 - jnp.exp(-kappa), a_min=0., a_max=1.)
                plant_height = deterministic(
                    'plant_height',
                    (_plant_height + r * (self.params['max_height'] - _plant_height)) *
                    jnp.heaviside(_plant_density - 1 / self.area, 1.)
                )

                # Ensure the plant_height's shape is correct
                assert plant_height.shape == (self.n_lots, self.n_species)

                # TODO is this plant specific
                plant_unit_biomass = jnp.clip(plant_height * self.plant_height_to_biomass, a_min=1e-16)
                survival_rate = jnp.expand_dims(_soil_biomass * coefficients['survival'][
                    'soil_biomass'] + _water_effect + _temp_effect + mod_survival + sn_eff1, -1) + _plant_density * \
                                coefficients['survival']['plant_density']

                # Ensure the survival_rate's shape is correct
                assert survival_rate.shape == (self.n_lots, self.n_species)

                # TODO is this plant specific
                kappa = (1 + survival_rate) * 8.
                r = deterministic('survival_probability', jnp.clip(1 - jnp.exp(-kappa), a_min=0., a_max=1.))
                new_plant_density = r * _plant_density + num_seeds

                # TODO is this plant specific
                harvest = jnp.expand_dims(jnp.any(self.name_integer['harvest-hemp'] == events, -1), -1)
                harvest_yield = jnp.clip(
                    new_plant_density * plant_unit_biomass, a_max=self.params['max_weekly_harvest']
                ) * harvest
                yield_per_m2 = deterministic('yield_density', harvest_yield)

                # Ensure the harvest_yield's shape is correct
                assert harvest_yield.shape == (self.n_lots, self.n_species)

                # TODO is this plant specific
                plant_density = deterministic('plant_density', new_plant_density - harvest_yield / plant_unit_biomass)

                self.likelihood(
                    soil_biomass, plant_density, plant_height, yield_per_m2, self.params, _mask, data_level=data_level
                )

            return (soil_nutrients, soil_biomass, plant_density, plant_height), None

        scan(step_fn, init, (self.data.policies, effects['water'], effects['temp'], effects['seeding'], mask))


class StochasticRootsAndCultureAgent(AbstractRootsAndCultureAgent):

    def __init__(self, data):
        # Call parent constructor
        super().__init__("Roots-and-Culture.roots-indoor1.Stochastic", data)

    def model(self, *args, **kwargs):
        return self.stochastic_dynamics(*self.global_parametric_model(*args, **kwargs))

    def stochastic_dynamics(self, effects, coefficients, mask, data_level):
        # Initial states
        soil_nutrients = jnp.ones(self.n_lots) * 0.1
        soil_biomass = jnp.ones(self.n_lots) * 0.1
        plant_density = jnp.zeros((self.n_lots, self.n_species))
        plant_height = jnp.zeros((self.n_lots, self.n_species))

        init = (soil_nutrients, soil_biomass, plant_density, plant_height)

        # Time steps
        def step_fn(carry, xs):
            interventions, _water_effect, _temp_effect, num_seeds, _mask = xs
            _soil_nutrients, _soil_biomass, _plant_density, _plant_height = carry

            mod_nutrients = jnp.stack([
                eff * jnp.any(interventions == self.name_integer[inter], -1)
                for (inter, eff) in effects['nutrients'].items()
            ], -1).sum(-1)

            mod_soil = jnp.stack([
                eff * jnp.any(interventions == self.name_integer[inter], -1)
                for (inter, eff) in effects['soil'].items()
            ], -1).sum(-1)

            mod_survival = jnp.stack([
                eff * jnp.any(interventions == self.name_integer[inter], -1)
                for (inter, eff) in effects['survival'].items()
            ], -1).sum(-1)

            mod_growth = jnp.stack([
                eff * jnp.any(interventions == self.name_integer[inter], -1)
                for (inter, eff) in effects['growth'].items()
            ], -1).sum(-1)

            with plate('num_lots', self.n_lots):
                # dynamics
                kappa = self.sample_folded_normal('nutrients_rate', Normal(4.5, 0.1))
                r = 1 - jnp.exp(-kappa)
                soil_nutrients = deterministic(
                    'soil_nutrients', jnp.clip(_soil_nutrients * r + mod_nutrients, a_min=0., a_max=1.)
                )
                sn_eff1 = soil_nutrients - 1.
                sn_eff2 = jnp.clip(soil_nutrients - .05, a_min=0., a_max=1) - 1.

                eta_max = 0.1 + 0.9 * jnp.heaviside(_plant_density - 1 / self.area, 1.).mean(-1)
                soil_bio_rate = mod_soil + _water_effect
                loc = (1 + soil_bio_rate) / 52
                kappa = self.sample_folded_normal('biomass_rate', Normal(loc, 0.1))
                r = 1 - jnp.exp(-kappa)
                soil_biomass = deterministic('soil_biomass', _soil_biomass + r * (eta_max - _soil_biomass))

                assert soil_biomass.shape == (self.n_lots,)

                growth_rate = jnp.expand_dims(
                    _soil_biomass * coefficients['height']['soil_biomass'] + _water_effect + _temp_effect +
                    sn_eff2 + mod_growth, -1) + _plant_density * coefficients['height']['plant_density']

                loc = (1 + growth_rate) / 4
                kappa = self.sample_folded_normal('growth_rate', Normal(loc, 0.1))
                r = 1 - jnp.exp(-kappa)
                plant_height = deterministic(
                    'plant_height',
                    (_plant_height + r * (self.params['max_height'] - _plant_height)) *
                    jnp.heaviside(_plant_density - 1 / self.area, 1.)
                )

                assert plant_height.shape == (self.n_lots, self.n_species)

                plant_unit_biomass = jnp.clip(plant_height * self.plant_height_to_biomass, a_min=1e-16)

                survival_rate = jnp.expand_dims(
                    _soil_biomass * coefficients['survival']['soil_biomass'] + _water_effect + _temp_effect +
                    mod_survival + sn_eff1, -1
                ) + _plant_density * coefficients['survival']['plant_density']

                assert survival_rate.shape == (self.n_lots, self.n_species)

                loc = 8. * (1 + survival_rate)
                kappa = self.sample_folded_normal('survival_rate', Normal(loc, 0.1))
                r = deterministic('survival_probability', 1 - jnp.exp(- kappa))
                new_plant_density = r * _plant_density + num_seeds

                harvest = jnp.expand_dims(jnp.any(self.name_integer['harvest-hemp'] == interventions, -1), -1)
                harvest_yield = jnp.clip(new_plant_density * plant_unit_biomass,
                                         a_max=self.params['max_weekly_harvest']) * harvest
                yield_per_m2 = deterministic('yield_density', harvest_yield)

                assert harvest_yield.shape == (self.n_lots, self.n_species)

                plant_density = deterministic('plant_density', new_plant_density - harvest_yield / plant_unit_biomass)

                self.likelihood(
                    soil_biomass, plant_density, plant_height, yield_per_m2, self.params, _mask, data_level=data_level
                )

                return (soil_nutrients, soil_biomass, plant_density, plant_height), None

        scan(step_fn, init, (self.data.policies, effects['water'], effects['temp'], effects['seeding'], mask))
