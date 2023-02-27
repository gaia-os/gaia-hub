from natural_models.ontology.v1.genetics.base.plant.Plant import PlantSpecies
from natural_models.ontology.v1.management.base.agriculture.Harvest import HarvestCrops as HarvestCrops
from natural_models.ontology.v1.management.base.agriculture.Planting import PlantingSeeds as PlantingSeeds
from natural_models.pydantic.Report import Report
from natural_models.pydantic.Observation import Observation
import jax.tree_util as jtu
from datetime import timedelta
from natural_models.agents.AgentInterface import AgentInterface
import jax.numpy as jnp
import numpy as np
from numpyro import sample, deterministic, plate
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Normal, Gamma
import jax


class GetStartedAgent(AgentInterface):

    # The model's species
    species = [
        PlantSpecies.__module__ + "." + PlantSpecies.__name__ + ".Hemp",
    ]

    # The model's actions
    actions = [
        HarvestCrops.__module__ + "." + HarvestCrops.__name__ + ".Hemp",
        PlantingSeeds.__module__ + "." + PlantingSeeds.__name__ + ".HempSeeds",
    ]

    def __init__(self, data):
        # Call parent constructor
        super().__init__("Tutorial.GetStarted", data)

        self.action_names = ["planting-hemp", "harvest-hemp"]
        self.obs_to_sample_site = {
            "v1.measurement.base.agriculture.Yield.HempYield.Continuous": "yield"
        }

        self.n_lots = len(self.data.project.lots)

        self.time_horizon = 1

        # collect names of all possible interventions
        self.interventions = set()
        for strategy in self.data.project.strategies:
            self.interventions = self.interventions.union(set(strategy.interventions))

        self.data = self.pre_process(self.data)
        self.lots = list({lot.name for lot in self.data.lots})

    def efe(self, samples, present_time):
        """
        Compute the expected free energy. This function assumes that :
           - the observation is called 'sr' which stands for state report
           - the log-probability of the observation is called 'log_sr'
        If these assumptions does not hold for a particular agent, this function must be overwritten
        :param samples: samples from the posterior distribution
        :param present_time: the present time step
        :return: the expected free energy
        """
        return self.expected_free_energy(samples, present_time, self.obs_to_sample_site.values())

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
                for obs_name in self.obs_to_sample_site.keys():
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

        # Get number of values in the reports
        n_values = 0
        for column in reports.iloc[0]:
            if isinstance(column, float):
                n_values += 1
            elif isinstance(column, list):
                n_values += len(column)
        n_values /= len(self.lots)

        # Extract report as numpy array
        np_report = np.zeros([len(reports.index) * len(self.lots), int(n_values)])
        for j, lot in enumerate(self.lots):
            k = 0
            for obs_name in self.obs_to_sample_site.keys():
                for column in reports[obs_name]:
                    if isinstance(column, float):
                        column = [column]
                    for value in column:
                        np_report[j][k] = value
                        k += 1
        np_report = jnp.expand_dims(np_report.transpose(), axis=(0, 2))
        np_report = np_report.mean(axis=-1).swapaxes(1, 2)

        # Merge new report to all reports
        self.reports = np_report if self.reports is None else jnp.concatenate((self.reports, np_report), axis=0)

        # Update time horizon and number of lots
        self.time_horizon = self.reports.shape[0]

    def to_sample_site(self, obs_name):
        """
        Getter
        :param obs_name: the observation name whose sample site should be returned
        :return: the name of the sample site associated with the observation name
        """
        return self.obs_to_sample_site[obs_name]

    def pre_process(self, data):
        """
        Preprocess the data to fit the model requirements
        :param data: the data
        :return: the pre-processed data
        """
        # Retrieve maximum number of interventions
        max_interventions = 1
        for policy in data.policy:
            mi = max([len(pi) for pi in policy])
            if mi + 1 > max_interventions:
                max_interventions = mi + 1

        # Pre-process policies
        name_integer = {action_name: i + 1 for i, action_name in enumerate(self.action_names)}
        policies = [self.policy_to_array(policy, max_interventions, name_integer) for policy in data.policy]
        self.policy = jnp.stack(policies, -2)
        return data

    @staticmethod
    def policy_to_array(policy, max_num_interventions, name_integer):
        int_policy = jtu.tree_map(lambda key: name_integer[key] if key is not None else key, policy)
        arrays = []
        for pi in int_policy:
            arr = jnp.array(pi)
            arr = jnp.pad(arr, (0, max_num_interventions - len(arr)))
            arrays.append(arr)

        return jnp.stack(arrays)

    def is_performed(self, action, actions_performed):
        """
        Check whether an action is performed
        :param action: the action for which the check is done
        :param actions_performed: all the performed actions
        :return: True if the action is performed, False otherwise
        """
        return jnp.any(self.index_of(action) + 1 == actions_performed, -1)

    def index_of(self, action_name):
        """
        Getter
        :param action_name: the name of the action whose must be returned
        :return: the action index
        """
        return self.action_names.index(action_name)

    def model_dynamic(self, states_t, values_t):
        """
        Implement the model dynamic
        :param states_t: the state of the lot at time t
        :param values_t: the current time step, as well as the growth rate and the actions performed at time t
        :return: the states at time t + 1
        """

        # Unpack the states and values at time t
        hemp_size_t, hemp_can_grow_t = states_t
        t, actions_performed = values_t

        # Ensure the model can be used with several lots, i.e., duplicate the model for each lot
        with plate('n_lots', self.n_lots):

            # Check if planting and harvesting are performed at time step t
            plant = self.is_performed("planting-hemp", actions_performed)
            harvest = self.is_performed("harvest-hemp", actions_performed)

            # Sample the hemp growth rate from a Gamma distribution
            hemp_grow_rate = sample("hemp_growth_rate", Gamma(3, 0.06))

            # Compute the states at time t + 1
            hemp_size_t1 = deterministic(f"hemp_size", (hemp_size_t + hemp_grow_rate * hemp_can_grow_t) * (1 - harvest))
            hemp_can_grow_t1 = hemp_can_grow_t * (1 - harvest) + (1 - hemp_can_grow_t) * plant
            hemp_can_grow_t1 = deterministic(f"hemp_can_grow", hemp_can_grow_t1)

            # Compute the yield at time t
            rng_key = jax.random.PRNGKey(0)  # TODO
            sample("yield", Normal(hemp_size_t * harvest, 0.1), rng_key=rng_key)

        return (hemp_size_t1, hemp_can_grow_t1), None

    def model(self, *args, time_horizon=-1, **kwargs):
        """
        The generative model of the agent
        """

        # Make sure the time horizon is value
        time_horizon = len(self.policy) if time_horizon == -1 else time_horizon

        # Ensure the model can be used with several lots, i.e., duplicate the model for each lot
        with plate('n_lots', self.n_lots):

            # Sample the initial hemp size and ensure that the hemp cannot grow initially
            # TODO why do I have to do the expand? Isn't it the role of the plate
            hemp_size = sample("hemp_size_0", Normal(1, 0.1), sample_shape=(1, 1, ))
            hemp_can_grow = jnp.zeros_like(hemp_size)

        # Make sure all the input of the scan function are of the same size
        time_indices = jnp.expand_dims(jnp.expand_dims(jnp.arange(0, time_horizon), axis=1), axis=2)
        time_indices = jnp.repeat(time_indices, self.n_lots, axis=1)
        policy = self.policy[:time_horizon]

        # Call the scan function that unroll the model over time
        scan(self.model_dynamic, (hemp_size, hemp_can_grow), (time_indices, policy))

    def guide(self, *args, **kwargs):
        """
        The variational distribution of the agent
        """
        return None

    @property
    def observations(self):
        """
        Observations getter
        :return: the name of the observations supported by the agent
        """
        return self.obs_to_sample_site.keys()
