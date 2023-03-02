from open_science_network.ontology.v1.measurement.base.agriculture.Yield import HempYield
from open_science_network.ontology.v1.genetics.base.plant.Plant import PlantSpecies
from open_science_network.ontology.v1.management.base.agriculture.Harvest import HarvestCrops as HarvestCrops
from open_science_network.ontology.v1.management.base.agriculture.Planting import PlantingSeeds as PlantingSeeds
from jax.tree_util import tree_map
from open_science_network.agents.AgentInterface import AgentInterface
import jax.numpy as jnp
from jax.numpy import stack, pad, array
import numpy as np
from numpyro import sample, deterministic, plate
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Normal, Gamma
from numpyro.infer.autoguide import AutoMultivariateNormal


class GetStartedAgent(AgentInterface):
    """
    A class serving as an example of custom agent, and helping newcomers to understand how to implement new agents.
    """

    # The agent's species
    species = [
        AgentInterface.ontology_name(PlantSpecies, "Hemp")
    ]

    # The agent's actions
    actions = [
        AgentInterface.ontology_name(HarvestCrops, "Hemp"),
        AgentInterface.ontology_name(PlantingSeeds, "HempSeeds")
    ]

    def __init__(self, data):
        """
        Construct the get started agent
        :param data: an instance of the data loader containing information about the available reports and lots.
        """

        # Call the AgentInterface constructor
        obs_to_site = {
            AgentInterface.ontology_name(HempYield, "Continuous"): "obs_yield"
        }
        super().__init__("Tutorial.GetStarted", data, obs_to_site=obs_to_site)

        # Store actions information
        self.action_names = ["planting-hemp", "harvest-hemp"]
        self.n_actions = len(self.action_names)

        # Store lots information
        self.n_lots = len(self.data.lots)

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

    def add_reports(self, reports):
        """
        Provide new reports to the model
        :param reports: the new reports
        """

        # Extract reports content as numpy array
        np_reports = stack([self.extract_measurements(reports, lot) for lot in range(self.n_lots)])
        np_reports = jnp.expand_dims(np_reports, axis=0)

        # Merge new report to already existing reports
        self.reports = np_reports if self.reports is None else jnp.concatenate((self.reports, np_reports), axis=0)

    def extract_measurements(self, reports, lot_id):
        """
        Extract the measurements associated with a lot
        :param reports: the reports available
        :param lot_id: the lot's index whose measurements need to be retrieved
        :return: the lot's measurements
        """

        # Select only the measurements correspond to i-th lot
        reports = reports[reports['lot'] == lot_id]

        # Create the numpy array that will contain the measurements
        observations = np.zeros(self.number_of_measurements(reports))

        # Iterate over the observation names
        obs_id = 0
        for obs_name in self.observations:

            # Iterate over the measurements associated with these observations
            for measurements in reports[obs_name]:

                # Extract the measurements
                measurements = [measurements] if isinstance(measurements, float) else measurements
                for measurement in measurements:
                    observations[obs_id] = measurement
                    obs_id += 1

        return observations

    def number_of_measurements(self, reports):
        """
        Getter
        :param reports: the reports
        :return: the number of measurements in the reports
        """

        # Get the measurements in the first row of the reports
        measurements = reports[self.observations].iloc[0].tolist()

        # Iterate over the measurements
        n_reports = 0
        for measurement in measurements:

            # Process each measurement
            if isinstance(measurement, float):
                n_reports += 1
            elif isinstance(measurement, list):
                n_reports += len(measurement)

        # Divide the total number of measurement by the number of lots
        return int(n_reports)

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
        :param action_name: the name of the action whose index must be returned
        :return: the action index
        """
        return self.action_names.index(action_name)

    def model_dynamic(self, states_t, values_t):
        """
        Implement the model dynamic
        :param states_t: the state of the lot at time t
        :param values_t: the current time step, the growth rate and the actions performed at time t
        :return: the states at time t + 1
        """

        # Unpack the states and values at time t
        hemp_size_t, hemp_can_grow_t, hemp_growth_rate = states_t
        t, actions_performed = values_t

        # Ensure the model can be used with several lots, i.e., duplicate the model for each lot
        with plate('n_lots', self.n_lots):

            # Check if planting and harvesting are performed at time step t
            plant = self.is_performed("planting-hemp", actions_performed)
            harvest = self.is_performed("harvest-hemp", actions_performed)

            # Compute the states at time t + 1
            hemp_size_t1 = (hemp_size_t + hemp_growth_rate * hemp_can_grow_t) * (1 - harvest)
            hemp_size_t1 = deterministic(f"hemp_size", hemp_size_t1)
            hemp_can_grow_t1 = hemp_can_grow_t * (1 - harvest) + (1 - hemp_can_grow_t) * plant
            hemp_can_grow_t1 = deterministic(f"hemp_can_grow", hemp_can_grow_t1)

            # Compute the yield at time t
            sample("obs_yield", Normal(hemp_size_t * harvest, 0.1), rng_key=self.rng_key)

        return (hemp_size_t1, hemp_can_grow_t1, hemp_growth_rate), None

    def model(self, *args, time_horizon=-1, **kwargs):
        """
        The generative model of the agent
        :param args: unused positional arguments
        :param time_horizon: the number of time steps the model needs to be unrolled for
        :param kwargs: unused keyword arguments
        """

        # Make sure the time horizon is valid
        time_horizon = len(self.policy) if time_horizon == -1 else time_horizon

        # Ensure the model can be used with several lots, i.e., duplicate the model for each lot
        with plate('n_lots', self.n_lots):

            # Sample the initial hemp size and ensure that the hemp cannot grow initially
            hemp_size = sample("hemp_size_0", Gamma(1, 0.1).expand((1, 1)))
            hemp_can_grow = jnp.zeros_like(hemp_size)

            # Sample the hemp growth rate from a Gamma distribution
            hemp_grow_rate = sample("hemp_growth_rate", Gamma(3, 0.06))

        # Make sure all the inputs of the scan function have the same size
        time_indices = jnp.expand_dims(jnp.expand_dims(jnp.arange(0, time_horizon), axis=1), axis=2)
        time_indices = jnp.repeat(time_indices, self.n_lots, axis=1)
        policy = self.policy[:time_horizon]

        # Call the scan function that unroll the model over time
        scan(self.model_dynamic, (hemp_size, hemp_can_grow, hemp_grow_rate), (time_indices, policy))

    def guide(self, model, *args, **kwargs):
        """
        The variational distribution of the agent
        :param model: the model to provide to the guide
        :param args: unused positional arguments
        :param kwargs: unused positional keyword arguments
        :return: the guide
        """
        return AutoMultivariateNormal(model)
