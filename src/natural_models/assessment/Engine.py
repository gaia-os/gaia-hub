from natural_models.assessment.DataLoader import DataLoader
from natural_models.assessment import ArgumentParser
from natural_models.agents.AgentFactory import AgentFactory
from jax.numpy import concatenate
import traceback


class Engine:
    """
    A class performing the assessment of a project
    """

    def __init__(self, verbose=False, debug=False):
        """
        Construct the assessment engine
        :param verbose: True if useful information should be displayed, False otherwise
        :param debug: True if debug information should be displayed, False otherwise
        """
        # Load arguments
        self.parser = ArgumentParser.ArgumentParser()
        args = self.parse_arguments()
        self.project_file = args.project
        self.reports_dir = args.reports_dir

        # Load the data and requested agents
        self.data = DataLoader(self.project_file, self.reports_dir, verbose=verbose)
        self.agents = AgentFactory.create(self.data, verbose=verbose)

        # Keep track of verbose and debug level
        self.verbose = verbose
        self.debug = debug

    def add_agent(self, agent):
        """
        Add an agent to the list of existing agents
        :param agent: the agent to be added
        """
        self.agents.append(agent)

    def parse_arguments(self):
        """
        Parse the arguments of the 'perform_assessment.py' script
        """
        return self.parser.parse()

    def perform_assessment(self):
        """
        Perform a project assessment
        """
        # Display functions arguments
        if self.verbose:
            print(f"Perform the assessment of the following project:")
            print(f"    - project_file = {self.project_file}")
            print(f"    - report_dir = {self.reports_dir}")

        # Perform the project assessment
        results = {}
        ignore_agent = []
        for t, report in sorted(self.data.reports, key=lambda x: x[0]):

            # Make sur t is an integer
            t = int(t)
            print(f"t: {t}")  # TODO

            # For each agent
            for agent in self.agents:  # TODO replace iterations over project by agent averaging

                # if t == 0:  # TODO
                #     continue  # TODO

                # Ignore the agent if it has already failed to produce an assessment
                if agent in ignore_agent:
                    continue

                try:
                    # Perform the assessment of the project at time step t, by providing an agent with a new report
                    result = self.assess_project(t, agent, report)

                    # Keep track of results
                    if agent.name not in results.keys():
                        results[agent.name] = []
                    results[agent.name].append(result)

                except Exception as e:

                    # Inform the user that an error occurred
                    print(f"[ERROR] Could not assess project at time {t} using agent {agent.name}.")
                    exclude_columns = {'report_id', 'datetime', 'location', 'reporter', 't', 'lot', 'index'}
                    columns = set(report.columns) - exclude_columns
                    print(f"[ERROR] The report contains the following columns:")
                    for column in columns:
                        print(f"[ERROR] - {column}")

                    # Display debug information if requested
                    if self.debug is True:
                        print(f"[DEBUG] \n[DEBUG] ", end='')
                        print(traceback.format_exc().replace("\n", "\n[DEBUG] "))

                    # Inform the user that the agent will be ignored from now on
                    print(f"[INFO] The agent {agent.name} will now be ignored.")
                    ignore_agent.append(agent.name)

        return results

    def assess_project(self, t, agent, report):
        # Provide a new report to the agent
        agent.add_reports(reports=report)

        # Get reports information sorted by agent's sample sites
        reports = agent.get_report_by_sample_site()

        # Perform inference using available data
        cond_model, cond_guide = agent.condition_all(reports)
        samples = agent.inference_algorithm().run_inference(
            model=cond_model,
            guide=cond_guide,
            inference_params={"time_horizon": t},
        )

        # Perform prediction of future random variables
        samples = agent.predict(
            model=agent.conditioned_model,
            posterior_samples=samples,
            return_sites=list(samples.keys())
        )

        # Concatenate report information with predictive samples
        for site_name, data in reports.items():
            predictive_samples = samples[site_name].mean(axis=0)[data.shape[0]:]
            reports[site_name] = concatenate((data, predictive_samples), axis=0)

        # Compute the expected free energy
        agent.condition_model(reports)
        efe = agent.efe(samples, t)
        if self.verbose:
            print(f"Expected Free Energy of {type(agent)} at time step {t}: {efe}")

        return samples, efe
