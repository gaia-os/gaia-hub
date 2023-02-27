import abc
import copy
import os


class AgentFactory:

    @staticmethod
    def get_all_classes(path, package):
        """
        Retrieve all the classes within a directory or file
        :param path: the path to the directory or file
        :param package: the classes package
        :return: all the classes
        """
        # Iterate over all files
        classes = {}
        files = os.listdir(path) if os.path.isdir(path) else [os.path.basename(path)]
        for file in files:
            # Check that the file is a python file but not the init.py
            if not file.endswith('.py') or file == '__init__.py':
                continue

            # Get the class and module
            class_name = file[:-3]
            class_module = __import__(package + class_name, fromlist=[class_name])

            # Get the frames' class
            module_dict = class_module.__dict__
            for obj in module_dict:
                if isinstance(module_dict[obj], type) and module_dict[obj].__module__ == class_module.__name__:
                    classes[obj] = getattr(class_module, obj)
        return classes

    @staticmethod
    def print_list(list_name, elements):
        """
        Print the list name and elements.
        """
        print(list_name)
        for element in elements:
            print(f"\t- {element}")
        print()

    @staticmethod
    def create(data, verbose=False):
        """
        Create the models corresponding to the project passed as parameters
        :param data: the data loader containing all the report, policy and project description
        :param verbose: True if useful information should be displayed, False otherwise
        :return: the created models
        """
        # Get all non-abstract agent classes
        agent_dir = os.path.abspath(os.path.dirname(__file__) + "/impl/")
        agents = AgentFactory.get_all_classes(agent_dir, "natural_models.agents.impl.").values()
        agents = [agent for agent in agents if abc.ABC not in agent.__bases__]
        if verbose:
            AgentFactory.print_list("Agents found:", agents)

        # Get the project interventions
        prefix = "natural_models.ontology."
        interventions = {
            prefix + value for strategy in data.project.strategies for key, value in strategy.interventions.items()
        }
        if verbose:
            AgentFactory.print_list("Project interventions:", interventions)

        # Get all agents supporting project interventions
        agents = [
            agent for agent in agents if hasattr(agent, 'actions') and interventions.issubset(set(agent.actions))
        ]
        if verbose:
            AgentFactory.print_list("Agents satisfying interventions:", agents)

        # Get all species associated to the project strategy
        species = {prefix + value for strategy in data.project.strategies for value in strategy.species}
        if verbose:
            AgentFactory.print_list("Project species:", species)

        # Get all models supporting project species
        agents = [agent for agent in agents if hasattr(agent, 'species') and species.issubset(set(agent.species))]
        if verbose:
            AgentFactory.print_list("Agents satisfying interventions and species:", agents)

        # Instantiate the compatible agents
        return [agent(copy.deepcopy(data)) for agent in agents]
