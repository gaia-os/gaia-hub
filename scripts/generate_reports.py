from natural_models.assessment.DataLoader import DataLoader
from natural_models.assessment.ArgumentParser import ArgumentParser
from natural_models.agents.AgentFactory import AgentFactory


def generate_reports():
    """
    Performing the project assessment
    """
    # Load the data and requested agents
    args = ArgumentParser().parse()
    data = DataLoader(args.project, args.reports_dir)
    agents = AgentFactory.create(data)

    # Generates the reports using all requested agents
    for agent in agents:
        agent.export_reports(args.reports_dir)


if __name__ == '__main__':
    # Entry point generating the reports
    generate_reports()
