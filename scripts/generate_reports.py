from open_science_network.assessment.DataLoader import DataLoader
from open_science_network.assessment.ArgumentParser import ArgumentParser
from open_science_network.agents.AgentFactory import AgentFactory


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
