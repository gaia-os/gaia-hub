# Import required classes from the natural_models package
from natural_models.agents.AgentFactory import AgentFactory
from natural_models.assessment.DataLoader import DataLoader
import os
from natural_models.visualisation.distributions import draw_beliefs
from natural_models.visualisation.reports import display_reports
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load the data corresponding to the GetStarted model
    root_dir = os.getenv('DG_NATURAL_MODELS_REPOSITORY')
    project_conf = f'{root_dir}/data/projects/Tutorial/GetStarted.json'
    report_dir = f'{root_dir}/data/reports/'
    data_loader = DataLoader(project_conf, report_dir)

    display_reports(data_loader.reports)

    # Load the agent(s) compatible with the loaded data, and retrieve the GetStarted agent
    agents = AgentFactory.create(data_loader)
    agent = next(filter(lambda a: a.name == "Tutorial.GetStarted", agents))

    # Predict the future using the GetStarted agent
    prediction_samples = agent.predict(model=agent.model, num_samples=1)

    # Visualise the predictions
    fig = draw_beliefs(prediction_samples, var_1={
        "hemp_size": "Hemp size",
        "hemp_can_grow": "Hemp can grow",
    }, var_2={
        "hemp_grow_rate": "Hemp growth rate",
        "obs_yield": "Yield",
    }, measured=[False, False])
    plt.show()
