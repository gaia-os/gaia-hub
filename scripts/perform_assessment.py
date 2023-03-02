from open_science_network.assessment.Engine import Engine as AssessmentEngine


def perform_assessment():
    """
    Performing the project assessment
    :return: a dictionary mapping model name to the model results, where the model results is a list of tuple
    containing the samples and expected free energy of the assessment at each time step
    """
    # Create the assessment engine and perform the assessment
    return AssessmentEngine(verbose=True, debug=True).perform_assessment()


if __name__ == '__main__':
    # Entry point performing project assessment
    perform_assessment()
