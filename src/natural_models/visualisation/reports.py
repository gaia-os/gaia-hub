def display_reports(reports, start="\n", end=""):
    """
    Display useful information about the reports
    :param reports: the reports whose information must be displayed
    :param start: the string to display at the beginning of the function
    :param end: the string to display just before existing the function
    """

    # Display the starting string, e.g., "\n"
    print(end=start)

    # Iterate over all reports
    for t, report in reports:

        # Extract the report's index, the report's time step and report's lot
        report_id = report['report_id'].values.tolist()[0]
        t = report["t"].values.tolist()[0]
        lot = int(report["lot"].values.tolist()[0])
        print(f"Report[id: {report_id}, t: {t}, lot: {lot}")

        # Extract the name of the observation variables
        obs_names = report.columns.tolist()
        for col_name in ["report_id", "location", "datetime", "reporter", "t", "lot"]:
            obs_names.remove(col_name)

        # Display the observations value
        for obs_name in obs_names:
            print(f"\t- {obs_name}: ", report[obs_name].values.tolist()[0])
            print()

        # Display the end string, e.g., "\n"
        print(end=end)
