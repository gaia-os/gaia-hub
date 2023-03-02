import math
import random
from geojson_pydantic import Point
from typing import List
import logging
from shapely.geometry import Point as ShPoint
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json
from open_science_network.pydantic.Project import Project
from open_science_network.pydantic.Report import Report


class DataLoader:
    """
    A class used to load the data required for a project assessment
    """

    def __init__(self, project_file, reports_dir=None, verbose=False):
        """
        Load the data
        :param project_file: the file describing the project to load
        :param reports_dir: the directory containing the reports
        :param verbose: True if useful information should be displayed, False otherwise
        """

        # Store verbose parameter
        self.verbose = verbose

        # Load project and time horizon
        if reports_dir is not None and isinstance(reports_dir, str):
            reports_dir = Path(reports_dir)
        self.project = self.load_project(project_file)
        self.T = self.project.duration_in_years * 52

        # Load lot information
        self.lots = self.project.lots
        self.lot_polys = gpd.GeoDataFrame([
            {'lot_id': i, 'lot_name': l.name, 'bounds': l.bounds} for i, l in enumerate(self.lots)
        ], geometry='bounds')

        # Load the policies
        strategies_by_name = {strategy.name: strategy for strategy in self.project.strategies}
        self.policies = [strategies_by_name[lot.strategy].policy for lot in self.lots]

        # Load the all the reports
        self.obs_names = []
        self.reports = self.load_reports(reports_dir)

    @staticmethod
    def load_project(project_file):
        """
        Create the project described the file passed as argument
        :param project_file: the file of the project to create
        :return: the created project
        """
        if project_file is None:
            raise FileNotFoundError("project file  as not found")
        project_json = json.load(open(project_file))
        return Project(**project_json)

    def load_reports(self, report_dir):
        """
        Given a directory path containing evidence reports, build up and return a dataframe recording the observations
        :param report_dir: the directory containing the reports
        :returns: list of reports
        """

        # If the report directory is None, return an empty dataframe
        if report_dir is None:
            return gpd.GeoDataFrame()

        # Create the list of report files
        files = self.get_report_files(report_dir)

        # Create a dataframe with two columns
        default_cols = ['report_id', 'datetime', 'location', 'reporter', 't', 'lot']
        reports = None

        # Load all the reports
        for file in files:

            # Load the report from the current file
            print(file)
            report_json = json.load(file.open())
            report = Report(**report_json)

            # Retrieve all the observation names
            for obs in report.observations:
                if obs.name not in self.obs_names:
                    self.obs_names.append(obs.name)

            # Compute time step associated to the report
            t = self.datetime_to_t(report.datetime)

            # Ensure the report is relevant
            if not self.is_report_relevant(report, t):
                continue

            # Get list of relevant lots
            lots = self.relevant_lots(report.location)

            # Get all the reports columns
            cols = default_cols + list({obs.name for obs in report.observations})

            # Add the reported observations to all relevant lots
            for lot in lots:

                # Find a point in the lot polygon
                location = self.find_point_in(self.lots[lot])

                # Create a new dataframe containing the data from the current report
                entry = [report.id, report.datetime, location, report.reporter, int(t), lot] + \
                        [obs.value for obs in report.observations if obs.lot_name == self.lots[lot].name]

                new_report = gpd.GeoDataFrame(data={key: val for key, val in zip(cols, entry)}, geometry='location')

                # If no reports added yet, create a dataframe containing the observation
                reports = new_report if reports is None else pd.concat([reports, new_report])

                # TODO Extends this function to work with vectors of values
                # TODO Extends this to reports with different measurements

        # If number of rows in the dataframe is zero, warn the user that no reports have been found
        if reports is None or len(reports.index) == 0:
            logging.warning("No compatible reports found.")
            return gpd.GeoDataFrame()

        # Display the reports, if requested
        if self.verbose:
            self.display_reports(reports)

        # Group the reports by time steps
        return reports.groupby('t', group_keys=True)

    @staticmethod
    def get_report_files(report_dir):
        """
        Getter
        :param report_dir: the directory to scoot for report files
        :return: the report files
        """
        files = [report_dir / file for file in report_dir.iterdir() if (report_dir / file).exists()]
        return filter(lambda file: file is not None, sorted(files))

    @staticmethod
    def display_reports(reports):
        """
        Display the reports passed as parameter
        :param reports: the reports to display
        """
        print(f"[INFO] Reports' columns: {', '.join(reports.columns.tolist())}")
        print("[INFO] Reports' rows:")
        print("[INFO] ")
        reports = reports.reset_index()
        for index, row in reports.iterrows():
            for col in reports.columns.tolist():
                print("[INFO] ", col, ": ", row[col])
            print("[INFO] ")

    @staticmethod
    def find_smallest_box(coordinates):
        """
        Find the smallest box containing the entire polygon
        :param coordinates: the coordinates of the polygon boundary
        :return: (x_min, x_max, y_min, y_max), i.e.

                y_max #===================#
                      #                   #
                      #                   #
                y_min #===================#
                    x_min               x_max
        """
        y_min = math.inf
        x_min = math.inf
        y_max = -math.inf
        x_max = -math.inf
        for (x, y) in coordinates:
            x_max = x if x > x_max else x_max
            x_min = x if x < x_min else x_min
            y_max = y if y > y_max else y_max
            y_min = y if y < y_min else y_min
        return x_min, x_max, y_min, y_max

    @staticmethod
    def find_point_in(lot):
        """
        Find a point located with the lot boundary
        :param lot: the lot
        :return: the point
        """

        # Get the boundary coordinates
        coordinates = lot.bounds.coordinates[0]

        # If no coordinates are available, then raise an error
        if len(coordinates) == 0:
            raise ValueError("Error: the lot boundary does not contain any coordinate.")

        # If only one coordinates is available, then return it
        if len(coordinates) == 1:
            return coordinates[0]

        # Find the smallest box containing the entire polygon
        x_min, x_max, y_min, y_max = DataLoader.find_smallest_box(coordinates)

        # Monte Carlo search for a point with the polygon
        polygon = gpd.GeoSeries([lot.bounds], index=range(0, 1))
        while True:
            point = ShPoint(
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max)
            )
            if polygon.contains(point).all():
                return point

    def is_report_relevant(self, report, t):
        """
        Check whether the report is relevant
        :param report: the report
        :param t: the time step of the report
        :return: True if the report is relevant, False otherwise
        """
        try:
            # Check that the time step is within the project duration
            if t < 0 or t > self.T:
                raise ValueError(f"A report has datetime outside project duration, skipping: {report.datetime}")

            # For project specific report, check that the report project matches the current project
            if report.project_name is not None and report.project_name != self.project.name:
                raise ValueError(f"Report {report.id} refers to another project called: {report.project_name}")

        except ValueError as e:
            # The report is irrelevant
            logging.warning(e)
            return False

        # The report is relevant
        return True

    def relevant_lots(self, loc):
        """
        Turn locations into lots
        :param loc: the location(s)
        :return: the lots
        """

        # If the location is a single point, create a list with a single element
        if isinstance(loc, Point):
            loc = [loc]

        # Ensure that the locations are stored in a list
        if not isinstance(loc, List):
            raise ValueError(f"Bad location type for {loc}")

        # Get the lots from the list of points
        points = gpd.GeoDataFrame(geometry=[ShPoint(location.coordinates) for location in loc])
        point_in_polys = gpd.tools.sjoin(points, self.lot_polys, predicate="within", how='left')
        lots = point_in_polys.lot_id.values.tolist()

        # Make sure no location is None
        return filter(lambda lot: lot is not None, lots)

    def datetime_to_t(self, d):
        """
        Turn a datetime into a time step
        :param d: the datetime
        :return: the time step
        """
        return (d - self.project.start_date).days // 7
