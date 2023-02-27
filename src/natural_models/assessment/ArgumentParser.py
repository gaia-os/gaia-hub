from pathlib import Path
import argparse


class ArgumentParser:
    """
    A class parsing the parameters of the 'perform_assessment.py' script
    """

    def __init__(self):
        """
        Construct the argument parser
        """
        self.parser = argparse.ArgumentParser(description='Script performing a project assessment.')
        self.parser.add_argument(
            '--project',
            type=str,
            help="Filepath of the project the Ent should watch.",
            default=None
        )
        self.parser.add_argument(
            '--update_frequency',
            type=int,
            default=-1,
            const=-1,
            nargs='?',
            help='Variational param update frequency: Every n steps. Always = -1 (default: %(default)s)'
        )
        self.parser.add_argument(
            '--var_param_dir',
            type=str,
            default='params',
            const='params',
            nargs='?',
            help='Directory with latest values of variational params'
        )

    def parse(self):
        """
        Parse the arguments of the 'perform_assessment.py' script
        """
        args = self.parser.parse_args()
        ddir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        args.reports_dir = ddir / "reports"
        args.var_param_dir = ddir / args.var_param_dir
        return args
