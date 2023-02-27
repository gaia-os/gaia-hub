from enum import IntEnum


description = "Rhizobacteria infect host plant root nodules and offer plants access to increased levels of nitrogen " \
              "and new evidence suggests they could have protective qualities against diseases and pests."


class UseRhizobacteria(IntEnum):
    """
    The variable representing whether to use Rhizobacteria on the soil.
    """
    No = 0
    Yes = 1
