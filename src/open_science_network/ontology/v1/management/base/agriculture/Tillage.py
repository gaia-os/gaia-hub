from enum import IntEnum


description = "Tillage is the agricultural preparation of soil by mechanical agitation of various types, such " \
              "as digging, stirring, and overturning."


class TillageSoil(IntEnum):
    """
    The variable representing whether tillage is performed.
    """
    No = 0
    Yes = 1
