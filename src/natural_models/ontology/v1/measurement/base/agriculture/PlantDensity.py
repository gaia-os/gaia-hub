from enum import IntEnum


description = "The plant density is the number of individual plants present per unit of ground area."


class HempDensity(IntEnum):
    """
    The variable representing the hemp density per unit of ground area.
    """
    Continuous = 0


class AlfalfaDensity(IntEnum):
    """
    The variable representing the alfalfa density per unit of ground area.
    """
    Continuous = 0
