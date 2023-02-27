from enum import IntEnum


description = "The crop yields which is the harvested production per unit of harvested area for crop products."


class HempYield(IntEnum):
    """
    The variable representing the hemp yield gathered during harvest.
    """
    Continuous = 0


class AlfalfaYield(IntEnum):
    """
    The variable representing the alfalfa yield gathered during harvest.
    """
    Continuous = 0
