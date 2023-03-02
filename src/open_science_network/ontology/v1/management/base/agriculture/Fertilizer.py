from enum import IntEnum


description = "A chemical or natural substance added to soil or land to increase its fertility."


class FertilizeSoil(IntEnum):
    """
    The variable representing whether to fertilize the soil.
    """
    No = 0
    Yes = 1
