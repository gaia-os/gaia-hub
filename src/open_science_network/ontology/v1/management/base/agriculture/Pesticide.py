from enum import IntEnum


description = "A substance used for destroying insects or other organisms harmful to cultivated plants or to animals."


class SpreadPesticide(IntEnum):
    """
    The variable representing whether to spread the pesticide on the crops.
    """
    No = 0
    Yes = 1
