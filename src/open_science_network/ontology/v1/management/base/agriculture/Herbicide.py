from enum import IntEnum


description = "A substance that is toxic to plants, used to destroy unwanted vegetation."


class SpreadHerbicide(IntEnum):
    """
    The variable representing whether to spread the herbicide on the unwanted vegetation.
    """
    No = 0
    Yes = 1
