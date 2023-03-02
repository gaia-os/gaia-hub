from enum import IntEnum


description = "Cow manure is rich in nutrients and has an approximately 3:2:1 ratio of NPK."


class UseCowManure(IntEnum):
    """
    The variable representing whether to use CowManure on the soil.
    """
    No = 0
    Yes = 1
