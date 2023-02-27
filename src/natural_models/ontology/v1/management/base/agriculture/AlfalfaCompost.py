from enum import IntEnum


description = "Compost made from alfalfa and other plant material and is rich in nitrogen and proteins."


class UseAlfalfaCompost(IntEnum):
    """
    The variable representing whether to use AlfalfaCompost on the soil.
    """
    No = 0
    Yes = 1
