from enum import IntEnum


description = "The supply of water to land or crops to help growth, typically by means of channels."


class IrrigateCrops(IntEnum):
    """
    The variable representing whether to irrigate the crops.
    """
    No = 0
    Yes = 1
