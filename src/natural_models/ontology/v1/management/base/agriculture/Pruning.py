from enum import IntEnum


description = "Pruning is the removal of plant parts like branches, buds, or roots for plant growth."


class PruneCrops(IntEnum):
    """
    The variable representing whether to prune the crops.
    """
    Yes = 1
    No = 0
