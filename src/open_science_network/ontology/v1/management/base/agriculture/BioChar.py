from enum import IntEnum


description = "Biochar is black carbon produced from biomass sources for the purpose of transforming the biomass " \
              "carbon into a more stable form (carbon sequestration)."


class UseBioChar(IntEnum):
    """
    The variable representing whether to use BioChar on the soil.
    """
    No = 0
    Yes = 1
