from enum import IntEnum


description = "The availability of a human labor force is often crucial to realizing agricultural outcomes " \
              "such as harvest, and can be effected by politics, natural disasters and more."


class LaborBool(IntEnum):
    """
    The variable representing whether Labor is available.
    """
    No = 0
    Yes = 1
