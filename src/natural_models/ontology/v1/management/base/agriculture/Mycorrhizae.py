from enum import IntEnum


description = "Mycorrhizae are fungi that form mutually beneficial relationships with plants. They  " \
              "can improve plant access to nutrients, nutrient uptake rate, and improve crop value " \
              "metrics like yield and quality."

class UseMycorrhizae(IntEnum):
    """
    The variable representing whether to use Mycorrhizae on the soil.
    """
    No = 0
    Yes = 1
