from enum import IntEnum


description = "The state of the atmosphere at a particular place and time as regards heat, cloudiness, " \
              "dryness, sunshine, wind, rain, etc..."


class WeatherCondition(IntEnum):
    """
    The variable representing the whether condition at a particular place and time.
    """
    Sun = 0
    Rain = 1
    Snow = 2
    Frost = 3
