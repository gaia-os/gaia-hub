from enum import IntEnum


description = "Vermicompost is known to improve many crop yields and quality. Vermicompost has high " \
              "porosity and improves aeration, drainage, water-holding capacity and microbial activity. "


class UseVermiCompost(IntEnum):
    """
    The variable representing whether to use VermiCompost on the soil.
    """
    No = 0
    Yes = 1
