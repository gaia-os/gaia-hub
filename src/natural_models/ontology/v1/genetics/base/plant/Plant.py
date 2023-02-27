from enum import IntEnum


description = "A living organism of the kind exemplified by trees, shrubs, herbs, grasses, ferns, and mosses, " \
              "typically growing in a permanent site, absorbing water and inorganic substances through its roots, " \
              "and synthesizing nutrients in its leaves by photosynthesis using the green pigment chlorophyll."


#  ==> RULE: every species added to this enum must also be added to the following enumerations:
#  ==> vX.intervention.base.agriculture.planting.PlantingSeeds
#  ==> vX.intervention.base.agriculture.harvest.HarvestCrops
class PlantSpecies(IntEnum):
    """
    A plant population/varietal whose evolution in a project is represented by one or more models.
    """
    Hemp = 0
    Copaiba = 1
    Andiroba = 2
    BrazilNut = 3
    Angelina = 4
    PauRose = 5
    Pupunha = 6
    Ucuuba = 7
    Cumaru = 8
    Puxuri = 9
    Muruci = 10
    Guarana = 11
    Cocoa = 12
    Cupuacu = 13
    Acai = 14
    Eucalyptus = 15
    Bacuri = 16
    Alfalfa = 17
    Vinifera = 18
    Almond = 19

