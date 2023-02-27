from enum import Enum


class LocalName(Enum):
    """
    Mapping between local variable names (used by the generative model) and global, self-describing variable names
    (used on reports and configs). Global names are represented as dictionaries to bypass issues with attribute
    ordering. It is not required for a variable to have a local name, but it makes the model specification much easier
    to understand.
    """
    soc = dict(var='soil_organic_carbon', measure='density', units='none', method='lab_analysis')
    plant_density_hemp = dict(var='plants', dims='species', measure='density', units='_m2', method='in_situ_observation', dim='hemp')
    plant_height_hemp = dict(var='plants', dims='species', measure='height', units='m', method='in_situ_observation', dim='hemp')
    cum_seeds_planted_hemp = dict(var='seeds_planted', dims='species', measure='count', cumulative='1', units='n', method='tractor_counter', dim='hemp')
    cum_harvest_yield_hemp = dict(var='harvest_yield', dims='species', measure='count', cumulative='1', units='kg', method='tractor_counter', dim='hemp')
    irrigation_vol = dict(var='irrigation', measure='volume', units='l', method='iot_counter_1234')
    interventions_base = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='base')
    interventions_irrigation = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='irrigation')
    interventions_fertilizer = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='fertilizer')
    interventions_pesticide = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='pesticide')
    interventions_herbicide = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='herbicide')
    interventions_biochar = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='biochar')
    interventions_tilling = dict(var='interventions', measure='one_hot', method='iot_counter_1234', dim='tilling')
    soc_albo = dict(var='soil_organic_carbon', measure='density', units='g_g', method='albo_soc_ag_model')
    tpb_albo = dict(var='plants', measure='biomass', units='kg', method='albo_plant_biomass_ag_model')
    pbc_albo = dict(var='plants', measure='biomass_carbon', units='kg', method='albo_biomass_carbon_ag_model')
