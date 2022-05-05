# -*- coding: utf-8 -*-
META_ANALYSIS_MODEL = {
    "continuos_features": [
        "Size PP [nm]",
        "Concentration [mg/mL]",
        "Energy Density [J/mL]",
        "Isoelectric Point",
        "Zeta Pot[mV]",
        "Volume [mL]",
        "Total Energy [J]",
    ],
    "cat_features": ["Coating", "Particle"],
}

LAB_DATA_MODEL = {
    "continuos_features": [
        "Size PP [nm]",
        "Concentration [mg/mL]",
        "Volume [mL]",
        "Time [min]",
        "Total Energy [J]",
        "Energy Density [J/mL]",
    ],
    "cat_features": ["Type"],
}


shap_global_feature_map_merged = [
    {
        "name": "size PP",
        "features": ["power_transform__Size PP [nm]"],
    },
    {
        "name": "concentration",
        "features": ["power_transform__Concentration [mg/mL]"],
    },
    {
        "name": "energy density",
        "features": ["power_transform__Energy Density [J/mL]"],
    },
    {
        "name": "zeta potential",
        "features": ["power_transform__Zeta Pot[mV]"],
    },
    {"name": "volume", "features": ["power_transform__Volume [mL]"]},
    {
        "name": "IEP",
        "features": ["power_transform__Isoelectric Point"],
    },
    {
        "name": "coating",
        "features": ["one_hot__Coating_Hydrophob"],
    },
    {
        "name": "energy",
        "features": ["power_transform__Total Energy [J]"],
    },
    {
        "name": "particle type",
        "features": [
            "one_hot__Particle_CeO2",
            "one_hot__Particle_SiO2",
            "one_hot__Particle_TiO2",
            "one_hot__Particle_ZnO",
        ],
    },
]

shap_global_feature_map_lab = [
    {
        "name": "size PP",
        "features": ["power_transform__Size PP [nm]"],
    },
    {
        "name": "concentration",
        "features": ["power_transform__Concentration [mg/mL]"],
    },
    {
        "name": "energy density",
        "features": ["power_transform__Energy Density [J/mL]"],
    },
    {
        "name": "energy",
        "features": ["power_transform__Total Energy [J]"],
    },
    {"name": "volume", "features": ["power_transform__Volume [mL]"]},
    {
        "name": "sonicator type",
        "features": ["one_hot__Type_1.0"],
    },
]
