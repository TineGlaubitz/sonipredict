# -*- coding: utf-8 -*-
META_ANALYSIS_MODEL = {
    "continuos_features": [
        "Size PP [nm]",
        "Concentration [mg/mL]",
        "Energy Density [J/mL]",
        "abs_zeta",
        "iep_stability",
        "BET [m2/g]",
        "Isoelectric Point",
        "Zeta Pot[mV]",
        "Volume [mL]",
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
