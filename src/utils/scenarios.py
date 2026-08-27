from enum import StrEnum

class Scenarios(StrEnum):
    """
    Enumeration of epidemiological scenarios used in the model.

    Each scenario represents a distinct vaccination or intervention
    setting under which the simulation can be run. These labels are
    used throughout the pipeline to select input data, configure
    model parameters, and generate comparative visualizations.

    Notes
    -----
    - BASELINE: the implemented real-world scenario.
    - VACC_LEVEL_*: hypothetical vaccination coverage levels.
    - NO_VACCINATION: counterfactual scenario without vaccination.
    - STARTING_*_YEAR_MINUS: hypothetical vaccination starting points.
    - DETECTION_RATE_*: scenarios to test the effect of detection rate.
    - NO_COVID_*: similar hypothetical scenarios, but as if no COVID‑19 pandemic had occurred.

    The enum ensures consistent naming and prevents accidental
    mismatches when referencing scenarios across modules.
    """
    BASELINE = "Implemented scenario"
    VACC_LEVEL_75 = "Vaccination level 75%"
    VACC_LEVEL_50 = "Vaccination level 50%"
    VACC_LEVEL_25 = "Vaccination level 25%"
    VACC_LEVEL_10 = "Vaccination level 10%"
    NO_VACCINATION = "No vaccination"
    STARTING_1_YEAR_MINUS = "Starting 1 year earlier"
    STARTING_2_YEAR_MINUS = "Starting 2 year earlier"
    STARTING_3_YEAR_MINUS = "Starting 3 year earlier"
    STARTING_5_YEAR_MINUS = "Starting 5 year earlier"
    STARTING_8_YEAR_MINUS = "Starting 8 year earlier"
    STARTING_13_YEAR_MINUS = "Starting 13 year earlier"
    STARTING_1_YEAR_PLUS = "Starting 1 year later"
    STARTING_2_YEAR_PLUS = "Starting 2 year later"
    STARTING_3_YEAR_PLUS = "Starting 3 year later"
    NO_COVID_BASELINE = "No COVID"
    NO_COVID_VACC_LEVEL_75 = "No COVID, vaccination level 75%"
    NO_COVID_VACC_LEVEL_50 = "No COVID, vaccination level 50%"
    NO_COVID_VACC_LEVEL_25 = "No COVID, vaccination level 25%"
    NO_COVID_NO_VACC = "No COVID and no vaccination"
    NO_COVID_VACC_1_YEAR_MINUS = "No COVID, starting vaccination level 1 year earlier"
    NO_COVID_VACC_2_YEAR_MINUS = "No COVID, starting vaccination level 2 year earlier"
    NO_COVID_VACC_3_YEAR_MINUS = "No COVID, starting vaccination level 3 year earlier"
    NO_COVID_VACC_5_YEAR_MINUS = "No COVID, starting vaccination level 5 year earlier"
    NO_COVID_VACC_8_YEAR_MINUS = "No COVID, starting vaccination level 8 year earlier"
    NO_COVID_VACC_13_YEAR_MINUS = "No COVID, starting vaccination level 13 year earlier"
    DETECTION_50_BASELINE = "Baseline scenario with a 50% detection rate"
    DETECTION_50_NO_VACCINATION = "No vaccination with a 50% detection rate"
    DETECTION_60_BASELINE = "Baseline scenario with a 60% detection rate"
    DETECTION_60_NO_VACCINATION = "No vaccination with a 60% detection rate"
    DETECTION_60_VACC_LEVEL_75 = "Vaccination level 75% with a 60% detection rate"
    DETECTION_60_VACC_LEVEL_25 = "Vaccination level 25% with a 60% detection rate"
    DETECTION_60_VACC_LEVEL_10 = "Vaccination level 10% with a 60% detection rate"
    DETECTION_70_BASELINE = "Baseline scenario with a 70% detection rate"
    DETECTION_70_NO_VACCINATION = "No vaccination with a 70% detection rate"
    DETECTION_70_VACC_LEVEL_75 = "Vaccination level 75% with a 70% detection rate"
    DETECTION_70_VACC_LEVEL_25 = "Vaccination level 25% with a 70% detection rate"
    DETECTION_70_VACC_LEVEL_10 = "Vaccination level 10% with a 70% detection rate"
    DETECTION_80_BASELINE = "Baseline scenario with a 70% detection rate"
    DETECTION_80_NO_VACCINATION = "No vaccination with a 80% detection rate"
    DETECTION_80_VACC_LEVEL_75 = "Vaccination level 75% with a 80% detection rate"
    DETECTION_80_VACC_LEVEL_25 = "Vaccination level 25% with a 80% detection rate"
    DETECTION_80_VACC_LEVEL_10 = "Vaccination level 10% with a 80% detection rate"
    DETECTION_90_BASELINE = "Baseline scenario with a 90% detection rate"
    DETECTION_90_NO_VACCINATION = "No vaccination with a 90% detection rate"
    DETECTION_90_VACC_LEVEL_75 = "Vaccination level 75% with a 90% detection rate"
    DETECTION_90_VACC_LEVEL_25 = "Vaccination level 25% with a 90% detection rate"
    DETECTION_90_VACC_LEVEL_10 = "Vaccination level 10% with a 90% detection rate"
    DETECTION_100_BASELINE = "Baseline scenario with a 100% detection rate"
    DETECTION_100_NO_VACCINATION = "No vaccination with a 100% detection rate"
    DETECTION_100_VACC_LEVEL_75 = "Vaccination level 75% with a 100% detection rate"
    DETECTION_100_VACC_LEVEL_25 = "Vaccination level 25% with a 100% detection rate"
    DETECTION_100_VACC_LEVEL_10 = "Vaccination level 10% with a 100% detection rate"
