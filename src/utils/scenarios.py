from enum import StrEnum

class Scenarios(StrEnum):
    BASELINE = "Implemented scenario"
    VACC_LEVEL_75 = "Vaccination level 75%"
    VACC_LEVEL_50 = "Vaccination level 50%"
    VACC_LEVEL_25 = "Vaccination level 25%"
    NO_VACCINATION = "No vaccination"
    STARTING_1_YEAR_MINUS = "Starting 1 year earlier"
    STARTING_2_YEAR_MINUS = "Starting 2 year earlier"
    STARTING_3_YEAR_MINUS = "Starting 3 year earlier"
    STARTING_5_YEAR_MINUS = "Starting 5 year earlier"
    STARTING_8_YEAR_MINUS = "Starting 8 year earlier"
    STARTING_13_YEAR_MINUS = "Starting 13 year earlier"
