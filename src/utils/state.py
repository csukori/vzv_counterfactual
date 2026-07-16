# utils/state.py

import numpy as np
import pandas as pd

from src.utils.models import Scenario
from src.utils.config import GlobalConfig

# --- Imported data ---
WEEKLY_CASES: pd.Series | None = None
FULL_INDEX: pd.DatetimeIndex | None = None
WEEKLY_CASES_FULL: pd.Series | None = None
WEEKLY_CASES_FILLED: pd.Series | None = None
BEFORE_VACCINATION_MASK: np.ndarray[bool] | None = None
WITH_VACCINATION_MASK: np.ndarray[bool] | None = None
CASES_MATRIX: np.ndarray[float] | None = None
GLOBAL_MIN: float | None = None
GLOBAL_MAX: float | None = None

ANNUAL_CASES: pd.Series | None = None

# mask that helps to work with data relevant only in time period where we have age structured data
AGE_STRUCTURED_DATA_MASK: np.ndarray[bool] | None = None
# the date indices of data relevant only in time period where we have age structured data
AGE_STRUCTURED_DATA_INDEX: pd.DatetimeIndex | None = None

DEATHS_MATRIX: np.ndarray[float] | None = None

BIRTH_MATRIX: np.ndarray[float] | None = None
WEEKLY_BIRTH_SERIES: pd.Series | None = None
WEEKLY_V1: np.ndarray[float] | None = None        # weekly number of vaccinated infants (1st dose)
WEEKLY_V2: np.ndarray[float] | None = None        # weekly number of vaccinated infants (2nd dose)
WEEKLY_V1_AGE_STRUCTURED: np.ndarray[float] | None = None     # weekly age structured number of vaccinated individuals (1st dose)
WEEKLY_V2_AGE_STRUCTURED: np.ndarray[float] | None = None     # weekly age structured number of vaccinated individuals (2nd dose)

S0_VECTOR: np.ndarray[float] | None = None
N0_VECTOR: np.ndarray[float] | None = None
CONTACTS0: np.ndarray[float] | None = None

# --- Baseline model results (kezdetben None) ---

BASELINE_SCENARIO: Scenario | None = None
POPULATION: np.ndarray[float] | None = None     # age structured population (A×T)
REMAINDERS: np.ndarray[float] | None = None
R_A: np.ndarray[float] | None = None
RHO: np.ndarray[float] | None = None

WEEKLY_INDEX: pd.Series | None = None   # time index (T = NR_TIMESTEPS)
AGE_GROUPS: np.ndarray[str] | None = None     # list of age groups (A = NR_AGE_GROUPS)
NR_AGE_GROUPS: int | None = None
YEARS: np.ndarray[int] | None = None
NR_TIMESTEPS: int | None = None
AFTER_2016_1_1_MASK: bool | None = None
AFTER_2015_1_1_MASK: bool | None = None

NO_COVID_REMAINDERS: np.ndarray[float] | None = None
NO_COVID_BIRTHS: np.ndarray[float] | None = None
NO_COVID_DEATHS: np.ndarray[float] | None = None
MEAN_I: np.ndarray[float] | None = None
MEAN_REMAINDERS: np.ndarray[float] | None = None
MEAN_I_SCENARIO: Scenario | None = None

def set_age_groups_and_nr_age_groups(age_groups):
    global AGE_GROUPS, NR_AGE_GROUPS
    AGE_GROUPS = age_groups
    NR_AGE_GROUPS = len(age_groups)

def set_years(years):
    global YEARS
    YEARS = years

def set_weekly_index_related_values(weekly_index):
    global WEEKLY_INDEX, AFTER_2016_1_1_MASK, AFTER_2015_1_1_MASK
    WEEKLY_INDEX = weekly_index
    AFTER_2016_1_1_MASK = WEEKLY_INDEX > pd.Timestamp(2016, 1, 1)
    AFTER_2015_1_1_MASK = WEEKLY_INDEX > pd.Timestamp(2015, 1, 1)

def set_nr_timesteps(timestep):
    global NR_TIMESTEPS
    NR_TIMESTEPS = timestep

def set_weekly_cases(cases):
    global WEEKLY_CASES
    WEEKLY_CASES = cases

def set_full_index(full_index):
    global FULL_INDEX
    FULL_INDEX = full_index

def set_weekly_cases_related_values():
    if WEEKLY_CASES is None:
        raise ValueError("WEEKLY_CASES has not been initialized")
    global WEEKLY_CASES_FULL, WEEKLY_CASES_FILLED, BEFORE_VACCINATION_MASK, WITH_VACCINATION_MASK
    WEEKLY_CASES_FULL = WEEKLY_CASES.reindex(FULL_INDEX)
    WEEKLY_CASES_FILLED = WEEKLY_CASES_FULL.interpolate(method='linear')
    BEFORE_VACCINATION_MASK = WEEKLY_CASES_FILLED.index < GlobalConfig.START_OF_VACCINATION
    WITH_VACCINATION_MASK = WEEKLY_CASES_FILLED.index >= GlobalConfig.START_OF_VACCINATION

def set_cases_matrix_and_global_min_max(cases_matrix):
    global CASES_MATRIX, GLOBAL_MIN, GLOBAL_MAX
    CASES_MATRIX = cases_matrix
    GLOBAL_MIN = cases_matrix.min()
    GLOBAL_MAX = cases_matrix.max()

def set_annual_cases(annual_cases):
    global ANNUAL_CASES
    ANNUAL_CASES = annual_cases

def set_age_structured_data_mask(age_structured_data_mask):
    global AGE_STRUCTURED_DATA_MASK
    AGE_STRUCTURED_DATA_MASK = age_structured_data_mask

def set_age_structured_data_index(age_structured_data_index):
    global AGE_STRUCTURED_DATA_INDEX
    AGE_STRUCTURED_DATA_INDEX = age_structured_data_index

def set_death_matrix(death_matrix):
    global DEATHS_MATRIX
    DEATHS_MATRIX = death_matrix

def set_birth_matrix(birth_matrix):
    global BIRTH_MATRIX
    BIRTH_MATRIX = birth_matrix

def set_weekly_births(weekly_births):
    global WEEKLY_BIRTH_SERIES
    WEEKLY_BIRTH_SERIES = weekly_births

def set_weekly_v1(weekly_v1):
    global WEEKLY_V1
    WEEKLY_V1 = weekly_v1

def set_weekly_v1_age_structured():
    if NR_AGE_GROUPS is None or NR_AGE_GROUPS is None:
        raise ValueError("NR_AGE_GROUPS or NR_AGE_GROUPS has not been initialized")
    elif WEEKLY_V1 is None:
        raise ValueError("WEEKLY_V1 has not been initialized")
    global WEEKLY_V1_AGE_STRUCTURED
    WEEKLY_V1_AGE_STRUCTURED = np.zeros((NR_AGE_GROUPS, NR_TIMESTEPS))
    WEEKLY_V1_AGE_STRUCTURED[1,:] = WEEKLY_V1

def set_weekly_v2_age_structured():
    if NR_AGE_GROUPS is None or NR_AGE_GROUPS is None:
        raise ValueError("NR_AGE_GROUPS or NR_AGE_GROUPS has not been initialized")
    elif WEEKLY_V2 is None:
        raise ValueError("WEEKLY_V2 has not been initialized")
    global WEEKLY_V2_AGE_STRUCTURED
    WEEKLY_V2_AGE_STRUCTURED = np.zeros((NR_AGE_GROUPS, NR_TIMESTEPS))

def set_weekly_v2(weekly_v2):
    global WEEKLY_V2
    WEEKLY_V2 = weekly_v2

def set_s0_vector(s0_vector):
    global S0_VECTOR
    S0_VECTOR = s0_vector

def set_n0_vector(n0_vector):
    global N0_VECTOR
    N0_VECTOR = n0_vector

def set_contacts0(contacts0):
    global CONTACTS0
    CONTACTS0 = contacts0



def set_baseline_scenario(scenario):
    global BASELINE_SCENARIO
    BASELINE_SCENARIO = scenario

def set_population(population):
    global POPULATION
    POPULATION = population

def set_remainders(remainders):
    global REMAINDERS
    REMAINDERS = remainders

def set_r_a(r_a):
    global R_A
    R_A = r_a

def set_rho(rho):
    global RHO
    RHO = rho

def set_no_covid_remainders(no_covid_remainders):
    global NO_COVID_REMAINDERS
    NO_COVID_REMAINDERS = no_covid_remainders

def set_no_covid_births(no_covid_births):
    global NO_COVID_BIRTHS
    NO_COVID_BIRTHS = no_covid_births

def set_no_covid_deaths(no_covid_deaths):
    global NO_COVID_DEATHS
    NO_COVID_DEATHS = no_covid_deaths

def set_mean_i(mean_i):
    global MEAN_I
    MEAN_I = mean_i

def set_mean_remainders(mean_remainders):
    global MEAN_REMAINDERS
    MEAN_REMAINDERS = mean_remainders

def set_mean_i_scenario(mean_i_scenario):
    global MEAN_I_SCENARIO
    MEAN_I_SCENARIO = mean_i_scenario
