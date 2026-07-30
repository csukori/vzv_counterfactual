# utils/state.py

import numpy as np
import pandas as pd

from src.utils.models import Scenario
from src.utils.config import GlobalConfig


"""
Global state container for the epidemiological model.

This module stores all shared data structures that are produced,
transformed, or consumed throughout the modelling pipeline. It acts
as a central repository for:

- imported raw data (cases, deaths, births),
- preprocessed weekly and annual datasets,
- age‑structured matrices (cases, deaths, births, vaccinations),
- initial population vectors (S0, N0),
- contact matrices,
- baseline scenario results,
- global masks and time indices,
- model outputs such as Rₜ components and remainder terms.

The state module ensures that all parts of the model operate on a
consistent set of data without repeatedly passing large arrays
between functions. Setter functions are provided to update individual
state components in a controlled manner.

Notes
-----
This module behaves like a lightweight in‑memory database. All global
variables start as None and are populated step‑by‑step during the
data‑loading and preprocessing phase. The modelling and plotting
modules rely on these values being correctly initialized.
"""


# Weekly reported case counts (raw imported data).
WEEKLY_CASES: pd.Series | None = None

# Full weekly date index covering the entire simulation period.
FULL_INDEX: pd.DatetimeIndex | None = None

# Weekly cases aligned to FULL_INDEX (with missing weeks).
WEEKLY_CASES_FULL: pd.Series | None = None

# Weekly cases with missing values interpolated.
WEEKLY_CASES_FILLED: pd.Series | None = None

# Boolean masks separating pre‑vaccination and post‑vaccination periods.
BEFORE_VACCINATION_MASK: np.ndarray[bool] | None = None
WITH_VACCINATION_MASK: np.ndarray[bool] | None = None

# Age‑structured weekly case matrix (A × T).
CASES_MATRIX: np.ndarray[float] | None = None

# Global min/max values used for heatmap scaling.
GLOBAL_MIN: float | None = None
GLOBAL_MAX: float | None = None

# Annual case counts per age group.
ANNUAL_CASES: pd.Series | None = None

# Mask and index for the period where age‑structured data is available.
AGE_STRUCTURED_DATA_MASK: np.ndarray[bool] | None = None
AGE_STRUCTURED_DATA_INDEX: pd.DatetimeIndex | None = None

# Age‑structured weekly deaths.
DEATHS_MATRIX: np.ndarray[float] | None = None

# Age‑structured weekly births.
BIRTH_MATRIX: np.ndarray[float] | None = None

# Weekly birth counts (not age‑structured).
WEEKLY_BIRTH_SERIES: pd.Series | None = None

# Weekly vaccination counts (infants only).
WEEKLY_V1: np.ndarray[float] | None = None
WEEKLY_V2: np.ndarray[float] | None = None

# Age‑structured vaccination matrices.
WEEKLY_V1_AGE_STRUCTURED: np.ndarray[float] | None = None
WEEKLY_V2_AGE_STRUCTURED: np.ndarray[float] | None = None

# Initial susceptible and population vectors.
S0_VECTOR: np.ndarray[float] | None = None
N0_VECTOR: np.ndarray[float] | None = None

# Contact matrix between age groups.
CONTACTS0: np.ndarray[float] | None = None

# Baseline scenario results and model outputs.
BASELINE_SCENARIO: Scenario | None = None
POPULATION: np.ndarray[float] | None = None
REMAINDERS: np.ndarray[float] | None = None
R_A: np.ndarray[float] | None = None
RHO: np.ndarray[float] | None = None

# Weekly time index and age group metadata.
WEEKLY_INDEX: pd.Series | None = None
AGE_GROUPS: np.ndarray[str] | None = None
NR_AGE_GROUPS: int | None = None
YEARS: np.ndarray[int] | None = None
NR_TIMESTEPS: int | None = None

# Masks for specific time periods used in analysis.
AFTER_2016_1_1_MASK: bool | None = None
AFTER_2015_1_1_MASK: bool | None = None

# No‑COVID counterfactual components.
NO_COVID_BIRTHS: np.ndarray[float] | None = None
NO_COVID_DEATHS: np.ndarray[float] | None = None

# Mean values used in No-COVID scenarios.
MEAN_I: np.ndarray[float] | None = None
MEAN_REMAINDERS: np.ndarray[float] | None = None
MEAN_I_SCENARIO: Scenario | None = None

def set_age_groups_and_nr_age_groups(age_groups):
    """Store the list of age groups and compute the number of age groups."""
    global AGE_GROUPS, NR_AGE_GROUPS
    AGE_GROUPS = age_groups
    NR_AGE_GROUPS = len(age_groups)

def set_years(years):
    global YEARS
    YEARS = years

def set_weekly_index_related_values(weekly_index):
    """
    Store the global weekly time index and compute helper masks for
    time‑based filtering.

    Parameters
    ----------
    weekly_index : DatetimeIndex
        Weekly dates covering the full simulation period.

    Notes
    -----
    The function initializes:
    - WEEKLY_INDEX: the global weekly timeline,
    - AFTER_2016_1_1_MASK: marks weeks after 2016‑01‑01,
    - AFTER_2015_1_1_MASK: marks weeks after 2015‑01‑01.

    These masks are used throughout the model to restrict analyses
    to specific time windows (e.g., periods with reliable data).
    """
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
    """
    Initialize weekly case structures:
    - align cases to the full index,
    - interpolate missing weeks,
    - compute pre/post‑vaccination masks.
    """
    if WEEKLY_CASES is None:
        raise ValueError("WEEKLY_CASES has not been initialized")
    global WEEKLY_CASES_FULL, WEEKLY_CASES_FILLED, BEFORE_VACCINATION_MASK, WITH_VACCINATION_MASK
    WEEKLY_CASES_FULL = WEEKLY_CASES.reindex(FULL_INDEX)
    WEEKLY_CASES_FILLED = WEEKLY_CASES_FULL.interpolate(method='linear')
    BEFORE_VACCINATION_MASK = WEEKLY_CASES_FILLED.index < GlobalConfig.START_OF_VACCINATION
    WITH_VACCINATION_MASK = WEEKLY_CASES_FILLED.index >= GlobalConfig.START_OF_VACCINATION

def set_cases_matrix_and_global_min_max(cases_matrix):
    """
    Store the global age‑structured weekly case matrix and compute its
    minimum and maximum values.

    Parameters
    ----------
    cases_matrix : ndarray
        Age‑structured weekly case counts with shape (A × T).

    Notes
    -----
    The function initializes:
    - CASES_MATRIX: the full age‑by‑week case matrix,
    - GLOBAL_MIN: the smallest value in the matrix,
    - GLOBAL_MAX: the largest value in the matrix.

    These global extrema are used for consistent scaling in heatmaps
    and other visualizations that rely on fixed color ranges.
    """
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
    """
    Create the age‑structured weekly first‑dose vaccination matrix.

    Notes
    -----
    This function initializes a matrix of shape (A × T), where A is the
    number of age groups and T is the number of weekly timesteps. The
    weekly first‑dose vaccination counts (WEEKLY_V1) are assigned to the
    appropriate age group, while all other age groups receive zeros.

    The resulting matrix is used in the epidemiological model to ensure
    that vaccination dynamics are represented in an age‑structured form.
    """
    if NR_AGE_GROUPS is None or NR_AGE_GROUPS is None:
        raise ValueError("NR_AGE_GROUPS or NR_AGE_GROUPS has not been initialized")
    elif WEEKLY_V1 is None:
        raise ValueError("WEEKLY_V1 has not been initialized")
    global WEEKLY_V1_AGE_STRUCTURED
    WEEKLY_V1_AGE_STRUCTURED = np.zeros((NR_AGE_GROUPS, NR_TIMESTEPS))
    WEEKLY_V1_AGE_STRUCTURED[1,:] = WEEKLY_V1

def set_weekly_v2_age_structured():
    """
    Create the age‑structured weekly second‑dose vaccination matrix.

    Notes
    -----
    This function initializes a matrix of shape (A × T), where A is the
    number of age groups and T is the number of weekly timesteps. The
    weekly second‑dose vaccination counts (WEEKLY_V2) are inserted into
    the matrix in an age‑structured format.

    The matrix is used by the model to incorporate second‑dose immunity
    dynamics into age‑specific transmission and population processes.
    """
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
