from numpy import ndarray
from pandas import DataFrame, Series, DatetimeIndex
import pandas as pd
import numpy as np

from src.utils.config import GlobalConfig
import src.utils.state as state

# Detection rate before vaccination rollout.
# Represents the estimated proportion of infections that were detected
# in surveillance data prior to the start of vaccination.
DETECTION_RATE_BEFORE: float = 0.4

# Detection rate after vaccination rollout.
# Assumes improved or stable detection efficiency in the post‑vaccination period.
DETECTION_RATE_AFTER: float = 0.4



def convert_df_cases_to_weekly_cases(df_cases: DataFrame) -> Series | DataFrame:
    """
    Convert a table of weekly case counts (e.g. '2020_1') into a
    time-indexed weekly case Series.

    Parameters
    ----------
    df_cases : DataFrame
        Input table containing columns:
        - 'Week' in the format 'YYYY_WW'
        - 'Cases' giving weekly case counts.

    Returns
    -------
    Series
        Weekly case counts indexed by the Monday of each ISO week.

    Notes
    -----
    The function:
    - splits 'YYYY_WW' into year and week number,
    - converts ISO week numbers into actual dates,
    - adjusts case counts using detection rates before/after vaccination start.
    """
    df_cases[["Year", "WeekNum"]] = df_cases["Week"].str.split("_", expand=True)
    df_cases["Year"] = df_cases["Year"].astype(int)
    df_cases["WeekNum"] = df_cases["WeekNum"].astype(int)

    df_cases["Date"] = pd.to_datetime(
        df_cases["Year"].astype(str) + df_cases["WeekNum"].astype(str) + "1",
        format="%G%V%u"
    )

    df_weekly_cases = df_cases.set_index("Date").sort_index()
    weekly_cases = df_weekly_cases["Cases"].copy()

    weekly_cases.loc[weekly_cases.index < GlobalConfig.START_OF_VACCINATION] *= 1 / DETECTION_RATE_BEFORE
    weekly_cases.loc[weekly_cases.index >= GlobalConfig.START_OF_VACCINATION] *= 1 / DETECTION_RATE_AFTER
    return weekly_cases


def create_full_index_date_range(max_date) -> DatetimeIndex:
    """
    Create a continuous weekly date index from the start of the
    age‑structured dataset by a given maximum date.

    Parameters
    ----------
    max_date : datetime-like
        Last date of the desired weekly index.

    Returns
    -------
    DatetimeIndex
        Weekly dates (Mondays) covering the full simulation period.
    """
    return pd.date_range(
        start=GlobalConfig.START_OF_AGE_STRUCTURED_DATE,
        end=max_date,
        freq="W-MON"
    )


def convert_df_age_structured_cases(df_age_structured_cases: DataFrame) -> Series | DataFrame:
    """
    Clean and standardize an age‑structured annual case table.

    Parameters
    ----------
    df_age_structured_cases : DataFrame
        Table where the first column contains age group labels.

    Returns
    -------
    DataFrame
        Annual case counts indexed by cleaned age group labels.

    Notes
    -----
    The function:
    - trims whitespace,
    - replaces unicode dashes with ASCII hyphens,
    - sets the first column as index.
    """
    annual_cases = df_age_structured_cases.set_index(df_age_structured_cases.columns[0])
    annual_cases.index = (
        annual_cases.index.map(str).str.strip().str.replace("–", "-", regex=False)
    )
    return annual_cases


def create_case_matrix(annual_cases, weekly_cases_filled, weekly_cases_full) -> ndarray:
    """
    Distribute annual age‑structured case counts into weekly values.

    Parameters
    ----------
    annual_cases : DataFrame
        Annual case counts per age group.
    weekly_cases_filled : Series
        Weekly total case counts with missing weeks filled.
    weekly_cases_full : Series
        Full weekly index for the simulation.

    Returns
    -------
    ndarray
        Weekly case matrix of shape (A × T), where:
        A = number of age groups,
        T = number of weeks.

    Notes
    -----
    Annual cases are distributed proportionally across weeks based on
    the relative share of weekly cases within each year.
    """
    weekly_age_structured = {}

    for age in state.AGE_GROUPS:
        s = pd.Series(np.zeros(len(weekly_cases_full.index), dtype=float), index=weekly_cases_full.index)
        for year in state.YEARS:
            annual_value = annual_cases.loc[age, year] * (1 / DETECTION_RATE_BEFORE)
            props = weekly_proportions_for_year(weekly_cases_filled, year)
            mask = weekly_cases_filled.index.year == year
            s.loc[mask] = props * annual_value
        weekly_age_structured[age] = s

    return np.column_stack([weekly_age_structured[age].values for age in state.AGE_GROUPS]).T


def weekly_proportions_for_year(weekly_cases_filled, year):
    """
    Compute the proportion of weekly cases within a given year.

    Parameters
    ----------
    weekly_cases_filled : Series
        Weekly case counts.
    year : int
        Target year.

    Returns
    -------
    Series
        Weekly proportions summing to 1 within the given year.
    """
    mask = weekly_cases_filled.index.year == year
    year_data = weekly_cases_filled[mask]
    total = year_data.sum()
    return year_data / total


def convert_annual_deaths(df_deaths: DataFrame) -> Series | DataFrame:
    """
    Clean and standardize an age‑structured annual death table.

    Parameters
    ----------
    df_deaths : DataFrame
        Table where the first column contains age group labels.

    Returns
    -------
    DataFrame
        Annual death counts indexed by cleaned age group labels.
    """
    annual_deaths = df_deaths.set_index(df_deaths.columns[0])
    annual_deaths.index = (
        annual_deaths.index.map(str).str.strip().str.replace("–", "-", regex=False)
    )
    return annual_deaths


def create_weekly_death_matrix(annual_deaths: Series, index) -> ndarray:
    """
    Convert annual death counts into weekly values.

    Parameters
    ----------
    annual_deaths : DataFrame or Series
        Annual death counts per age group.
    index : DatetimeIndex
        Weekly date index.

    Returns
    -------
    ndarray
        Weekly death matrix of shape (A × T).
    """
    weekly_deaths = {}
    for age in state.AGE_GROUPS:
        s = pd.Series(np.zeros(len(index), dtype=float), index=index)
        for year in state.YEARS:
            annual_value = annual_deaths.loc[age, year]
            mask = index.year == year
            s.loc[mask] = annual_value / 52
        weekly_deaths[age] = s

    return np.column_stack([weekly_deaths[age].values for age in state.AGE_GROUPS]).T


def convert_annual_v1_data_to_weekly_v1(df_vaccines: DataFrame) -> Series:
    """
    Convert annual first‑dose vaccination counts into weekly values.

    Parameters
    ----------
    df_vaccines : DataFrame
        Table containing columns 'Year' and '1st dose'.

    Returns
    -------
    Series
        Weekly first‑dose vaccination counts indexed by week.

    Notes
    -----
    Weekly vaccination counts are distributed proportionally based on
    weekly birth counts. This ensures that age‑specific vaccination
    coverage follows real‑world dynamics and that the intra‑year
    distribution of immune individuals mirrors the seasonal pattern
    of infections.
    """
    v1_yearly = dict(zip(df_vaccines["Year"], df_vaccines["1st dose"]))
    weekly_v1 = pd.Series(index=state.WEEKLY_INDEX, dtype=float)

    annual_birth_series = state.WEEKLY_BIRTH_SERIES.groupby(state.WEEKLY_BIRTH_SERIES.index.year).sum()
    mask_2019_after_vaccination = (
        (GlobalConfig.START_OF_VACCINATION < state.WEEKLY_INDEX) &
        (state.WEEKLY_INDEX <= pd.Timestamp('2019-12-31'))
    )
    births_in_2019_after_vaccination = (
        state.WEEKLY_BIRTH_SERIES[mask_2019_after_vaccination]
        .groupby(state.WEEKLY_BIRTH_SERIES.index[mask_2019_after_vaccination].year)
        .sum()
    )

    for date in state.WEEKLY_INDEX:
        year = date.year
        if date < GlobalConfig.START_OF_VACCINATION:
            weekly_v1.loc[date] = 0
        else:
            target = date + pd.DateOffset(months=-13)
            monday = target - pd.Timedelta(days=target.weekday())
            total_births = (
                births_in_2019_after_vaccination[date.year]
                if date.year == 2019
                else annual_birth_series.loc[date.year]
            )
            birth_prop = state.WEEKLY_BIRTH_SERIES.loc[monday] / total_births
            weekly_v1.loc[date] = v1_yearly[year] * birth_prop

    return weekly_v1


def convert_annual_v2_data_to_weekly_v2(df_vaccines: DataFrame) -> Series:
    """
    Convert annual second‑dose vaccination counts into weekly values.

    Parameters
    ----------
    df_vaccines : DataFrame
        Table containing columns 'Year' and '2nd dose'.

    Returns
    -------
    Series
        Weekly second‑dose vaccination counts indexed by week.

    Notes
    -----
    Second‑dose vaccinations begin only after a delay following the
    start of vaccination, reflecting real-world rollout timing.
    """
    v2_yearly = dict(zip(df_vaccines["Year"], df_vaccines["2nd dose"]))
    weekly_v2 = pd.Series(index=state.WEEKLY_INDEX, dtype=float)

    for date in state.WEEKLY_INDEX:
        year = date.year
        if date < GlobalConfig.START_OF_VACCINATION + pd.DateOffset(months=3):
            weekly_v2.loc[date] = 0
        else:
            weekly_v2.loc[date] = v2_yearly[year] / 52

    return weekly_v2


def convert_annual_birth_data_to_birth_mtx(df_births: DataFrame) -> ndarray:
    """
    Convert annual birth counts into a weekly birth matrix.

    Parameters
    ----------
    df_births : DataFrame
        Table containing columns 'Year' and 'Cases' (annual births).

    Returns
    -------
    ndarray
        Weekly birth matrix of shape (A × T), with births assigned to
        the youngest age group.
    """
    births_yearly = dict(zip(df_births["Year"], df_births["Cases"]))
    weekly_births = pd.Series(index=state.WEEKLY_INDEX, dtype=float)

    for date in state.WEEKLY_INDEX:
        weekly_births.loc[date] = births_yearly[date.year] / 52

    birth_mtx = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    birth_mtx[0, :] = weekly_births
    state.set_weekly_births(weekly_births)
    return birth_mtx


def convert_daily_birth_data_to_birth_mtx(df_daily_births: DataFrame) -> ndarray:
    """
    Convert daily birth counts into a weekly birth matrix.

    Parameters
    ----------
    df_daily_births : DataFrame
        Table containing daily birth counts in a wide format.

    Returns
    -------
    ndarray
        Weekly birth matrix of shape (A × T), with births assigned to
        the youngest age group.

    Notes
    -----
    The function:
    - reshapes daily data into long format,
    - constructs valid dates,
    - resamples to weekly totals,
    - aligns weeks to Monday,
    - stores weekly births in the global state.
    """
    years_row = df_daily_births.iloc[0].ffill()
    months_row = df_daily_births.iloc[1]
    days = df_daily_births.iloc[2:]
    days.index = days.index - 1

    births_long = (
        days.melt(ignore_index=False, var_name="col", value_name="births")
        .reset_index()
        .rename(columns={"index": "day"})
    )
    births_long["year"] = births_long["col"].map(years_row.to_dict())
    births_long["month"] = births_long["col"].map(months_row.to_dict())
    births_long["date"] = pd.to_datetime(
        births_long[["year", "month", "day"]],
        errors="coerce"
    )

    daily_births = births_long.dropna(subset=["date"]).set_index("date")["births"]
    first_monday = daily_births.index[daily_births.index.weekday == 0][0]
    daily_births_from_monday = daily_births.loc[first_monday:]

    weekly_births = daily_births_from_monday.resample("W-SUN").sum()
    weekly_births.index = weekly_births.index - pd.Timedelta(days=6)

    birth_mtx = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    birth_mtx[0, :] = weekly_births
    state.set_weekly_births(weekly_births)
    return birth_mtx


def convert_weekly_death_prop_data_to_death_prop_mtx(df_death_proportions: DataFrame) -> ndarray:
    """
    Convert weekly age‑structured death proportions into a matrix.

    Parameters
    ----------
    df_death_proportions : DataFrame
        Table containing columns:
        - 'Week' in 'YYYY_WW' format,
        - age‑specific death proportions.

    Returns
    -------
    ndarray
        Weekly death proportion matrix of shape (A × T).
    """
    df_death_proportions[["Year", "WeekNum"]] = df_death_proportions["Week"].str.split("_", expand=True)
    df_death_proportions["Year"] = df_death_proportions["Year"].astype(int)
    df_death_proportions["WeekNum"] = df_death_proportions["WeekNum"].astype(int)

    df_death_proportions["Date"] = pd.to_datetime(
        df_death_proportions["Year"].astype(str)
        + "-W"
        + df_death_proportions["WeekNum"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    df_weekly_death_proportions = df_death_proportions.set_index("Date").sort_index()
    age_cols = [col for col in df_weekly_death_proportions.columns if col not in ["Week", "Year", "WeekNum"]]
    return df_weekly_death_proportions[age_cols].T.values


def convert_data_to_s0(df_susceptible: DataFrame) -> ndarray:
    """
    Convert an age‑structured susceptible population table into an array.

    Parameters
    ----------
    df_susceptible : DataFrame
        Table containing age group labels and susceptible counts.

    Returns
    -------
    ndarray
        Initial susceptible counts per age group.
    """
    df_susceptible.iloc[:, 0] = (
        df_susceptible.iloc[:, 0].astype(str).str.strip().str.replace("–", "-", regex=False)
    )
    df_susceptible.iloc[:, 1] = pd.to_numeric(df_susceptible.iloc[:, 1], errors="coerce")
    df_s0 = df_susceptible.set_index(df_susceptible.columns[0])
    return df_s0.iloc[:, 0].values.astype(float)


def convert_data_to_n0(df_population: DataFrame) -> ndarray:
    """
    Convert an age‑structured population table into an array.

    Parameters
    ----------
    df_population : DataFrame
        Table containing age group labels and population counts.

    Returns
    -------
    ndarray
        Initial population counts per age group.
    """
    df_population.iloc[:, 0] = (
        df_population.iloc[:, 0].astype(str).str.strip().str.replace("–", "-", regex=False)
    )
    df_population.iloc[:, 1] = pd.to_numeric(df_population.iloc[:, 1], errors="coerce")
    df_n0 = df_population.set_index(df_population.columns[0])
    return df_n0.iloc[:, 0].values.astype(float)

def convert_data_to_contact_matrix(df_contacts: DataFrame) -> ndarray:
    """
    Convert a contact matrix table into a numerical matrix.

    Parameters
    ----------
    df_contacts : DataFrame
        Table where rows and columns represent age groups and values
        represent average contact rates.

    Returns
    -------
    ndarray
        Contact matrix of shape (A × A).

    Notes
    -----
    The function cleans age group labels by removing whitespace and
    replacing unicode dashes with ASCII hyphens.
    """
    df_contacts.index = (
        df_contacts.index.astype(str).str.strip().str.replace("–", "-", regex=False)
    )
    df_contacts.columns = (
        df_contacts.columns.astype(str).str.strip().str.replace("–", "-", regex=False)
    )
    return df_contacts.values