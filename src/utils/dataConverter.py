from numpy import ndarray
from pandas import DataFrame, Series, DatetimeIndex
import pandas as pd
import numpy as np

from src.utils.config import GlobalConfig
import src.utils.state as state

DETECTION_RATE_BEFORE: float = 0.4
DETECTION_RATE_AFTER: float = 0.4

def convert_df_cases_to_weekly_cases(df_cases: DataFrame) -> Series | DataFrame:
    # "2020_1" → split year and week
    df_cases[["Year", "WeekNum"]] = df_cases["Week"].str.split("_", expand=True)
    df_cases["Year"] = df_cases["Year"].astype(int)
    df_cases["WeekNum"] = df_cases["WeekNum"].astype(int)

    # ISO week -> date (the first day of the week, Monday)
    df_cases["Date"] = pd.to_datetime(df_cases["Year"].astype(str) + df_cases["WeekNum"].astype(str) + "1",
                                      format="%G%V%u")

    # set date index
    df_weekly_cases = df_cases.set_index("Date").sort_index()

    weekly_cases = df_weekly_cases["Cases"].copy()

    weekly_cases.loc[weekly_cases.index < GlobalConfig.START_OF_VACCINATION] *= 1 / DETECTION_RATE_BEFORE
    weekly_cases.loc[weekly_cases.index >= GlobalConfig.START_OF_VACCINATION] *= 1 / DETECTION_RATE_AFTER
    return weekly_cases

def create_full_index_date_range(max_date) -> DatetimeIndex:
    return pd.date_range(
        start=GlobalConfig.START_OF_AGE_STRUCTURED_DATE,
        end=max_date,
        freq="W-MON"
    )

def convert_df_age_structured_cases(df_age_structured_cases: DataFrame) -> Series | DataFrame:
    annual_cases = df_age_structured_cases.set_index(df_age_structured_cases.columns[0])
    annual_cases.index = (
        annual_cases.index
        .map(str)  # convert all element to string
        .str.strip()  # remove whitespaces
        .str.replace("–", "-", regex=False)  # unicode dash → hyphen
    )
    return annual_cases

def create_case_matrix(annual_cases, weekly_cases_filled, weekly_cases_full) -> ndarray:
    weekly_age_structured = {}

    for age in state.AGE_GROUPS:
        s = pd.Series(np.zeros(len(weekly_cases_full.index), dtype=float), index=weekly_cases_full.index)
        for year in state.YEARS:
            annual_value = annual_cases.loc[age, year] * (1 / DETECTION_RATE_BEFORE)
            props = weekly_proportions_for_year(weekly_cases_filled, year)
            mask = weekly_cases_filled.index.year == year
            s.loc[mask] = props * annual_value
        weekly_age_structured[age] = s

    return np.column_stack(
        [weekly_age_structured[age].values for age in state.AGE_GROUPS]
    ).T  # transpose to have shape A × T

def weekly_proportions_for_year(weekly_cases_filled, year):
    mask = weekly_cases_filled.index.year == year
    year_data = weekly_cases_filled[mask]
    total = year_data.sum()
    return year_data / total

def convert_annual_deaths(df_deaths: DataFrame) -> Series | DataFrame:
    annual_deaths = df_deaths.set_index(df_deaths.columns[0])
    annual_deaths.index = (
        annual_deaths.index
        .map(str)  # convert all element to string
        .str.strip()  # remove whitespaces
        .str.replace("–", "-", regex=False)  # unicode dash → hyphen
    )
    return annual_deaths


def create_weekly_death_matrix(annual_deaths: Series, index) -> ndarray:
    weekly_deaths = {}
    for age in state.AGE_GROUPS:
        s = pd.Series(np.zeros(len(index), dtype=float), index=index)
        for year in state.YEARS:
            annual_value = annual_deaths.loc[age, year]
            mask = index.year == year
            s.loc[mask] = annual_value / 52
        weekly_deaths[age] = s

    death_matrix = np.column_stack(
        [weekly_deaths[age].values for age in state.AGE_GROUPS]
    ).T  # transpose to have shape A × T
    return death_matrix

def convert_annual_v1_data_to_weekly_v1(df_vaccines: DataFrame) -> Series:
    v1_yearly = dict(zip(df_vaccines["Year"], df_vaccines["1st dose"]))
    weekly_v1 = pd.Series(index=state.WEEKLY_INDEX, dtype=float)
    for date in state.WEEKLY_INDEX:
        year = date.year
        if date < GlobalConfig.START_OF_VACCINATION:
            weekly_v1.loc[date] = 0
        elif (date >= GlobalConfig.START_OF_VACCINATION) & (date < pd.Timestamp("2020-01-01")):
            weekly_v1.loc[date] = v1_yearly[year] / (52 - GlobalConfig.START_OF_VACCINATION.week)
        else:
            weekly_v1.loc[date] = v1_yearly[year] / 52
    return weekly_v1

def convert_annual_v2_data_to_weekly_v2(df_vaccines: DataFrame) -> Series:
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
    births_yearly = dict(zip(df_births["Year"], df_births["Cases"]))
    weekly_births = pd.Series(index=state.WEEKLY_INDEX, dtype=float)
    # Filling
    for date in state.WEEKLY_INDEX:
        year = date.year
        weekly_births.loc[date] = births_yearly[year] / 52
    birth_mtx = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    birth_mtx[0, :] = weekly_births
    state.set_weekly_births(weekly_births)
    return birth_mtx


def convert_daily_birth_data_to_birth_mtx(df_daily_births: DataFrame) -> ndarray:
    # births_df: the complete imported table
    years_row = df_daily_births.iloc[0].ffill()  # first raw
    months_row = df_daily_births.iloc[1]  # second raw
    days = df_daily_births.iloc[2:]
    days.index = days.index - 1

    births_long = (
        days
        .melt(ignore_index=False, var_name="col", value_name="births")
        .reset_index()
        .rename(columns={"index": "day"})
    )
    births_long["year"] = births_long["col"].map(years_row.to_dict())
    births_long["month"] = births_long["col"].map(months_row.to_dict())
    births_long["date"] = pd.to_datetime(
        births_long[["year", "month", "day"]],
        errors="coerce"
    )
    daily_births = births_long.dropna(subset=["date"]).set_index("date")["births"]  #
    first_monday = daily_births.index[daily_births.index.weekday == 0][0]
    daily_births_from_monday = daily_births.loc[first_monday:]

    weekly_births = daily_births_from_monday.resample("W-SUN").sum()
    weekly_births.index = weekly_births.index - pd.Timedelta(days=6)
    birth_mtx = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    birth_mtx[0, :] = weekly_births
    state.set_weekly_births(weekly_births)
    return birth_mtx

def convert_weekly_death_prop_data_to_death_prop_mtx(df_death_proportions: DataFrame) -> ndarray:
    # "2020_1" → Year, WeekNum
    df_death_proportions[["Year", "WeekNum"]] = df_death_proportions["Week"].str.split("_", expand=True)
    df_death_proportions["Year"] = df_death_proportions["Year"].astype(int)
    df_death_proportions["WeekNum"] = df_death_proportions["WeekNum"].astype(int)

    # ISO week → date of Mondays
    df_death_proportions["Date"] = pd.to_datetime(
        df_death_proportions["Year"].astype(str)
        + "-W"
        + df_death_proportions["WeekNum"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    # date index
    df_weekly_death_proportions = df_death_proportions.set_index("Date").sort_index()
    age_cols = [col for col in df_weekly_death_proportions.columns
                if col not in ["Week", "Year", "WeekNum"]]
    deaths_prop_matrix = df_weekly_death_proportions[age_cols].T.values
    return deaths_prop_matrix

def convert_data_to_s0(df_susceptible: DataFrame) -> ndarray:
    df_susceptible.iloc[:, 0] = (
        df_susceptible.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
    )
    df_susceptible.iloc[:, 1] = pd.to_numeric(df_susceptible.iloc[:, 1], errors="coerce")
    df_s0 = df_susceptible.set_index(df_susceptible.columns[0])
    return df_s0.iloc[:, 0].values.astype(float)

def convert_data_to_n0(df_population: DataFrame) -> ndarray:
    df_population.iloc[:, 0] = (
        df_population.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
    )
    df_population.iloc[:, 1] = pd.to_numeric(df_population.iloc[:, 1], errors="coerce")
    df_n0 = df_population.set_index(df_population.columns[0])
    return df_n0.iloc[:, 0].values.astype(float)

def convert_data_to_contact_matrix(df_contacts: DataFrame) -> ndarray:
    df_contacts.index = (
        df_contacts.index.astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
    )
    df_contacts.columns = (
        df_contacts.columns.astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
    )
    return df_contacts.values