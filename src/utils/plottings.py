import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from numpy import ndarray
from matplotlib.dates import RRuleLocator, rrulewrapper
from dateutil.rrule import YEARLY

import src.utils.state as state
from src.utils.config import GlobalConfig
from src.utils.models import Scenario


def plot_vector_with_index(data: ndarray, index, title: str, x_label: str, y_label: str):
    """
    Plot a one‑dimensional numerical vector against a given time index.

    Parameters
    ----------
    data : ndarray
        A 1D array containing the values to plot.
    index : array-like
        A sequence of timestamps or labels corresponding to each element of `data`.
    title : str
        Title of the plot.
    x_label : str
        Label for the horizontal axis.
    y_label : str
        Label for the vertical axis.

    Notes
    -----
    This is a simple helper function used throughout the project to visualize
    time‑indexed numerical data such as weekly infections, susceptibles, or
    vaccination counts.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(pd.Series(data, index=index))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_remainders():
    """
    Plot the estimated effective reproduction number Rₜ over time.

    Notes
    -----
    Rₜ is computed from:
        - the model's remainder term (state.REMAINDERS),
        - the proportion of susceptibles in each age group,
        - and aggregated across all age groups.

    The red dashed line at Rₜ = 1 serves as a visual reference indicating
    the threshold between epidemic growth (Rₜ > 1) and decline (Rₜ < 1).
    """
    r_t = np.zeros(state.NR_TIMESTEPS)
    for t in range(math.ceil(GlobalConfig.MAX_TAU), state.NR_TIMESTEPS):
        s_t = state.BASELINE_SCENARIO.S[:, t] / state.POPULATION[:, t]
        r_t[t] = (state.REMAINDERS[:, t] * s_t).sum()

    weekly_remainders_t = pd.Series(r_t, index=state.AGE_STRUCTURED_DATA_INDEX)

    plt.figure(figsize=(12, 6))
    weekly_remainders_t.plot()

    plt.axhline(1.0, color='red', linestyle='--', linewidth=1)
    plt.title("Weekly estimated Rₜ")
    plt.ylabel("Rₜ value")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()


def plot_model_actual_scat_comparison(scenario: Scenario):
    """
    Compare model‑estimated weekly infections with observed case data.

    Parameters
    ----------
    scenario : Scenario
        The epidemiological scenario containing weekly infection estimates.

    Notes
    -----
    The function overlays:
        - a line plot of model‑estimated weekly infections,
        - scatter points representing observed weekly case counts.

    This visualization helps assess how well the model fits real-world data.
    """
    model = scenario.weekly_i_series
    plt.figure(figsize=(10, 5))

    plt.plot(model.index, model.values,
             label="Estimated number of cases", color="royalblue", linewidth=1)

    plt.scatter(
        state.AGE_STRUCTURED_DATA_INDEX,
        np.sum(state.CASES_MATRIX, axis=0)[:len(model)],
        color="black",
        s=4,
        label="Observed data"
    )

    plt.title("Estimated and observed cases: " + scenario.name)
    plt.xlabel("Date")
    plt.ylabel("Weekly number of cases")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_susceptibles(scenario: Scenario):
    """
    Plot the proportion of susceptible individuals over time.

    Parameters
    ----------
    scenario : Scenario
        The scenario containing S (susceptible counts) for each age group.

    Notes
    -----
    The function aggregates susceptibles across all age groups except the last,
    divides by the corresponding population, and visualizes the resulting
    proportion over time.
    """
    prop = (np.sum(scenario.S[:state.NR_AGE_GROUPS-1,:], axis=0) /
            np.sum(state.POPULATION[:state.NR_AGE_GROUPS-1,:], axis=0))
    model = pd.Series(prop, index=state.AGE_STRUCTURED_DATA_INDEX)
    plt.figure(figsize=(10, 5))

    plt.plot(model.index, model.values,
             label="Estimated number of susceptibles", color="royalblue", linewidth=1)

    plt.title("Number of susceptibles: " + scenario.name)
    plt.xlabel("Date")
    plt.ylabel("Weekly number of susceptibles")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_annual_cases(scenario: Scenario):
    """
    Plot annual infection totals for a given scenario.

    Parameters
    ----------
    scenario : Scenario
        Scenario containing annual infection counts.

    Notes
    -----
    The function:
        - plots annual totals,
        - adds scatter points for emphasis,
        - uses yearly grid lines for readability.
    """
    annual_infections_scen = scenario.annual_i_series
    fig, ax = plt.subplots()
    ax.plot(annual_infections_scen.index, annual_infections_scen)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.3)

    plt.scatter(
        annual_infections_scen.index,
        annual_infections_scen,
        color="black",
        s=4,
        label="Observed data"
    )

    plt.title("Annual cases: " + scenario.name)
    plt.xlabel("Date")
    plt.ylabel("Annual number of cases")
    plt.tight_layout()
    plt.show()


def plot_model_results_age_range_in_range(model, min_age: int, max_age: int, scenario_name: str):
    """
    Plot model results for a selected age range.

    Parameters
    ----------
    model : ndarray
        Age‑structured weekly model output (e.g., infections).
    min_age : int
        Minimum age group index.
    max_age : int
        Maximum age group index.
    scenario_name : str
        Name of the scenario.

    Notes
    -----
    Each age group is plotted as a separate line. The x‑axis uses a custom
    date locator to highlight specific yearly intervals (e.g., every 2 years
    on September 1).
    """
    plt.figure(figsize=(10, 2))

    for a in range(min_age, max_age + 1):
        plt.plot(
            state.AGE_STRUCTURED_DATA_INDEX,
            pd.Series(model[a], index=state.AGE_STRUCTURED_DATA_INDEX).values,
            label=state.AGE_GROUPS[a], linewidth=1
        )

    ax = plt.gca()
    ax.set_xlim(state.AGE_STRUCTURED_DATA_INDEX.min(), state.AGE_STRUCTURED_DATA_INDEX.max())
    rule = rrulewrapper(YEARLY, interval=2, bymonth=9, bymonthday=1)
    ax.xaxis.set_major_locator(RRuleLocator(rule))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.title("Estimated cases: " + scenario_name)
    plt.xlabel("Date")
    plt.ylabel("Weekly number of cases")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scat_age_range_in_range(model, index, min_age: int, max_age: int, actual_case_matrix, date_mask):
    """
    Plot estimated vs. observed weekly cases for a selected age range.

    Parameters
    ----------
    model : ndarray
        Model‑estimated weekly cases per age group.
    index : array-like
        Time index for the data.
    min_age : int
        Minimum age group index.
    max_age : int
        Maximum age group index.
    actual_case_matrix : ndarray
        Observed weekly cases per age group.
    date_mask : ndarray of bool
        Mask selecting the time interval to plot.

    Notes
    -----
    Each age group is plotted with:
        - a line for model estimates,
        - scatter points for observed data.
    """
    plt.figure(figsize=(10, 5))

    for a in range(min_age, max_age):
        plt.plot(index, pd.Series(model[a], index=index).values,
                 label=state.AGE_GROUPS[a], linewidth=1)
        plt.scatter(index, actual_case_matrix[a][date_mask], s=4)

    plt.title("Estimated and observed cases")
    plt.xlabel("Date")
    plt.ylabel("Weekly number of cases")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_scenario_to_actual_case(scen_i: pd.Series, scenario_name: str):
    """
    Compare annual model estimates with annual observed cases.

    Parameters
    ----------
    scen_i : pd.Series
        Weekly model‑estimated infections.
    scenario_name : str
        Name of the scenario.

    Notes
    -----
    The function aggregates weekly infections into annual totals and
    overlays them with observed annual case counts from the baseline scenario.
    """
    annual_infections_scen = scen_i.groupby(scen_i.index.year).sum()
    annual_infections_scen.index = pd.to_datetime(annual_infections_scen.index, format="%Y")

    fig, ax = plt.subplots()
    ax.plot(annual_infections_scen.index, annual_infections_scen, color="royalblue", linewidth=1)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.3)

    plt.scatter(
        annual_infections_scen.index,
        annual_infections_scen,
        color="royalblue",
        s=6,
        label="Estimated data"
    )

    ax.plot(
        state.BASELINE_SCENARIO.annual_i_series.index,
        state.BASELINE_SCENARIO.annual_i_series,
        color="orange", linewidth=1)
    plt.scatter(
        state.BASELINE_SCENARIO.annual_i_series.index,
        state.BASELINE_SCENARIO.annual_i_series,
        color="orange",
        s=6,
        label="Observed data"
    )

    plt.title("Compare annual cases: " + scenario_name)
    plt.xlabel("Date")
    plt.ylabel("Annual number of cases")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_annual_number_of_cases_for_age_groups(scen_i: np.ndarray, min_age: int, max_age: int, scenario_name: str):
    """
    Plot annual model estimates for multiple age groups.

    Parameters
    ----------
    scen_i : ndarray
        Weekly model‑estimated infections per age group.
    min_age : int
        Minimum age group index.
    max_age : int
        Maximum age group index.
    scenario_name : str
        Name of the scenario.

    Notes
    -----
    The function:
        - converts weekly data into a DataFrame,
        - aggregates by year,
        - plots annual totals for each selected age group.
    """
    cases_df = pd.DataFrame(scen_i.T, index=state.WEEKLY_INDEX)
    annual_cases = cases_df.groupby(cases_df.index.year).sum()
    annual_cases.index = pd.to_datetime(annual_cases.index.astype(str))

    fig, ax = plt.subplots()

    for a in range(min_age, max_age + 1):
        ax.plot(annual_cases.index, pd.Series(annual_cases[a], index=annual_cases.index).values,
                label=state.AGE_GROUPS[a], linewidth=1)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.3)

    plt.title("Annual cases: " + scenario_name)
    plt.xlabel("Date")
    plt.ylabel("Annual number of cases")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


def heatmap(data_frame: pd.DataFrame, ax: plt.Axes):
    """
    Render a heatmap of weekly cases across age groups.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Rows correspond to age groups, columns correspond to weeks.
    ax : plt.Axes
        Matplotlib axes object on which the heatmap is drawn.

    Notes
    -----
    The heatmap uses a color scale to represent case intensity, with
    younger age groups placed at the bottom for intuitive reading.
    """
    return ax.imshow(
        data_frame.values,
        aspect="auto",
        cmap="summer",
        origin="lower",
        vmin=state.GLOBAL_MIN,
        vmax=state.GLOBAL_MAX
    )


def plot_heatmap(dataframe: pd.DataFrame, title: str):
    """
    Plot a heatmap of weekly cases across age groups.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Age‑by‑week case matrix.
    title : str
        Title of the plot.

    Notes
    -----
    The function:
        - draws the heatmap,
        - labels age groups on the y‑axis,
        - reduces x‑axis tick frequency for readability,
        - adds a colorbar indicating case intensity.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    im = heatmap(dataframe, ax=ax)
    age_groups = dataframe.index
    weekly_index = dataframe.columns

    ax.set_yticks(np.arange(len(age_groups)))
    ax.set_yticklabels(age_groups)

    ax.set_xticks(np.linspace(0, len(weekly_index)-1, 10).astype(int))
    ax.set_xticklabels(
        [weekly_index[i].strftime("%Y-%m-%d")
         for i in np.linspace(0, len(weekly_index)-1, 10).astype(int)],
        rotation=45,
        ha="right"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Age group")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weekly cases")

    plt.tight_layout()
    plt.show()


def plot_cumulative_cases(scenarios: list[Scenario], index, mask):
    """
    Plot cumulative infections for multiple scenarios.

    Parameters
    ----------
    scenarios : list of Scenario
        Scenarios to compare.
    index : array-like
        Time index.
    mask : ndarray of bool
        Mask selecting the time interval to plot.

    Notes
    -----
    Each scenario is plotted as a cumulative sum of weekly infections,
    with a filled area under the curve for visual emphasis.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for scenario in scenarios:
        i_total = scenario.I.sum(axis=0)
        cum = i_total[mask].cumsum()
        ax.plot(index, cum, label=scenario.name)
        ax.fill_between(index, cum, alpha=0.5)

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.grid(alpha=0.3)
    ax.set_title("Cumulative I")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_i_and_v(scenario: Scenario, starting_date):
    """
    Plot cumulative infections and vaccinations from a given starting date.

    Parameters
    ----------
    scenario : Scenario
        Scenario containing infection and vaccination data.
    starting_date : datetime-like
        Date from which cumulative counts should be plotted.

    Notes
    -----
    The function visualizes:
        - cumulative infections (I),
        - cumulative first‑dose vaccinations (V1),
    allowing comparison of epidemic progression and vaccination rollout.
    """
    mask = state.WEEKLY_INDEX > starting_date
    fig, ax = plt.subplots(figsize=(10, 5))

    i_cum = scenario.I.sum(axis=0).cumsum()
    v_cum = scenario.V1.sum(axis=0).cumsum()

    ax.plot(state.WEEKLY_INDEX[mask], i_cum[mask], label="Infected (cum)", color="red")
    ax.plot(state.WEEKLY_INDEX[mask], v_cum[mask], label="Vaccinated (cum)", color="green")

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.grid(alpha=0.3)
    ax.set_title("Cumulative I and V")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_cases_barchart(scenarios: list[Scenario], mask=None, title="Cumulative cases by vaccination rate"):
    """
    Barchart a szcenáriók kumulatív esetszámáról az index végén.

    scenarios: lista szcenárió objektumokkal
    index: pandas index (heti dátumok)
    mask: opcionális boolean maszk (ha csak egy időszakot akarsz nézni)
    title: ábra címe
    """

    vaccination_rates = []
    cumulative_values = []

    for scen in scenarios:
        i_total = scen.I.sum(axis=0)
        if mask is not None:
            i_total = i_total[mask]
        cum = i_total.cumsum()
        cumulative_values.append(cum[-1])
        vaccination_rates.append(scen.name)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.bar(vaccination_rates, cumulative_values, color="steelblue", alpha=0.8, width=0.6)
    ax.set_ylim(1500000)
    ax.set_title(title)
    ax.set_xlabel("Vaccination rate scenarios")
    ax.set_ylabel("Cumulative cases at end of period")
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def plot_compare_annual_incidences_for_scenarios(scen_i: list[Scenario]):
    fig, ax = plt.subplots()

    for scen in scen_i:
        ax.plot(scen.annual_i_series.index, scen.annual_i_series.values, label=scen.name, linewidth=1)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.3)

    plt.title("Compare scenarios with different detection rate and same vaccination rate")
    plt.xlabel("Date")
    plt.ylabel("Annual number of cases")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()
