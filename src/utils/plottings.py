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
    plt.figure(figsize=(12, 6))
    plt.plot(pd.Series(data, index=index))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_remainders():
    r_t = np.zeros(state.NR_TIMESTEPS)
    for t in range(GlobalConfig.MAX_TAU, state.NR_TIMESTEPS):
        s_t = state.BASELINE_SCENARIO.S[:, t] / state.POPULATION[:, t]
        r_t[t] = (state.REMAINDERS[:, t] * s_t).sum()

    weekly_remainders_t = pd.Series(r_t, index=state.AGE_STRUCTURED_DATA_INDEX)

    plt.figure(figsize=(12, 6))
    weekly_remainders_t.plot()

    plt.axhline(1.0, color='red', linestyle='--', linewidth=1)  # R=1 as reference
    plt.title("Weekly estimated Rₜ")
    plt.ylabel("Rₜ value")
    plt.xlabel("Date")
    plt.grid(True)

    plt.show()

def plot_model_actual_scat_comparison(scenario: Scenario):
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

def plot_annual_cases(scenario: Scenario):
    annual_infections_scen = scenario.annual_i_series
    fig, ax = plt.subplots()
    ax.plot(annual_infections_scen.index, annual_infections_scen)

    # Annual grids and titles
    ax.xaxis.set_major_locator(mdates.YearLocator(3))  # each year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # display only the year

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
    plt.figure(figsize=(10, 2))

    for a in range(min_age, max_age + 1):
        plt.plot(state.AGE_STRUCTURED_DATA_INDEX, pd.Series(model[a], index=state.AGE_STRUCTURED_DATA_INDEX).values,
             label=state.AGE_GROUPS[a], linewidth=1)

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
    plt.figure(figsize=(10, 5))

    for a in range(min_age, max_age):
        plt.plot(index, pd.Series(model[a], index=index).values,
                 label=state.AGE_GROUPS[a], linewidth=1)
        plt.scatter(
            index, actual_case_matrix[a][date_mask], s=4
        )

    plt.title("Estimated and observed cases")
    plt.xlabel("Date")
    plt.ylabel("Weekly number of cases")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_compare_scenario_to_actual_case(scen: pd.Series, scenario_name: str):
    annual_infections_scen = scen.groupby(scen.index.year).sum()
    annual_infections_scen.index = pd.to_datetime(annual_infections_scen.index, format="%Y")

    fig, ax = plt.subplots()
    ax.plot(annual_infections_scen.index, annual_infections_scen, color="royalblue", linewidth=1)

    # Annual grids and titles
    ax.xaxis.set_major_locator(mdates.YearLocator(3))        # each year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # display only the year

    ax.grid(alpha=0.3)

    plt.scatter(
        annual_infections_scen.index,
        annual_infections_scen,
        color="royalblue",
        s=6,
        label="Estimated data"
    )

    ax.plot(state.BASELINE_SCENARIO.annual_i_series.index, state.BASELINE_SCENARIO.annual_i_series, color="orange", linewidth=1)
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

def plot_compare_scenario_to_actual_case_for_age_groups(scen: np.ndarray, min_age: int, max_age: int, scenario_name: str):
    cases_df = pd.DataFrame(scen.T, index=state.WEEKLY_INDEX)
    annual_cases = cases_df.groupby(cases_df.index.year).sum()
    annual_cases.index = pd.to_datetime(annual_cases.index.astype(str))

    fig, ax = plt.subplots()

    for a in range(min_age, max_age + 1):
        ax.plot(annual_cases.index, pd.Series(annual_cases[a], index=annual_cases.index).values,
                label=state.AGE_GROUPS[a], linewidth=1)

    # Annual grids and titles
    ax.xaxis.set_major_locator(mdates.YearLocator(3))        # each year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # display only the year

    ax.grid(alpha=0.3)

    plt.title("Annual cases: " + scenario_name)
    plt.xlabel("Date")
    plt.ylabel("Annual number of cases")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()

def heatmap(data_frame: pd.DataFrame, ax: plt.Axes):
    return ax.imshow(
        data_frame.values,
        aspect="auto",
        # cmap="afmhot",
        # cmap="bone",
        cmap="summer",
        # cmap="copper",
        origin="lower",       # position younger age groups to the bottom
        vmin=state.GLOBAL_MIN,
        vmax=state.GLOBAL_MAX
    )

def plot_heatmap(dataframe: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    im = heatmap(dataframe, ax=ax)

    ax.set_yticks(np.arange(len(state.AGE_GROUPS)))
    ax.set_yticklabels(state.AGE_GROUPS)

    # reducing the frequency of grids
    ax.set_xticks(np.linspace(0, len(state.WEEKLY_INDEX)-1, 10).astype(int))
    ax.set_xticklabels(
        [state.WEEKLY_INDEX[i].strftime("%Y-%m-%d") for i in np.linspace(0, len(state.WEEKLY_INDEX)-1, 10).astype(int)],
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
    fig, ax = plt.subplots(figsize=(10, 5))

    for scenario in scenarios:
        cum = scenario.I.sum(axis=0).cumsum()
        ax.plot(index, cum[mask], label = scenario.name)
        ax.fill_between(index, cum[mask], alpha=0.5)

    # annual grids
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.grid(alpha=0.3)
    ax.set_title(f"Cumulative I")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_cumulative_i_and_v(scenario: Scenario):
    fig, ax = plt.subplots(figsize=(10, 5))

    i_cum = scenario.I.sum(axis=0).cumsum()
    v_cum = scenario.V1.sum(axis=0).cumsum()

    ax.plot(state.WEEKLY_INDEX, i_cum, label="Infected (cum)", color="red")
    ax.plot(state.WEEKLY_INDEX, v_cum, label="Vaccinated (cum)", color="green")

    # annual grids
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.grid(alpha=0.3)
    ax.set_title(f"Cumulative I and V")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    plt.tight_layout()
    plt.show()
