import numpy as np
import pandas as pd

import src.utils.state as state
from src.utils.config import GlobalConfig
from src.utils.models import Scenario
from src.utils.scenarios import Scenarios

L = 2  # length of latent period in weeks
D = 1  # length of infectious period in weeks

SIGMA = 1/2  # length of latent period is 2 weeks
GAMMA = 1  # length of infectious period is 1 week
TAU = np.arange(1, GlobalConfig.MAX_TAU + 1)
G = (SIGMA * GAMMA/(GAMMA-SIGMA)) * (np.exp(-SIGMA * TAU)-np.exp(-GAMMA * TAU))
G = G/G.sum()     # normalizing

V1_EFFICACY = 0.81
V2_EFFICACY = 0.92

#class ScenarioCalculator:
    # @staticmethod
    # def calculate_number_of_susceptible_cases_in_baseline():
    #     baseline_scenario: Scenario = init_baseline_scenario()
    #     state.set_baseline_scenario(baseline_scenario)
    #     state.set_population(np.zeros((state.NR_AGE_GROUPS,state.NR_TIMESTEPS)))
    #     state.POPULATION[:, 0] = state.N0_VECTOR
    #
    #     for t in range(1,state.NR_TIMESTEPS):
    #         # if weekly_cases_filled.index[t] > start_of_vaccination + pd.DateOffset(months=3): v2[1,t] = v1[1,t-3*4] * (1-V1_efficacy)
    #         calculate_next_step_in_baseline(t)

    # @staticmethod
    # def calculate_remainders():
        # remainders_t = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
        # r_a_t = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
        # rho = np.zeros(state.NR_TIMESTEPS)

        # for t in range(GlobalConfig.MAX_TAU, state.NR_TIMESTEPS):
        #     cm = calculate_simetrized_contact_matrix(t)
        #     # weighted sum of past incidences
        #     denominator = calculate_denominator(state.CASES_MATRIX, t)
        #
        #     if denominator.all() > 0 and state.BASELINE_SCENARIO.S[:, t].all() > 0:
        #         s_t = state.BASELINE_SCENARIO.S[:, t] / state.POPULATION[:, t]
        #         remainders_t[:, t] = state.CASES_MATRIX[:, t] / (s_t * denominator)
        #         r_a_t[:, t] = remainders_t[:, t] * s_t
        #         rho[t] = max(abs(np.linalg.eigvals(cm * r_a_t[:, t])))
        #     else:
        #         remainders_t[:, t] = np.nan
        #
        # state.set_remainders(remainders_t)
        # state.set_r_a(r_a_t)
        # state.set_rho(rho)

    # @staticmethod
    # def calculate_number_of_incidences_based_on_remainders():
    #     A = state.NR_AGE_GROUPS
    #     T = state.NR_TIMESTEPS
    #     i_model = np.zeros((A, T))
    #     for t in range(0, GlobalConfig.MAX_TAU):
    #         i_model[:, t] = state.CASES_MATRIX[:, t]
    #
    #     for t in range(GlobalConfig.MAX_TAU, T - 1):
    #         denominator = calculate_denominator(state.CASES_MATRIX, t)
    #         i_model[:, t] = state.R_A[:, t] * denominator
    #
    #     state.BASELINE_SCENARIO.I = i_model
    #     state.BASELINE_SCENARIO.compute_i_series()


def calculate_number_of_susceptible_cases_in_baseline(with_covid: bool):
    baseline_scenario: Scenario = init_baseline_scenario() if with_covid else init_scenario("no covid baseline")
    if with_covid:
        state.set_baseline_scenario(baseline_scenario)
        state.set_population(np.zeros((state.NR_AGE_GROUPS,state.NR_TIMESTEPS)))
        state.POPULATION[:, 0] = state.N0_VECTOR
    else: state.set_mean_i_scenario(baseline_scenario)

    for t in range(1,state.NR_TIMESTEPS):
        # if weekly_cases_filled.index[t] > start_of_vaccination + pd.DateOffset(months=3): v2[1,t] = v1[1,t-3*4] * (1-V1_efficacy)
        calculate_next_step_in_baseline(baseline_scenario, with_covid, t)

def calculate_remainders(baseline: bool):
    remainders_t = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    r_a_t = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    rho = np.zeros(state.NR_TIMESTEPS)
    cases_matrix = state.CASES_MATRIX if baseline else state.MEAN_I_SCENARIO.I
    s_matrix = state.BASELINE_SCENARIO.S if baseline else state.MEAN_I_SCENARIO.S

    for t in range(GlobalConfig.MAX_TAU, state.NR_TIMESTEPS):
        cm = calculate_simetrized_contact_matrix(t)
        # weighted sum of past incidences
        denominator = calculate_denominator(cases_matrix, t)

        if denominator.all() > 0 and s_matrix[:, t].all() > 0:
            s_t = s_matrix[:, t] / state.POPULATION[:, t]
            remainders_t[:, t] = cases_matrix[:, t] / (s_t * denominator)
            r_a_t[:, t] = remainders_t[:, t] * s_t
            rho[t] = max(abs(np.linalg.eigvals(cm * r_a_t[:, t])))
        else:
            remainders_t[:, t] = np.nan

    if baseline:
        state.set_remainders(remainders_t)
        state.set_r_a(r_a_t)
        state.set_rho(rho)
    else:
        state.set_mean_remainders(remainders_t)

def calculate_number_of_incidences_based_on_remainders():
    A = state.NR_AGE_GROUPS
    T = state.NR_TIMESTEPS
    i_model = np.zeros((A, T))
    for t in range(0, GlobalConfig.MAX_TAU):
        i_model[:, t] = state.CASES_MATRIX[:, t]

    for t in range(GlobalConfig.MAX_TAU, T - 1):
        denominator = calculate_denominator(state.CASES_MATRIX, t)
        i_model[:, t] = state.R_A[:, t] * denominator

    state.BASELINE_SCENARIO.I = i_model
    state.BASELINE_SCENARIO.compute_i_series()

def calculate_number_of_cases_for_scenario(scenario_name: str, vacc_level: float, start_of_vaccination: pd.Timestamp, with_covid: bool) -> Scenario:
    scenario = init_scenario(scenario_name)
    calculate_the_number_of_vaccinated_individuals(start_of_vaccination, vacc_level, scenario.V1)
    for t in range(0, state.NR_TIMESTEPS - 1):
        if state.WEEKLY_CASES_FILLED.index[t] < start_of_vaccination:
            scenario.I[:, t] = state.CASES_MATRIX[:, t]
            scenario.S[:, t] = state.BASELINE_SCENARIO.S[:, t]

    for t in range(GlobalConfig.MAX_TAU, state.NR_TIMESTEPS):
        if state.WEEKLY_CASES_FILLED.index[t] >= start_of_vaccination:
            calculate_next_step_for_scenario(scenario, with_covid, start_of_vaccination, t)
    scenario.compute_i_series()
    return scenario


def init_baseline_scenario():
    s_scen = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    s_scen[:, 0] = state.S0_VECTOR
    return Scenario(Scenarios.BASELINE, S=s_scen, V1=state.WEEKLY_V1_AGE_STRUCTURED, V2=state.WEEKLY_V2_AGE_STRUCTURED)

def init_scenario(scenario_name: str) -> Scenario:
    s_scen = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    i_scen = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    for t in range(0,  GlobalConfig.MAX_TAU):
        i_scen[:, t] = state.BASELINE_SCENARIO.I[:, t]
        s_scen[:, t] = state.BASELINE_SCENARIO.S[:, t]
    return Scenario(scenario_name, S=s_scen, I=i_scen)


def update_scenario(scenario: Scenario):
    weekly_incidences = pd.Series(np.sum(scenario.I, axis=0), index=state.AGE_STRUCTURED_DATA_INDEX)
    annual_incidences = weekly_incidences.groupby(weekly_incidences.index.year).sum()
    annual_incidences.index = pd.to_datetime(annual_incidences.index, format="%Y")
    scenario.weekly_i_series = weekly_incidences
    scenario.annual_i_series = annual_incidences

def calculate_next_step_in_baseline(scenario: Scenario, with_covid: bool, t: int):
    calculate_number_of_susceptible_cases_in_next_step_in_baseline(scenario, with_covid, t)
    update_scenario_when_aging(scenario, t, True, with_covid)
    if with_covid: calculate_size_of_population_in_next_step(t)

def calculate_size_of_population_in_next_step(t: int):
    is_not_the_first_monday_of_september = not ((state.WEEKLY_CASES_FILLED.index[t].month == 9) &
                                    (state.WEEKLY_CASES_FILLED.index[t].day <= 7))
    if (t <= state.NR_TIMESTEPS - 1) & is_not_the_first_monday_of_september:
        state.POPULATION[:, t] = state.POPULATION[:, t - 1] + state.BIRTH_MATRIX[:, t - 1] - state.DEATHS_MATRIX[:, t - 1]


def calculate_number_of_susceptible_cases_in_next_step_in_baseline(scenario: Scenario, with_covid: bool, t: int):
    # effectively_vaccinated_v1 = scenario.V1[:, t-1] * GlobalConfig.V1_EFFICACY if with_covid else np.zeros(state.NR_AGE_GROUPS)
    effectively_vaccinated_v1 = scenario.V1[:, t-1] * GlobalConfig.V1_EFFICACY
    effectively_vaccinated_v2 = scenario.V2[:, t-1] * GlobalConfig.V2_EFFICACY
    # effectively_vaccinated_v2 = scenario.V2[:, t-1] * GlobalConfig.V2_EFFICACY if with_covid else np.zeros(state.NR_AGE_GROUPS)
    effectively_vaccinated = effectively_vaccinated_v1 + effectively_vaccinated_v2
    cases_matrix = state.CASES_MATRIX[:, t-1] if with_covid else state.MEAN_I[:, t-1]
    births = state.BIRTH_MATRIX[:, t-1] if with_covid else state.NO_COVID_BIRTHS[:,t-1]
    deaths = state.DEATHS_MATRIX[:, t-1] if with_covid else state.NO_COVID_DEATHS[:,t-1]
    demographic_changes = (births - deaths * (scenario.S[:, t-1] / state.POPULATION[:, t - 1]))
    scenario.S[:, t] = (scenario.S[:, t - 1] - cases_matrix - effectively_vaccinated + demographic_changes)


def update_scenario_when_aging(scenario: Scenario, t: int, baseline: bool, with_covid: bool):
    if is_time_for_aging(t):
        if baseline:
            handle_aging_baseline(scenario.S, t, with_covid)
        else:
            handle_aging(scenario.S, scenario.I, t)

def is_time_for_aging(t: int) -> bool:
    is_first_monday_of_september = ((state.WEEKLY_CASES_FILLED.index[t].month == 9) &
                                    (state.WEEKLY_CASES_FILLED.index[t].day <= 7))
    return (t <= state.NR_TIMESTEPS - 1) & is_first_monday_of_september


def handle_aging(s_scen: np.ndarray, i_scen: np.ndarray, t: int):
    handle_aging_of_susceptible_cases(s_scen, t, False)
    handle_aging_of_infectious_cases(i_scen, t)

def handle_aging_baseline(s_scen: np.ndarray, t: int, with_covid: bool):
    handle_aging_of_susceptible_cases(s_scen, t, True)
    if with_covid: handle_aging_of_population(t)

def handle_aging_of_population(t: int):
    A = state.NR_AGE_GROUPS
    population = state.POPULATION
    demographic_changes = state.BIRTH_MATRIX[:, t-1] - state.DEATHS_MATRIX[:, t-1]
    # age group 20+
    population[A - 1, t] = population[A - 1, t - 1] + population[A - 2, t - 1] / 5 + demographic_changes[A - 1]
    # age group 15-19
    population[A - 2, t] = population[A - 2, t - 1] * 4 / 5 + population[A - 3, t - 1] / 5 + demographic_changes[A - 2]
    # age group 10-14
    population[A - 3, t] = population[A - 3, t - 1] * 4 / 5 + population[A - 4, t - 1] + demographic_changes[A - 3]
    # age groups 1-9
    for a in range(A - 4, 0, -1):
        population[a, t] = population[a - 1, t - 1] + demographic_changes[a]
    population[0, t] = 0 + demographic_changes[0]


def handle_aging_of_susceptible_cases(s_scen: np.ndarray, t: int, baseline: bool):
    A = state.NR_AGE_GROUPS
    # age group 20+
    s_scen[A - 1, t] = s_scen[A - 1, t] + s_scen[A - 2, t] / 5
    # age group 15-19
    s_scen[A - 2, t] = s_scen[A - 2, t] * 4 / 5 + s_scen[A - 3, t] / 5
    # age group 10-14
    s_scen[A - 3, t] = s_scen[A - 3, t] * 4 / 5 + s_scen[A - 4, t]
    # age groups 1-9
    for a in range(A - 4, 0, -1):
        s_scen[a, t] = s_scen[a - 1, t]
    if baseline:
        s_scen[0,t] = state.BIRTH_MATRIX[0,t-1]-state.DEATHS_MATRIX[0,t-1]
    else :
        s_scen[0, t] = 1


def handle_aging_of_infectious_cases(i_scen: np.ndarray, t: int):
    i_scen[0, t] = 0


def calculate_denominator(i_scen: np.ndarray, t: int) -> np.ndarray:
    cm = calculate_simetrized_contact_matrix(t)
    mu_t = (state.DEATHS_MATRIX[:, t] / state.POPULATION[:, t]).reshape(1, -1)  # shape: (1,A)
    tau = np.arange(1, GlobalConfig.MAX_TAU + 1)  # shape: (A,1)
    kernel = G.reshape(-1, 1) * np.exp(-mu_t * tau.reshape(-1, 1))
    return np.sum(i_scen[:,t-tau] * kernel.T, axis=1) @ cm

def calculate_simetrized_contact_matrix(t: int) -> np.ndarray:
    ## the size of the age groups at the previous week as a column vector
    population_i = state.POPULATION[:, t - 1].reshape(-1,1)
    ## the size of the age groups at the previous week as a row vector
    population_j = state.POPULATION[:, t - 1].reshape(1,-1)
    return (state.CONTACTS0 * population_i + state.CONTACTS0.T * population_j) / (2 * population_i)


def calculate_nr_of_susceptible_cases_in_next_step_for_scenario(scenario: Scenario, with_covid: bool, start_of_vaccination: pd.Timestamp, t: int) -> np.ndarray:
    effectively_vaccinated_v1 = scenario.V1[:, t - 1] * GlobalConfig.V1_EFFICACY
    effectively_vaccinated_v2 = scenario.V2[:, t - 1] * GlobalConfig.V2_EFFICACY
    effectively_vaccinated = effectively_vaccinated_v1 + effectively_vaccinated_v2
    birth_mtx: np.ndarray[float] = state.BIRTH_MATRIX if with_covid else state.NO_COVID_BIRTHS
    death_matrix: np.ndarray[float] = state.DEATHS_MATRIX if with_covid else state.NO_COVID_DEATHS
    demographic_changes = (birth_mtx[:,t-1] - death_matrix[:,t-1] * (scenario.S[:,t-1] / state.POPULATION[:,t-1]))

    if state.WEEKLY_CASES_FILLED.index[t] < start_of_vaccination:
        s_scen = state.BASELINE_SCENARIO.S[:, t]
    else:
        s_scen = scenario.S[:, t-1] - scenario.I[:, t-1] - effectively_vaccinated + demographic_changes

    return s_scen


def calculate_nr_of_infectious_incidences_in_next_step_for_scenario(scenario: Scenario, with_covid: bool, t: int) -> np.ndarray:
    s_t = scenario.S[:,t] / state.POPULATION[:, t]
    remainders: np.ndarray[float] = state.REMAINDERS if with_covid else state.MEAN_REMAINDERS
    #remainders: np.ndarray[float] = state.REMAINDERS if with_covid else state.NO_COVID_REMAINDERS
    r_t_scen = remainders[:, t] * s_t
    denominator = calculate_denominator(scenario.I, t)
    return r_t_scen * denominator

def calculate_next_step_for_scenario(scenario: Scenario, with_covid: bool, start_of_vaccination: pd.Timestamp, t: int):
    scenario.S[:, t] = calculate_nr_of_susceptible_cases_in_next_step_for_scenario(scenario, with_covid, start_of_vaccination, t)
    if is_time_for_aging(t) & (state.WEEKLY_CASES_FILLED.index[t] >= start_of_vaccination): handle_aging_of_susceptible_cases(scenario.S, t, False)
    scenario.I[:, t] = calculate_nr_of_infectious_incidences_in_next_step_for_scenario(scenario, with_covid, t)
    if is_time_for_aging(t) : handle_aging_of_infectious_cases(scenario.I, t)


def calculate_the_number_of_vaccinated_individuals_with_negative_offset(
        start_of_vaccination: pd.Timestamp, vacc_level: float, vacc1: np.ndarray):
    # Empty series for the weekly data
    v1_weekly = pd.Series(index=state.WEEKLY_INDEX, dtype=float)
    if start_of_vaccination + pd.DateOffset(months=-13) < state.WEEKLY_INDEX[0]:
        print("number of vaccinated individuals can not be calculated before " + str(state.WEEKLY_INDEX[0] + pd.DateOffset(months=13)))
    # Filling
    for date in state.WEEKLY_INDEX:
        if (date < start_of_vaccination) or (date + pd.DateOffset(months=-13) < pd.Timestamp(2005,1,3)):
            v1_weekly.loc[date] = 0
        elif date < GlobalConfig.START_OF_VACCINATION:
            v1_weekly.loc[date] = state.WEEKLY_BIRTH_SERIES.asof(date + pd.DateOffset(months=-13)) * 0.99 * vacc_level
        else:
            v1_weekly.loc[date] = state.WEEKLY_V1.loc[date] * vacc_level
        vacc1[1, :] = v1_weekly.values

def calculate_the_number_of_vaccinated_individuals(start_of_vaccination: pd.Timestamp, vacc_level: float, vacc1: np.ndarray):
    if start_of_vaccination <= GlobalConfig.START_OF_VACCINATION:
        calculate_the_number_of_vaccinated_individuals_with_negative_offset(start_of_vaccination, vacc_level, vacc1)

def calculate_mean_remainders():
    df_remainders = pd.DataFrame(state.REMAINDERS[:, 3:].T, index=state.WEEKLY_INDEX[3:], columns=state.AGE_GROUPS)
    timerange_for_mean = (df_remainders.index > GlobalConfig.START_OF_VACCINATION + pd.DateOffset(years=-5)) & (
                df_remainders.index < GlobalConfig.START_OF_VACCINATION)
    remainders_mean: pd.DataFrame = (df_remainders[timerange_for_mean].groupby(df_remainders.index[timerange_for_mean]
                                                                               .map(lambda index: index.week)).mean())
    no_covid_remainders: np.ndarray = np.zeros((len(state.AGE_GROUPS), len(state.WEEKLY_INDEX)))
    for t in range(0, state.NR_TIMESTEPS):
        if (state.WEEKLY_INDEX[t] < GlobalConfig.START_OF_RESTRICTIONS) or (
                state.WEEKLY_INDEX[t] > GlobalConfig.END_OF_RESTRICTIONS):
            no_covid_remainders[:, t] = state.REMAINDERS[:, t]
        else:
            week_of_year: int = state.WEEKLY_INDEX[t].week
            no_covid_remainders[:, t] = remainders_mean.loc[week_of_year]
    state.set_no_covid_remainders(no_covid_remainders)

def calculate_mean_during_covid(data: np.ndarray[float], index: pd.Series) -> np.ndarray[float]:
    df: pd.DataFrame = pd.DataFrame(data.T, index=index, columns=state.AGE_GROUPS)
    timerange_for_mean = ((df.index > GlobalConfig.START_OF_VACCINATION + pd.DateOffset(years=-5))
                          & (df.index < GlobalConfig.START_OF_VACCINATION))
    mean_data: pd.DataFrame = df[timerange_for_mean].groupby(df.index[timerange_for_mean].map(lambda i: i.week)).mean()
    no_cov_data: np.ndarray = np.zeros((len(state.AGE_GROUPS), len(state.WEEKLY_INDEX)))
    min_t = len(state.WEEKLY_INDEX) - len(index)
    max_t = len(state.WEEKLY_INDEX)
    for t in range(min_t, max_t):
        if (state.WEEKLY_INDEX[t] < GlobalConfig.START_OF_RESTRICTIONS) or (
                state.WEEKLY_INDEX[t] > GlobalConfig.END_OF_RESTRICTIONS):
            no_cov_data[:, t] = data[:, t - min_t]
        else:
            week_of_year: int = state.WEEKLY_INDEX[t].week
            no_cov_data[:, t] = mean_data.loc[week_of_year]
    return no_cov_data
