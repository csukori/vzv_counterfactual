# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
#     jupytext_version: 1.14.4
# ---

# %%
import numpy as np
import pandas as pd
import math

import src.utils.state as state
from src.utils.config import GlobalConfig
from src.utils.models import Scenario
from src.utils.scenarios import Scenarios

# %% [markdown]
# This cell defines the constants used in the model.
#
# **G**: represents generational interval derived from continuous SEIR model with exponential transitions:
# $$
# g(\tau) = \frac{\sigma \gamma}{\gamma - \sigma}(e^{-\sigma \tau} - e^{-\gamma \tau}),
# $$
# where the length of both the latent and infectious period follow exponential distribution, Exp($\sigma$) and Exp($\gamma$), respectively.
#
# **TAU**: represents the discretized form of $\tau$ as a vector containing the time grids: $\tau_k = k \cdot \Delta t$.
#
# (**L**: length of latent period, **D**: length of infectious period)

# %%
SIGMA = 1/GlobalConfig.L
GAMMA = 1/GlobalConfig.D
TAU = np.arange(1, GlobalConfig.MAX_TAU + 1)
G = (SIGMA * GAMMA/(GAMMA-SIGMA)) * (np.exp(-SIGMA * TAU)-np.exp(-GAMMA * TAU))
G = G/G.sum()     # normalizing

V1_EFFICACY = 0.81
V2_EFFICACY = 0.92

# %% [markdown]
# Function calculating the number of susceptible cases in the actual baseline scenario based on the following renewal equation model:
# $$
# \begin{aligned}
# S_{0,t} &= S_{0,t-1} + b_{t-1} - d_{0,t-1} \cdot \frac{S_{0,t-1}}{N_{0,t-1}} - I_{0,t-1} \\
# S_{a,t} &= S_{a,t-1} - d_{a,t-1} \cdot \frac{S_{a,t-1}}{N_{a,t-1}} - I_{a,t-1} - V_{1_{a,t-1}} * c_1, \quad a > 0
# \end{aligned}
# $$
# where $S_{a,t}, I_{a,t}, N_{a,t}, V_{1_{a,t}}, d_{a,t}, b_t, c_1$ denote, respectively, the number of susceptibles, new infections, total population,  first‑dose vaccinations administered in age group a during week $t$, deaths in age group $a$, and births at time $t$, and the efficacy of the first dose of vaccine. The size of the total population is calculated by the following system of equations:
# $$
# \begin{aligned}
# N_{0,t} = N_{0,t-1} + b_{t-1} - d_{0,t-1}, \\
# N_{a,t} = N_{a,t-1} - d_{a,t-1}, \quad a>0.
# \end{aligned}
# $$
# The number of new infections and first-dose vaccinations are derived from the reported number of cases incorporating underreporting ratio.
#
# This baseline scenario is used to calculate the effective reproduction number $R_{ab,t}$, quantifying how many secondary cases are generated in age group $a$ by on primary case in age group $b$ at time $t$.
# $$
# I_{a,t} = \sum_{k=1}^{\tau_{max}} I_{a,t-k} * g_{k,\mu}
# $$

# %%
def calculate_number_of_susceptible_cases_in_baseline(with_covid: bool):
    """
    Compute the baseline scenario’s weekly susceptible counts.

    Parameters
    ----------
    with_covid : bool
        If True, the baseline scenario includes COVID dynamics.
        If False, a counterfactual “no‑COVID” baseline is used.

    Notes
    -----
    The function initializes the baseline scenario, sets the initial
    population, and iteratively computes weekly susceptible counts
    using infection data, vaccination effects, demographic changes,
    and aging. The result forms the backbone of all subsequent
    scenario comparisons.
    """
    baseline_scenario: Scenario = init_baseline_scenario() if with_covid else init_scenario(Scenarios.NO_COVID_BASELINE)
    if with_covid:
        state.set_baseline_scenario(baseline_scenario)
        state.set_population(np.zeros((state.NR_AGE_GROUPS,state.NR_TIMESTEPS)))
        state.POPULATION[:, 0] = state.N0_VECTOR
    else:
        state.set_no_covid_population((np.zeros((state.NR_AGE_GROUPS,state.NR_TIMESTEPS))))
        state.NO_COVID_POPULATION[:, 0] = state.N0_VECTOR
        state.set_mean_i_scenario(baseline_scenario)

    for t in range(1,state.NR_TIMESTEPS):
        calculate_next_step_in_baseline(baseline_scenario, with_covid, t)

def calculate_no_covid_population():
    state.set_no_covid_population((np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))))
    state.NO_COVID_POPULATION[:, 0] = state.N0_VECTOR
    for t in range(1, state.NR_TIMESTEPS):
        calculate_size_of_population_in_next_step(False, t)

# %% [markdown]
# Function calculating remainder ($q_{a,t}$) based on Kayano, T., Ko, Y., Otani, K., Kobayashi, T., Suzuki, M., & Nishiura, H. (2023). Evaluating the COVID-19 vaccination program in Japan, 2021 using the counterfactual reproduction number. Scientific Reports, 13(1), 17762.
#
# Here, the effective reproduction number, $R_{ab,t}$, is calculated as follows:
# $$
# R_{ab,t} = s_{a,t} \cdot K_{ab,t} \cdot q_t,
# $$
# where $s_{a,t}$ denotes the fraction of susceptibles in age group $a$ at time $t$. $K_{ab}$ represents the next-generation matrix modeled as $K_{ab,t} = m_{ab} r_a$, where $m_{ab}$ and $r_a$ denote the contact matrix and relative susceptibility, respectively, and  $q_t$ incorporates all other phenomenon influencing the transmission, e.g. changes in mobility, influence of consecutive holiday etc. Let $q_{a,t} = r_a q_t$. For convenience, we will call the quantity $q_{a,t}$ the **remainders**, and use this terminology consistently in the code as well. Then
# $$
# \begin{aligned}
# R_{ab,t} &= s_{a,t} \cdot m_{ab} \cdot r_a q_t \\
# R_{ab,t} &= s_{a,t} \cdot m_{ab} \cdot q_{a,t}
# \end{aligned}
# $$
# Hence, the transmission model is as follows:
# $$
# \begin{aligned}
# I_{a,t} &= \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} R_{ab,t} I_{b,t-\tau} g_{\tau} \\
# I_{a,t} &= \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} s_{a,t} m_{ab} q_{a,t} I_{b,t-\tau} g_{\tau} \\
# I_{a,t} &= s_{a,t}  q_{a,t} \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} m_{ab} I_{b,t-\tau} g_{\tau} \\
# q_{a,t} &= \frac{I_{a,t}}{s_{a,t} \cdot \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} m_{ab} I_{b,t-\tau} g_{\tau}}
# \end{aligned}
# $$
# where $g_{\tau}$ denotes generational interval derived from continuous SEIR model with exponential transitions.
#
# Note that in the codebase, the expression
# $$
# \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} m_{ab} I_{b,t-\tau} g_{\tau}
# $$
# is implemented as the variable **denominator**, which captures the full renewal contribution from all age groups and past infectiousness.

# %%
def calculate_remainders(with_covid: bool):
    """
    Compute weekly remainder terms and related quantities.

    Parameters
    ----------
    with_covid : bool
        If True, compute remainders for the baseline scenario.
        If False, compute remainders for the mean‑incidence scenario.

    Notes
    -----
    Remainders quantify the relationship between observed cases,
    susceptibility, and past infections weighted by the generation
    interval kernel. They are used to estimate age‑specific
    transmission strength (rₐₜ) and the dominant eigenvalue ρₜ,
    which reflects overall transmissibility.
    """
    remainders_t = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    r_a_t = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    rho = np.zeros(state.NR_TIMESTEPS)
    cases_matrix = state.CASES_MATRIX if with_covid else state.MEAN_I_SCENARIO.I
    s_matrix = state.BASELINE_SCENARIO.S if with_covid else state.MEAN_I_SCENARIO.S
    population_mtx = state.POPULATION if with_covid else state.NO_COVID_POPULATION

    for t in range(math.ceil(GlobalConfig.MAX_TAU), state.NR_TIMESTEPS):
        cm = calculate_symmetrized_contact_matrix(with_covid, t)
        # weighted sum of past incidences
        denominator = calculate_denominator(cases_matrix, with_covid, t)

        if denominator.all() > 0 and s_matrix[:, t].all() > 0:
            s_t = s_matrix[:, t] / population_mtx[:, t]
            remainders_t[:, t] = cases_matrix[:, t] / (s_t * denominator)
            r_a_t[:, t] = remainders_t[:, t] * s_t
            rho[t] = max(abs(np.linalg.eigvals(cm * r_a_t[:, t])))
        else:
            remainders_t[:, t] = np.nan

    if with_covid:
        state.set_remainders(remainders_t)
        state.set_r_a(r_a_t)
        state.set_rho(rho)
    else:
        state.set_mean_remainders(remainders_t)

# %% [markdown]
# After determining the remainders from the reported number of cases, the incidence counts for the remaining scenarios are calculated using the model state from the previous step together with the corresponding remainders.
#
# In the following function the number of incidences is calculated in the baseline scenario.

# %%
def calculate_number_of_incidences_based_on_remainders(with_covid: bool):
    """
    Reconstruct weekly incidences using remainder terms.

    Notes
    -----
    For early weeks (t < τ), observed case counts are used directly.
    For later weeks, incidences are computed as rₐₜ multiplied by the
    weighted sum of past infections. This produces a model‑consistent
    incidence trajectory for the baseline scenario.
    """
    A = state.NR_AGE_GROUPS
    T = state.NR_TIMESTEPS
    i_model = np.zeros((A, T))
    for t in range(0, math.ceil(GlobalConfig.MAX_TAU)):
        i_model[:, t] = state.CASES_MATRIX[:, t]

    for t in range(math.ceil(GlobalConfig.MAX_TAU), T - 1):
        denominator = calculate_denominator(state.CASES_MATRIX, with_covid, t)
        i_model[:, t] = state.R_A[:, t] * denominator

    state.BASELINE_SCENARIO.I = i_model
    state.BASELINE_SCENARIO.compute_i_series(state.AGE_STRUCTURED_DATA_INDEX)

# %% [markdown]
# Function calculating the number of incidences in different counterfactual vaccination scenarios. The start of the vaccination and the level of vaccination coverage can be changed.
#
# Before the start of the vaccination, the number of cases will be the same as in the baseline scenario. After the vaccination is started, the number of cases is calculated based on the following model:
# $$
# \begin{aligned}
# S_{0,t} &= S_{0,t-1} + b_{t-1} - d_{0,t-1} \cdot \frac{S_{0,t-1}}{N_{0,t-1}} - I_{0,t-1}, \\
# S_{a,t} &= S_{a,t-1} - d_{a,t-1} \cdot \frac{S_{a,t-1}}{N_{a,t-1}} - I_{a,t-1} - V_{1_{a,t-1}} * c_1 * p_1, \quad a > 0 \\
# I_{a,t} &=  q_{a,t} \frac{S_{a,t}}{N_{a,t}} \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} m_{ab} I_{b,t-\tau} g_{\tau},
# \end{aligned}
# $$
# where $p_1$ is the vaccination level relative to the actually implemented level.

# %%
def calculate_number_of_cases_for_scenario(scenario_name: str, vacc_level: float, start_of_vaccination: pd.Timestamp, with_covid: bool) -> Scenario:
    """
    Compute weekly susceptible and infectious cases for a given scenario.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario (e.g., “Vaccination level 75%”).
    vacc_level : float
        Target vaccination coverage level.
    start_of_vaccination : Timestamp
        Date when vaccination begins in the scenario.
    with_covid : bool
        Whether COVID dynamics are included.

    Notes
    -----
    The function:
    - initializes the scenario,
    - assigns vaccination counts,
    - copies baseline values before vaccination starts,
    - iteratively computes weekly S and I after vaccination begins.

    The result is a fully populated Scenario object.
    """
    scenario = init_scenario(scenario_name)
    calculate_the_number_of_vaccinated_individuals(start_of_vaccination, vacc_level, scenario.V1)
    for t in range(0, state.NR_TIMESTEPS - 1):
        if state.WEEKLY_CASES_FILLED.index[t] < start_of_vaccination:
            scenario.I[:, t] = state.CASES_MATRIX[:, t]
            #scenario.I[:, t] = state.CASES_MATRIX[:, t] if with_covid else state.MEAN_I_SCENARIO.I[:, t]
            scenario.S[:, t] = state.BASELINE_SCENARIO.S[:, t]
            #scenario.S[:, t] = state.BASELINE_SCENARIO.S[:, t] if with_covid else state.MEAN_I_SCENARIO.S[:, t]

    for t in range(math.ceil(GlobalConfig.MAX_TAU), state.NR_TIMESTEPS):
        is_next_step_calculated: bool = (
                ((start_of_vaccination <= GlobalConfig.START_OF_VACCINATION)
                 & (state.WEEKLY_CASES_FILLED.index[t] >= start_of_vaccination))
                | ((start_of_vaccination > GlobalConfig.START_OF_VACCINATION)
                   & (state.WEEKLY_CASES_FILLED.index[t] >= GlobalConfig.START_OF_VACCINATION)))
        if is_next_step_calculated:
             calculate_next_step_for_scenario(scenario, with_covid, start_of_vaccination, t)
    scenario.compute_i_series(state.AGE_STRUCTURED_DATA_INDEX)
    return scenario

# %% [markdown]
# This function initializes the baseline scenario with the following parameters:
# - name: "Implemented scenario"
# - S (number of susceptible individuals): an array of zeros but at the first position, where the initial distribution is set
# - V1 (vaccinated with one dose): actual reported data of vaccinated individuals distributed in time according to the birth distribution
# - V2 (vaccinated with two doses): (should be) actual reported data of vaccinated individuals distributed in time according to the birth distribution

# %%
def init_baseline_scenario():
    """
    Initialize the baseline scenario with initial susceptible counts
    and age‑structured vaccination matrices.

    Notes
    -----
    The baseline scenario serves as the reference trajectory against
    which all hypothetical scenarios are compared.
    """
    s_scen = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    s_scen[:, 0] = state.S0_VECTOR
    return Scenario(Scenarios.BASELINE, S=s_scen, V1=state.WEEKLY_V1_AGE_STRUCTURED, V2=state.WEEKLY_V2_AGE_STRUCTURED)

# %%
def init_scenario(scenario_name: str) -> Scenario:
    """
    Initialize a new scenario using baseline values for the first τ weeks.

    Notes
    -----
    Early weeks use baseline S and I because the generation interval
    kernel requires past incidence values. After initialization, the
    scenario is ready for iterative updates.
    """
    s_scen = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    i_scen = np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS))
    for t in range(0, math.ceil(GlobalConfig.MAX_TAU)):
        i_scen[:, t] = state.BASELINE_SCENARIO.I[:, t]
        s_scen[:, t] = state.BASELINE_SCENARIO.S[:, t]
    return Scenario(scenario_name, S=s_scen, I=i_scen)

# %%
def calculate_next_step_in_baseline(scenario: Scenario, with_covid: bool, t: int):
    """
    Compute susceptible counts, apply aging, and update population
    for the baseline scenario at week t.

    Notes
    -----
    This function performs one full model step:
    - update susceptible counts,
    - apply demographic aging,
    - update population (if COVID dynamics are included).
    """
    calculate_number_of_susceptible_cases_in_next_step_in_baseline(scenario, with_covid, t)
    calculate_size_of_population_in_next_step(with_covid, t)

# %%
def calculate_size_of_population_in_next_step(with_covid: bool, t: int):
    """
    Update the age‑structured population for week t.

    Notes
    -----
    Population changes are applied weekly except during the first
    Monday of September, when aging is handled separately. Births
    and deaths are incorporated into each age group.
    """
    population_mtx = state.POPULATION if with_covid else state.NO_COVID_POPULATION
    prev_population = get_prev_step(t, population_mtx, state.SUSCEPIBLE_AGING_MTX)
    birth = state.BIRTH_MATRIX[:, t - 1] if with_covid else state.NO_COVID_BIRTHS[:, t - 1]
    death = state.DEATHS_MATRIX[:, t - 1] if with_covid else state.NO_COVID_DEATHS[:, t - 1]
    population_mtx[:, t] = prev_population + birth - death

# %%
def calculate_number_of_susceptible_cases_in_next_step_in_baseline(scenario: Scenario, with_covid: bool, t: int):
    """
    Compute susceptible counts for week t in the baseline scenario.

    Notes
    -----
    Susceptibles are updated using:
    - infections,
    - effective vaccination,
    - births and deaths,
    - demographic changes.

    This forms the core recurrence relation for Sₜ.
    """
    effectively_vaccinated_v1 = scenario.V1[:, t-1] * GlobalConfig.V1_EFFICACY
    effectively_vaccinated_v2 = scenario.V2[:, t-1] * GlobalConfig.V2_EFFICACY
    effectively_vaccinated = effectively_vaccinated_v1 + effectively_vaccinated_v2
    cases_mtx = state.CASES_MATRIX if with_covid else state.MEAN_I
    prev_cases = get_prev_step(t, cases_mtx, state.INFECTIOUS_AGING_MTX)
    births = state.BIRTH_MATRIX[:, t-1] if with_covid else state.NO_COVID_BIRTHS[:,t-1]
    deaths = state.DEATHS_MATRIX[:, t-1] if with_covid else state.NO_COVID_DEATHS[:,t-1]
    population = state.POPULATION if with_covid else state.NO_COVID_POPULATION
    prev_population = get_prev_step(t, population, state.SUSCEPIBLE_AGING_MTX)
    prev_s = get_prev_step(t, scenario.S, state.SUSCEPIBLE_AGING_MTX)
    demographic_changes = (births - deaths * (prev_s / prev_population))
    scenario.S[:, t] = (prev_s - prev_cases - effectively_vaccinated + demographic_changes)

# %% [markdown]
# We use realistic age‑structured models in which individuals shift to the next age group at fixed periodic points in time, with no intermediate transfers between age groups. The timing for age shift is set to the first Monday of September in each year.
#
# This function updates the compartments at time $t$ if it is the first Monday of September according to the above described aging process.

# %%
def is_time_for_aging(t: int) -> bool:
    """
    Determine whether week t corresponds to the annual aging event.

    Notes
    -----
    Aging occurs during the first Monday of September.
    """
    is_first_monday_of_september = ((state.WEEKLY_CASES_FILLED.index[t].month == 9) &
                                    (state.WEEKLY_CASES_FILLED.index[t].day <= 7))
    return (t <= state.NR_TIMESTEPS - 1) & is_first_monday_of_september

# %% [markdown]
# This function updates the infectious compartments at time $t$ (if it is the first Monday of September) according to the above described aging process. In the age groups 10–14, 15–19, and 20+, only a fraction of individuals moves to the next age group. Specifically, we assume that one tithe advances, whereas nine tithe remain in the same age group.

# %%
def handle_aging_of_infectious_cases(i_scen: np.ndarray, t: int):
    """Reset infectious counts for the youngest age group during aging."""
    A = state.NR_AGE_GROUPS
    q: float = 0.2
    # age group 20+
    i_scen[A - 1, t] = i_scen[A - 1, t] + i_scen[A - 2, t] * q
    # age group 15-19
    i_scen[A - 2, t] = i_scen[A - 2, t] * (1-q) + i_scen[A - 3, t] * q
    # age group 10-14
    i_scen[A - 3, t] = i_scen[A - 3, t] * (1-q) + i_scen[A - 4, t]
    # age groups 1-9
    for a in range(A - 4, 0, -1):
        i_scen[a, t] = i_scen[a - 1, t]
    i_scen[0, t] = 1

# %% [markdown]
# As mentioned above, the expression
# $$
# \sum_{b=0}^{b_{max}} \sum_{\tau=1}^{t-1} m_{ab} I_{b,t-\tau} g_{\tau}
# $$
# in the denominator of the formula for the remainders ($q_{a,t}$) is referred as **denominator** in the implementation. This function computes the value of this expression.

# %%
def calculate_denominator(i_scen: np.ndarray, with_covid: bool, t: int) -> np.ndarray:
    """
    Compute the weighted sum of past incidences for week t.

    Notes
    -----
    The denominator combines:
    - the generation interval kernel,
    - mortality adjustments,
    - past incidence values,
    - and the symmetrized contact matrix.

    It represents the force of infection contributed by past weeks.
    """
    population_mtx = state.POPULATION if with_covid else state.NO_COVID_POPULATION
    death_mtx = state.DEATHS_MATRIX if with_covid else state.NO_COVID_DEATHS
    cm = calculate_symmetrized_contact_matrix(with_covid, t)
    mu_t = (death_mtx[:, t] / population_mtx[:, t]).reshape(1, -1)  # shape: (1,A)
    tau = np.arange(1, GlobalConfig.MAX_TAU + 1)  # shape: (A,1)
    kernel = G.reshape(-1, 1) * np.exp(-mu_t * tau.reshape(-1, 1))
    infectious = i_scen[:, t - tau].copy()
    aging_mask = is_time_for_aging(t - tau)
    if np.any(aging_mask):
        infectious[:, aging_mask] = (
                state.INFECTIOUS_AGING_MTX
                @ infectious[:, aging_mask]
        )
    return np.sum(infectious * kernel.T, axis=1) @ cm

# %% [markdown]
# Contact matrix is taken from https://github.com/zsvizi/covid19hun (published in Röst, G., Bartha, F. A., Bogya, N., Boldog, P., Dénes, A., Ferenci, T., Horváth, K. J., Juhász, A., Nagy, C., Tekeli, T., Vizi, Z., & Oroszi, B. (2020). Early Phase of the COVID-19 Outbreak in Hungary and Post-Lockdown Scenarios. Viruses, 12(7), 708. https://doi.org/10.3390/v12070708 ).
#
# This function symmetrize the measured matrix.

# %%
def calculate_symmetrized_contact_matrix(with_covid: bool, t: int) -> np.ndarray:
    """
    Compute the symmetrized contact matrix for week t.

    Notes
    -----
    Symmetrization accounts for differences in age‑group sizes and
    ensures that contact rates are consistent in both directions.
    """
    ## the size of the age groups at the previous week as a column vector
    population = state.POPULATION if with_covid else state.NO_COVID_POPULATION
    prev_population = get_prev_step(t, population, state.SUSCEPIBLE_AGING_MTX)
    population_i = prev_population.reshape(-1,1)
    ## the size of the age groups at the previous week as a row vector
    population_j = prev_population.reshape(1,-1)
    return (state.CONTACTS0 * population_i + state.CONTACTS0.T * population_j) / (2 * population_i)

# %%
def calculate_nr_of_susceptible_cases_in_next_step_for_scenario(scenario: Scenario, with_covid: bool, start_of_vaccination: pd.Timestamp, t: int) -> np.ndarray:
    """
    Compute susceptible counts for week t in a hypothetical scenario.

    Notes
    -----
    Uses scenario‑specific vaccination, infections, births, deaths,
    and demographic changes. Before vaccination starts, baseline
    susceptible counts are used.
    """
    effectively_vaccinated_v1 = scenario.V1[:, t - 1] * GlobalConfig.V1_EFFICACY
    effectively_vaccinated_v2 = scenario.V2[:, t - 1] * GlobalConfig.V2_EFFICACY
    effectively_vaccinated = effectively_vaccinated_v1 + effectively_vaccinated_v2
    birth_mtx: np.ndarray[float] = state.BIRTH_MATRIX if with_covid else state.NO_COVID_BIRTHS
    death_mtx: np.ndarray[float] = state.DEATHS_MATRIX if with_covid else state.NO_COVID_DEATHS
    population_mtx: np.ndarray[float] = state.POPULATION if with_covid else state.NO_COVID_POPULATION
    prev_population = get_prev_step(t, population_mtx, state.SUSCEPIBLE_AGING_MTX)
    prev_s = get_prev_step(t, scenario.S, state.SUSCEPIBLE_AGING_MTX)
    prev_i = get_prev_step(t, scenario.I, state.INFECTIOUS_AGING_MTX)
    demographic_changes = (birth_mtx[:,t-1] - death_mtx[:,t-1] * (prev_s / prev_population))

    is_baseline_used: bool =((start_of_vaccination < GlobalConfig.START_OF_VACCINATION) & (
                state.WEEKLY_CASES_FILLED.index[t] < start_of_vaccination)) \
    | ((start_of_vaccination > GlobalConfig.START_OF_VACCINATION) & (state.WEEKLY_CASES_FILLED.index[
        t] < GlobalConfig.START_OF_VACCINATION))

    if is_baseline_used:
        s_scen = state.BASELINE_SCENARIO.S[:, t]
    else:
        s_scen = prev_s - prev_i - effectively_vaccinated + demographic_changes

    return s_scen

# %%
def calculate_nr_of_infectious_incidences_in_next_step_for_scenario(scenario: Scenario, with_covid: bool, t: int) -> np.ndarray:
    """
    Compute infectious incidences for week t in a hypothetical scenario.

    Notes
    -----
    Incidences are derived from remainder terms and the weighted sum
    of past infections, scaled by susceptibility.
    """
    population = state.POPULATION if with_covid else state.NO_COVID_POPULATION
    s_t = scenario.S[:,t] / population[:, t]
    remainders: np.ndarray[float] = state.REMAINDERS if with_covid else state.MEAN_REMAINDERS
    r_t_scen = remainders[:, t] * s_t
    denominator = calculate_denominator(scenario.I, with_covid, t)
    return r_t_scen * denominator

# %%
def calculate_next_step_for_scenario(scenario: Scenario, with_covid: bool, start_of_vaccination: pd.Timestamp, t: int):
    """
    Perform one full model step for a hypothetical scenario.

    Notes
    -----
    Updates susceptible counts, applies aging, and computes new
    infectious incidences for week t.
    """
    scenario.S[:, t] = calculate_nr_of_susceptible_cases_in_next_step_for_scenario(scenario, with_covid, start_of_vaccination, t)
    scenario.I[:, t] = calculate_nr_of_infectious_incidences_in_next_step_for_scenario(scenario, with_covid, t)

# %% [markdown]
# This function calculates the number of individuals vaccinated with one dose in counterfactual scenarios where the vaccination against varicella has been part of the routine immunization program earlier than in the actual baseline scenario (**start_of_vaccination**) with vaccination level relative to the actual implemented scenario (**vacc_level**). In this case, the number of individuals vaccinated with one dose is based on the number of births 13 months before $t$ assuming that 99% of the infants would get vaccinated at most before 01/09/2019 (the actual start of vaccination) and the actual number of vaccinated individuals after 01/09/2019. That is:
# $$
# \begin{aligned}
# v_{1_t}&=0, \quad &\text{if $t$ is before the start of vaccination} \\
# v_{1_t}&=b_{(t-13 months)} * 0.99 * c_1 \quad &\text{if $t$ is after the start of vaccination but before 01/09/2019} \\
# v_{1_t}&=V_{1_t} * c_1 \quad &\text{if $t$ is after 01/09/2019}
# \end{aligned}
# $$

# %%
def calculate_the_number_of_vaccinated_individuals_with_negative_offset(
        start_of_vaccination: pd.Timestamp, vacc_level: float, vacc1: np.ndarray):
    """
    Compute weekly first‑dose vaccination counts when vaccination
    begins before the reference vaccination start date.

    Notes
    -----
    Uses birth‑based proportional allocation for early weeks and
    observed vaccination data afterwards.
    """
    # Empty series for the weekly data
    v1_weekly = pd.Series(index=state.WEEKLY_INDEX, dtype=float)
    if start_of_vaccination + pd.DateOffset(months=-13) < state.WEEKLY_INDEX[0]:
        print("number of vaccinated individuals can not be calculated before " + str(state.WEEKLY_INDEX[0] + pd.DateOffset(months=13)))
    # Filling
    for date in state.WEEKLY_INDEX:
        if (date <= start_of_vaccination) or (date + pd.DateOffset(months=-13) < pd.Timestamp(2005,1,3)):
            v1_weekly.loc[date] = 0
        elif date < GlobalConfig.START_OF_VACCINATION:
            v1_weekly.loc[date] = state.WEEKLY_BIRTH_SERIES.asof(date + pd.DateOffset(months=-13)) * 0.99 * vacc_level
        else:
            v1_weekly.loc[date] = state.WEEKLY_V1.loc[date] * vacc_level
        vacc1[1, :] = v1_weekly.values

def calculate_the_number_of_vaccinated_individuals_with_positive_offset(
        start_of_vaccination: pd.Timestamp, vacc_level: float, vacc1: np.ndarray):
    v1_weekly = pd.Series(index=state.WEEKLY_INDEX, dtype=float)
    for date in state.WEEKLY_INDEX:
        if date < start_of_vaccination:
            v1_weekly.loc[date] = 0
        else:
            v1_weekly.loc[date] = state.WEEKLY_V1.loc[date] * vacc_level
        vacc1[1, :] = v1_weekly.values

# %%
def calculate_the_number_of_vaccinated_individuals(start_of_vaccination: pd.Timestamp, vacc_level: float, vacc1: np.ndarray):
    """
    Wrapper for computing weekly vaccination counts, handling cases
    where vaccination starts earlier than the reference date.
    """
    if start_of_vaccination <= GlobalConfig.START_OF_VACCINATION:
        calculate_the_number_of_vaccinated_individuals_with_negative_offset(start_of_vaccination, vacc_level, vacc1)
    else:
        calculate_the_number_of_vaccinated_individuals_with_positive_offset(start_of_vaccination, vacc_level, vacc1)

# %% [markdown]
# This function calculates the mean of the given **data** in the timerange 01/09/2014 - 01/09/2019. This mean values are used then during the restrictions and the actual data otherwise. (The mean value of births, deaths and varicella cases are calculated by this function and used during the restrictions. The mean value of remainders is calculated then based on the mean number of varicella cases.)

# %%
def calculate_mean_during_covid(data: np.ndarray[float], index: pd.Series) -> np.ndarray[float]:
    """
    Compute seasonal mean values for a dataset and construct a
    no‑COVID counterfactual version.

    Notes
    -----
    Weekly seasonal means are used to replace values during COVID
    restriction periods, producing a smooth counterfactual dataset.
    """
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

def estimate_rt_during_covid(data: np.ndarray[float], index: pd.Series) -> np.ndarray:
    df: pd.DataFrame = pd.DataFrame(data.T, index=index, columns=state.AGE_GROUPS)
    timerange_for_mean = ((df.index > GlobalConfig.START_OF_VACCINATION + pd.DateOffset(years=-5))
                         & (df.index < GlobalConfig.START_OF_VACCINATION))
    q_pre = df.loc[timerange_for_mean]   # q: DataFrame, columns: age groups
    mu = q_pre.mean()   # the mean of all age groups
    weeks_pre = q_pre.index.isocalendar().week
    seasonal = q_pre.groupby(weeks_pre).mean()   # weekly mean per age group
    seasonal = seasonal - mu                     # demeaning

    weeks_full = df.index.isocalendar().week
    q_nocovid = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:   # all age groups
        q_nocovid[col] = mu[col] + seasonal[col].reindex(weeks_full).values

    covid_mask = (df.index >= pd.Timestamp("2020-03-16")) & (df.index <= pd.Timestamp("2025-09-01"))

    q_final = df.copy()   # real data
    q_final.loc[covid_mask] = q_nocovid.loc[covid_mask]*0.9   # estimation (only) during COVID

    return q_final.values.T

def calculate_mean_during_covid(data: np.ndarray[float], index: pd.Series) -> np.ndarray[float]:
    df: pd.DataFrame = pd.DataFrame(data.T, index=index, columns=state.AGE_GROUPS)
    timerange_for_mean = ((df.index > GlobalConfig.START_OF_VACCINATION + pd.DateOffset(years=-5))
                         & (df.index < GlobalConfig.START_OF_VACCINATION))
    df_sub = df[timerange_for_mean].copy()
    current_year = df.index.max().year
    df_sub["weight"] = current_year - df_sub.index.year + 1


    mean_data: pd.DataFrame = df[timerange_for_mean].groupby(df.index[timerange_for_mean].map(lambda i: i.week)).mean()
    no_cov_data: np.ndarray = np.zeros((len(state.AGE_GROUPS), len(state.WEEKLY_INDEX)))
    for t in range(0, len(state.WEEKLY_INDEX)):
        no_cov_data[:, t] = data[:, t]
        if (state.WEEKLY_INDEX[t] > pd.Timestamp("2020-03-16")) and (state.WEEKLY_INDEX[t] < pd.Timestamp("2025-09-01")):
            week_of_year: int = state.WEEKLY_INDEX[t].week
            if week_of_year > len(mean_data):
                no_cov_data[:, t] = mean_data.loc[week_of_year-1][:]
            else:
                no_cov_data[:, t] = mean_data.loc[week_of_year][:]
    return no_cov_data

def get_prev_step(t: int, data: np.ndarray[float], aging_matrix: np.ndarray[float]) -> np.ndarray[float]:
    aging_corrector_vector: np.ndarray[float] = np.zeros(state.NR_AGE_GROUPS)
    aging_corrector_vector[0] = 1
    return aging_matrix @ data[:, t - 1] + aging_corrector_vector if is_time_for_aging(t - 1) else data[:, t - 1]
