import src.utils.dataConverter as dataConverter
import src.utils.state as state
#import src.utils.dataLoader as dataLoader
from src.utils.dataLoader import DataLoader

def init_all(loader: DataLoader, detection_rate_after: float =None):
    init_varicella_data(loader, detection_rate_after)
    init_demographic_data(loader)
    init_model_initial_values(loader)
    # stb.

def init_varicella_data(loader: DataLoader, detection_rate_after: float =None):
    init_weekly_varicella_data(loader, detection_rate_after)
    init_annual_varicella_data(loader)
    init_age_structured_weekly_cases(detection_rate_after)
    set_age_structured_data_related_variables()

def init_weekly_varicella_data(loader: DataLoader, detection_rate_after: float =None):
    weekly_cases = dataConverter.convert_df_cases_to_weekly_cases(loader.df_cases, detection_rate_after)
    state.set_weekly_cases(weekly_cases)

    # fill missing values
    full_index = dataConverter.create_full_index_date_range(weekly_cases.index.max())
    state.set_full_index(full_index)

    state.set_weekly_cases_related_values()

def init_annual_varicella_data(loader: DataLoader):
    state.set_age_groups_and_nr_age_groups(loader.df_age_structured_cases.iloc[:, 0].astype(str).tolist())
    state.set_years(loader.df_age_structured_cases.columns[1:].astype(int).tolist())

    annual_cases = dataConverter.convert_df_age_structured_cases(loader.df_age_structured_cases)
    state.set_annual_cases(annual_cases)

def init_age_structured_weekly_cases(detection_rate_after: float =None):
    case_matrix = dataConverter.create_case_matrix(state.ANNUAL_CASES, state.WEEKLY_CASES_FILLED, state.WEEKLY_CASES_FULL, detection_rate_after)
    state.set_cases_matrix_and_global_min_max(case_matrix)

def set_age_structured_data_related_variables():
    state.set_age_structured_data_mask(state.WEEKLY_CASES_FILLED.index.year <= max(state.YEARS))
    state.set_age_structured_data_index(state.WEEKLY_CASES_FILLED.loc[state.AGE_STRUCTURED_DATA_MASK].index)
    #i = case_matrix
    state.set_nr_timesteps(len(state.WEEKLY_CASES_FILLED[state.AGE_STRUCTURED_DATA_MASK]))

def init_demographic_data(loader: DataLoader):
    init_death_mtx(loader)
    init_birth_mtx(loader)
    init_vaccination_matrices(loader)

def init_death_mtx(loader: DataLoader):
    state.set_weekly_index_related_values(state.WEEKLY_CASES_FILLED[state.AGE_STRUCTURED_DATA_MASK].index)

    annual_deaths = dataConverter.convert_annual_deaths(loader.df_deaths)
    death_matrix = dataConverter.create_weekly_death_matrix(annual_deaths, state.WEEKLY_INDEX)
    # death_matrix = dataConverter.create_weekly_death_matrix(annual_deaths, state.WEEKLY_CASES_FULL)
    state.set_death_matrix(death_matrix)

    # deaths_prop_matrix = dataConverter.convert_weekly_death_prop_data_to_death_prop_mtx(df_death_proportions)
    # state.set_death_matrix(deaths_prop_matrix)

def init_birth_mtx(loader: DataLoader):
    birth_mtx = dataConverter.convert_daily_birth_data_to_birth_mtx(loader.df_daily_births)
    state.set_birth_matrix(birth_mtx)

def init_vaccination_matrices(loader: DataLoader):
    weekly_v1 = dataConverter.convert_annual_v1_data_to_weekly_v1(loader.df_vaccines)
    state.set_weekly_v1(weekly_v1)
    weekly_v2 = dataConverter.convert_annual_v2_data_to_weekly_v2(loader.df_vaccines)
    state.set_weekly_v2(weekly_v2)
    state.set_weekly_v1_age_structured()
    state.set_weekly_v2_age_structured()

def init_model_initial_values(loader: DataLoader):
    init_s0(loader)
    init_n0(loader)
    init_contact_mtx(loader)

def init_s0(loader: DataLoader):
    s0_vector = dataConverter.convert_data_to_s0(loader.df_susceptibles)
    state.set_s0_vector(s0_vector)

def init_n0(loader: DataLoader):
    n0_vector = dataConverter.convert_data_to_n0(loader.df_population)
    state.set_n0_vector(n0_vector)

def init_contact_mtx(loader: DataLoader):
    contacts0 = dataConverter.convert_data_to_contact_matrix(loader.df_contacts)
    state.set_contacts0(contacts0)