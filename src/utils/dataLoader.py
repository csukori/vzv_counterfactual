import pandas as pd

class DataLoader:
    def __init__(self):
        self._df_cases: pd.DataFrame | None = None
        self._df_age_structured_cases: pd.DataFrame | None = None
        self._df_susceptibles: pd.DataFrame | None = None
        self._df_population: pd.DataFrame | None = None
        self._df_births: pd.DataFrame | None = None
        self._df_daily_births: pd.DataFrame | None = None
        self._df_deaths: pd.DataFrame | None = None
        self._df_death_proportions: pd.DataFrame | None = None
        self._df_vaccines: pd.DataFrame | None = None
        self._df_contacts: pd.DataFrame | None = None

    def load_data(self):
        self._df_cases = pd.read_excel("../resources/weekly_varicella_hun.xlsx", sheet_name="weekly_cases")
        self._df_age_structured_cases = pd.read_excel("../resources/age_structured_vzv_hun.xlsx", sheet_name="varicella")
        self._df_susceptibles = pd.read_excel("../resources/number_of_susceptibles_hun.xlsx", sheet_name="s0")
        self._df_population = pd.read_excel("../resources/age_structured_population.xlsx", sheet_name="population")
        self._df_births = pd.read_excel("../resources/births_hun.xlsx", sheet_name="births")
        self._df_daily_births = pd.read_excel("../resources/births_hun.xlsx", sheet_name="daily", header=None)
        self._df_deaths = pd.read_excel("../resources/age_structured_deaths_hun.xlsx", sheet_name="deaths")
        self._df_death_proportions = pd.read_excel("../resources/age_structured_deaths_hun.xlsx", sheet_name="proportions")
        self._df_vaccines = pd.read_excel("../resources/vaccine_coverage_hun.xlsx", sheet_name="vaccines")
        self._df_contacts = pd.read_excel("../resources/contact_mtx_hun.xlsx", sheet_name="contacts", index_col=0)

    @property
    def df_cases(self) -> pd.DataFrame:
        return self._df_cases

    @property
    def df_age_structured_cases(self) -> pd.DataFrame:
        return self._df_age_structured_cases

    @property
    def df_susceptibles(self) -> pd.DataFrame:
        return self._df_susceptibles

    @property
    def df_population(self) -> pd.DataFrame:
        return self._df_population

    @property
    def df_births(self) -> pd.DataFrame:
        return self._df_births

    @property
    def df_daily_births(self) -> pd.DataFrame:
        return self._df_daily_births

    @property
    def df_deaths(self) -> pd.DataFrame:
        return self._df_deaths

    @property
    def df_death_proportions(self) -> pd.DataFrame:
        return self._df_death_proportions

    @property
    def df_vaccines(self) -> pd.DataFrame:
        return self._df_vaccines

    @property
    def df_contacts(self) -> pd.DataFrame:
        return self._df_contacts
