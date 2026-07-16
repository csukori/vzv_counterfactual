import pandas as pd

class GlobalConfig:
    START_OF_VACCINATION: pd.Timestamp = pd.Timestamp("2019-09-01")
    START_OF_AGE_STRUCTURED_DATE: pd.Timestamp = pd.Timestamp("2005-01-01")
    START_OF_RESTRICTIONS: pd.Timestamp = pd.Timestamp("2020-03-16")
    END_OF_RESTRICTIONS: pd.Timestamp = pd.Timestamp("2021-09-01")
    V1_EFFICACY: float = 0.81
    V2_EFFICACY: float = 0.92

    MAX_TAU: int = 3
