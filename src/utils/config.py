import pandas as pd

class GlobalConfig:
    START_OF_VACCINATION = pd.Timestamp("2019-09-01")
    START_OF_AGE_STRUCTURED_DATE = pd.Timestamp("2005-01-01")
    V1_EFFICACY = 0.81
    V2_EFFICACY = 0.92

    MAX_TAU = 3
