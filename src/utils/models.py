# utils/models.py

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

import src.utils.state as state

@dataclass
class Scenario:
    name: str
    S: Optional[np.ndarray]
    I: Optional[np.ndarray]
    V1: Optional[np.ndarray]
    V2: Optional[np.ndarray]
    weekly_i_series: Optional[pd.Series]
    annual_i_series: Optional[pd.Series]

    def __init__(self, scenario_name: str, **kwargs):
        self.name = scenario_name
        fields = ["S", "I", "V1", "V2"]
        for field in fields:
            value = kwargs.get(field, None)
            if value is None:
                setattr(self, field, np.zeros((state.NR_AGE_GROUPS, state.NR_TIMESTEPS)))
            else:
                setattr(self, field, value)
        self.compute_i_series()

    def compute_i_series(self):
        # weekly_i_series: egyszerűen pandas Series az I-ből
        self.weekly_i_series = pd.Series(np.sum(self.I, axis=0), index=state.AGE_STRUCTURED_DATA_INDEX)

        # annual_i_series: 52 hetes összevonás
        # (ha más logika kell, ide rakod)
        self.annual_i_series = (
            self.weekly_i_series
            .groupby(self.weekly_i_series.index.year)
            .sum()
        )
        self.annual_i_series.index = pd.to_datetime(self.annual_i_series.index, format="%Y")