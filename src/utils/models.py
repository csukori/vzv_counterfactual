# utils/models.py

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

import src.utils.state as state

@dataclass
class Scenario:
    """
    Represents an epidemiological scenario containing age‑structured
    weekly data for susceptible individuals (S), new infections (I),
    first‑dose vaccinations (V1), second‑dose vaccinations (V2), and
    aggregated weekly/annual infection counts.

    Each variable is stored as a 2D NumPy array with shape:
        (number of age groups, number of weeks)

    If the user does not provide S, I, V1, or V2, the model initializes
    them as zero‑filled arrays. This ensures that every Scenario object
    is complete and internally consistent, even when only partial data
    is supplied.

    Attributes
    ----------
    name : str
        Human‑readable name of the scenario.
    S : np.ndarray
        Number of susceptible individuals in each age group and week.
    I : np.ndarray
        Number of new infections in each age group and week.
    V1 : np.ndarray
        Number of individuals receiving their first vaccine dose in
        each age group and week.
    V2 : np.ndarray
        Number of individuals receiving their second vaccine dose in
        each age group and week.
    weekly_i_series : pd.Series
        Total weekly infections summed over all age groups.
    annual_i_series : pd.Series
        Total annual infections summed over all weeks of each year.
    """
    name: str
    S: Optional[np.ndarray]
    I: Optional[np.ndarray]
    V1: Optional[np.ndarray]
    V2: Optional[np.ndarray]
    weekly_i_series: Optional[pd.Series]
    annual_i_series: Optional[pd.Series]

    def __init__(self, scenario_name: str, **kwargs):
        """
        Initializes the Scenario object.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario.
        **kwargs :
            Optional keyword arguments for S, I, V1, and V2. If any of
            these are missing, the model creates a zero‑filled array of
            the correct shape so that the scenario remains usable.

        Notes
        -----
        After initialization, the method automatically computes:
        - weekly_i_series : total infections per week
        - annual_i_series : total infections per year
        """
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
        """
        Computes weekly and annual infection totals.

        weekly_i_series:
            Sums the new infections (I) across all age groups for each week.
            The result is a pandas Series indexed by calendar dates.

        annual_i_series:
            Groups the weekly totals by calendar year and sums them.
            This produces a high-level view of how many infections occurred
            in each year of the simulation.
        """
        self.weekly_i_series = pd.Series(np.sum(self.I, axis=0), index=state.AGE_STRUCTURED_DATA_INDEX)

        self.annual_i_series = (
            self.weekly_i_series
            .groupby(self.weekly_i_series.index.year)
            .sum()
        )
        self.annual_i_series.index = pd.to_datetime(self.annual_i_series.index, format="%Y")