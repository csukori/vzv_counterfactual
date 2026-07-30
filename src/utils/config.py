import pandas as pd

class GlobalConfig:
    """
    Global configuration parameters for the epidemiological model.

    This class stores fixed values that define the structure and
    time frame of the simulation, such as the number of age groups,
    the weekly date index, and key reference dates (e.g. the start
    of vaccination). These constants ensure that all modules use a
    consistent configuration and make the model reproducible.

    Notes
    -----
    - Changing these values affects the entire model pipeline.
    - The parameters reflect real-world demographic and epidemiological
      assumptions used throughout the project.
    """

    # Date when vaccination rollout begins in the model.
    # Used to switch detection rates and to start distributing vaccine doses.
    START_OF_VACCINATION: pd.Timestamp = pd.Timestamp("2019-09-01")

    # First date for which age‑structured population and case data are available.
    # Defines the earliest point of the simulation timeline.
    START_OF_AGE_STRUCTURED_DATE: pd.Timestamp = pd.Timestamp("2005-01-01")

    # Start date of non‑pharmaceutical interventions (e.g., restrictions, lockdowns).
    # Used to adjust transmission dynamics during the intervention period.
    START_OF_RESTRICTIONS: pd.Timestamp = pd.Timestamp("2020-03-16")

    # End date of non‑pharmaceutical interventions.
    # Marks the point where transmission dynamics return to baseline.
    END_OF_RESTRICTIONS: pd.Timestamp = pd.Timestamp("2021-09-01")

    # Vaccine efficacy after the first dose.
    # Represents the proportion of vaccinated individuals who become immune.
    V1_EFFICACY: float = 0.81

    # Vaccine efficacy after the second dose.
    # Represents the proportion of individuals who achieve full immunity.
    V2_EFFICACY: float = 0.92

    # Length of the latent period (time between infection and becoming infectious), in weeks.
    L: float = 2

    # Length of the infectious period, in weeks.
    D: float = 1

    # Total generation interval (latent + infectious period).
    # Used in Rₜ estimation and transmission dynamics.
    MAX_TAU: float = L + D
