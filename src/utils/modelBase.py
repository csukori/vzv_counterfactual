import src.utils.scenarioCalculator as scenarioCalculator

def calculate_baseline():
    scenarioCalculator.calculate_number_of_susceptible_cases_in_baseline(True)
    scenarioCalculator.calculate_remainders(True)
    scenarioCalculator.calculate_number_of_incidences_based_on_remainders(True)

