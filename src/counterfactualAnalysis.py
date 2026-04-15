import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Reading data
# ---------------------------------------------------------

df_cases = pd.read_excel("../resources/weekly_varicella_hun.xlsx", sheet_name="weekly_cases")
df_births = pd.read_excel("../resources/births_hun.xlsx", sheet_name="births")
df_deaths = pd.read_excel("../resources/deaths_hun.xlsx", sheet_name="deaths")

# ---------------------------------------------------------
# 2. Convert dates and set indices
# ---------------------------------------------------------

# VARICELLA DATA

# "2020_1" → split year and week
df_cases[["Year", "WeekNum"]] = df_cases["Week"].str.split("_", expand=True)
df_cases["Year"] = df_cases["Year"].astype(int)
df_cases["WeekNum"] = df_cases["WeekNum"].astype(int)

# ISO week -> date (the first day of the week, Monday)
df_cases["Date"] = pd.to_datetime(df_cases["Year"].astype(str) + df_cases["WeekNum"].astype(str) + "1", format="%G%V%u")

# set date index
df_weekly_cases = df_cases.set_index("Date").sort_index()
weekly_cases = df_weekly_cases["Cases"] * (1/0.4)

# fill missing values
full_index = pd.date_range(
    start=weekly_cases.index.min(),
    end=weekly_cases.index.max(),
    freq="W-MON"
)
weekly_cases_full = weekly_cases.reindex(full_index)
weekly_cases_filled = weekly_cases_full.interpolate(method="linear")
print(weekly_cases_filled)

# BIRTH AND DEATH DATA

weekly_index = weekly_cases_filled.index

# Yearly data in dictionary format
births_yearly = dict(zip(df_births["Year"], df_births["Cases"]))
deaths_yearly = dict(zip(df_deaths["Year"], df_deaths["Cases"]))

# Empty series for the weekly data
weekly_births = pd.Series(index=weekly_index, dtype=float)
weekly_deaths = pd.Series(index=weekly_index, dtype=float)

# Filling
for date in weekly_index:
    year = date.year
    weekly_births.loc[date] = births_yearly[year] / 52
    weekly_deaths.loc[date] = deaths_yearly[year] / 52


# ---------------------------------------------------------
# 2. CALCULATING THE NUMBER OF SUSCEPTIBLES
# ---------------------------------------------------------

i = weekly_cases_filled.values
T = len(i)

S = np.zeros(T)
N = np.zeros(T)

# Values are from number_of_susceptibles_hun.xlsx
N0 = 10200198
S[0] = 1222086
N[0] = N0

for t in range(1, T):
    # Update the number of susceptibles
    S[t] = S[t-1] + weekly_births.iloc[t] - i[t] - weekly_deaths.iloc[t] * (S[t-1] / N[t-1])

    # Update the population
    N[t] = N[t-1] + weekly_births.iloc[t] - weekly_deaths.iloc[t]

# ---------------------------------------------------------
# 3. ESTIMATE R_t USING RENEWAL EQUATION
# ---------------------------------------------------------

L = 2 # length of latent period in weeks
D = 1 # length of infectious period in weeks

sigma = 1/2   # length of latent period is 2 weeks
gamma = 1   # length of infectious period is 1 week

max_tau = 3
tau = np.arange(1, max_tau+1)
g = (sigma * gamma / (gamma - sigma)) * (np.exp(-sigma * tau) - np.exp(-gamma * tau))
g = g / g.sum()          # normalizing

R_t = np.zeros(T)

for t in range(max_tau, T):
    mu_t = weekly_deaths.iloc[t] / N[t]
    year = weekly_cases_filled.index[t].year

    # infectiousness kernel considering demography
    kernel = g * np.exp(-mu_t * np.arange(1, max_tau+1))

    # weighted sum of past incidences
    denom = np.sum(i[t - np.arange(1, max_tau+1)] * kernel)

    if denom > 0 and S[t] > 0:
        s_t = S[t] / N[t]
        R_t[t] = i[t] / (s_t * denom)
    else:
        R_t[t] = np.nan

# ---------------------------------------------------------
# 4. RESULTS FOR R_T
# ---------------------------------------------------------

weekly_Rt = pd.Series(R_t, index=weekly_cases_filled.index)
print(weekly_Rt)


# plt.figure(figsize=(12, 6))
# weekly_Rt.plot()
#
# plt.axhline(1.0, color='red', linestyle='--', linewidth=1)  # R=1 as reference
# plt.title("Weekly estimated Rₜ")
# plt.ylabel("Rₜ value")
# plt.xlabel("Date")
# plt.grid(True)

# plt.show()

# ---------------------------------------------------------
# 4. RESULTS FOR NUMBER OF INCIDENCES
# ---------------------------------------------------------

i_model = np.zeros(T)
for t in range(0, max_tau):
    i_model[t] = i[t]

for t in range(max_tau, T-1):
    s_t = S[t] / N[t]
    i_model[t] = s_t * R_t[t] * np.sum(i_model[t - np.arange(1, max_tau+1)] * g)

weekly_i_model = pd.Series(i_model, index=weekly_cases_filled.index)

# plt.figure(figsize=(10, 5))
#
# plt.plot(weekly_i_model.index, weekly_i_model.values,
#          label="Estimated number of cases", color="royalblue", linewidth=1)
#
# plt.scatter(
#     weekly_cases_filled.index,
#     weekly_cases_filled.values,
#     color="black",
#     s=4,
#     label="Observed data"
# )
#
#
# plt.title("Estimated and observed cases")
# plt.xlabel("Date")
# plt.ylabel("Weekly number of cases")
# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# ---------------------------------------------------------
# 4. INCIDENCES WITHOUT VACCINATION
# ---------------------------------------------------------

# 4. a) estimate R_t using mean

df = weekly_Rt.to_frame(name="Rt")
df["year"] = df.index.year
df["week"] = df.index.isocalendar().week

weekly_Rt_mean = df.groupby("week")["Rt"].mean()
df_2018 = df.loc[df.year == 2018].copy()
df_2018["week"] = df_2018.index.isocalendar().week-1
Rt_2018 = df_2018.set_index("week")["Rt"]
# Rt_2018 = Rt_2018.iloc[:-1]

counterfactual_Rt = weekly_Rt.copy()
start_of_vaccination = pd.Timestamp("2019-09-01")

T = 0
for t, date in enumerate(counterfactual_Rt.index):
    vaccination_starting_week = 1
    if date >= start_of_vaccination:
        vaccination_starting_week = t if T == 0 else vaccination_starting_week
        T=1
        week_since_vaccination = t - vaccination_starting_week
        week = date.isocalendar().week
        s_t = S[t] / N[t]
        counterfactual_Rt.iloc[t] = weekly_Rt_mean.loc[week]
        # counterfactual_Rt.iloc[t] = Rt_2018.loc[week % 52]


# plt.figure(figsize=(12, 6))
# weekly_Rt.plot()
# counterfactual_Rt.plot()
#
# plt.axhline(1.0, color='red', linestyle='--', linewidth=1)  # R=1 as reference
# plt.title("Weekly estimated Rₜ")
# plt.ylabel("Rₜ value")
# plt.xlabel("Date")
# plt.grid(True)
#
# plt.show()

T = len(weekly_cases_filled)
i_cf = np.zeros(T)

# indulás: az első max_tau hétben a megfigyelt adatot használjuk
i_cf[:max_tau] = weekly_cases_filled.iloc[:max_tau].values

for t in range(max_tau, T):
    past = i_cf[t - np.arange(1, max_tau+1)]
    denom = np.sum(past * g)
    s_t = S[t] / N[t]
    i_cf[t] = s_t * counterfactual_Rt.iloc[t] * denom

weekly_i_cf = pd.Series(i_cf, index=weekly_cases_filled.index)

mask = (weekly_cases_filled.index >= start_of_vaccination) & (weekly_cases_filled.index < "2026-01-01")
plt.plot(weekly_i_cf.index[mask], weekly_i_cf.values[mask],
         label="Estimated number of cases", color="royalblue", linewidth=1)

plt.scatter(
    weekly_cases_filled.index[mask],
    weekly_cases_filled.values[mask],
    color="black",
    s=4,
    label="Observed data"
)
plt.show()

weekly_prevented = weekly_i_cf - weekly_cases_filled
cumulative_prevented = weekly_prevented.cumsum()

# csak oltás utáni időszakra
total_prevented = weekly_prevented[mask].sum()

# plt.figure(figsize=(12,5))
# cumulative_prevented[mask].plot()
# plt.title("Kumulatív megelőzött esetek (counterfactual – megfigyelt)")
# plt.ylabel("Esetszám")
# plt.grid(True)
# plt.show()

# ------------------------------------------
# 4 b) esimate R_t using growth-rate profile

# I = weekly_cases_filled.copy()
#
# # kis simítás, hogy ne legyen log(0)
# I = I.replace(0, 1e-6)
#
# r = np.log(I) - np.log(I.shift(1))
# r = r.dropna()
#
# df_r = r.to_frame(name="r")
# df_r["week"] = df_r.index.isocalendar().week
#
# weekly_r_mean = df_r.groupby("week")["r"].mean()
#
# weekly_r_median = df_r.groupby("week")["r"].median()
#
# r_cf = r.copy()
#
# mask = r_cf.index >= start_of_vaccination
# weeks = r_cf.index[mask].isocalendar().week
#
# r_cf.loc[mask] = weekly_r_mean.reindex(weeks).values
#
# I_cf = I.copy()
#
# for t in range(1, len(I_cf)):
#     date = I_cf.index[t]
#     I_cf.iloc[t] = I_cf.iloc[t-1] * np.exp(r_cf.loc[date])
#
# M = np.sum(g * np.exp(-r_cf.values[:, None] * np.arange(1, len(g)+1)), axis=1)
# Rt_cf = 1 / M
# Rt_cf = pd.Series(Rt_cf, index=r_cf.index)
#
# weekly_I_cf = pd.Series(I_cf, index=weekly_cases_filled.index)
#
# plt.plot(weekly_I_cf.index, weekly_I_cf.values,
#          label="Estimated number of cases", color="royalblue", linewidth=1)
#
# plt.show()
