"""
10_descriptive_stats.py  (was 07_descriptive_stats.py)
Generate Table 1 descriptive statistics for Section 4.1 Results.

Must run LAST — depends on outputs from 06_experience.py and 08_usability.py.

Inputs:
    data/processed/{CITY}_da_supply.gpkg
    data/processed/{CITY}_da_experience.csv
    data/google-reviews/processed/08c-park-metrics.csv
    data/processed/{CITY}_da_usability.csv      (from 08_usability.py)
    outputs/tables/{CITY}_amenity_kappa.csv     (from 08_usability.py, if available)
    config.CANUE_CSV                            (for salience equity check, if CANUE_AVAILABLE)

Outputs:
    outputs/tables/table1-descriptive-stats.csv
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd
import numpy as np
import geopandas as gpd

CITY = config.CITY

SUPPLY_PATH   = f"data/processed/{CITY}_da_supply.gpkg"
EXP_PATH      = f"data/processed/{CITY}_da_experience.csv"
PARKS_PATH    = "data/google-reviews/processed/08c-park-metrics.csv"
USABILITY_PATH= f"data/processed/{CITY}_da_usability.csv"
KAPPA_PATH    = f"outputs/tables/{CITY}_amenity_kappa.csv"

TAB_DIR = "outputs/tables"
os.makedirs(TAB_DIR, exist_ok=True)

print("Ready.")


# %% 1. SUPPLY DIMENSION
da_supply    = gpd.read_file(SUPPLY_PATH)
park_metrics = pd.read_csv(PARKS_PATH)

n_das        = len(da_supply)
REACH_THRESH = config.SUPPLY_REACH_THRESH
qty_med      = da_supply["qty_cap20"].median()

cov_mean = da_supply["DA_reach_400"].mean() * 100
cov_full = (da_supply["DA_reach_400"] == 1).sum() / n_das * 100
cov_none = (da_supply["DA_reach_400"] == 0).sum() / n_das * 100
area_mean = da_supply["qty_cap20"].mean()
area_med  = da_supply["qty_cap20"].median()

da_supply["reach_hi"] = (da_supply["DA_reach_400"] >= REACH_THRESH).astype(int)
da_supply["qty_hi"]   = (da_supply["qty_cap20"]    >= qty_med).astype(int)

def classify_supply(r, q):
    if pd.isna(r) or pd.isna(q): return "No data"
    if r==1 and q==1: return "HH"
    if r==1 and q==0: return "HL"
    if r==0 and q==1: return "LH"
    return "LL"

da_supply["supply_type"] = [
    classify_supply(r, q)
    for r, q in zip(da_supply["reach_hi"], da_supply["qty_hi"])
]
typology_pct = da_supply["supply_type"].value_counts(normalize=True) * 100

print(f"\n--- SUPPLY ---")
print(f"DAs:                      {n_das}")
print(f"Coverage mean:            {cov_mean:.1f}%")
print(f"Coverage full (100%):     {cov_full:.1f}%")
print(f"Coverage none (0%):       {cov_none:.1f}%")
print(f"Park area mean:           {area_mean:.1f} ha/1,000")
print(f"Park area median:         {area_med:.1f} ha/1,000")
print(f"\nSupply typology (%):")
print(typology_pct.round(1).to_string())


# %% 2. EXPERIENCE DIMENSION
da_exp = pd.read_csv(EXP_PATH)

sal_mean   = da_exp["salience"].mean()
sal_median = da_exp["salience"].median()
sal_std    = da_exp["salience"].std()

sent_mean   = da_exp["satisfaction_sentiment"].mean()
sent_median = da_exp["satisfaction_sentiment"].median()
sent_std    = da_exp["satisfaction_sentiment"].std()
sent_nas    = da_exp["satisfaction_sentiment"].isna().sum()

star_mean   = da_exp["satisfaction_star"].mean()
star_median = da_exp["satisfaction_star"].median()
n_qualifying = da_exp["n_qualifying_parks"].mean()

print(f"\n--- EXPERIENCE ---")
print(f"Salience mean (reviews/1,000 residents): {sal_mean:.1f}")
print(f"Salience median:                         {sal_median:.1f}")
print(f"Salience std:                            {sal_std:.1f}")
print(f"\nSentiment mean (RoBERTa):                {sent_mean:.3f}")
print(f"Sentiment median:                        {sent_median:.3f}")
print(f"Sentiment std:                           {sent_std:.3f}")
print(f"DAs with no sentiment data:              {sent_nas}")
print(f"\nStar rating mean:                        {star_mean:.2f}")
print(f"Star rating median:                      {star_median:.2f}")
print(f"Mean qualifying parks per DA:            {n_qualifying:.1f}")


# %% 3. PARK-LEVEL EXPERIENCE SUMMARY
print(f"\n--- PARK-LEVEL EXPERIENCE ---")
print(f"Parks with valid sentiment (>=10 text):  {park_metrics['has_valid_sentiment'].sum()}")
print(f"Parks without valid sentiment:           {(~park_metrics['has_valid_sentiment']).sum()}")
print(f"\nMeanSentiment (park level):")
print(park_metrics["MeanSentiment"].describe().round(3))
print(f"\nAvgRating (park level):")
print(park_metrics["AvgRating"].describe().round(3))
print(f"\nTotalReviews (park level):")
print(park_metrics["TotalReviews"].describe().round(0))


# %% 4. USABILITY DIMENSION
da_usability   = pd.read_csv(USABILITY_PATH)
usability_mean   = da_usability["amenity_type_count"].mean()
usability_median = da_usability["amenity_type_count"].median()
usability_std    = da_usability["amenity_type_count"].std()

if os.path.exists(KAPPA_PATH):
    kappa_df   = pd.read_csv(KAPPA_PATH)
    mean_kappa = kappa_df["Cohen's kappa"].mean()
else:
    mean_kappa = np.nan
    print(f"Note: {KAPPA_PATH} not found — kappa not included.")

print(f"\n--- USABILITY ---")
print(f"Amenity type count mean (SD): {usability_mean:.1f} ({usability_std:.1f})")
print(f"Amenity type count median:    {usability_median:.1f}")
if not np.isnan(mean_kappa):
    print(f"Mean Cohen's kappa:           {mean_kappa:.2f}")


# %% 5. BUILD TABLE 1
rows = [
    ("Supply", "(1) Park Coverage", ""),
    ("", "Mean park coverage (%)", f"{cov_mean:.1f}"),
    ("", "Full coverage (100%) (% of DAs)", f"{cov_full:.1f}"),
    ("", "No coverage (0%) (% of DAs)", f"{cov_none:.1f}"),
    ("", "(2) Accessible park area", ""),
    ("", "Mean park area (ha per 1,000)", f"{area_mean:.1f}"),
    ("", "Median park area (ha per 1,000)", f"{area_med:.1f}"),
    ("", f"Supply typology (coverage ≥{REACH_THRESH*100:.0f}%, area ≥ median)", ""),
    ("", "HH: broad coverage, high area (% DAs)", f"{typology_pct.get('HH', 0):.1f}"),
    ("", "HL: broad coverage, low area (% DAs)",  f"{typology_pct.get('HL', 0):.1f}"),
    ("", "LH: limited coverage, high area (% DAs)", f"{typology_pct.get('LH', 0):.1f}"),
    ("", "LL: limited coverage, low area (% DAs)",  f"{typology_pct.get('LL', 0):.1f}"),
    ("Experience", "(1) Digital salience (Google reviews per 1,000 residents)", ""),
    ("", "Salience: Mean (SD)", f"{sal_mean:.0f} ({sal_std:.0f})"),
    ("", "Salience: Median", f"{sal_median:.0f}"),
    ("", "(2) Expressed satisfaction", ""),
    ("", "DAs with sentiment data: n (%)",
     f"{int(da_exp['satisfaction_sentiment'].notna().sum())} "
     f"({da_exp['satisfaction_sentiment'].notna().mean()*100:.1f}%)"),
    ("", "Mean qualifying parks per DA", f"{n_qualifying:.1f}"),
    ("", "Sentiment score: Mean (SD)", f"{sent_mean:.2f} ({sent_std:.2f})"),
    ("", "Sentiment score: Median", f"{sent_median:.2f}"),
    ("", "Star rating: Mean (SD)", f"{star_mean:.2f} ({da_exp['satisfaction_star'].std():.2f})"),
    ("", "Star rating: Median", f"{star_median:.2f}"),
    ("Usability", "(3) Perceived usability", ""),
    ("", "Amenity type count: Mean (SD)", f"{usability_mean:.1f} ({usability_std:.1f})"),
    ("", "Amenity type count: Median", f"{usability_median:.1f}"),
]
if not np.isnan(mean_kappa):
    rows.append(("", "Mean Cohen's kappa (amenity validation)", f"{mean_kappa:.2f}"))

table1 = pd.DataFrame(rows, columns=["Dimension", "Indicator", "Value"])
print(f"\n{table1.to_string(index=False)}")
table1.to_csv("outputs/tables/table1-descriptive-stats.csv", index=False)
print("\nSaved: outputs/tables/table1-descriptive-stats.csv")


# %% 6. SALIENCE EQUITY CHECK (requires CANUE data)
if config.CANUE_AVAILABLE:
    da_census = pd.read_csv(config.CANUE_CSV, dtype={"DAUID": str})
    da_supply["DAUID"] = da_supply["DAUID"].astype(str)
    da_exp["DAUID"]    = da_exp["DAUID"].astype(str)

    da_joined = da_supply.merge(da_exp, on="DAUID", how="left")
    da_joined = da_joined.merge(
        da_census[["DAUID", "medhhinc", "inc_LIM_AT", "inc_totalpop"]],
        on="DAUID", how="left"
    )
    da_joined["pct_LIM_AT"] = (
        da_joined["inc_LIM_AT"] / da_joined["inc_totalpop"] * 100
    ).round(1)

    city_med_inc = da_joined["medhhinc"].median()
    da_joined["inc_group"] = pd.cut(
        da_joined["medhhinc"],
        bins=[0, city_med_inc * 0.6, city_med_inc * 1.4, float("inf")],
        labels=["Low income", "Middle income", "High income"]
    )

    sal_by_inc = da_joined.groupby("inc_group")["salience"].agg(["mean", "median"]).round(1)
    print(f"\n--- SALIENCE BY INCOME GROUP ---")
    print(sal_by_inc.to_string())
    print("\n(Lower salience in low-income areas = reduced civic voice in park discourse)")
else:
    print("\nNote: CANUE not available — skipping salience equity check.")

print("\nDone.")
