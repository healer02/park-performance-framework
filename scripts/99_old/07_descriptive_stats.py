# =============================================================================
# 07-descriptive-stats.py
# Purpose: Generate Table 1 descriptive statistics for 4.1 Results
#
# Inputs:
#   - data/processed/vancouver_da_supply.gpkg
#   - data/processed/vancouver_da_experience.csv
#   - data/google-reviews/processed/08c-park-metrics.csv
#
# Outputs:
#   - outputs/tables/table1-descriptive-stats.csv
# =============================================================================

# %%
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import numpy as np
import geopandas as gpd

SUPPLY_PATH  = "data/processed/vancouver_da_supply.gpkg"
EXP_PATH     = "data/processed/vancouver_da_experience.csv"
PARKS_PATH   = "data/google-reviews/processed/08c-park-metrics.csv"
USABILITY_PATH = "data/processed/vancouver_da_usability.csv"
KAPPA_PATH   = "outputs/tables/vancouver_amenity_kappa.csv"

OUT_DIR = "data/processed"
TAB_DIR = "outputs/tables"

os.makedirs(TAB_DIR, exist_ok=True)

print("Ready.")


# %% 1. SUPPLY DIMENSION
da_supply = gpd.read_file(SUPPLY_PATH)
park_metrics = pd.read_csv(PARKS_PATH)

n_das = len(da_supply)
REACH_THRESH = 0.8
qty_med      = da_supply["qty_cap20"].median()

# Coverage stats
cov_mean  = da_supply["DA_reach_400"].mean() * 100
cov_full  = (da_supply["DA_reach_400"] == 1).sum() / n_das * 100
cov_none  = (da_supply["DA_reach_400"] == 0).sum() / n_das * 100

# Area stats
area_mean = da_supply["qty_cap20"].mean()
area_med  = da_supply["qty_cap20"].median()

# Supply typology
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
print(f"\nText review ratio:")
print(park_metrics["text_review_ratio"].describe().round(3))


# %% 6. USABILITY DIMENSION
da_usability = pd.read_csv(f"{OUT_DIR}/vancouver_da_usability.csv")
kappa_df     = pd.read_csv(f"{TAB_DIR}/vancouver_amenity_kappa.csv")

usability_mean   = da_usability["amenity_type_count"].mean()
usability_median = da_usability["amenity_type_count"].median()
usability_std    = da_usability["amenity_type_count"].std()
mean_kappa       = kappa_df["Cohen's kappa"].mean()

print(f"\n--- USABILITY ---")
print(f"Amenity type count mean (SD): {usability_mean:.1f} ({usability_std:.1f})")
print(f"Amenity type count median:    {usability_median:.1f}")
print(f"Mean Cohen's kappa:           {mean_kappa:.2f}")
# %%


# %% 4. BUILD TABLE 1
rows = [
    # Supply
    ("Supply", "(1) Park Coverage", ""),
    ("", "Mean park coverage (%)", f"{cov_mean:.1f}"),
    ("", "Full coverage (100%) (% of DAs)", f"{cov_full:.1f}"),
    ("", "No coverage (0%) (% of DAs)", f"{cov_none:.1f}"),
    ("", "(2) Accessible park area", ""),
    ("", "Mean park area (ha per 1,000)", f"{area_mean:.1f}"),
    ("", "Median park area (ha per 1,000)", f"{area_med:.1f}"),
    ("", "Supply typology (high coverage ≥80%, high area ≥ median)", ""),
    ("", "HH: broad coverage, high area (% DAs)", f"{typology_pct.get('HH', 0):.1f}"),
    ("", "HL: broad coverage, low area (% DAs)", f"{typology_pct.get('HL', 0):.1f}"),
    ("", "LH: limited coverage, high area (% DAs)", f"{typology_pct.get('LH', 0):.1f}"),
    ("", "LL: limited coverage, low area (% DAs)", f"{typology_pct.get('LL', 0):.1f}"),
    # Experience
    ("Experience", "(1) Digital salience (Google reviews per 1,000 residents)", ""),
    ("", "Salience: Mean (SD)", f"{sal_mean:.0f} ({sal_std:.0f})"),
    ("", "Salience: Median", f"{sal_median:.0f}"),
    ("", "(2) Expressed satisfaction", ""),
    ("", "DAs with sentiment data: n (%)", f"{int(da_exp['satisfaction_sentiment'].notna().sum())} ({da_exp['satisfaction_sentiment'].notna().mean()*100:.1f}%)"),
    ("", "Mean qualifying parks per DA", f"{n_qualifying:.1f}"),
    ("", "Sentiment score: Mean (SD)", f"{sent_mean:.2f} ({sent_std:.2f})"),
    ("", "Sentiment score: Median", f"{sent_median:.2f}"),
    ("", "Star rating: Mean (SD)", f"{star_mean:.2f} ({da_exp['satisfaction_star'].std():.2f})"),
    ("", "Star rating: Median", f"{star_median:.2f}"),
    # Usability
    ("Usability", "(3) Perceived usability", ""),
    ("", "Amenity type count: Mean (SD)", f"{usability_mean:.1f} ({usability_std:.1f})"),
    ("", "Amenity type count: Median",    f"{usability_median:.1f}"),
]

table1 = pd.DataFrame(rows, columns=["Dimension", "Indicator", "Value"])
print(table1.to_string(index=False))
table1.to_csv("outputs/tables/table1-descriptive-stats.csv", index=False)
print("\nSaved: outputs/tables/table1-descriptive-stats.csv")


# %% 5. SALIENCE EQUITY CHECK (foreshadows 4.3)
# Join supply + experience + census to check if low-income DAs have lower salience
CENSUS_PATH = "data/census/raw/census_CANUE_DA_nearVan.csv"
da_census   = pd.read_csv(CENSUS_PATH, dtype={"DAUID": str})
da_supply["DAUID"] = da_supply["DAUID"].astype(str)
da_exp["DAUID"]    = da_exp["DAUID"].astype(str)

da_joined = da_supply.merge(da_exp, on="DAUID", how="left")
da_joined = da_joined.merge(
    da_census[["DAUID", "medhhinc", "inc_LIM_AT", "inc_totalpop"]], on="DAUID", how="left"
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
