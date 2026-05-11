# =============================================================================
# 09-equity.py
# Purpose: Equity analysis of supply-experience divergence
#
# Inputs:
#   - data/processed/vancouver_da_divergence.gpkg
#   - data/census/raw/census_CANUE_DA_nearVan.csv
#
# Outputs:
#   - data/processed/vancouver_da_equity.csv
#   - outputs/figures/vancouver_equity_socioeconomic.png
#   - outputs/figures/vancouver_equity_demographic_builtenv.png
# =============================================================================

# %% 1. IMPORTS AND PATHS
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIV_PATH    = "data/processed/vancouver_da_divergence.gpkg"
CENSUS_PATH = "data/census/raw/census_CANUE_DA_nearVan.csv"
OUT_DIR     = "data/processed"
FIG_DIR     = "outputs/figures"

colours_2x2 = {
    "HH": "#2c7bb6",
    "HL": "#d7191c",
    "LH": "#abd9e9",
    "LL": "#fdae61",
}
legend_labels = {
    "HH": "High supply / high experience",
    "HL": "High supply / low experience",
    "LH": "Low supply / high experience",
    "LL": "Low supply / low experience",
}

print("Ready.")


# %% 2. LOAD, JOIN, AND CLASSIFY DIVERGENCE
da_div    = gpd.read_file(DIV_PATH)
da_census = pd.read_csv(CENSUS_PATH)

# Recompute supply typology and 2x2 divergence (mirrors 08-experience.py)
REACH_THRESH  = 0.8
qty_med       = da_div["qty_cap20"].median()
sentiment_med = da_div["satisfaction_sentiment"].median()

da_div["supply_type"] = da_div.apply(
    lambda r: (
        "HH" if (r["DA_reach_400"] >= REACH_THRESH and r["qty_cap20"] >= qty_med)
        else "HL" if (r["DA_reach_400"] >= REACH_THRESH and r["qty_cap20"] < qty_med)
        else "LH" if (r["DA_reach_400"] < REACH_THRESH and r["qty_cap20"] >= qty_med)
        else "LL" if pd.notna(r["DA_reach_400"]) else "No data"
    ), axis=1
)

da_div["supply_binary"] = (da_div["supply_type"] == "HH").astype(int)
da_div["experience_hi"] = (da_div["satisfaction_sentiment"] >= sentiment_med).astype(int)

def classify_2x2(s, e):
    if pd.isna(s) or pd.isna(e): return "No data"
    if s == 1 and e == 1: return "HH"
    if s == 1 and e == 0: return "HL"
    if s == 0 and e == 1: return "LH"
    return "LL"

da_div["divergence_2x2"] = [
    classify_2x2(s, e)
    for s, e in zip(da_div["supply_binary"], da_div["experience_hi"])
]

print(f"Supply typology:\n{da_div['supply_type'].value_counts()}")
print(f"\nDivergence 2x2:\n{da_div['divergence_2x2'].value_counts()}")
print(f"\nThresholds — reachability: {REACH_THRESH}, "
      f"quantity median: {qty_med:.1f} ha/1,000, "
      f"sentiment median: {sentiment_med:.3f}")


# %% 3. JOIN CENSUS AND DERIVE EQUITY VARIABLES
da_census["DAUID"] = da_census["DAUID"].astype(str)
da_div["DAUID"]    = da_div["DAUID"].astype(str)

# Derived proportions
da_census["pct_visible_minority"] = (
    da_census["visible_minority"] / da_census["visible_minority_totalpop"] * 100
).round(1)
da_census["pct_age_65plus"] = (
    da_census["age_65plus"] / da_census["pop_total"] * 100
).round(1)
da_census["pct_LIM_AT"] = (
    da_census["inc_LIM_AT"] / da_census["inc_totalpop"] * 100
).round(1)
da_census["pct_immigrant"] = (
    da_census["immigrant_immigrant"] / da_census["immigrant_totalpop"] * 100
).round(1)
da_census["pct_bachelor_plus"] = (
    da_census["education_bachelor_plus"] / da_census["education_totalpop"] * 100
).round(1)

equity_cols = [
    "DAUID", "medhhinc", "pop_total",
    "pct_visible_minority", "pct_age_65plus",
    "pct_LIM_AT", "pct_immigrant", "pct_bachelor_plus",
    "inc_LIM_AT", "inc_totalpop",
    "immigrant_immigrant", "immigrant_totalpop",
    "education_bachelor_plus", "education_totalpop",
    "ale16_08",
]
da_eq = da_div.merge(da_census[equity_cols], on="DAUID", how="left")

print(f"\nDAs with equity data: {da_eq['medhhinc'].notna().sum()} / {len(da_eq)}")
print(da_eq[["medhhinc", "pct_visible_minority", "pct_age_65plus",
             "pct_LIM_AT", "pct_immigrant", "pct_bachelor_plus"]].describe().round(1))


# %% 4. DEFINE ALL EQUITY STRATA
city_median_inc = da_eq["medhhinc"].median()
print(f"\nCity median household income: ${city_median_inc:,.0f}")

# Income (median household income)
da_eq["inc_stratum"] = pd.cut(
    da_eq["medhhinc"],
    bins=[0, city_median_inc * 0.6, city_median_inc * 1.4, float("inf")],
    labels=["Low income", "Middle income", "High income"]
)

# Visible minority
da_eq["vm_stratum"] = pd.cut(
    da_eq["pct_visible_minority"],
    bins=[0, 20, 50, 100],
    labels=["Low VM (<20%)", "Mid VM (20-50%)", "High VM (>50%)"]
)

# Age composition
da_eq["age_stratum"] = pd.cut(
    da_eq["pct_age_65plus"],
    bins=[0, 10, 20, 100],
    labels=["Young (<10% 65+)", "Mid age (10-20% 65+)", "Older (>20% 65+)"]
)

# Low income measure (LIM-AT)
da_eq["limat_stratum"] = pd.cut(
    da_eq["pct_LIM_AT"],
    bins=[0, 20, 35, 100],
    labels=["Low poverty (<20%)", "Mid poverty (20-35%)", "High poverty (>35%)"]
)

# Immigrant share
da_eq["immigrant_stratum"] = pd.cut(
    da_eq["pct_immigrant"],
    bins=[0, 30, 50, 100],
    labels=["Low immigrant (<30%)", "Mid immigrant (30-50%)", "High immigrant (>50%)"]
)

# Education
da_eq["edu_stratum"] = pd.cut(
    da_eq["pct_bachelor_plus"],
    bins=[0, 30, 50, 100],
    labels=["Low education (<30%)", "Mid education (30-50%)", "High education (>50%)"]
)

# Active Living Environment (tertile split)
da_eq["ale_stratum"] = pd.qcut(
    da_eq["ale16_08"],
    q=3,
    labels=["Low ALE", "Mid ALE", "High ALE"]
)

# Print stratum counts
for col, label in [
    ("inc_stratum",       "Income"),
    ("vm_stratum",        "Visible Minority"),
    ("age_stratum",       "Age (65+)"),
    ("limat_stratum",     "LIM-AT"),
    ("immigrant_stratum", "Immigrant Share"),
    ("edu_stratum",       "Education"),
    ("ale_stratum",       "ALE"),
]:
    print(f"\n{label} stratum counts:")
    print(da_eq[col].value_counts().to_string())

# Save equity file
da_eq.drop(columns="geometry").to_csv(f"{OUT_DIR}/vancouver_da_equity.csv", index=False)
print(f"\nSaved: {OUT_DIR}/vancouver_da_equity.csv")


# %% 5. EQUITY CROSSTABS (all strata)
for stratum_col, label in [
    ("inc_stratum",       "Income"),
    ("vm_stratum",        "Visible Minority"),
    ("age_stratum",       "Age (65+)"),
    ("limat_stratum",     "Low Income (LIM-AT)"),
    ("immigrant_stratum", "Immigrant Share"),
    ("edu_stratum",       "Education (Bachelor+)"),
    ("ale_stratum",       "Active Living Environment"),
]:
    print(f"\n{'='*50}")
    print(f"Divergence by {label}:")
    ct = pd.crosstab(
        da_eq[stratum_col],
        da_eq["divergence_2x2"],
        normalize="index"
    ).round(3) * 100
    print(ct.to_string())


# %% 6. FIGURE A: SOCIOECONOMIC INDICATORS
# Income, LIM-AT, Education
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

strata_a = [
    ("inc_stratum",   "Income",
     ["Low income", "Middle income", "High income"]),
    ("limat_stratum", "Low Income (LIM-AT %)",
     ["Low poverty (<20%)", "Mid poverty (20-35%)", "High poverty (>35%)"]),
    ("edu_stratum",   "Education (Bachelor+ %)",
     ["Low education (<30%)", "Mid education (30-50%)", "High education (>50%)"]),
]

for ax, (col, title, order) in zip(axes, strata_a):
    ct = pd.crosstab(
        da_eq[col],
        da_eq["divergence_2x2"],
        normalize="index"
    ) * 100
    ct = ct.reindex(index=order, columns=["HH", "HL", "LH", "LL"])

    bottom = np.zeros(len(ct))
    for quad in ["HH", "HL", "LH", "LL"]:
        if quad in ct.columns:
            vals = ct[quad].fillna(0).values
            ax.bar(range(len(ct)), vals, bottom=bottom,
                   color=colours_2x2[quad], label=quad, width=0.6)
            bottom += vals

    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of DAs")
    ax.set_title(title)
    ax.set_ylim(0, 100)

patches = [mpatches.Patch(color=colours_2x2[q], label=legend_labels[q])
           for q in ["HH", "HL", "LH", "LL"]]
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
plt.suptitle(
    "Supply–Experience Divergence by Socioeconomic Indicators — Vancouver",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_equity_socioeconomic.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vancouver_equity_socioeconomic.png")


# %% 7. FIGURE B: DEMOGRAPHIC + BUILT ENVIRONMENT
# Visible minority, Age, Immigrant share, ALE
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

strata_b = [
    ("vm_stratum",        "Visible Minority (%)",
     ["Low VM (<20%)", "Mid VM (20-50%)", "High VM (>50%)"]),
    ("age_stratum",       "Age Composition (65+)",
     ["Young (<10% 65+)", "Mid age (10-20% 65+)", "Older (>20% 65+)"]),
    ("immigrant_stratum", "Immigrant Share (%)",
     ["Low immigrant (<30%)", "Mid immigrant (30-50%)", "High immigrant (>50%)"]),
    ("ale_stratum",       "Active Living Environment",
     ["Low ALE", "Mid ALE", "High ALE"]),
]

for ax, (col, title, order) in zip(axes, strata_b):
    ct = pd.crosstab(
        da_eq[col],
        da_eq["divergence_2x2"],
        normalize="index"
    ) * 100
    ct = ct.reindex(index=order, columns=["HH", "HL", "LH", "LL"])

    bottom = np.zeros(len(ct))
    for quad in ["HH", "HL", "LH", "LL"]:
        if quad in ct.columns:
            vals = ct[quad].fillna(0).values
            ax.bar(range(len(ct)), vals, bottom=bottom,
                   color=colours_2x2[quad], label=quad, width=0.6)
            bottom += vals

    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of DAs")
    ax.set_title(title)
    ax.set_ylim(0, 100)

patches = [mpatches.Patch(color=colours_2x2[q], label=legend_labels[q])
           for q in ["HH", "HL", "LH", "LL"]]
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
plt.suptitle(
    "Supply–Experience Divergence by Demographic and Built Environment Indicators — Vancouver",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_equity_demographic_builtenv.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vancouver_equity_demographic_builtenv.png")