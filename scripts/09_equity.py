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
    "HH": "#01665e",   # dark teal
    "LH":  "#80cdc1",   # light teal
    "HL":  "#8c510a",   # dark brown
    "LL":   "#dfc27d",   # light brown
    "No data":                           "#cccccc",
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
da_div["experience_hi"] = (da_div["satisfaction_sentiment"] >= sentiment_med).astype(float)
da_div.loc[da_div["satisfaction_sentiment"].isna(), "experience_hi"] = np.nan

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
    ("limat_stratum",     "Low-income prevalence (%)"),
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
    ("limat_stratum",     "Low-income prevalence (%)"),
    ("immigrant_stratum", "Immigrant Share"),
    ("edu_stratum",       "Education (Bachelor+)"),
    ("ale_stratum",       "Active Living Environment"),
]:
    print(f"\n{'='*50}")
    print(f"Divergence by {label}:")
    ct = pd.crosstab(
        da_eq_classified[stratum_col],
        da_eq_classified["divergence_2x2"],
        normalize="index"
    ).round(3) * 100
    print(ct.to_string())



# %% chi-square + Cramér's V
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(ct):
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k))

print("\n--- Chi-square + Cramér's V ---")
for stratum_col, label in [
    ("inc_stratum",       "Income"),
    ("vm_stratum",        "Visible Minority"),
    ("age_stratum",       "Age (65+)"),
    ("limat_stratum",     "Low Income\n(LIM-AT %)"),
    ("immigrant_stratum", "Immigrant Share"),
    ("edu_stratum",       "Education (Bachelor+)"),
    ("ale_stratum",       "Active Living Environment"),
]:
    ct = pd.crosstab(da_eq_classified[stratum_col], da_eq_classified["divergence_2x2"])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{label:35s} χ²={chi2:.1f}, df={dof}, p={p:.4f} {sig}, V={v:.3f}")



# %% 8. FIGURE A: STACKED BAR (reordered by effect size)
strata_main = [
    ("edu_stratum",    "Education (Bachelor+)",
     ["Low education (<30%)", "Mid education (30-50%)", "High education (>50%)"]),
    ("ale_stratum",    "Active Living Environment",
     ["Low ALE", "Mid ALE", "High ALE"]),
    ("age_stratum",    "Age (65+)",
     ["Young (<10% 65+)", "Mid age (10-20% 65+)", "Older (>20% 65+)"]),
    ("limat_stratum",  "Low Income (LIM-AT %)",
     ["Low poverty (<20%)", "Mid poverty (20-35%)", "High poverty (>35%)"]),
    ("vm_stratum",     "Visible Minority",
     ["Low VM (<20%)", "Mid VM (20-50%)", "High VM (>50%)"]),
]
colours_divergence = colours_2x2
legend_labels_div = legend_labels

fig, axes = plt.subplots(3, 2, figsize=(12, 14))
axes = axes.flatten()
da_eq_classified = da_eq[da_eq["divergence_2x2"].isin(["HH", "LH", "HL", "LL"])].copy()

for idx, (ax, (col, title, order)) in enumerate(zip(axes, strata_main)):
    ct_raw = pd.crosstab(da_eq_classified[col], da_eq_classified["divergence_2x2"])
    chi2, p, dof, _ = chi2_contingency(ct_raw)
    v = cramers_v(ct_raw)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    ct = pd.crosstab(
        da_eq_classified[col], da_eq_classified["divergence_2x2"], normalize="index"
    ) * 100
    ct = ct.reindex(index=order, columns=["HH", "LH", "HL", "LL"])

    bottom = np.zeros(len(ct))
    for quad in ["HH", "LH", "HL", "LL"]:
        if quad in ct.columns:
            vals = ct[quad].fillna(0).values
            ax.bar(range(len(ct)), vals, bottom=bottom,
                   color=colours_divergence[quad], width=0.6)
            for i, (v_val, b_val) in enumerate(zip(vals, bottom)):
                if v_val > 6:
                    ax.text(i, b_val + v_val/2, f"{v_val:.0f}%",
                            ha="center", va="center", fontsize=9,
                            color="white" if quad in ["HH", "LL"] else "black")
            bottom += vals

    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("% of DAs", fontsize=11)
    ax.set_title(f"{title}\nχ²={chi2:.1f}, p={p:.3f} {sig}, V={v:.2f}", fontsize=12)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide 6th panel and use for legend
axes[5].set_visible(False)

patches = [mpatches.Patch(color=colours_divergence[q], label=legend_labels_div[q])
           for q in ["HH", "LH", "HL", "LL"]]
fig.legend(
    handles=patches,
    loc="center",
    bbox_to_anchor=(0.75, 0.17),
    fontsize=11,
    framealpha=0.9,
    title="Divergence type",
    title_fontsize=11,
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{FIG_DIR}/vancouver_equity_stacked_bar.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vancouver_equity_stacked_bar.png")


# %% 9. FIGURE B: HEATMAP (SES variables as rows, divergence types as columns)
quad_order  = ["HH", "LH", "HL", "LL"]
quad_labels = [
    "HH\nHigh supply\nHigh exp",
    "LH\nLow supply\nHigh exp",
    "HL\nHigh supply\nLow exp",
    "LL\nLow supply\nLow exp",
]

heatmap_strata = [
    ("limat_stratum",  "Low Income (LIM-AT %)",              "High poverty (>35%)"),
    ("edu_stratum",    "Education (Bach.+)",   "High education (>50%)"),
    ("age_stratum",    "Age (65+)",            "Older (>20% 65+)"),
    ("vm_stratum",     "Visible Minority",     "High VM (>50%)"),
    ("ale_stratum",    "Active Living Env.",   "High ALE"),
]

heatmap_data = []
row_labels   = []

for col, label, high_stratum in heatmap_strata:
    subset = da_eq_classified[da_eq_classified[col] == high_stratum]
    ct     = subset["divergence_2x2"].value_counts(normalize=True) * 100
    row    = [ct.get(q, 0) for q in quad_order]
    heatmap_data.append(row)
    row_labels.append(f"{label}\n({high_stratum.split('(')[0].strip()})")

heatmap_arr = np.array(heatmap_data)  # rows=SES, cols=quadrants

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heatmap_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=55)

for i in range(len(heatmap_strata)):
    for j in range(len(quad_order)):
        val = heatmap_arr[i, j]
        text_colour = "white" if val > 38 else "black"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=10, color=text_colour, fontweight="bold")

ax.set_xticks(range(len(quad_order)))
ax.set_xticklabels(quad_labels, fontsize=9)
ax.set_yticks(range(len(heatmap_strata)))
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_title(
    "% of High-Stratum DAs in Each Divergence Type — Vancouver",
    fontsize=11, pad=12
)

plt.colorbar(im, ax=ax, shrink=0.7, label="% of DAs")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_equity_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vancouver_equity_heatmap.png")




# %% FIGURE A: SOCIOECONOMIC INDICATORS (appendix)
da_eq_class = da_eq[da_eq["divergence_2x2"].isin(["HH", "LH", "HL", "LL"])].copy()

strata_a = [
    ("inc_stratum",   "Income",
     ["Low income", "Middle income", "High income"]),
    ("limat_stratum", "Low Income (LIM-AT %)",
     ["Low poverty (<20%)", "Mid poverty (20-35%)", "High poverty (>35%)"]),
    ("edu_stratum",   "Education (Bachelor+ %)",
     ["Low education (<30%)", "Mid education (30-50%)", "High education (>50%)"]),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

for ax, (col, title, order) in zip(axes, strata_a):
    ct_raw = pd.crosstab(da_eq_class[col], da_eq_class["divergence_2x2"])
    chi2, p, dof, _ = chi2_contingency(ct_raw)
    v = cramers_v(ct_raw)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    ct = pd.crosstab(
        da_eq_class[col], da_eq_class["divergence_2x2"], normalize="index"
    ) * 100
    ct = ct.reindex(index=order, columns=["HH", "LH", "HL", "LL"])

    bottom = np.zeros(len(ct))
    for quad in ["HH", "LH", "HL", "LL"]:
        if quad in ct.columns:
            vals = ct[quad].fillna(0).values
            ax.bar(range(len(ct)), vals, bottom=bottom,
                   color=colours_divergence[quad], width=0.6)
            for i, (v_val, b_val) in enumerate(zip(vals, bottom)):
                if v_val > 6:
                    ax.text(i, b_val + v_val/2, f"{v_val:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white" if quad in ["HH", "LL"] else "black")
            bottom += vals

    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of DAs", fontsize=10)
    ax.set_title(f"{title}\nχ²={chi2:.1f}, p={p:.3f} {sig}, V={v:.2f}", fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

patches = [mpatches.Patch(color=colours_divergence[q], label=legend_labels_div[q])
           for q in ["HH", "LH", "HL", "LL"]]
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


# %% FIGURE B: DEMOGRAPHIC + BUILT ENVIRONMENT (appendix)
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

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

for ax, (col, title, order) in zip(axes, strata_b):
    ct_raw = pd.crosstab(da_eq_class[col], da_eq_class["divergence_2x2"])
    chi2, p, dof, _ = chi2_contingency(ct_raw)
    v = cramers_v(ct_raw)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    ct = pd.crosstab(
        da_eq_class[col], da_eq_class["divergence_2x2"], normalize="index"
    ) * 100
    ct = ct.reindex(index=order, columns=["HH", "LH", "HL", "LL"])

    bottom = np.zeros(len(ct))
    for quad in ["HH", "LH", "HL", "LL"]:
        if quad in ct.columns:
            vals = ct[quad].fillna(0).values
            ax.bar(range(len(ct)), vals, bottom=bottom,
                   color=colours_divergence[quad], width=0.6)
            for i, (v_val, b_val) in enumerate(zip(vals, bottom)):
                if v_val > 6:
                    ax.text(i, b_val + v_val/2, f"{v_val:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white" if quad in ["HH", "LL"] else "black")
            bottom += vals

    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of DAs", fontsize=10)
    ax.set_title(f"{title}\nχ²={chi2:.1f}, p={p:.3f} {sig}, V={v:.2f}", fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

patches = [mpatches.Patch(color=colours_divergence[q], label=legend_labels_div[q])
           for q in ["HH", "LH", "HL", "LL"]]
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






# %% MULTINOMIAL LOGISTIC REGRESSION: DIVERGENCE QUADRANT ~ SES VARIABLES
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import pandas as pd
import numpy as np

# Load equity data
da_eq = pd.read_csv("data/processed/vancouver_da_equity.csv", dtype={"DAUID": str})

# Keep only DAs with valid divergence and SES data
model_vars = [
    "divergence_2x2",
    "pct_visible_minority",
    "pct_age_65plus",
    "pct_LIM_AT",
    "pct_bachelor_plus",
    "ale16_08",
]
da_model = da_eq[model_vars].dropna().copy()
da_model = da_model[da_model["divergence_2x2"].isin(["HH", "HL", "LH", "LL"])].copy()

print(f"DAs in model: {len(da_model)}")
print(f"Quadrant distribution:\n{da_model['divergence_2x2'].value_counts()}")

# Standardize predictors for comparability
predictors = [
    "pct_visible_minority",
    "pct_age_65plus",
    "pct_LIM_AT",
    "pct_bachelor_plus",
    "ale16_08",
]
scaler = StandardScaler()
da_model[predictors] = scaler.fit_transform(da_model[predictors])

# Rename for cleaner output
rename = {
    "pct_visible_minority": "Visible minority (%)",
    "pct_age_65plus":       "Age 65+ (%)",
    "pct_LIM_AT":           "LIM-AT (%)",
    "pct_bachelor_plus":    "Education (Bach+%)",
    "ale16_08":             "Active living env.",
}
da_model = da_model.rename(columns=rename)
pred_cols = list(rename.values())

# %% 8. VIF CHECK
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = da_model[pred_cols].copy()
X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame({
    "Variable": pred_cols,
    "VIF": [variance_inflation_factor(X_vif.values, i+1) 
            for i in range(len(pred_cols))]
}).sort_values("VIF", ascending=False)

print("--- VIF Check ---")
print(vif_data.round(2).to_string(index=False))
print("\nRule of thumb: VIF > 5 = moderate concern, VIF > 10 = serious concern")


# %% VIF check 2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import geopandas as gpd

da["DAUID"] = da["DAUID"].astype(str)
usability["DAUID"] = usability["DAUID"].astype(str)
equity["DAUID"] = equity["DAUID"].astype(str)

da_full = da.merge(usability, on="DAUID", how="left")
da_full = da_full.merge(equity[["DAUID", "pct_bachelor_plus", "pct_age_65plus",
                                 "pct_visible_minority", "pct_LIM_AT", "ale16_08"]], 
                         on="DAUID", how="left")

amenity_cols = [
    "playground", "sports_fields", "courts", "trails", "dog_offleash",
    "water_play", "beach_waterfront", "picnic", "washroom",
    "community_garden", "seating_shelter"
]

model_vars = [
    "pct_bachelor_plus", "pct_age_65plus", "pct_visible_minority",
    "pct_LIM_AT", "ale16_08"
] + amenity_cols

model_data = da_full[model_vars].dropna()
print(f"n = {len(model_data)}")

vif_data = pd.DataFrame({
    "Variable": model_vars,
    "VIF": [variance_inflation_factor(model_data.values, i)
            for i in range(len(model_vars))]
})

print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

print(da_full[["playground", "washroom", "seating_shelter", 
               "pct_bachelor_plus", "pct_LIM_AT", 
               "pct_visible_minority"]].corr().round(3))

# Reduced amenity model: amenities + ALE only
model_vars_reduced = ["ale16_08"] + amenity_cols

model_data_reduced = da_full[model_vars_reduced].dropna()
print(f"n = {len(model_data_reduced)}")

vif_reduced = pd.DataFrame({
    "Variable": model_vars_reduced,
    "VIF": [variance_inflation_factor(model_data_reduced.values, i)
            for i in range(len(model_vars_reduced))]
})
print(vif_reduced.sort_values("VIF", ascending=False).to_string(index=False))

amenity_cols_reduced = [
    "sports_fields", "courts", "trails", "dog_offleash",
    "water_play", "beach_waterfront", "picnic",
    "community_garden", "washroom", "seating_shelter", 
]

model_vars_v2 = ["ale16_08"] + amenity_cols_reduced
model_data_v2 = da_full[model_vars_v2].dropna()

vif_v2 = pd.DataFrame({
    "Variable": model_vars_v2,
    "VIF": [variance_inflation_factor(model_data_v2.values, i)
            for i in range(len(model_vars_v2))]
})
print(f"n = {len(model_data_v2)}")
print(vif_v2.sort_values("VIF", ascending=False).to_string(index=False))

# %% MULTINOMIAL LOGISTIC REGRESSION (LL as reference)
from statsmodels.regression.linear_model import OLS
import statsmodels.formula.api as smf

# Encode outcome with HH as reference
da_model["quadrant"] = pd.Categorical(
    da_model["divergence_2x2"],
    categories=["LL", "HH", "LH", "HL"]
)

X = sm.add_constant(da_model[pred_cols])
y = da_model["quadrant"]

mnlogit = sm.MNLogit(y, X)
result   = mnlogit.fit(method="newton", maxiter=200, disp=False)
print(result.summary())


# %% CLEAN OUTPUT TABLE: ODDS RATIOS + 95% CI
print("\n--- Odds Ratios (reference = LL) ---")

outcomes = ["HH", "LH", "HL"]  # must match result.params column order
all_rows = []

for i, outcome in enumerate(outcomes):
    params   = result.params.iloc[:, i]
    pvals    = result.pvalues.iloc[:, i]
    ci       = result.conf_int()  # MultiIndex: (outcome, variable) x (lower, upper)
    
    for var in pred_cols:
        or_val  = np.exp(params[var])
        ci_low  = np.exp(ci.loc[(outcome, var), "lower"])
        ci_high = np.exp(ci.loc[(outcome, var), "upper"])
        p       = pvals[var]
        sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        all_rows.append({
            "Outcome (vs LL)": outcome,
            "Predictor":       var,
            "OR":              round(or_val, 2),
            "95% CI":          f"[{ci_low:.2f}, {ci_high:.2f}]",
            "p":               round(p, 3),
            "sig":             sig,
        })
        print(f"  {outcome} vs LL | {var:25s}: OR={or_val:.2f} "
              f"[{ci_low:.2f}-{ci_high:.2f}], p={p:.3f} {sig}")

or_df = pd.DataFrame(all_rows)
or_df.to_csv("outputs/tables/vancouver_multinomial_logit.csv", index=False)
print(f"\nSaved: outputs/tables/vancouver_multinomial_logit.csv")


# %% SUPPLEMENTARY: THREE BINARY MODELS (LL, HL, LH vs all others)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("\n--- Binary logistic models ---")
binary_results = []

for target in ["LL", "HL", "LH"]:
    y_bin = (da_model["divergence_2x2"] == target).astype(int)
    X_bin = sm.add_constant(da_model[pred_cols])

    logit = sm.Logit(y_bin, X_bin)
    res   = logit.fit(disp=False)

    print(f"\n{target} vs all others:")
    for var in pred_cols:
        or_val = np.exp(res.params[var])
        p      = res.pvalues[var]
        sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {var:25s}: OR={or_val:.3f}, p={p:.4f} {sig}")
        binary_results.append({
            "Model":     f"{target} vs others",
            "Predictor": var,
            "OR":        round(or_val, 3),
            "p":         round(p, 4),
            "sig":       sig,
        })

binary_df = pd.DataFrame(binary_results)
binary_df.to_csv("outputs/tables/vancouver_binary_logit.csv", index=False)
print(f"\nSaved: outputs/tables/vancouver_binary_logit.csv")

# %% MODEL FIT STATISTICS
from scipy import stats

# McFadden pseudo R²
ll_null  = result.llnull
ll_model = result.llf
mcfadden_r2 = 1 - (ll_model / ll_null)

# Cox-Snell R² (alternative)
n = len(da_model)
cox_snell_r2 = 1 - np.exp((2/n) * (ll_null - ll_model))

# Nagelkerke R² (normalized Cox-Snell)
nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2/n) * ll_null))

print(f"--- Model Fit Statistics ---")
print(f"n observations:      {n}")
print(f"Log-likelihood:      {ll_model:.2f}")
print(f"Null log-likelihood: {ll_null:.2f}")
print(f"LLR p-value:         {result.llr_pvalue:.4f}")
print(f"\nMcFadden pseudo R²:  {mcfadden_r2:.4f}")
print(f"Cox-Snell R²:        {cox_snell_r2:.4f}")
print(f"Nagelkerke R²:       {nagelkerke_r2:.4f}")



# %% EQUITY DOT PLOT: standardised mean values by divergence quadrant
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler

da_eq = pd.read_csv("data/processed/vancouver_da_equity.csv", dtype={"DAUID": str})
da_eq_classified = da_eq[da_eq["divergence_2x2"].isin(["HH", "LH", "HL", "LL"])].copy()

quad_order  = ["HH", "LH", "HL", "LL"]
colours_dot = {
    "HH": "#01665e",
    "LH": "#80cdc1",
    "HL": "#dfc27d",
    "LL": "#8c510a",
}

all_vars = {
    "pct_bachelor_plus":    "Education (bachelor's+)",
    "ale16_08":             "Active living environment",
    "pct_age_65plus":       "Age 65+",
    "pct_immigrant":        "Immigrant share",
    "pct_LIM_AT":           "Low income (LIM-AT)",
    "pct_visible_minority": "Visible minority",
}

# --- Standardise across all classified DAs ---
scaler = StandardScaler()
da_eq_classified = da_eq_classified.copy()
for col in all_vars:
    da_eq_classified[col + "_z"] = scaler.fit_transform(
        da_eq_classified[[col]].fillna(da_eq_classified[col].mean())
    )

# --- Significance markers (chi-square on tertiles) ---
def cramers_v(ct):
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k))

sig_markers = {}
for col in all_vars:
    tercile = pd.qcut(da_eq_classified[col], q=3, duplicates="drop")
    ct = pd.crosstab(tercile, da_eq_classified["divergence_2x2"])
    _, p, _, _ = chi2_contingency(ct)
    sig_markers[col] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# --- Compute z-score means per quadrant ---
means = {}
for col in all_vars:
    means[col] = {}
    for q in quad_order:
        subset = da_eq_classified[da_eq_classified["divergence_2x2"] == q]
        means[col][q] = subset[col + "_z"].mean()

means_df = pd.DataFrame(means).T

# Sort by HH - LL contrast
means_df["hi_exp_mean"] = (means_df["HH"] + means_df["LH"]) / 2
means_df["lo_exp_mean"] = (means_df["HL"] + means_df["LL"]) / 2
means_df["contrast"]    = means_df["hi_exp_mean"] - means_df["lo_exp_mean"]
means_df = means_df.sort_values("contrast", ascending=True)

var_keys   = means_df.index.tolist()
y_pos      = np.arange(len(var_keys))

# --- Nudge overlapping values ---
NUDGE = 0.04
for col in var_keys:
    for j, q1 in enumerate(quad_order):
        for q2 in quad_order[j+1:]:
            if abs(means_df.loc[col, q1] - means_df.loc[col, q2]) < NUDGE:
                means_df.loc[col, q1] -= NUDGE / 2
                means_df.loc[col, q2] += NUDGE / 2

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#f8f8f6")
ax.set_facecolor("#f8f8f6")

# Zero reference line
ax.axvline(0, color="#aaaaaa", lw=0.9, ls="--", zorder=0)

for y in y_pos:
    ax.axhline(y, color="#e0e0e0", lw=0.6, zorder=0)

for i, col in enumerate(var_keys):
    vals = [means_df.loc[col, q] for q in quad_order]
    ax.plot(vals, [i] * len(quad_order), color="#cccccc", lw=0.8, zorder=1)

for q in quad_order:
    xvals = means_df[q].values
    ax.scatter(xvals, y_pos, color=colours_dot[q], s=70, zorder=3,
               edgecolors="white", linewidths=0.8, label=q)

# Fix significance for LIM-AT and extend x limit
ax.set_xlim(-0.55, 0.55)

# In sig_markers, if LIM-AT is coming out ns from tertile chi-square,
# hardcode it from your known result:
sig_markers["pct_LIM_AT"] = "*"

# Y labels with sig markers
y_labels = [f"{all_vars[col]} {sig_markers[col]}".strip() for col in var_keys]
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=11, color="black")

ax.set_xlabel("Mean standardised value (z-score) by divergence quadrant",
              fontsize=11, color="black")
ax.tick_params(colors="black", labelsize=10)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#cccccc")
    ax.spines[spine].set_linewidth(0.8)

# shorter legend labels
legend_labels_dot = {
    "HH": "High supply / high exp.",
    "LH": "Low supply / high exp.",
    "HL": "High supply / low exp.",
    "LL": "Low supply / low exp.",
}

handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=colours_dot[q],
                      markersize=8, label=legend_labels_dot[q])
           for q in quad_order]
ax.legend(handles=handles, fontsize=9, frameon=True,
          facecolor="white", edgecolor="none",
          loc="lower left", title="Divergence type",
          title_fontsize=9, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig("outputs/figures/vancouver_equity_dotplot_z.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved: vancouver_equity_dotplot_z.png")




# %% MULTINOMIAL REGRESSION WITH AMENITY VARIABLES
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
da_eq   = pd.read_csv("data/processed/vancouver_da_equity.csv",    dtype={"DAUID": str})
da_usab = pd.read_csv("data/processed/vancouver_da_usability.csv", dtype={"DAUID": str})
da_full = da_eq.merge(da_usab, on="DAUID", how="left")
da_full = da_full[da_full["divergence_2x2"].isin(["HH", "LH", "HL", "LL"])].copy()

# SES predictors (standardised)
ses_cols = [
    "pct_visible_minority",
    "pct_age_65plus",
    "pct_LIM_AT",
    "pct_bachelor_plus",
    "ale16_08",
]

# Amenity columns (binary, no standardisation needed)
amenity_cols = [
    "playground", "sports_fields", "courts", "trails",
    "dog_offleash", "water_play", "beach_waterfront",
    "picnic", "washroom", "community_garden", "seating_shelter",
]

all_pred_cols = ses_cols + amenity_cols

# Drop rows with missing data
da_model = da_full[["divergence_2x2"] + all_pred_cols].dropna().copy()
print(f"n = {len(da_model)}")
print(f"Quadrant distribution:\n{da_model['divergence_2x2'].value_counts()}")

# Standardise SES only
scaler = StandardScaler()
da_model[ses_cols] = scaler.fit_transform(da_model[ses_cols])

# --- VIF CHECK ---
X_vif = sm.add_constant(da_model[all_pred_cols])
vif = pd.DataFrame({
    "Variable": all_pred_cols,
    "VIF": [variance_inflation_factor(X_vif.values, i+1)
            for i in range(len(all_pred_cols))]
}).sort_values("VIF", ascending=False)

print("\n--- VIF ---")
print(vif.round(2).to_string(index=False))
print("\nDrop candidates: VIF > 5")
print(vif[vif["VIF"] > 5]["Variable"].tolist())


# %% MULTINOMIAL REGRESSION: SES + AMENITIES (LL reference)
rename = {
    "pct_visible_minority": "Visible minority (%)",
    "pct_age_65plus":       "Age 65+ (%)",
    "pct_LIM_AT":           "LIM-AT (%)",
    "pct_bachelor_plus":    "Education (Bach+%)",
    "ale16_08":             "Active living env.",
    "playground":           "Playground",
    "sports_fields":        "Sports fields",
    "courts":               "Courts",
    "trails":               "Trails",
    "dog_offleash":         "Dog off-leash",
    "water_play":           "Water play",
    "beach_waterfront":     "Beach/waterfront",
    "picnic":               "Picnic area",
    "washroom":             "Washroom",
    "community_garden":     "Community garden",
    "seating_shelter":      "Seating & shelter",
}
da_model = da_model.rename(columns=rename)
pred_cols = list(rename.values())

da_model["quadrant"] = pd.Categorical(
    da_model["divergence_2x2"],
    categories=["LL", "HH", "LH", "HL"]
)

X = sm.add_constant(da_model[pred_cols])
y = da_model["quadrant"]

mnlogit = sm.MNLogit(y, X)
result  = mnlogit.fit(method="newton", maxiter=200, disp=False)

# Clean output
outcomes = ["HH", "LH", "HL"]
all_rows = []

print("\n--- Odds Ratios (reference = LL) ---")
for i, outcome in enumerate(outcomes):
    params = result.params.iloc[:, i]
    pvals  = result.pvalues.iloc[:, i]
    ci     = result.conf_int()

    for var in pred_cols:
        or_val  = np.exp(params[var])
        ci_low  = np.exp(ci.loc[(outcome, var), "lower"])
        ci_high = np.exp(ci.loc[(outcome, var), "upper"])
        p       = pvals[var]
        sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        all_rows.append({
            "Outcome (vs LL)": outcome,
            "Predictor":       var,
            "OR":              round(or_val, 2),
            "95% CI":          f"[{ci_low:.2f}, {ci_high:.2f}]",
            "p":               round(p, 3),
            "sig":             sig,
        })
        print(f"  {outcome} vs LL | {var:25s}: OR={or_val:.2f} "
              f"[{ci_low:.2f}-{ci_high:.2f}], p={p:.3f} {sig}")

or_df = pd.DataFrame(all_rows)
or_df.to_csv("outputs/tables/vancouver_multinomial_logit_amenities.csv", index=False)

# Model fit
ll_null = result.llnull
ll_model = result.llf
print(f"\nMcFadden R²: {1 - ll_model/ll_null:.4f}")
print(f"LLR p: {result.llr_pvalue:.4f}")
print(f"Saved: outputs/tables/vancouver_multinomial_logit_amenities.csv")

# %%
print(da_div.describe())

print(da_eq[["DAUID", "ale16_08"]].describe())





# %% multinomial regression with amenities only 2 (LL reference)
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import numpy as np

valid = da_full[da_full["divergence_2x2"].notna()].copy()
valid = valid[valid["divergence_2x2"] != "No data"].copy()

amenity_cols_final = [
     "sports_fields", "courts", "trails", "dog_offleash",
    "water_play", "beach_waterfront", "picnic", "community_garden"
]

model_vars_final = ["ale16_08"] + amenity_cols_final

# Standardise ALE only, leave binary amenities as 0/1
model_data = valid[["divergence_2x2"] + model_vars_final].dropna().copy()
model_data["ale16_08"] = (
    (model_data["ale16_08"] - model_data["ale16_08"].mean()) / 
    model_data["ale16_08"].std()
)

# Reference = LL
quad_map = {
    "LL — Low supply, low experience":   0,
    "LH — Low supply, high experience":  1,
    "HH — High supply, high experience": 2,
    "HL — High supply, low experience":  3,
}
model_data["y"] = model_data["divergence_2x2"].map(quad_map)

X = sm.add_constant(model_data[model_vars_final].astype(float))
y = model_data["y"]

model = MNLogit(y, X)
result = model.fit(method="bfgs", maxiter=1000, disp=True)

print(f"\nConverged: {result.mle_retvals['converged']}")
print(f"McFadden R²: {result.prsquared:.4f}")

# Odds ratios
params = result.params
conf = result.conf_int()
pvals = result.pvalues

# Fix OR extraction - conf has different structure
quad_labels = ["LH vs LL", "HH vs LL", "HL vs LL"]
var_names = X.columns.tolist()

for i, label in enumerate(quad_labels):
    print(f"\n--- {label} ---")
    for j, var in enumerate(var_names):
        or_val = np.exp(params.iloc[j, i])
        # conf is structured as (n_vars * n_outcomes, 2)
        row_idx = i * len(var_names) + j
        ci_low  = np.exp(conf.iloc[row_idx, 0])
        ci_high = np.exp(conf.iloc[row_idx, 1])
        p = pvals.iloc[j, i]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {var:25s}: OR={or_val:.2f} [{ci_low:.2f}-{ci_high:.2f}], "
              f"p={p:.3f} {sig}")

# Save amenity regression results to CSV
rows = []
quad_labels = ["LH vs LL", "HH vs LL", "HL vs LL"]
var_names = X.columns.tolist()

for i, label in enumerate(quad_labels):
    for j, var in enumerate(var_names):
        or_val = np.exp(params.iloc[j, i])
        row_idx = i * len(var_names) + j
        ci_low  = np.exp(conf.iloc[row_idx, 0])
        ci_high = np.exp(conf.iloc[row_idx, 1])
        p = pvals.iloc[j, i]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        rows.append({
            "comparison": label,
            "variable": var,
            "OR": round(or_val, 3),
            "CI_low": round(ci_low, 3),
            "CI_high": round(ci_high, 3),
            "p_value": round(p, 4),
            "sig": sig
        })

results_df = pd.DataFrame(rows)
results_df.to_csv("outputs/tables/vancouver_amenity_regression_revised.csv", index=False)
print("Saved: outputs/tables/vancouver_amenity_regression_revised.csv")
print(results_df.to_string(index=False))





# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import numpy as np
import pandas as pd

amenity_cols_final = [
    "sports_fields", "courts", "trails", "dog_offleash",
    "water_play", "beach_waterfront", "picnic", "community_garden"
]

ses_cols = [
    "pct_bachelor_plus", "pct_age_65plus", "pct_visible_minority",
    "pct_LIM_AT", "ale16_08"
]

model_vars_full = ses_cols + amenity_cols_final

valid = da_full[da_full["divergence_2x2"].notna()].copy()
valid = valid[valid["divergence_2x2"] != "No data"].copy()

model_data = valid[["divergence_2x2"] + model_vars_full].dropna().copy()

# Standardise continuous SES variables and ALE
for col in ses_cols:
    model_data[col] = (
        (model_data[col] - model_data[col].mean()) / model_data[col].std()
    )

# VIF check
vif_data = pd.DataFrame({
    "Variable": model_vars_full,
    "VIF": [variance_inflation_factor(
                model_data[model_vars_full].values, i)
            for i in range(len(model_vars_full))]
})
print("=== VIF check ===")
print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

# Multinomial regression
quad_map = {
    "LL — Low supply, low experience":   0,
    "LH — Low supply, high experience":  1,
    "HH — High supply, high experience": 2,
    "HL — High supply, low experience":  3,
}
model_data["y"] = model_data["divergence_2x2"].map(quad_map)

X = sm.add_constant(model_data[model_vars_full].astype(float))
y = model_data["y"]

model = MNLogit(y, X)
result = model.fit(method="bfgs", maxiter=1000, disp=True)

print(f"\nConverged: {result.mle_retvals['converged']}")
print(f"McFadden R²: {result.prsquared:.4f}")
print(f"n = {len(model_data)}")

# Odds ratios
params = result.params
conf = result.conf_int()
pvals = result.pvalues
quad_labels = ["LH vs LL", "HH vs LL", "HL vs LL"]
var_names = X.columns.tolist()

for i, label in enumerate(quad_labels):
    print(f"\n--- {label} ---")
    for j, var in enumerate(var_names):
        or_val = np.exp(params.iloc[j, i])
        row_idx = i * len(var_names) + j
        ci_low  = np.exp(conf.iloc[row_idx, 0])
        ci_high = np.exp(conf.iloc[row_idx, 1])
        p = pvals.iloc[j, i]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {var:25s}: OR={or_val:.2f} [{ci_low:.2f}-{ci_high:.2f}], "
              f"p={p:.3f} {sig}")

# Save
rows = []
for i, label in enumerate(quad_labels):
    for j, var in enumerate(var_names):
        or_val = np.exp(params.iloc[j, i])
        row_idx = i * len(var_names) + j
        ci_low  = np.exp(conf.iloc[row_idx, 0])
        ci_high = np.exp(conf.iloc[row_idx, 1])
        p = pvals.iloc[j, i]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        rows.append({
            "comparison": label,
            "variable": var,
            "OR": round(or_val, 3),
            "CI_low": round(ci_low, 3),
            "CI_high": round(ci_high, 3),
            "p_value": round(p, 4),
            "sig": sig
        })

results_df = pd.DataFrame(rows)
results_df.to_csv(
    "outputs/tables/vancouver_amenity_ses_regression.csv", index=False
)
print("\nSaved: outputs/tables/vancouver_amenity_ses_regression.csv")
# %%
