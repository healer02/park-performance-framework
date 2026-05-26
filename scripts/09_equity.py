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



# %% EXTRA: EQUITY CROSSTABS (binary supply/experience)
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

# %% MULTINOMIAL LOGISTIC REGRESSION (HH as reference)
from statsmodels.regression.linear_model import OLS
import statsmodels.formula.api as smf

# Encode outcome with HH as reference
da_model["quadrant"] = pd.Categorical(
    da_model["divergence_2x2"],
    categories=["HH", "LH", "HL", "LL"]
)

X = sm.add_constant(da_model[pred_cols])
y = da_model["quadrant"]

mnlogit = sm.MNLogit(y, X)
result   = mnlogit.fit(method="newton", maxiter=200, disp=False)
print(result.summary())


# %% CLEAN OUTPUT TABLE: ODDS RATIOS + 95% CI
print("\n--- Odds Ratios (reference = HH) ---")

outcomes = ["LH", "HL", "LL"]
all_rows = []

conf_int = result.conf_int()

print("\nConfidence interval columns:")
print(conf_int.columns)

for i, outcome in enumerate(outcomes):
    params = result.params.iloc[:, i]
    pvals  = result.pvalues.iloc[:, i]
    ci_block = result.conf_int().loc[outcome]

    for var in pred_cols:
        or_val  = np.exp(params[var])
        ci_low  = np.exp(ci_block.loc[var, "lower"])
        ci_high = np.exp(ci_block.loc[var, "upper"])
        p       = pvals[var]
        sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        all_rows.append({
            "Outcome (vs HH)": outcome,
            "Predictor":       var,
            "OR":              round(or_val, 2),
            "95% CI":          f"[{ci_low:.2f}, {ci_high:.2f}]",
            "p":               round(p, 3),
            "sig":             sig,
        })
        print(f"  {outcome} vs HH | {var:25s}: OR={or_val:.2f} "
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

# %%
