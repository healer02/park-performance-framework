"""
07_equity.py  (was 09_equity.py)
Equity analysis of supply-experience divergence.

Requires CANUE sociodemographic data. Set config.CANUE_AVAILABLE = True once
the CANUE dataset has been downloaded for this city's CSDs (see config.py).

Inputs:
    data/processed/{CITY}_da_divergence.gpkg
    config.CANUE_CSV  (data/census/raw/census_CANUE_DA_nearVan.csv for Vancouver)

Outputs:
    data/processed/{CITY}_da_equity.csv
    outputs/figures/{CITY}_equity_socioeconomic.png
    outputs/figures/{CITY}_equity_demographic_builtenv.png
    outputs/tables/{CITY}_multinomial_logit.csv
    outputs/tables/{CITY}_binary_logit.csv
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

# ── Guard: CANUE data required ────────────────────────────────────────────────
if not config.CANUE_AVAILABLE:
    print(
        "Equity analysis requires CANUE sociodemographic data.\n"
        f"config.CANUE_AVAILABLE = False for city '{config.CITY}'.\n\n"
        "To enable:\n"
        "  1. Submit a CANUE data request covering the required CSDs\n"
        "     (see comments in config.py for CSD codes).\n"
        "  2. Place the downloaded CSV at config.CANUE_CSV.\n"
        "  3. Set config.CANUE_AVAILABLE = True and re-run this script."
    )
    sys.exit(0)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency
import statsmodels.api as sm

CITY = config.CITY

DIV_PATH    = f"data/processed/{CITY}_da_divergence.gpkg"
CENSUS_PATH = config.CANUE_CSV
OUT_DIR     = "data/processed"
FIG_DIR     = "outputs/figures"
TAB_DIR     = "outputs/tables"
os.makedirs(TAB_DIR, exist_ok=True)

# Experience column: depends on which source was used in 06_experience.py
EXP_COL = "satisfaction_sentiment" if config.EXPERIENCE_SOURCE == "sentiment" \
          else "satisfaction_star"

colours_2x2 = config.COLOURS_2X2
legend_labels = {
    "HH": "High supply / high experience",
    "HL": "High supply / low experience",
    "LH": "Low supply / high experience",
    "LL": "Low supply / low experience",
}

print("Ready.")


# %% 1. LOAD AND CLASSIFY DIVERGENCE
da_div    = gpd.read_file(DIV_PATH)
da_census = pd.read_csv(CENSUS_PATH)

REACH_THRESH  = config.SUPPLY_REACH_THRESH
qty_med       = da_div["qty_cap20"].median()
sentiment_med = da_div[EXP_COL].median()

da_div["supply_type"] = da_div.apply(
    lambda r: (
        "HH" if (r["DA_reach_400"] >= REACH_THRESH and r["qty_cap20"] >= qty_med)
        else "HL" if (r["DA_reach_400"] >= REACH_THRESH and r["qty_cap20"] < qty_med)
        else "LH" if (r["DA_reach_400"] < REACH_THRESH and r["qty_cap20"] >= qty_med)
        else "LL" if pd.notna(r["DA_reach_400"]) else "No data"
    ), axis=1
)

da_div["supply_binary"] = (da_div["supply_type"] == "HH").astype(int)
da_div["experience_hi"] = (da_div[EXP_COL] >= sentiment_med).astype(float)
da_div.loc[da_div[EXP_COL].isna(), "experience_hi"] = np.nan

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
      f"{EXP_COL} median: {sentiment_med:.3f}")


# %% 2. JOIN CENSUS AND DERIVE EQUITY VARIABLES
da_census["DAUID"] = da_census["DAUID"].astype(str)
da_div["DAUID"]    = da_div["DAUID"].astype(str)

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


# %% 3. DEFINE EQUITY STRATA
city_median_inc = da_eq["medhhinc"].median()

da_eq["inc_stratum"] = pd.cut(
    da_eq["medhhinc"],
    bins=[0, city_median_inc * 0.6, city_median_inc * 1.4, float("inf")],
    labels=["Low income", "Middle income", "High income"]
)
da_eq["vm_stratum"] = pd.cut(
    da_eq["pct_visible_minority"], bins=[0, 20, 50, 100],
    labels=["Low VM (<20%)", "Mid VM (20-50%)", "High VM (>50%)"]
)
da_eq["age_stratum"] = pd.cut(
    da_eq["pct_age_65plus"], bins=[0, 10, 20, 100],
    labels=["Young (<10% 65+)", "Mid age (10-20% 65+)", "Older (>20% 65+)"]
)
da_eq["limat_stratum"] = pd.cut(
    da_eq["pct_LIM_AT"], bins=[0, 20, 35, 100],
    labels=["Low poverty (<20%)", "Mid poverty (20-35%)", "High poverty (>35%)"]
)
da_eq["immigrant_stratum"] = pd.cut(
    da_eq["pct_immigrant"], bins=[0, 30, 50, 100],
    labels=["Low immigrant (<30%)", "Mid immigrant (30-50%)", "High immigrant (>50%)"]
)
da_eq["edu_stratum"] = pd.cut(
    da_eq["pct_bachelor_plus"], bins=[0, 30, 50, 100],
    labels=["Low education (<30%)", "Mid education (30-50%)", "High education (>50%)"]
)
da_eq["ale_stratum"] = pd.qcut(
    da_eq["ale16_08"], q=3, labels=["Low ALE", "Mid ALE", "High ALE"]
)

# Save equity file
da_eq.drop(columns="geometry").to_csv(f"{OUT_DIR}/{CITY}_da_equity.csv", index=False)
print(f"\nSaved: {OUT_DIR}/{CITY}_da_equity.csv")


# %% 4. CHI-SQUARE + CRAMÉR'S V

def cramers_v(ct):
    chi2 = chi2_contingency(ct)[0]
    n    = ct.sum().sum()
    k    = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k))

da_eq_classified = da_eq[da_eq["divergence_2x2"].isin(["HH", "LH", "HL", "LL"])].copy()

print("\n--- Chi-square + Cramér's V ---")
strata_all = [
    ("inc_stratum",       "Income"),
    ("vm_stratum",        "Visible Minority"),
    ("age_stratum",       "Age (65+)"),
    ("limat_stratum",     "Low Income (LIM-AT %)"),
    ("immigrant_stratum", "Immigrant Share"),
    ("edu_stratum",       "Education (Bachelor+)"),
    ("ale_stratum",       "Active Living Environment"),
]
for stratum_col, label in strata_all:
    ct    = pd.crosstab(da_eq_classified[stratum_col], da_eq_classified["divergence_2x2"])
    chi2, p, dof, _ = chi2_contingency(ct)
    v     = cramers_v(ct)
    sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{label:35s} χ²={chi2:.1f}, df={dof}, p={p:.4f} {sig}, V={v:.3f}")


# %% 5. FIGURE A: SOCIOECONOMIC STACKED BAR (appendix)
colours_divergence = colours_2x2
legend_labels_div  = legend_labels

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
            ax.bar(range(len(ct)), vals, bottom=bottom, color=colours_divergence[quad], width=0.6)
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
    f"Supply–Experience Divergence by Socioeconomic Indicators — {CITY.title()}",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_equity_socioeconomic.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {CITY}_equity_socioeconomic.png")


# %% 6. FIGURE B: DEMOGRAPHIC + BUILT ENVIRONMENT (appendix)
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
            ax.bar(range(len(ct)), vals, bottom=bottom, color=colours_divergence[quad], width=0.6)
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
    f"Supply–Experience Divergence by Demographic and Built Environment — {CITY.title()}",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_equity_demographic_builtenv.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {CITY}_equity_demographic_builtenv.png")


# %% 7. MULTINOMIAL LOGISTIC REGRESSION
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

da_eq_full = pd.read_csv(f"{OUT_DIR}/{CITY}_da_equity.csv", dtype={"DAUID": str})

model_vars = [
    "divergence_2x2",
    "pct_visible_minority", "pct_age_65plus", "pct_LIM_AT",
    "pct_bachelor_plus", "ale16_08",
]
da_model = da_eq_full[model_vars].dropna().copy()
da_model = da_model[da_model["divergence_2x2"].isin(["HH", "HL", "LH", "LL"])].copy()
print(f"\nDAs in model: {len(da_model)}")

predictors = ["pct_visible_minority", "pct_age_65plus", "pct_LIM_AT",
              "pct_bachelor_plus", "ale16_08"]
scaler = StandardScaler()
da_model[predictors] = scaler.fit_transform(da_model[predictors])

rename = {
    "pct_visible_minority": "Visible minority (%)",
    "pct_age_65plus":       "Age 65+ (%)",
    "pct_LIM_AT":           "LIM-AT (%)",
    "pct_bachelor_plus":    "Education (Bach+%)",
    "ale16_08":             "Active living env.",
}
da_model = da_model.rename(columns=rename)
pred_cols = list(rename.values())

# VIF check
X_vif = sm.add_constant(da_model[pred_cols])
vif_data = pd.DataFrame({
    "Variable": pred_cols,
    "VIF": [variance_inflation_factor(X_vif.values, i+1) for i in range(len(pred_cols))]
}).sort_values("VIF", ascending=False)
print("\n--- VIF Check ---")
print(vif_data.round(2).to_string(index=False))

# Multinomial logit (LL = reference)
da_model["quadrant"] = pd.Categorical(
    da_model["divergence_2x2"], categories=["LL", "HH", "LH", "HL"]
)
X = sm.add_constant(da_model[pred_cols])
y = da_model["quadrant"]
mnlogit_model = sm.MNLogit(y, X)
result        = mnlogit_model.fit(method="newton", maxiter=200, disp=False)

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
            "Outcome (vs LL)": outcome, "Predictor": var,
            "OR": round(or_val, 2), "95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "p": round(p, 3), "sig": sig,
        })
        print(f"  {outcome} vs LL | {var:25s}: OR={or_val:.2f} "
              f"[{ci_low:.2f}-{ci_high:.2f}], p={p:.3f} {sig}")

or_df = pd.DataFrame(all_rows)
or_df.to_csv(f"{TAB_DIR}/{CITY}_multinomial_logit.csv", index=False)
print(f"\nSaved: {TAB_DIR}/{CITY}_multinomial_logit.csv")

ll_null, ll_model_val = result.llnull, result.llf
print(f"McFadden pseudo R²: {1 - ll_model_val/ll_null:.4f}, LLR p: {result.llr_pvalue:.4f}")


# %% 8. BINARY LOGISTIC REGRESSION
binary_results = []
print("\n--- Binary logistic models ---")
for target in ["LL", "HL", "LH"]:
    y_bin = (da_model["divergence_2x2"] == target).astype(int)
    X_bin = sm.add_constant(da_model[pred_cols])
    res   = sm.Logit(y_bin, X_bin).fit(disp=False)
    print(f"\n{target} vs all others:")
    for var in pred_cols:
        or_val = np.exp(res.params[var])
        p      = res.pvalues[var]
        sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {var:25s}: OR={or_val:.3f}, p={p:.4f} {sig}")
        binary_results.append({
            "Model": f"{target} vs others", "Predictor": var,
            "OR": round(or_val, 3), "p": round(p, 4), "sig": sig,
        })

binary_df = pd.DataFrame(binary_results)
binary_df.to_csv(f"{TAB_DIR}/{CITY}_binary_logit.csv", index=False)
print(f"\nSaved: {TAB_DIR}/{CITY}_binary_logit.csv")

print("\nDone.")
