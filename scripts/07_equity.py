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

# %% Imports and setup
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
da_census = pd.read_csv(CENSUS_PATH)  # load census first
da_census["DAUID"] = da_census["DAUID"].astype(str)

# Merge Can-ALE 2021 into census
canale21 = pd.read_csv("data/census/raw/CanALE_2021.csv", dtype={"DAUID": str})
da_census = da_census.merge(canale21[["DAUID", "ALE_index"]], on="DAUID", how="left")

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
da_census["pct_age_0_14"] = (
    da_census["age_0_14"] / da_census["pop_total"] * 100
).round(1)

equity_cols = [
    "DAUID", "medhhinc", "pop_total",
    "pct_visible_minority", "pct_age_65plus", "pct_age_0_14", 
    "pct_LIM_AT", "pct_immigrant", "pct_bachelor_plus",
    "inc_LIM_AT", "inc_totalpop", 
    "immigrant_immigrant", "immigrant_totalpop",
    "education_bachelor_plus", "education_totalpop",
    "ALE_index",
]
da_eq = da_div.merge(da_census[equity_cols], on="DAUID", how="left")
print(f"\nDAs with equity data: {da_eq['medhhinc'].notna().sum()} / {len(da_eq)}")


# %% 3. KRUSKAL-WALLIS + ETA-SQUARED
from scipy.stats import kruskal

da_eq_classified = da_eq[da_eq["divergence_2x2"].isin(["HH", "LH", "HL", "LL"])].copy()

continuous_vars = [
    ("medhhinc",            "Income"),
    ("pct_visible_minority","Visible Minority"),
    ("pct_age_65plus",      "Age (65+)"),
    ("pct_age_0_14",        "Age 0-14 (children)"),
    ("pct_LIM_AT",          "Low Income (LIM-AT %)"),
    ("pct_immigrant",       "Immigrant Share"),
    ("pct_bachelor_plus",   "Education (Bachelor+)"),
    ("ALE_index",           "Active Living Environment"),
]

print("\n--- Kruskal-Wallis test ---")
kw_results = {}
for col, label in continuous_vars:
    groups = [
        da_eq_classified.loc[da_eq_classified["divergence_2x2"] == q, col].dropna().values
        for q in ["HH", "LH", "HL", "LL"]
    ]
    H, p = kruskal(*groups)
    # Eta-squared as effect size: H / (n - 1)
    n = sum(len(g) for g in groups)
    eta2 = (H - len(groups) + 1) / (n - len(groups))
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    kw_results[col] = {"H": H, "p": p, "eta2": eta2, "sig": sig}
    print(f"{label:35s} H={H:.1f}, p={p:.4f} {sig}, η²={eta2:.3f}")

# %% 7. MULTINOMIAL LOGISTIC REGRESSION
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

da_eq_full = pd.read_csv(f"{OUT_DIR}/{CITY}_da_equity.csv", dtype={"DAUID": str})

model_vars = [
    "divergence_2x2","pct_age_0_14",
    "pct_visible_minority", "pct_age_65plus", "pct_LIM_AT",
    "pct_bachelor_plus", "ALE_index",
]
da_model = da_eq_full[model_vars].dropna().copy()
da_model = da_model[da_model["divergence_2x2"].isin(["HH", "HL", "LH", "LL"])].copy()
print(f"\nDAs in model: {len(da_model)}")

predictors = ["pct_visible_minority", "pct_age_65plus", "pct_LIM_AT","pct_age_0_14",
              "pct_bachelor_plus", "ALE_index"]
scaler = StandardScaler()
da_model[predictors] = scaler.fit_transform(da_model[predictors])

rename = {
    "pct_age_0_14":         "Age 0-14 (%)",
    "pct_age_65plus":       "Age 65+ (%)",
    "pct_visible_minority": "Visible minority (%)",
    "pct_LIM_AT":           "LIM-AT (%)",
    "pct_bachelor_plus":    "Education (Bach+%)",
    "ALE_index":             "Active living env.",
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
    "ALE_index":             "Active living environment",
    "pct_age_65plus":       "Older adults (age 65+)",
    "pct_immigrant":        "Immigrant share",
    "pct_LIM_AT":           "Low income (LIM-AT)",
    "pct_visible_minority": "Visible minority",
    "pct_age_0_14": "Children (age 0–14)",
}

# --- Standardise across all classified DAs ---
scaler = StandardScaler()
da_eq_classified = da_eq_classified.copy()
for col in all_vars:
    da_eq_classified[col + "_z"] = scaler.fit_transform(
        da_eq_classified[[col]].fillna(da_eq_classified[col].mean())
    )

# --- Significance markers  ---
sig_markers = {}
for col in all_vars:
    if col in kw_results:
        sig_markers[col] = kw_results[col]["sig"]
    else:
        sig_markers[col] = ""

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




# %%
