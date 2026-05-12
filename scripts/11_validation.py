# =============================================================================
# 11-validation.py
# Purpose: Validate sentiment vs star rating agreement at park and DA level
#
# Inputs:
#   - data/google-reviews/processed/08c-park-metrics.csv
#   - data/processed/vancouver_da_experience.csv
#   - data/processed/vancouver_da_divergence.gpkg
#
# Outputs:
#   - outputs/figures/vancouver_validation_scatter.png
#   - outputs/tables/vancouver_validation_summary.csv
# =============================================================================

# %%
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PARK_METRICS_PATH = "data/google-reviews/processed/08c-park-metrics.csv"
EXP_PATH          = "data/processed/vancouver_da_experience.csv"
DIV_PATH          = "data/processed/vancouver_da_divergence.gpkg"

FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"

print("Ready.")


# %% 1. PARK-LEVEL VALIDATION
park_metrics = pd.read_csv(PARK_METRICS_PATH)

# Keep only parks with both sentiment and rating
park_val = park_metrics[
    park_metrics["MeanSentiment"].notna() &
    park_metrics["AvgRating"].notna()
].copy()

print(f"Parks with both sentiment and rating: {len(park_val)}")

r_p, p_p   = pearsonr(park_val["MeanSentiment"], park_val["AvgRating"])
r_sp, p_sp = spearmanr(park_val["MeanSentiment"], park_val["AvgRating"])

print(f"\n--- Park-level: Sentiment vs Rating ---")
print(f"Pearson r:  {r_p:.3f}, p={p_p:.4f}")
print(f"Spearman r: {r_sp:.3f}, p={p_sp:.4f}")


# %% 2. DA-LEVEL VALIDATION
da_exp = pd.read_csv(EXP_PATH)

da_val = da_exp[
    da_exp["satisfaction_sentiment"].notna() &
    da_exp["satisfaction_star"].notna()
].copy()

print(f"\nDAs with both sentiment and rating: {len(da_val)}")

r_da_p, p_da_p   = pearsonr(da_val["satisfaction_sentiment"], da_val["satisfaction_star"])
r_da_sp, p_da_sp = spearmanr(da_val["satisfaction_sentiment"], da_val["satisfaction_star"])

print(f"\n--- DA-level: Sentiment vs Rating ---")
print(f"Pearson r:  {r_da_p:.3f}, p={p_da_p:.4f}")
print(f"Spearman r: {r_da_sp:.3f}, p={p_da_sp:.4f}")

# %%
print([c for c in da_div_val.columns if 'sat' in c or 'sent' in c])

# %% 3. QUADRANT AGREEMENT
# Use da_val directly -- no need to merge with da_div
sentiment_med = da_val["satisfaction_sentiment"].median()
rating_med    = da_val["satisfaction_star"].median()

da_val["exp_hi_sentiment"] = (da_val["satisfaction_sentiment"] >= sentiment_med).astype(int)
da_val["exp_hi_rating"]    = (da_val["satisfaction_star"]      >= rating_med).astype(int)

agree = (da_val["exp_hi_sentiment"] == da_val["exp_hi_rating"]).mean() * 100
disagree = 100 - agree

print(f"\n--- DA-level quadrant agreement ---")
print(f"Sentiment median: {sentiment_med:.3f}")
print(f"Rating median:    {rating_med:.3f}")
print(f"DAs agreeing on high/low classification: {agree:.1f}%")
print(f"DAs disagreeing: {disagree:.1f}%")

n_sent_hi_rat_lo = ((da_val["exp_hi_sentiment"]==1) & (da_val["exp_hi_rating"]==0)).sum()
n_sent_lo_rat_hi = ((da_val["exp_hi_sentiment"]==0) & (da_val["exp_hi_rating"]==1)).sum()
print(f"  Sentiment high, rating low: {n_sent_hi_rat_lo}")
print(f"  Sentiment low, rating high: {n_sent_lo_rat_hi}")


# %% 4. SCATTERPLOTS
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Park level
ax = axes[0]
ax.scatter(park_val["MeanSentiment"], park_val["AvgRating"],
           alpha=0.5, s=30, color="#01665e")
m, b = np.polyfit(park_val["MeanSentiment"], park_val["AvgRating"], 1)
x_line = np.linspace(park_val["MeanSentiment"].min(),
                     park_val["MeanSentiment"].max(), 100)
ax.plot(x_line, m*x_line + b, color="#8c510a", linewidth=1.5)
ax.set_xlabel("Mean Sentiment Score (RoBERTa)", fontsize=10)
ax.set_ylabel("Mean Star Rating", fontsize=10)
ax.set_title(f"Park level (n={len(park_val)})\n"
             f"Pearson r={r_p:.3f}, Spearman r={r_sp:.3f}", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# DA level
ax = axes[1]
ax.scatter(da_val["satisfaction_sentiment"], da_val["satisfaction_star"],
           alpha=0.4, s=20, color="#01665e")
m2, b2 = np.polyfit(da_val["satisfaction_sentiment"], da_val["satisfaction_star"], 1)
x_line2 = np.linspace(da_val["satisfaction_sentiment"].min(),
                      da_val["satisfaction_sentiment"].max(), 100)
ax.plot(x_line2, m2*x_line2 + b2, color="#8c510a", linewidth=1.5)
ax.set_xlabel("Mean Sentiment Score (RoBERTa)", fontsize=10)
ax.set_ylabel("Mean Star Rating", fontsize=10)
ax.set_title(f"DA level (n={len(da_val)})\n"
             f"Pearson r={r_da_p:.3f}, Spearman r={r_da_sp:.3f}", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.suptitle("Convergent Validity: Sentiment Score vs Star Rating — Vancouver",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_validation_scatter.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {FIG_DIR}/vancouver_validation_scatter.png")


# %% 5. SAVE SUMMARY
summary = pd.DataFrame([
    {"Level": "Park", "Metric": "Pearson r",  "Value": round(r_p, 3),    "p": round(p_p, 4)},
    {"Level": "Park", "Metric": "Spearman r", "Value": round(r_sp, 3),   "p": round(p_sp, 4)},
    {"Level": "DA",   "Metric": "Pearson r",  "Value": round(r_da_p, 3), "p": round(p_da_p, 4)},
    {"Level": "DA",   "Metric": "Spearman r", "Value": round(r_da_sp, 3),"p": round(p_da_sp, 4)},
    {"Level": "DA",   "Metric": "Quadrant agreement (%)", "Value": round(agree, 1), "p": ""},
])
summary.to_csv(f"{TAB_DIR}/vancouver_validation_summary.csv", index=False)
print(f"Saved: {TAB_DIR}/vancouver_validation_summary.csv")
print(summary.to_string(index=False))


# %% 6. COHEN'S KAPPA FOR HIGH/LOW CLASSIFICATION AGREEMENT
from sklearn.metrics import cohen_kappa_score

# Use existing binary classifications from Cell 3
kappa_quad = cohen_kappa_score(
    da_val["exp_hi_sentiment"],
    da_val["exp_hi_rating"]
)

print(f"\n--- DA-level classification agreement ---")
print(f"Cohen's kappa: {kappa_quad:.3f}")

# Update summary table
summary = pd.DataFrame([
    {"Level": "Park", "Metric": "Pearson r",  "Value": round(r_p, 3),    "p": round(p_p, 4)},
    {"Level": "Park", "Metric": "Spearman r", "Value": round(r_sp, 3),   "p": round(p_sp, 4)},
    {"Level": "DA",   "Metric": "Pearson r",  "Value": round(r_da_p, 3), "p": round(p_da_p, 4)},
    {"Level": "DA",   "Metric": "Spearman r", "Value": round(r_da_sp, 3),"p": round(p_da_sp, 4)},
    {"Level": "DA",   "Metric": "Quadrant agreement (%)",
     "Value": round(agree, 1), "p": ""},
    {"Level": "DA",   "Metric": "Cohen's kappa",
     "Value": round(kappa_quad, 3), "p": ""},
])

summary.to_csv(f"{TAB_DIR}/vancouver_validation_summary.csv", index=False)

print(f"\nSaved: {TAB_DIR}/vancouver_validation_summary.csv")
print(summary.to_string(index=False))
# %%