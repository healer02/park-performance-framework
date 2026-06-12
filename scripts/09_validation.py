"""
09_validation.py  (was 11_validation.py)
Validate RoBERTa sentiment against star ratings at park and DA level.

Requires sentiment — skips automatically for star-rating cities.

Inputs:
    data/google-reviews/processed/08c-park-metrics.csv
    data/google-reviews/processed/08a-text-reviews-with-sentiment.csv
    data/processed/{CITY}_da_experience.csv
    data/processed/{CITY}_da_divergence.gpkg

Outputs:
    outputs/figures/{CITY}_validation_scatter.png
    outputs/figures/{CITY}_validation_scatter2.png
    outputs/figures/{CITY}_validation_scatter3.png
    outputs/tables/{CITY}_validation_summary.csv
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

# ── Guard: sentiment required ─────────────────────────────────────────────────
if config.EXPERIENCE_SOURCE == "star_rating":
    print(
        "Sentiment validation requires RoBERTa sentiment scores.\n"
        "Skipping for star-rating cities "
        f"(config.EXPERIENCE_SOURCE = 'star_rating' for city '{config.CITY}')."
    )
    sys.exit(0)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from scipy import stats

CITY = config.CITY

PARK_METRICS_PATH = "data/google-reviews/processed/08c-park-metrics.csv"
REVIEWS_PATH      = "data/google-reviews/processed/08a-text-reviews-with-sentiment.csv"
EXP_PATH          = f"data/processed/{CITY}_da_experience.csv"
DIV_PATH          = f"data/processed/{CITY}_da_divergence.gpkg"

FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"

print("Ready.")


# %% 1. PARK-LEVEL VALIDATION
park_metrics = pd.read_csv(PARK_METRICS_PATH)
park_val     = park_metrics[
    park_metrics["MeanSentiment"].notna() &
    park_metrics["AvgRating"].notna()
].copy()

print(f"Parks with both sentiment and rating: {len(park_val)}")

r_p,  p_p  = pearsonr(park_val["MeanSentiment"], park_val["AvgRating"])
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

r_da_p,  p_da_p  = pearsonr(da_val["satisfaction_sentiment"], da_val["satisfaction_star"])
r_da_sp, p_da_sp = spearmanr(da_val["satisfaction_sentiment"], da_val["satisfaction_star"])

print(f"\n--- DA-level: Sentiment vs Rating ---")
print(f"Pearson r:  {r_da_p:.3f}, p={p_da_p:.4f}")
print(f"Spearman r: {r_da_sp:.3f}, p={p_da_sp:.4f}")


# %% 3. QUADRANT AGREEMENT
sentiment_med = da_val["satisfaction_sentiment"].median()
rating_med    = da_val["satisfaction_star"].median()

da_val["exp_hi_sentiment"] = (da_val["satisfaction_sentiment"] >= sentiment_med).astype(int)
da_val["exp_hi_rating"]    = (da_val["satisfaction_star"]      >= rating_med).astype(int)

agree    = (da_val["exp_hi_sentiment"] == da_val["exp_hi_rating"]).mean() * 100
disagree = 100 - agree

print(f"\n--- DA-level quadrant agreement ---")
print(f"Sentiment median: {sentiment_med:.3f}")
print(f"Rating median:    {rating_med:.3f}")
print(f"Agreement: {agree:.1f}%  Disagreement: {disagree:.1f}%")
print(f"  Sentiment high, rating low: {((da_val['exp_hi_sentiment']==1)&(da_val['exp_hi_rating']==0)).sum()}")
print(f"  Sentiment low, rating high: {((da_val['exp_hi_sentiment']==0)&(da_val['exp_hi_rating']==1)).sum()}")


# %% 4. FIGURE 1: PARK + DA SCATTERPLOTS
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(park_val["MeanSentiment"], park_val["AvgRating"], alpha=0.5, s=30, color="#01665e")
m, b = np.polyfit(park_val["MeanSentiment"], park_val["AvgRating"], 1)
x_line = np.linspace(park_val["MeanSentiment"].min(), park_val["MeanSentiment"].max(), 100)
ax.plot(x_line, m*x_line + b, color="#8c510a", linewidth=1.5)
ax.set_xlabel("Mean Sentiment Score (RoBERTa)", fontsize=10)
ax.set_ylabel("Mean Star Rating", fontsize=10)
ax.set_title(f"Park level (n={len(park_val)})\n"
             f"Pearson r={r_p:.3f}, Spearman r={r_sp:.3f}", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

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

plt.suptitle(f"Convergent Validity: Sentiment Score vs Star Rating — {CITY.title()}", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_validation_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}/{CITY}_validation_scatter.png")


# %% 5. FIGURE 2: DA SCATTER COLOURED BY DIVERGENCE QUADRANT
da_div = gpd.read_file(DIV_PATH)
valid  = da_div[
    da_div["satisfaction_sentiment"].notna() &
    da_div["satisfaction_star"].notna()
].copy()

colours = {
    "HH — High supply, high experience": "#3a2f36",
    "LH — Low supply, high experience":  "#d29a6a",
    "HL — High supply, low experience":  "#6fa3c8",
    "LL — Low supply, low experience":   "#efe7dc",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
r1, p1   = stats.pearsonr(park_val["MeanSentiment"], park_val["AvgRating"])
r1s, p1s = stats.spearmanr(park_val["MeanSentiment"], park_val["AvgRating"])
ax.scatter(park_val["MeanSentiment"], park_val["AvgRating"], alpha=0.5, s=20, color="#2d6a4f")
m, b = np.polyfit(park_val["MeanSentiment"], park_val["AvgRating"], 1)
x_line = np.linspace(park_val["MeanSentiment"].min(), park_val["MeanSentiment"].max(), 100)
ax.plot(x_line, m*x_line + b, color="#8b4513", linewidth=1.5)
ax.set_xlabel("Mean sentiment score (RoBERTa)", fontsize=11)
ax.set_ylabel("Mean star rating", fontsize=11)
ax.set_title(f"A. Park level (n={len(park_val)})\n"
             f"Pearson r={r1:.3f}, Spearman r={r1s:.3f}", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
for quad, colour in colours.items():
    subset = valid[valid["divergence_2x2"] == quad]
    ax.scatter(subset["satisfaction_sentiment"], subset["satisfaction_star"],
               color=colour, alpha=0.5, s=15, label=quad.split(" — ")[0])
r2, p2   = stats.pearsonr(valid["satisfaction_sentiment"], valid["satisfaction_star"])
r2s, p2s = stats.spearmanr(valid["satisfaction_sentiment"], valid["satisfaction_star"])
m2, b2 = np.polyfit(valid["satisfaction_sentiment"], valid["satisfaction_star"], 1)
x_line2 = np.linspace(valid["satisfaction_sentiment"].min(),
                       valid["satisfaction_sentiment"].max(), 100)
ax.plot(x_line2, m2*x_line2 + b2, color="#8b4513", linewidth=1.5)
ax.set_xlabel("Mean sentiment score (RoBERTa)", fontsize=11)
ax.set_ylabel("Mean star rating", fontsize=11)
ax.set_title(f"B. DA level (n={len(valid)})\n"
             f"Pearson r={r2:.3f}, Spearman r={r2s:.3f}", fontsize=11)
ax.legend(title="Divergence quadrant", fontsize=8, loc="upper left", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.suptitle(
    f"Agreement between sentiment scores and star ratings\n"
    f"at park and neighbourhood levels, {CITY.title()}",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_validation_scatter2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}/{CITY}_validation_scatter2.png")


# %% 6. FIGURE 3: THREE-PANEL (review / park / DA level)
reviews_sentiment = pd.read_csv(REVIEWS_PATH, low_memory=False)
review_valid      = reviews_sentiment[
    reviews_sentiment["Sentiment"].notna() &
    reviews_sentiment["Rating"].notna()
].copy()

r0,  p0  = stats.pearsonr(review_valid["Sentiment"], review_valid["Rating"])
r0s, p0s = stats.spearmanr(review_valid["Sentiment"], review_valid["Rating"])
print(f"\nReview-level Pearson r={r0:.3f}, p={p0:.4f}, n={len(review_valid)}")
print(f"Review-level Spearman r={r0s:.3f}, p={p0s:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: violin by star rating
ax = axes[0]
star_groups = [
    review_valid[review_valid["Rating"] == s]["Sentiment"].values
    for s in [1, 2, 3, 4, 5]
]
parts = ax.violinplot(star_groups, positions=[1, 2, 3, 4, 5],
                      showmedians=True, showextrema=False, vert=False)
for pc in parts["bodies"]:
    pc.set_facecolor("#2d6a4f"); pc.set_alpha(0.6)
parts["cmedians"].set_color("#8b4513"); parts["cmedians"].set_linewidth(1.5)
ax.set_ylabel("Google Star rating", fontsize=12)
ax.set_xlabel("Sentiment score (RoBERTa)", fontsize=12)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_title(f"A. Review level (n={len(review_valid):,})\n"
             f"Pearson r={r0:.3f}, Spearman r={r0s:.3f}", fontsize=14)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: park level
ax = axes[1]
r1, p1   = stats.pearsonr(park_val["MeanSentiment"], park_val["AvgRating"])
r1s, p1s = stats.spearmanr(park_val["MeanSentiment"], park_val["AvgRating"])
ax.scatter(park_val["MeanSentiment"], park_val["AvgRating"], alpha=0.5, s=20, color="#2d6a4f")
m1, b1 = np.polyfit(park_val["MeanSentiment"], park_val["AvgRating"], 1)
x1 = np.linspace(park_val["MeanSentiment"].min(), park_val["MeanSentiment"].max(), 100)
ax.plot(x1, m1*x1 + b1, color="#8b4513", linewidth=1.5)
ax.set_xlabel("Mean sentiment score (RoBERTa)", fontsize=12)
ax.set_ylabel("Mean star rating", fontsize=12)
ax.set_title(f"B. Park level (n={len(park_val)})\n"
             f"Pearson r={r1:.3f}, Spearman r={r1s:.3f}", fontsize=14)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel C: DA level
ax = axes[2]
r2, p2   = stats.pearsonr(da_valid["satisfaction_sentiment"], da_valid["satisfaction_star"]) \
           if len(valid) > 0 else (np.nan, np.nan)
r2s, p2s = stats.spearmanr(da_valid["satisfaction_sentiment"], da_valid["satisfaction_star"]) \
           if len(valid) > 0 else (np.nan, np.nan)
da_valid = da_div[da_div["satisfaction_sentiment"].notna() &
                  da_div["satisfaction_star"].notna()].copy()
ax.scatter(da_valid["satisfaction_sentiment"], da_valid["satisfaction_star"],
           alpha=0.3, s=15, color="#2d6a4f")
m2, b2 = np.polyfit(da_valid["satisfaction_sentiment"], da_valid["satisfaction_star"], 1)
x2 = np.linspace(da_valid["satisfaction_sentiment"].min(),
                  da_valid["satisfaction_sentiment"].max(), 100)
ax.plot(x2, m2*x2 + b2, color="#8b4513", linewidth=1.5)
r2,  p2  = stats.pearsonr(da_valid["satisfaction_sentiment"], da_valid["satisfaction_star"])
r2s, p2s = stats.spearmanr(da_valid["satisfaction_sentiment"], da_valid["satisfaction_star"])
ax.set_xlabel("Mean sentiment score (RoBERTa)", fontsize=12)
ax.set_ylabel("Mean star rating", fontsize=12)
ax.set_title(f"C. DA level (n={len(da_valid)})\n"
             f"Pearson r={r2:.3f}, Spearman r={r2s:.3f}", fontsize=14)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_validation_scatter3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}/{CITY}_validation_scatter3.png")


# %% 7. SAVE SUMMARY TABLE
kappa_quad = cohen_kappa_score(da_val["exp_hi_sentiment"], da_val["exp_hi_rating"])
print(f"\nDA-level Cohen's kappa: {kappa_quad:.3f}")

summary = pd.DataFrame([
    {"Level": "Park", "Metric": "Pearson r",              "Value": round(r_p, 3),    "p": round(p_p, 4)},
    {"Level": "Park", "Metric": "Spearman r",             "Value": round(r_sp, 3),   "p": round(p_sp, 4)},
    {"Level": "DA",   "Metric": "Pearson r",              "Value": round(r_da_p, 3), "p": round(p_da_p, 4)},
    {"Level": "DA",   "Metric": "Spearman r",             "Value": round(r_da_sp, 3),"p": round(p_da_sp, 4)},
    {"Level": "DA",   "Metric": "Quadrant agreement (%)", "Value": round(agree, 1),  "p": ""},
    {"Level": "DA",   "Metric": "Cohen's kappa",          "Value": round(kappa_quad, 3), "p": ""},
])
summary.to_csv(f"{TAB_DIR}/{CITY}_validation_summary.csv", index=False)
print(f"Saved: {TAB_DIR}/{CITY}_validation_summary.csv")
print(summary.to_string(index=False))

print("\nDone.")
