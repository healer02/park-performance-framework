# =============================================================================
# 08-experience.py
# Purpose: Compute DA-level experience exposure from reachable park metrics
#
# Inputs:
#   - data/google-reviews/processed/08c-park-metrics.csv  (from 06d)
#   - data/osm/Vancouver_walk.graphml
#   - data/parks/processed/vancouver_park_entrances.shp
#   - data/census/processed/vancouver_db_centroids.gpkg
#   - data/processed/vancouver_da_supply.gpkg
#
# Outputs:
#   - data/processed/vancouver_da_experience.csv
#   - data/processed/vancouver_da_divergence.gpkg
#   - outputs/figures/vancouver_da_divergence_2x2.png
#   - outputs/figures/vancouver_da_divergence_prototype.png
# =============================================================================

# %% 1. IMPORTS AND PATHS
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import numpy as np
import json

PARK_METRICS_PATH = "data/google-reviews/processed/08c-park-metrics.csv"
GRAPH_PATH        = "data/osm/Vancouver_walk.graphml"
ENT_PATH          = "data/parks/processed/vancouver_park_entrances.shp"
DB_PATH           = "data/census/processed/vancouver_db_centroids.gpkg"
SUPPLY_PATH       = "data/processed/vancouver_da_supply.gpkg"
DA_PARK_SETS_PATH = "data/processed/vancouver_da_park_sets.json"

OUT_DIR = "data/processed"
FIG_DIR = "outputs/figures"

print("Ready.")


# %% 2. LOAD PARK METRICS (from 06d)
park_metrics = pd.read_csv(PARK_METRICS_PATH)

# Rename for consistency with downstream pipeline
park_metrics = park_metrics.rename(columns={
    "MeanSentiment": "AvgSentiment",
})

# Keep only parks with valid data
park_metrics = park_metrics[
    park_metrics["park_id"].notna()
].copy()

print(f"Parks loaded:                    {len(park_metrics)}")
print(f"Parks with valid sentiment:      {park_metrics['has_valid_sentiment'].sum()}")
print(f"Parks without valid sentiment:   {(~park_metrics['has_valid_sentiment']).sum()}")
print(f"\nAvgSentiment summary:")
print(park_metrics["AvgSentiment"].describe().round(3))
print(f"\nAvgRating summary:")
print(park_metrics["AvgRating"].describe().round(3))
print(f"\nTotalReviews summary:")
print(park_metrics["TotalReviews"].describe().round(0))


# %% 3. LOAD OR REBUILD DA PARK SETS
# Load from disk if available (saves 5-15 min network loop)
import geopandas as gpd

if os.path.exists(DA_PARK_SETS_PATH):
    print("Loading da_park_sets from disk...")
    with open(DA_PARK_SETS_PATH, "r") as f:
        da_park_sets = {k: set(v) for k, v in json.load(f).items()}
    print(f"Loaded. DAs with reachable parks: {len(da_park_sets)}")
else:
    print("da_park_sets not found -- rebuilding (5-15 min)...")
    import osmnx as ox
    import networkx as nx

    G            = ox.load_graphml(GRAPH_PATH)
    entrances    = gpd.read_file(ENT_PATH)
    db_centroids = gpd.read_file(DB_PATH)

    node_col = 'nearest_no' if 'nearest_no' in entrances.columns else 'nearest_node'
    node_to_parks = {}
    for _, row in entrances.iterrows():
        node = row[node_col]
        pid  = row['park_id']
        if pd.notna(node):
            node = int(node)
            node_to_parks.setdefault(node, set()).add(pid)

    entrance_node_set = set(node_to_parks.keys())
    print(f"Entrance nodes: {len(entrance_node_set)}")

    THRESHOLD    = 400
    da_park_sets = {}
    n_processed  = 0

    print(f"Running DB loop ({len(db_centroids)} DBs)...")
    for _, db_row in db_centroids.iterrows():
        dauid  = db_row['DAUID']
        db_pop = int(db_row['db_pop'])

        if db_pop == 0 or pd.isna(db_row['nearest_node']):
            n_processed += 1
            continue

        db_node = int(db_row['nearest_node'])
        try:
            distances = nx.single_source_dijkstra_path_length(
                G, db_node, cutoff=THRESHOLD, weight='length'
            )
        except nx.NodeNotFound:
            n_processed += 1
            continue

        reachable_parks = set()
        for node in set(distances.keys()) & entrance_node_set:
            reachable_parks.update(node_to_parks[node])

        da_park_sets.setdefault(dauid, set()).update(reachable_parks)
        n_processed += 1
        if n_processed % 500 == 0:
            print(f"  {n_processed}/{len(db_centroids)} DBs...")

    print(f"Done. DAs with reachable parks: {len(da_park_sets)}")

    with open(DA_PARK_SETS_PATH, "w") as f:
        json.dump({k: list(v) for k, v in da_park_sets.items()}, f)
    print("Saved da_park_sets to disk.")


# %% 4. COMPUTE DA-LEVEL EXPERIENCE EXPOSURE
db_centroids = gpd.read_file(DB_PATH)
da_pop_lookup = (
    db_centroids.groupby("DAUID")["db_pop"]
    .sum()
    .to_dict()
)
print(f"DA population lookup built: {len(da_pop_lookup)} DAs")

# Build park metrics index
exp_index = park_metrics.set_index("park_id")

experience_records = []
for dauid, park_ids in da_park_sets.items():
    subset = exp_index[exp_index.index.isin(park_ids)]
    da_pop = da_pop_lookup.get(dauid, 0)

    if len(subset) == 0 or da_pop == 0:
        experience_records.append({
            "DAUID":                  dauid,
            "salience":               np.nan,
            "satisfaction_sentiment": np.nan,
            "satisfaction_star":      np.nan,
            "coverage_pct":           np.nan,
            "n_reachable_parks":      len(subset),
            "n_qualifying_parks":     0,
        })
        continue

    # SALIENCE: total Google reviews across all reachable parks per 1,000 residents
    total_reviews = subset["TotalReviews"].sum()
    salience      = total_reviews / da_pop * 1000

    # SATISFACTION: unweighted mean across reachable parks with >=10 text reviews
    # (has_valid_sentiment flags parks meeting this threshold)
    qualifying             = subset[subset["has_valid_sentiment"] == True]
    n_qualifying           = len(qualifying)
    satisfaction_sentiment = qualifying["AvgSentiment"].mean() if n_qualifying > 0 else np.nan
    satisfaction_star      = qualifying["AvgRating"].mean()    if n_qualifying > 0 else np.nan
    coverage_pct           = n_qualifying / len(subset) * 100

    experience_records.append({
        "DAUID":                  dauid,
        "salience":               round(salience, 4),
        "satisfaction_sentiment": round(satisfaction_sentiment, 4) if not np.isnan(satisfaction_sentiment) else np.nan,
        "satisfaction_star":      round(satisfaction_star, 4)      if not np.isnan(satisfaction_star)      else np.nan,
        "coverage_pct":           round(coverage_pct, 1),
        "n_reachable_parks":      len(subset),
        "n_qualifying_parks":     n_qualifying,
    })

da_experience = pd.DataFrame(experience_records)
da_experience.to_csv(f"{OUT_DIR}/vancouver_da_experience.csv", index=False)

print(f"\nDAs with salience data:               {da_experience['salience'].notna().sum()}")
print(f"DAs with satisfaction (sentiment):     {da_experience['satisfaction_sentiment'].notna().sum()}")
print(f"DAs with no qualifying parks:          {(da_experience['n_qualifying_parks']==0).sum()}")
print(f"\nSalience summary (reviews per 1,000 residents):")
print(da_experience["salience"].describe().round(2))
print(f"\nSatisfaction summary (sentiment):")
print(da_experience["satisfaction_sentiment"].describe().round(3))
print(f"\nCoverage % summary:")
print(da_experience["coverage_pct"].describe().round(1))


# %% 5. JOIN SUPPLY + EXPERIENCE AND CLASSIFY DIVERGENCE
da_supply = gpd.read_file(SUPPLY_PATH)

REACH_THRESH = 0.8
qty_med      = da_supply["qty_cap20"].median()

da_supply["reach_cat"] = (da_supply["DA_reach_400"] >= REACH_THRESH).astype(int)
da_supply["qty_cat"]   = (da_supply["qty_cap20"]    >= qty_med).astype(int)

def classify_supply(r, q):
    if pd.isna(r) or pd.isna(q): return "No data"
    if r==1 and q==1: return "HH"
    if r==1 and q==0: return "HL"
    if r==0 and q==1: return "LH"
    return "LL"

def classify_divergence(s, e):
    if pd.isna(e) or s == "No data": return "No data"
    return f"{s}_{'hi' if e == 1 else 'lo'}"

def classify_2x2(s, e):
    if pd.isna(s) or pd.isna(e): return "No data"
    if s == 1 and e == 1: return "HH — High supply, high experience"
    if s == 1 and e == 0: return "HL — High supply, low experience"
    if s == 0 and e == 1: return "LH — Low supply, high experience"
    return "LL — Low supply, low experience"

da_supply["supply_type"] = [
    classify_supply(r, q)
    for r, q in zip(da_supply["reach_cat"], da_supply["qty_cat"])
]

print("Supply typology:")
print(da_supply["supply_type"].value_counts())
print(f"Thresholds — reachability: {REACH_THRESH}, quantity median: {qty_med:.1f} ha/1,000")

# Join experience
da_div = da_supply.merge(da_experience, on="DAUID", how="left")

# Binary experience classification: median split on sentiment
sentiment_med = da_div["satisfaction_sentiment"].median()
da_div["experience_hi"] = (da_div["satisfaction_sentiment"] >= sentiment_med).astype(int)

# Reclassify NaN sentiment DAs as No data -- not low experience
da_div.loc[da_div["satisfaction_sentiment"].isna(), "experience_hi"] = np.nan

# Update divergence classification
da_div["divergence_type"] = [
    classify_divergence(s, e)
    for s, e in zip(da_div["supply_type"], da_div["experience_hi"])
]

# supply_binary: HH only = high supply
da_div["supply_binary"] = (da_div["supply_type"] == "HH").astype(int)

da_div["divergence_2x2"] = [
    classify_2x2(s, e)
    for s, e in zip(da_div["supply_binary"], da_div["experience_hi"])
]

da_div["divergence_2x2"] = [
    classify_2x2(s, e)
    for s, e in zip(da_div["supply_binary"], da_div["experience_hi"])
]

print(f"\nSentiment median: {sentiment_med:.3f}")
print(f"\nFull 4x2 divergence distribution:")
print(da_div["divergence_type"].value_counts().sort_index())
print(f"\nProblem cells (low experience):")
print(da_div[da_div["experience_hi"]==0]["supply_type"].value_counts())

da_div.to_file(f"{OUT_DIR}/vancouver_da_divergence.gpkg", driver="GPKG")
print("\nSaved vancouver_da_divergence.gpkg")


# %% 5b. EXPERIENCE MAPS: Salience and Satisfaction (side by side)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import numpy as np

parks_gdf = gpd.read_file("data/parks/processed/vancouver_parks_merged.shp")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# --- Left: Digital Salience ---
ax = axes[0]
da_div_plot = da_div.copy()

# Log-transform salience for display (highly skewed)
da_div_plot["salience_log"] = np.log1p(da_div_plot["salience"])

da_div_plot.plot(
    ax=ax,
    column="salience_log",
    cmap="YlOrRd",
    legend=True,
    missing_kwds={"color": "#cccccc", "label": "No data"},
    legend_kwds={
        "label": "Log(reviews per 1,000 residents + 1)",
        "shrink": 0.5
    }
)
parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.6, zorder=2)
ax.set_title(
    "Digital Salience\nGoogle reviews per 1,000 residents (log-transformed)",
    fontsize=11
)
ax.set_axis_off()

# --- Right: Expressed Satisfaction (Sentiment) ---
ax = axes[1]
da_div.plot(
    ax=ax,
    column="satisfaction_sentiment",
    cmap="RdYlGn",
    vmin=0.4,
    vmax=0.9,
    legend=True,
    missing_kwds={"color": "#cccccc", "label": "No data"},
    legend_kwds={
        "label": "Mean sentiment score (RoBERTa)",
        "shrink": 0.5
    }
)
parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.6, zorder=2)
ax.set_title(
    "Expressed Satisfaction\nMean RoBERTa sentiment score across reachable parks",
    fontsize=11
)
ax.set_axis_off()

plt.suptitle(
    "Park Experience Dimensions — Vancouver Dissemination Areas (2021)",
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_da_experience_2maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: vancouver_da_experience_2maps.png")

# %%
da_equity = pd.read_csv("data/processed/vancouver_da_equity.csv", dtype={"DAUID": str})
print(da_equity[da_equity["satisfaction_sentiment"].isna()]["divergence_2x2"].value_counts())
print(f"\nTotal NaN sentiment DAs: {da_equity['satisfaction_sentiment'].isna().sum()}")


# %% 6a. SIMPLE 2x2 DIVERGENCE MAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

da_div["supply_binary"] = (da_div["supply_type"] == "HH").astype(int)

def classify_2x2(s, e):
    if pd.isna(s) or pd.isna(e): return "No data"
    if s == 1 and e == 1: return "HH — High supply, high experience"
    if s == 1 and e == 0: return "HL — High supply, low experience"
    if s == 0 and e == 1: return "LH — Low supply, high experience"
    return "LL — Low supply, low experience"

da_div["divergence_2x2"] = [
    classify_2x2(s, e)
    for s, e in zip(da_div["supply_binary"], da_div["experience_hi"])
]

colours_2x2 = {
    "HH — High supply, high experience": "#01665e",   # dark teal
    "LH — Low supply, high experience":  "#80cdc1",   # light teal
    "HL — High supply, low experience":  "#dfc27d",   # dark brown
    "LL — Low supply, low experience":   "#8c510a",   # light brown
    "No data":                           "#cccccc",
}

parks_gdf   = gpd.read_file("data/parks/processed/vancouver_parks_merged.shp")
counts_2x2  = da_div["divergence_2x2"].value_counts()

fig, ax = plt.subplots(figsize=(13, 10))
for dtype, colour in colours_2x2.items():
    subset = da_div[da_div["divergence_2x2"] == dtype]
    if len(subset):
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)

parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.8, zorder=2)

legend_labels = {
    "HH": "High supply, high experience",
    "HL": "High supply, low experience",
    "LH": "Low supply, high experience",
    "LL": "Low supply, low experience",
}
# Add count annotations to legend
patches = [
    mpatches.Patch(color=colours_2x2["HH — High supply, high experience"],
                   label=f"HH — High supply, high experience (n={counts_2x2.get('HH — High supply, high experience', 0)})"),
    mpatches.Patch(color=colours_2x2["HL — High supply, low experience"],
                   label=f"HL — High supply, low experience (n={counts_2x2.get('HL — High supply, low experience', 0)})"),
    mpatches.Patch(color=colours_2x2["LH — Low supply, high experience"],
                   label=f"LH — Low supply, high experience (n={counts_2x2.get('LH — Low supply, high experience', 0)})"),
    mpatches.Patch(color=colours_2x2["LL — Low supply, low experience"],
                   label=f"LL — Low supply, low experience (n={counts_2x2.get('LL — Low supply, low experience', 0)})"),
]

# Add 2x2 matrix inset
ax_inset = fig.add_axes([0.02, 0.02, 0.18, 0.18])  # [left, bottom, width, height]
matrix_data = np.array([
    [counts_2x2.get("LH — Low supply, high experience", 0),
     counts_2x2.get("HH — High supply, high experience", 0)],
    [counts_2x2.get("LL — Low supply, low experience", 0),
     counts_2x2.get("HL — High supply, low experience", 0)],
])
matrix_colours = np.array([
    ["#80cdc1", "#01665e"],  # high exp: LH=light teal, HH=dark teal
    ["#8c510a", "#dfc27d"],  # low exp: LL=light brown, HL=dark brown
])
for i in range(2):
    for j in range(2):
        ax_inset.add_patch(plt.Rectangle(
            (j, 1-i), 1, 1,
            color=matrix_colours[i, j], ec="white", lw=1.5
        ))
        ax_inset.text(
            j + 0.5, 1.5 - i,
            str(matrix_data[i, j]),
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white"
        )

ax_inset.set_xlim(0, 2)
ax_inset.set_ylim(0, 2)
ax_inset.set_xticks([0.5, 1.5])
ax_inset.set_xticklabels(["Low supply", "High supply"], fontsize=7)
ax_inset.set_yticks([0.5, 1.5])
ax_inset.set_yticklabels(["Low exp", "High exp"], fontsize=7)
ax_inset.tick_params(length=0)
ax.annotate(
    f"Grey = No data (insufficient reviews; n={counts_2x2.get('No data', 0)})",
    xy=(0.02, 0.22),
    xycoords="axes fraction",
    fontsize=9,
    color="#555555",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none")
)
ax_inset.set_title("n by quadrant", fontsize=7, pad=3)
for spine in ax_inset.spines.values():
    spine.set_visible(False)
    

ax.set_title(
    "Supply–Experience Divergence — Vancouver DAs\n"
    f"Supply: HH only (reachability ≥ {REACH_THRESH} & quantity ≥ {qty_med:.0f} ha/1,000) | "
    f"Experience: sentiment ≥ median ({sentiment_med:.3f})",
    fontsize=10
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_da_divergence_2x2_20260521.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 2x2 map.")
print(counts_2x2)


# %% 6b. PROTOTYPE DIVERGENCE MAP (4x2, low experience highlighted)
HIGH_EXP_COLOUR = "#f0ebe3"

colours = {
    "HH_low_exp":  "#4b3f44",
    "HL_low_exp":  "#7ea6c2",
    "LH_low_exp":  "#d0a07a",
    "LL_low_exp":  "#c0392b",
    "HH_high_exp": HIGH_EXP_COLOUR,
    "HL_high_exp": HIGH_EXP_COLOUR,
    "LH_high_exp": HIGH_EXP_COLOUR,
    "LL_high_exp": HIGH_EXP_COLOUR,
    "No data":     "#cccccc",
}

counts = da_div["divergence_type"].value_counts()

fig, ax = plt.subplots(figsize=(13, 10))
for dtype, colour in colours.items():
    subset = da_div[da_div["divergence_type"] == dtype]
    if len(subset):
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)

parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.8, zorder=2)

patches = [
    mpatches.Patch(color=colours["HH_low_exp"], label=f"HH — Strong supply, low experience (n={counts.get('HH_low_exp', 0)})"),
    mpatches.Patch(color=colours["HL_low_exp"], label=f"HL — Coverage-oriented, low experience (n={counts.get('HL_low_exp', 0)})"),
    mpatches.Patch(color=colours["LH_low_exp"], label=f"LH — Area-oriented, low experience (n={counts.get('LH_low_exp', 0)})"),
    mpatches.Patch(color=colours["LL_low_exp"], label=f"LL — Weak supply, low experience (n={counts.get('LL_low_exp', 0)})"),
    mpatches.Patch(color=HIGH_EXP_COLOUR,       label=f"High experience (n={sum(counts.get(k, 0) for k in ['HH_high_exp','HL_high_exp','LH_high_exp','LL_high_exp'])})"),
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)
ax.set_title(
    f"Supply–Experience Divergence — Vancouver DAs\n"
    f"Supply: reachability ≥ {REACH_THRESH} & quantity ≥ median ({qty_med:.0f} ha/1,000) | "
    f"Experience: mean sentiment ≥ median ({sentiment_med:.3f})",
    fontsize=10
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_da_divergence_prototype.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved prototype map.")


# %% 7. SENSITIVITY: Review-weighted sentiment
exp_index_w      = park_metrics.set_index("park_id")
sensitivity_records = []

for dauid, park_ids in da_park_sets.items():
    subset     = exp_index_w[exp_index_w.index.isin(park_ids)]
    qualifying = subset[subset["has_valid_sentiment"] == True]

    if len(qualifying) == 0:
        sensitivity_records.append({"DAUID": dauid, "exp_sentiment_weighted": np.nan})
        continue

    weighted_mean = np.average(
        qualifying["AvgSentiment"],
        weights=np.log1p(qualifying["TotalReviews"])
    )
    sensitivity_records.append({
        "DAUID":                 dauid,
        "exp_sentiment_weighted": weighted_mean,
    })

da_sensitivity = pd.DataFrame(sensitivity_records)
da_div_s       = da_div.merge(da_sensitivity, on="DAUID", how="left")

weighted_med = da_div_s["exp_sentiment_weighted"].median()
da_div_s["experience_hi_weighted"] = (
    da_div_s["exp_sentiment_weighted"] >= weighted_med
).astype(int)

da_div_s["divergence_type_weighted"] = [
    classify_divergence(s, e)
    for s, e in zip(da_div_s["supply_type"], da_div_s["experience_hi_weighted"])
]

changed = (da_div_s["divergence_type"] != da_div_s["divergence_type_weighted"]).sum()
total   = da_div_s["divergence_type"].notna().sum()
print(f"\nSensitivity — DAs changing quadrant (weighted vs unweighted): {changed} / {total} ({100*changed/total:.1f}%)")
print(f"Primary sentiment median:  {sentiment_med:.3f}")
print(f"Weighted sentiment median: {weighted_med:.3f}")


# %%
da_exp = pd.read_csv("data/processed/vancouver_da_experience.csv", dtype={"DAUID": str})
park_metrics = pd.read_csv("data/google-reviews/processed/08c-park-metrics.csv")

with open("data/processed/vancouver_da_park_sets.json") as f:
    da_park_sets = {k: set(v) for k, v in json.load(f).items()}

# Identify the 180 No data DAs
no_data_daus = da_exp[da_exp["satisfaction_sentiment"].isna()]["DAUID"].tolist()
print(f"No data DAs: {len(no_data_daus)}")

# For each no-data DA, check what parks are reachable and why they fail
reasons = []
for dauid in no_data_daus:
    park_ids = da_park_sets.get(dauid, set())
    if len(park_ids) == 0:
        reasons.append("no reachable parks")
        continue
    subset = park_metrics[park_metrics["park_id"].isin(park_ids)]
    if len(subset) == 0:
        reasons.append("parks not in metrics file")
    elif subset["n_text_reviews"].max() < 10:
        reasons.append("all parks below 10 review threshold")
    else:
        reasons.append("other")

import pandas as pd
print(pd.Series(reasons).value_counts())

# %% 8. SUPPLY-EXPERIENCE CONCORDANCE (for 4.1 prose)
# Requires: da_div (from cell 5)
# --- reload if needed ---
# import pandas as pd, geopandas as gpd, numpy as np, json
# da_div = gpd.read_file("data/processed/vancouver_da_divergence.gpkg")
# da_div["DAUID"] = da_div["DAUID"].astype(str)
# ------------------------

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import itertools

df = da_div[da_div["divergence_2x2"] != "No data"].copy()

variables = {
    "DA_reach_400":             ("Park coverage (prop. within 400 m)",          False),
    "qty_cap20":                ("Accessible park area (ha/1,000 res.)",         True),
    "salience":                 ("Digital salience (reviews/1,000 res.)",        True),
    "satisfaction_sentiment":   ("Mean sentiment score",                         False),
}

var_keys = list(variables.keys())
pairs = list(itertools.combinations(var_keys, 2))  # 6 pairs
n_pairs = len(pairs)

# --- Print all Spearman r ---
print("Spearman correlations:")
for v1, v2 in pairs:
    valid = df[[v1, v2]].dropna()
    r, p = spearmanr(valid[v1], valid[v2])
    p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    print(f"  {v1} × {v2}: r = {r:.3f}, p {p_str}, n = {len(valid)}")

# --- 2x3 scatterplot grid ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Pairwise relationships — park supply and experience, Vancouver DAs", fontsize=13)

for ax, (v1, v2) in zip(axes.flat, pairs):
    label1, log1 = variables[v1]
    label2, log2 = variables[v2]

    valid = df[[v1, v2]].dropna()
    # drop zeros before log transform
    if log1:
        valid = valid[valid[v1] > 0]
    if log2:
        valid = valid[valid[v2] > 0]

    r, p = spearmanr(valid[v1], valid[v2])
    p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"

    ax.scatter(valid[v1], valid[v2], c="#4a7c6f", alpha=0.4, s=14, zorder=2)
    ax.axvline(valid[v1].median(), color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(valid[v2].median(), color="black", lw=0.8, ls="--", alpha=0.6)

    if log1:
        ax.set_xscale("log")
        ax.set_xlabel(label1 + " (log)", fontsize=9)
    else:
        ax.set_xlabel(label1, fontsize=9)

    if log2:
        ax.set_yscale("log")
        ax.set_ylabel(label2 + " (log)", fontsize=9)
    else:
        ax.set_ylabel(label2, fontsize=9)

    ax.set_title(f"r = {r:.3f}, p {p_str}, n = {len(valid)}", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/vancouver_supply_experience_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 2x3 grid.")

# %% 8b. STANDALONE FIGURE: area vs sentiment (for main text)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = da_div[da_div["divergence_2x2"] != "No data"].copy()
valid = df[["qty_cap20", "satisfaction_sentiment"]].dropna()
valid = valid[valid["qty_cap20"] > 0]

r, p = spearmanr(valid["qty_cap20"], valid["satisfaction_sentiment"])
p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#f8f8f6")
ax.set_facecolor("#f8f8f6")

# Scatter
ax.scatter(valid["qty_cap20"], valid["satisfaction_sentiment"],
           c="#1f5c4d", alpha=0.5, s=20, zorder=2, linewidths=0)

# Median lines — subtle, no label
ax.axvline(valid["qty_cap20"].median(),
           color="#999999", lw=1.0, ls=(0, (4, 4)), zorder=1)
ax.axhline(valid["satisfaction_sentiment"].median(),
           color="#999999", lw=1.0, ls=(0, (4, 4)), zorder=1)

# Annotation instead of title for r
ax.annotate(f"Spearman r = {r:.3f}, p {p_str}\nn = {len(valid)}",
            xy=(0.04, 0.04), xycoords="axes fraction",
            fontsize=10, color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="none", alpha=0.7))

# Spines — keep only bottom and left, thin
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#cccccc")
    ax.spines[spine].set_linewidth(0.8)

ax.tick_params(colors="#666666", labelsize=9)
ax.set_xscale("log")
ax.set_xlabel("Accessible park area (ha per 1,000 residents)", fontsize=11, color="#333333", labelpad=8)
ax.set_ylabel("Mean sentiment score (reachable parks)",         fontsize=11, color="#333333", labelpad=8)
ax.set_title("Park area and expressed satisfaction — Vancouver DAs",
             fontsize=12, color="#222222", pad=14, loc="left", fontweight="bold")

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}"
))
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)

plt.tight_layout()
plt.savefig("outputs/figures/vancouver_area_sentiment_scatter.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved area-sentiment scatterplot.")


# %% 8c. PARK AREA–SENTIMENT CORRELATION: sensitivity across area caps
from scipy.stats import pearsonr, spearmanr

area_caps = [
    ("qty_cap10", "10 ha cap"),
    ("qty_cap20", "20 ha cap (primary)"),
    ("qty_raw",   "Uncapped"),
]

df_corr = da_div[da_div["divergence_2x2"] != "No data"].copy()

print("--- Park area–sentiment correlation (DA level) ---")
print(f"{'Cap':<22} {'r_pearson':>10} {'p_pearson':>10} {'r_spearman':>11} {'p_spearman':>11} {'n':>5}")
print("-" * 72)

for col, label in area_caps:
    subset = df_corr[[col, "satisfaction_sentiment"]].dropna()
    n = len(subset)
    r_p, p_p = pearsonr(subset[col], subset["satisfaction_sentiment"])
    r_s, p_s = spearmanr(subset[col], subset["satisfaction_sentiment"])
    sig_p = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns"
    sig_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "ns"
    print(f"{label:<22} {r_p:>+9.3f}{sig_p:>2} {p_p:>10.4f} {r_s:>+10.3f}{sig_s:>2} {p_s:>10.4f} {n:>5}")





# %% QA figure: extreme DAs by quadrant

import geopandas as gpd
import matplotlib.pyplot as plt

da = gpd.read_file("data/processed/vancouver_da_divergence.gpkg")

supply_var = "qty_cap20"
exp_var    = "satisfaction_sentiment"

# Map short codes to full labels
quad_labels = {
    "HH": "HH — High supply, high experience",
    "HL": "HL — High supply, low experience",
    "LH": "LH — Low supply, high experience",
    "LL": "LL — Low supply, low experience",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

quadrants = {
    "HH": ("High supply / High experience", axes[0, 0]),
    "HL": ("High supply / Low experience",  axes[0, 1]),
    "LH": ("Low supply / High experience",  axes[1, 0]),
    "LL": ("Low supply / Low experience",   axes[1, 1]),
}

for quad, (title, ax) in quadrants.items():
    subset = da[da["divergence_2x2"] == quad_labels[quad]].copy()
    subset = subset[subset[supply_var].notna() & subset[exp_var].notna()]

    if quad == "HH":
        subset["score"] = subset[supply_var].rank() + subset[exp_var].rank()
    elif quad == "HL":
        subset["score"] = subset[supply_var].rank() - subset[exp_var].rank()
    elif quad == "LH":
        subset["score"] = subset[exp_var].rank() - subset[supply_var].rank()
    elif quad == "LL":
        subset["score"] = -(subset[supply_var].rank() + subset[exp_var].rank())

    top10 = subset.nlargest(10, "score")

    da.plot(ax=ax, color="lightgrey", linewidth=0)
    subset.plot(ax=ax, color="#dddddd", linewidth=0)
    top10.plot(ax=ax, color="red", edgecolor="black", linewidth=0.5)

    for _, row in top10.iterrows():
        x = row.geometry.centroid.x
        y = row.geometry.centroid.y
        ax.annotate(
            row["DAUID"][-4:],
            xy=(x, y),
            fontsize=6,
            ha="center",
            color="black"
        )

    ax.set_title(f"{title}\nTop 10 extreme DAs (n={len(subset)})")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/figures/vancouver_extreme_DA_QA.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved.")


# %% Moran's I check
from libpysal.weights import Queen
from esda.moran import Moran

da = gpd.read_file("data/processed/vancouver_da_divergence.gpkg")

# Encode divergence quadrant as numeric for Moran's I
quad_map = {
    "HH — High supply, high experience": 3,
    "LH — Low supply, high experience":  2,
    "HL — High supply, low experience":  1,
    "LL — Low supply, low experience":   0,
}
da["divergence_numeric"] = da["divergence_2x2"].map(quad_map)

variables = [
    "qty_cap20",
    "satisfaction_sentiment",
    "divergence_numeric",
]

for var in variables:
    subset = da[["DAUID", var, "geometry"]].dropna().copy()
    subset = subset.reset_index(drop=True)

    w = Queen.from_dataframe(subset)
    w.transform = "r"

    mi = Moran(subset[var], w)
    print(f"\n{var} (n={len(subset)})")
    print(f"  Moran's I = {mi.I:.3f}")
    print(f"  p-value   = {mi.p_sim:.4f}")



# %%
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from scipy import stats

da = gpd.read_file("data/processed/vancouver_da_divergence.gpkg")
valid = da[da["satisfaction_sentiment"].notna() & 
           da["DA_reach_400"].notna() & 
           da["qty_cap20"].notna()].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel A: Coverage × Sentiment ---
ax = axes[0]

ceiling     = valid[valid["DA_reach_400"] == 1.0]
non_ceiling = valid[valid["DA_reach_400"] < 1.0]

ax.hexbin(
    non_ceiling["DA_reach_400"],
    non_ceiling["satisfaction_sentiment"],
    gridsize=20, cmap="YlGn", mincnt=1, alpha=0.8
)

np.random.seed(42)
jitter = np.random.normal(1.0, 0.008, len(ceiling))
ax.scatter(
    jitter,
    ceiling["satisfaction_sentiment"],
    alpha=0.15, s=8, color="#2d6a4f"
)

r, p = stats.spearmanr(valid["DA_reach_400"], valid["satisfaction_sentiment"])
ax.set_xlabel("Park coverage (proportion within 400 m)", fontsize=11)
ax.set_ylabel("Mean sentiment score", fontsize=11)
ax.set_title(f"A. Coverage × Satisfaction\nSpearman r = {r:.3f}, ns, n = {len(valid)}", 
             fontsize=11)
ax.set_xlim(-0.05, 1.12)
ax.set_ylim(0.15, 0.95)

# Annotation moved to bottom-right, outside hexbin area
ax.annotate(
    f"n={len(ceiling)} DAs\nat full coverage",
    xy=(1.02, 0.5),
    xytext=(1.04, 0.25),
    xycoords=("data", "data"),
    fontsize=9, color="dimgrey",
    ha="left",
    arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.8)
)

# --- Panel B: Area × Sentiment ---
ax = axes[1]

r2, p2 = stats.spearmanr(valid["qty_cap20"], valid["satisfaction_sentiment"])
ax.scatter(
    valid["qty_cap20"],
    valid["satisfaction_sentiment"],
    alpha=0.25, s=12, color="#2d6a4f"
)
ax.set_xscale("log")

# Median lines — darker and labelled
med_x = valid["qty_cap20"].median()
med_y = valid["satisfaction_sentiment"].median()
ax.axvline(med_x, color="#555555", linestyle="--", linewidth=1.2)
ax.axhline(med_y, color="#555555", linestyle="--", linewidth=1.2)
ax.text(med_x * 1.08, 0.18, "median area", fontsize=8, color="#555555")
ax.text(0.11, med_y + 0.01, "median sentiment", fontsize=8, color="#555555")

ax.set_xlabel("Accessible park area (ha per 1,000 residents, log scale)", fontsize=11)
ax.set_ylabel("Mean sentiment score", fontsize=11)
ax.set_title(f"B. Park Area × Satisfaction\nSpearman r = {r2:.3f}, p < 0.001, n = {len(valid)}", 
             fontsize=11)
ax.set_ylim(0.15, 0.95)

plt.suptitle(
    "Park supply and expressed satisfaction — Vancouver dissemination areas",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(
    "outputs/figures/vancouver_figure1_supply_experience.png",
    dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved.")
# %%
