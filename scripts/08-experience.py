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

# Full 4x2 divergence type
def classify_divergence(supply, exp):
    if pd.isna(supply) or supply == "No data": return "No data"
    exp_label = "high_exp" if exp == 1 else "low_exp"
    return f"{supply}_{exp_label}"

da_div["divergence_type"] = [
    classify_divergence(s, e)
    for s, e in zip(da_div["supply_type"], da_div["experience_hi"])
]

print(f"\nSentiment median: {sentiment_med:.3f}")
print(f"\nFull 4x2 divergence distribution:")
print(da_div["divergence_type"].value_counts().sort_index())
print(f"\nProblem cells (low experience):")
print(da_div[da_div["experience_hi"]==0]["supply_type"].value_counts())

da_div.to_file(f"{OUT_DIR}/vancouver_da_divergence.gpkg", driver="GPKG")
print("\nSaved vancouver_da_divergence.gpkg")


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
    "HL — High supply, low experience":  "#8c510a",   # dark brown
    "LL — Low supply, low experience":   "#dfc27d",   # light brown
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
    ["#dfc27d", "#8c510a"],  # low exp: LL=light brown, HL=dark brown
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
plt.savefig(f"{FIG_DIR}/vancouver_da_divergence_2x2.png", dpi=150, bbox_inches="tight")
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
