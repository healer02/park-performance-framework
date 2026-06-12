"""
06_experience.py  (was 08-experience.py)
Compute DA-level experience exposure from reachable park metrics,
then classify supply–experience divergence into 2×2 matrix.

Supports two experience sources controlled by config.EXPERIENCE_SOURCE:
  "sentiment"   — RoBERTa sentiment scores (requires vancouver/ sub-track)
  "star_rating" — Google star ratings (replication cities, no text scraping needed)

Inputs:
    data/google-reviews/processed/08c-park-metrics.csv   (from v03_sentiment.py)
    data/osm/{CITY}_walk.graphml
    data/parks/processed/{CITY}_park_entrances.shp
    data/census/processed/{CITY}_db_centroids.gpkg
    data/processed/{CITY}_da_supply.gpkg
    data/processed/{CITY}_da_park_sets.json              (cached from 05_supply.py)

Outputs:
    data/processed/{CITY}_da_experience.csv
    data/processed/{CITY}_da_divergence.gpkg
    outputs/figures/{CITY}_da_divergence_2x2.png
    outputs/figures/{CITY}_da_experience_2maps.png
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd
import numpy as np
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CITY = config.CITY

PARK_METRICS_PATH = "data/google-reviews/processed/08c-park-metrics.csv"
SUPPLY_PATH       = f"data/processed/{CITY}_da_supply.gpkg"
DA_PARK_SETS_PATH = f"data/processed/{CITY}_da_park_sets.json"
DB_PATH           = f"data/census/processed/{CITY}_db_centroids.gpkg"
OUT_DIR           = "data/processed"
FIG_DIR           = "outputs/figures"

REACH_THRESH = config.SUPPLY_REACH_THRESH


# %% 1. LOAD PARK METRICS
#NOTE: for other citirs, EXPERIENCE_SOURCE should be "star_rating" not "sentiment" fields.

park_metrics = pd.read_csv(PARK_METRICS_PATH)
park_metrics = park_metrics.rename(columns={"MeanSentiment": "AvgSentiment"})
park_metrics = park_metrics[park_metrics["park_id"].notna()].copy()

print(f"Parks loaded:                    {len(park_metrics)}")
print(f"Parks with valid sentiment:      {park_metrics['has_valid_sentiment'].sum()}") 
print(f"AvgSentiment summary:")
print(park_metrics["AvgSentiment"].describe().round(3))
print(f"\nAvgRating summary:")
print(park_metrics["AvgRating"].describe().round(3))
print(f"\nExperience source: {config.EXPERIENCE_SOURCE}")


# %% 2. LOAD OR REBUILD DA PARK SETS
if os.path.exists(DA_PARK_SETS_PATH):
    print(f"\nLoading da_park_sets from disk ({DA_PARK_SETS_PATH})...")
    with open(DA_PARK_SETS_PATH, "r") as f:
        da_park_sets = {k: set(v) for k, v in json.load(f).items()}
    print(f"Loaded. DAs with reachable parks: {len(da_park_sets)}")
else:
    print("da_park_sets not found — rebuilding (5–15 min)...")
    import osmnx as ox
    import networkx as nx

    GRAPH_PATH = f'data/osm/{CITY}_walk.graphml'
    ENT_PATH   = f'data/parks/processed/{CITY}_park_entrances.shp'

    G            = ox.load_graphml(GRAPH_PATH)
    entrances    = gpd.read_file(ENT_PATH)
    db_centroids = gpd.read_file(DB_PATH)

    node_col = 'nearest_no' if 'nearest_no' in entrances.columns else 'nearest_node'
    node_to_parks = {}
    for _, row in entrances.iterrows():
        node = row[node_col]
        if pd.notna(node):
            node_to_parks.setdefault(int(node), set()).add(row['park_id'])

    entrance_node_set = set(node_to_parks.keys())
    THRESHOLD = config.REACH_PRIMARY
    da_park_sets = {}

    for _, db_row in db_centroids.iterrows():
        dauid  = db_row['DAUID']
        db_pop = int(db_row['db_pop'])
        if db_pop == 0 or pd.isna(db_row['nearest_node']):
            continue
        db_node = int(db_row['nearest_node'])
        try:
            distances = nx.single_source_dijkstra_path_length(
                G, db_node, cutoff=THRESHOLD, weight='length'
            )
        except nx.NodeNotFound:
            continue
        reachable_parks = set()
        for node in set(distances.keys()) & entrance_node_set:
            reachable_parks.update(node_to_parks[node])
        da_park_sets.setdefault(dauid, set()).update(reachable_parks)

    print(f"Done. DAs with reachable parks: {len(da_park_sets)}")
    with open(DA_PARK_SETS_PATH, "w") as f:
        json.dump({k: list(v) for k, v in da_park_sets.items()}, f)
    print(f"Saved: {DA_PARK_SETS_PATH}")


# %% 3. COMPUTE DA-LEVEL EXPERIENCE EXPOSURE
db_centroids_gdf = gpd.read_file(DB_PATH)
da_pop_lookup    = db_centroids_gdf.groupby("DAUID")["db_pop"].sum().to_dict()
exp_index        = park_metrics.set_index("park_id")

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

    total_reviews = subset["TotalReviews"].sum()
    salience      = total_reviews / da_pop * 1000

    qualifying   = subset[subset["has_valid_sentiment"] == True]
    n_qualifying = len(qualifying)

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
da_experience.to_csv(f"{OUT_DIR}/{CITY}_da_experience.csv", index=False)

print(f"\nDAs with salience data:           {da_experience['salience'].notna().sum()}")
print(f"DAs with sentiment satisfaction:   {da_experience['satisfaction_sentiment'].notna().sum()}")
print(f"DAs with star satisfaction:        {da_experience['satisfaction_star'].notna().sum()}")


# %% 4. JOIN SUPPLY + EXPERIENCE AND CLASSIFY DIVERGENCE
da_supply = gpd.read_file(SUPPLY_PATH)
qty_med   = da_supply["qty_cap20"].median()

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
print(f"\nSupply typology: {da_supply['supply_type'].value_counts().to_dict()}")
print(f"Thresholds — reachability: {REACH_THRESH}, quantity median: {qty_med:.1f} ha/1,000")

# Join experience
da_div = da_supply.merge(da_experience, on="DAUID", how="left")

# Experience classification: select column based on EXPERIENCE_SOURCE
if config.EXPERIENCE_SOURCE == "sentiment":
    exp_col   = "satisfaction_sentiment"
    exp_label = "mean sentiment score"
elif config.EXPERIENCE_SOURCE == "star_rating":
    exp_col   = "satisfaction_star"
    exp_label = "mean star rating"
else:
    raise ValueError(f"Unknown EXPERIENCE_SOURCE: {config.EXPERIENCE_SOURCE!r}. "
                     "Set to 'sentiment' or 'star_rating' in config.py.")

exp_med = da_div[exp_col].median()
da_div["experience_hi"] = (da_div[exp_col] >= exp_med).astype(float)
da_div.loc[da_div[exp_col].isna(), "experience_hi"] = np.nan
print(f"\nExperience column: {exp_col}, median: {exp_med:.4f}")

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

counts_2x2 = da_div["divergence_2x2"].value_counts()
print(f"\n2x2 divergence distribution:")
print(counts_2x2)
if (counts_2x2.max() / len(da_div)) > 0.5:
    print("WARNING: >50% of DAs in one quadrant — check thresholds.")

da_div.to_file(f"{OUT_DIR}/{CITY}_da_divergence.gpkg", driver="GPKG")
print(f"\nSaved {CITY}_da_divergence.gpkg")


# %% 5. MAPS
colours_2x2 = config.COLOURS_2X2

parks_gdf  = gpd.read_file(f"data/parks/processed/{CITY}_parks_merged.shp")
full_labels = {
    "HH — High supply, high experience": colours_2x2["HH"],
    "LH — Low supply, high experience":  colours_2x2["LH"],
    "HL — High supply, low experience":  colours_2x2["HL"],
    "LL — Low supply, low experience":   colours_2x2["LL"],
    "No data":                           "#cccccc",
}

# 2x2 map
fig, ax = plt.subplots(figsize=(13, 10))
for dtype, colour in full_labels.items():
    subset = da_div[da_div["divergence_2x2"] == dtype]
    if len(subset):
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)
parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.8, zorder=2)
patches = [
    mpatches.Patch(color=full_labels[k],
                   label=f"{k} (n={counts_2x2.get(k, 0)})")
    for k in full_labels if k != "No data"
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)
ax.set_title(
    f"Supply–Experience Divergence — {CITY.title()} DAs\n"
    f"Supply: reachability ≥ {REACH_THRESH} & quantity ≥ median ({qty_med:.0f} ha/1,000) | "
    f"Experience: {exp_label} ≥ median ({exp_med:.3f})",
    fontsize=10
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_da_divergence_2x2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 2x2 divergence map.")

# Experience side-by-side
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

da_div_plot = da_div.copy()
da_div_plot["salience_log"] = np.log1p(da_div_plot["salience"])

da_div_plot.plot(ax=axes[0], column="salience_log", cmap="YlOrRd", legend=True,
    missing_kwds={"color": "#cccccc", "label": "No data"},
    legend_kwds={"label": "Log(reviews per 1,000 residents + 1)", "shrink": 0.5})
parks_gdf.plot(ax=axes[0], facecolor="none", edgecolor="#2d6a2d", linewidth=0.6, zorder=2)
axes[0].set_title("Digital Salience\nGoogle reviews per 1,000 residents (log)", fontsize=11)
axes[0].set_axis_off()

da_div.plot(ax=axes[1], column=exp_col, cmap="RdYlGn", legend=True,
    missing_kwds={"color": "#cccccc", "label": "No data"},
    legend_kwds={"label": f"Mean {exp_label}", "shrink": 0.5})
parks_gdf.plot(ax=axes[1], facecolor="none", edgecolor="#2d6a2d", linewidth=0.6, zorder=2)
axes[1].set_title(f"Expressed Satisfaction\n{exp_label.title()} across reachable parks", fontsize=11)
axes[1].set_axis_off()

plt.suptitle(f"Park Experience Dimensions — {CITY.title()} DAs (2021)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{CITY}_da_experience_2maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved experience 2-panel map.")

print("\nDone.")
