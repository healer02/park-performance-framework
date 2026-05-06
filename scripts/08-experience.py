# =============================================================================
# 08-experience.py
# Purpose: Compute DA-level experience exposure from reachable park metrics
#
# Inputs:
#   - data/parks/processed/06-master-park-placeids.csv
#   - data/google-reviews/processed/10-avgsentiment-avgrating.csv
#   - data/google-reviews/processed/09a-all-place-metadata.csv
#   - data/osm/Vancouver_walk.graphml
#   - data/parks/processed/vancouver_park_entrances.shp
#   - data/census/processed/vancouver_db_centroids.gpkg
#   - data/processed/vancouver_da_supply.gpkg   (for final join)
#
# Output:
#   - data/processed/vancouver_da_experience.csv
#   - data/processed/vancouver_da_divergence.gpkg
#   - outputs/figures/vancouver_da_divergence_prototype.png
# =============================================================================

# %% 1. IMPORTS AND PATHS
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import numpy as np

MASTER_PATH    = "data/parks/processed/06-master-park-placeids.csv"
SENTIMENT_PATH = "data/google-reviews/processed/10-avgsentiment-avgrating.csv"
METADATA_PATH  = "data/google-reviews/processed/09a-all-place-metadata.csv"
GRAPH_PATH     = "data/osm/Vancouver_walk.graphml"
ENT_PATH       = "data/parks/processed/vancouver_park_entrances.shp"
DB_PATH        = "data/census/processed/vancouver_db_centroids.gpkg"
SUPPLY_PATH    = "data/processed/vancouver_da_supply.gpkg"

OUT_DIR = "data/processed"
FIG_DIR = "outputs/figures"

print("Ready.")


# %% 2. BUILD PARK-LEVEL EXPERIENCE METRICS
# Reuse same join logic from 07-divergence-pilot.py

master    = pd.read_csv(MASTER_PATH)
sentiment = pd.read_csv(SENTIMENT_PATH)
metadata  = pd.read_csv(METADATA_PATH)

master["place_id_list"] = master["place_id"].str.split(",").apply(
    lambda ids: [x.strip() for x in ids]
)
master_exploded = master.explode("place_id_list").rename(columns={"place_id_list": "PlaceID"})

joined = master_exploded.merge(sentiment, on="PlaceID", how="left")
joined = joined.merge(metadata[["PlaceID", "TotalReviews_All"]], on="PlaceID", how="left")

park_metrics = joined.groupby("park_id").agg(
    source       = ("source", "first"),
    park_name    = ("park_name", "first"),
    area_ha      = ("area_ha", "first"),
    AvgSentiment = ("AvgSentiment", "mean"),
    AvgRating    = ("AvgRating", "mean"),
    TotalReviews = ("TotalReviews_All", "sum"),
).reset_index()

# Keep only parks with experience data and >0 reviews
park_metrics = park_metrics[
    park_metrics["AvgSentiment"].notna() &
    (park_metrics["TotalReviews"] > 0)
].copy()

print(f"Parks with experience data: {len(park_metrics)}")
print(f"park_id sample: {park_metrics['park_id'].head(3).tolist()}")


# %% 3. REBUILD DA PARK SETS (network walk, ~5-15 min)
# Same logic as 05-quantity.py step 3
import geopandas as gpd
import osmnx as ox
import networkx as nx

print("Loading network and entrances...")
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

THRESHOLD = 400
da_park_sets = {}
n_processed  = 0

print(f"Running DB loop ({len(db_centroids)} DBs)... this takes 5-15 min")
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

# Save to disk so you don't have to rerun this again
import json
with open("data/processed/vancouver_da_park_sets.json", "w") as f:
    json.dump({k: list(v) for k, v in da_park_sets.items()}, f)
print("Saved da_park_sets to disk.")


# %% 4. COMPUTE DA-LEVEL EXPERIENCE EXPOSURE

print("Computing DA-level experience metrics...")
exp_index = park_metrics.set_index("park_id")

experience_records = []
for dauid, park_ids in da_park_sets.items():
    subset = exp_index[exp_index.index.isin(park_ids)]

    if len(subset) == 0:
        experience_records.append({
            "DAUID":         dauid,
            "exp_rating":    None,
            "exp_sentiment": None,
            "exp_salience":  None,
            "n_exp_parks":   0,
        })
        continue

    experience_records.append({
        "DAUID":         dauid,
        "exp_rating":    subset["AvgRating"].mean(),
        "exp_sentiment": subset["AvgSentiment"].mean(),
        "exp_salience":  np.log1p(subset["TotalReviews"]).sum(),
        "n_exp_parks":   len(subset),
    })

da_experience = pd.DataFrame(experience_records)
da_experience.to_csv(f"{OUT_DIR}/vancouver_da_experience.csv", index=False)

print(f"DAs with experience data: {da_experience['exp_rating'].notna().sum()}")
print(da_experience[["exp_rating", "exp_sentiment", "exp_salience"]].describe().round(3))


# %% 5. JOIN SUPPLY + EXPERIENCE AND CLASSIFY DIVERGENCE

da_supply = gpd.read_file(SUPPLY_PATH)
da_div = da_supply.merge(da_experience, on="DAUID", how="left")

# Supply: use existing typology thresholds from 05-quantity.py
REACH_THRESH = 0.8
qty_med      = da_div["qty_cap20"].median()

da_div["supply_hi"] = (
    (da_div["DA_reach_400"] >= REACH_THRESH) &
    (da_div["qty_cap20"]    >= qty_med)
).astype(int)

# Experience: median split on rating (primary satisfaction signal)
exp_rating_med = da_div["exp_rating"].median()
da_div["experience_hi"] = (da_div["exp_rating"] >= exp_rating_med).astype(int)

def classify_divergence(s, e):
    if pd.isna(s) or pd.isna(e): return "No data"
    if s == 1 and e == 1: return "HH — High supply, high experience"
    if s == 1 and e == 0: return "HL — High supply, low experience"
    if s == 0 and e == 1: return "LH — Low supply, high experience"
    return "LL — Low supply, low experience"

da_div["divergence_type"] = [
    classify_divergence(s, e)
    for s, e in zip(da_div["supply_hi"], da_div["experience_hi"])
]

print("\nDivergence quadrant distribution:")
print(da_div["divergence_type"].value_counts())

da_div.to_file(f"{OUT_DIR}/vancouver_da_divergence.gpkg", driver="GPKG")
print("Saved divergence GeoPackage.")


# %% 6. PROTOTYPE DIVERGENCE MAP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
colours = {
    "HH — High supply, high experience": "#2c7bb6",
    "HL — High supply, low experience":  "#d7191c",
    "LH — Low supply, high experience":  "#abd9e9",
    "LL — Low supply, low experience":   "#fdae61",
    "No data":                           "#cccccc",
}

parks_gdf = gpd.read_file("data/parks/processed/vancouver_parks_merged.shp")

fig, ax = plt.subplots(figsize=(13, 10))
for dtype, colour in colours.items():
    subset = da_div[da_div["divergence_type"] == dtype]
    if len(subset):
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)

parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.8, zorder=2)

counts = da_div["divergence_type"].value_counts()
patches = [
    mpatches.Patch(color=c, label=f"{t} (n={counts.get(t, 0)})")
    for t, c in colours.items() if t != "No data"
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)
ax.set_title(
    f"Supply–Experience Divergence — Vancouver DAs (PROTOTYPE)\n"
    f"Supply: reachability ≥ {REACH_THRESH} & quantity ≥ median ({qty_med:.1f} ha/1,000) | "
    f"Experience: mean reachable rating ≥ median ({exp_rating_med:.3f})",
    fontsize=10
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_da_divergence_prototype.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved prototype divergence map.")
# %%
