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
import geopandas as gpd

# Build DA population lookup from DB centroids
db_centroids = gpd.read_file(DB_PATH)
da_pop_lookup = (
    db_centroids.groupby("DAUID")["db_pop"]
    .sum()
    .to_dict()
)
print(f"DA population lookup built: {len(da_pop_lookup)} DAs")

# Build park metrics index for fast lookup
exp_index = park_metrics.set_index("park_id")

experience_records = []
for dauid, park_ids in da_park_sets.items():
    subset = exp_index[exp_index.index.isin(park_ids)]
    da_pop = da_pop_lookup.get(dauid, 0)

    if len(subset) == 0 or da_pop == 0:
        experience_records.append({
            "DAUID":                  dauid,
            "salience":               None,
            "satisfaction_sentiment": None,
            "satisfaction_star":      None,
            "coverage_pct":           None,
            "n_reachable_parks":      len(subset),
            "n_qualifying_parks":     0,
        })
        continue

    # SALIENCE: total reviews across all reachable parks, per 1,000 residents
    total_reviews = subset["TotalReviews"].sum()
    salience = total_reviews / da_pop * 1000

    # SATISFACTION: only parks with >=10 reviews (reduces noise)
    qualifying = subset[subset["TotalReviews"] >= 10]
    satisfaction_sentiment = qualifying["AvgSentiment"].mean() if len(qualifying) > 0 else None
    satisfaction_star      = qualifying["AvgRating"].mean()    if len(qualifying) > 0 else None
    coverage_pct           = len(qualifying) / len(subset) * 100

    experience_records.append({
        "DAUID":                  dauid,
        "salience":               round(salience, 4),
        "satisfaction_sentiment": round(satisfaction_sentiment, 4) if satisfaction_sentiment else None,
        "satisfaction_star":      round(satisfaction_star, 4)      if satisfaction_star      else None,
        "coverage_pct":           round(coverage_pct, 1),
        "n_reachable_parks":      len(subset),
        "n_qualifying_parks":     len(qualifying),
    })

da_experience = pd.DataFrame(experience_records)
da_experience.to_csv(f"{OUT_DIR}/vancouver_da_experience.csv", index=False)

print(f"\nDAs with salience data:               {da_experience['salience'].notna().sum()}")
print(f"DAs with satisfaction data:            {da_experience['satisfaction_sentiment'].notna().sum()}")
print(f"DAs with no qualifying parks (0 data): {(da_experience['n_qualifying_parks']==0).sum()}")
print(f"\nSalience summary (reviews per 1,000 residents):")
print(da_experience["salience"].describe().round(2))
print(f"\nSatisfaction summary (sentiment):")
print(da_experience["satisfaction_sentiment"].describe().round(3))
print(f"\nCoverage % summary:")
print(da_experience["coverage_pct"].describe().round(1))



# %% 5. JOIN SUPPLY + EXPERIENCE AND CLASSIFY DIVERGENCE
da_supply = gpd.read_file(SUPPLY_PATH)

# Compute supply typology (mirrors 05-quantity.py logic)
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

# Experience: sentiment-only binary
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

# Key problem cells (low experience only)
print(f"\nProblem cells (low experience):")
problem = da_div[da_div["experience_hi"]==0]["supply_type"].value_counts()
print(problem)

da_div.to_file(f"{OUT_DIR}/vancouver_da_divergence.gpkg", driver="GPKG")
print("\nSaved.")

# %% 6a. SIMPLE 2x2 DIVERGENCE MAP (binary supply x binary experience)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

# Collapse supply to binary: HH = high supply, everything else = low supply
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
    "HH — High supply, high experience": "#2c7bb6",
    "HL — High supply, low experience":  "#d7191c",
    "LH — Low supply, high experience":  "#abd9e9",
    "LL — Low supply, low experience":   "#fdae61",
    "No data":                           "#cccccc",
}

parks_gdf = gpd.read_file("data/parks/processed/vancouver_parks_merged.shp")
counts_2x2 = da_div["divergence_2x2"].value_counts()

fig, ax = plt.subplots(figsize=(13, 10))
for dtype, colour in colours_2x2.items():
    subset = da_div[da_div["divergence_2x2"] == dtype]
    if len(subset):
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)

parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.8, zorder=2)

patches = [
    mpatches.Patch(color=c, label=f"{t} (n={counts_2x2.get(t, 0)})")
    for t, c in colours_2x2.items() if t != "No data"
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)
ax.set_title(
    "Supply–Experience Divergence — Vancouver DAs\n"
    f"Supply: HH only (reachability ≥ {REACH_THRESH} & quantity ≥ {qty_med:.0f} ha/1,000) | "
    f"Experience: sentiment ≥ median ({sentiment_med:.2f})",
    fontsize=10
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_da_divergence_2x2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 2x2 map.")
print(counts_2x2)


# %% 6b. PROTOTYPE DIVERGENCE MAP (low experience cells only)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

HIGH_EXP_COLOUR = "#f0ebe3"

colours = {
    "HH_low_exp": "#4b3f44",   # dark -- strong supply failure, most concerning
    "HL_low_exp": "#7ea6c2",   # blue -- coverage without quality
    "LH_low_exp": "#d0a07a",   # warm -- area without access
    "LL_low_exp": "#c0392b",   # red -- compounded disadvantage, most urgent
    "HH_high_exp": HIGH_EXP_COLOUR,  # high experience -- all same colour
    "HL_high_exp": HIGH_EXP_COLOUR,
    "LH_high_exp": HIGH_EXP_COLOUR,
    "LL_high_exp": HIGH_EXP_COLOUR,
    "No data":     "#cccccc",
}

parks_gdf = gpd.read_file("data/parks/processed/vancouver_parks_merged.shp")
counts = da_div["divergence_type"].value_counts()

fig, ax = plt.subplots(figsize=(13, 10))
for dtype, colour in colours.items():
    subset = da_div[da_div["divergence_type"] == dtype]
    if len(subset):
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)

parks_gdf.plot(ax=ax, facecolor="none", edgecolor="#2d6a2d", linewidth=0.8, zorder=2)

# Legend: problem cells prominent, success cells minimal
problem_labels = {
    "HH_low_exp": "HH — Strong supply, low experience",
    "HL_low_exp": "HL — Coverage-oriented, low experience",
    "LH_low_exp": "LH — Area-oriented, low experience",
    "LL_low_exp": "LL — Weak supply, low experience",
}
patches = [
    mpatches.Patch(color=colours["HH_low_exp"], label=f"HH — Strong supply, low experience (n={counts.get('HH_low_exp', 0)})"),
    mpatches.Patch(color=colours["HL_low_exp"], label=f"HL — Coverage-oriented, low experience (n={counts.get('HL_low_exp', 0)})"),
    mpatches.Patch(color=colours["LH_low_exp"], label=f"LH — Area-oriented, low experience (n={counts.get('LH_low_exp', 0)})"),
    mpatches.Patch(color=colours["LL_low_exp"], label=f"LL — Weak supply, low experience (n={counts.get('LL_low_exp', 0)})"),
    mpatches.Patch(color=HIGH_EXP_COLOUR, label=f"High experience (n={sum(counts.get(k,0) for k in ['HH_high_exp','HL_high_exp','LH_high_exp','LL_high_exp'])})"),
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)

ax.set_title(
    f"Supply–Experience Divergence — Vancouver DAs\n"
    f"Supply: reachability ≥ {REACH_THRESH} & quantity ≥ median ({qty_med:.0f} ha/1,000) | "
    f"Experience: mean sentiment ≥ median ({sentiment_med:.2f})",
    fontsize=10
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_da_divergence_prototype.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved map.")




# %% SENSITIVITY: Review-weighted sentiment (drop this anaysis)
import numpy as np

# no park-level weighted variable needed

# DA-level weighted sentiment (parks >=10 reviews only)
exp_index_w = park_metrics.set_index("park_id")

sensitivity_records = []
for dauid, park_ids in da_park_sets.items():
    subset = exp_index_w[exp_index_w.index.isin(park_ids)]
    qualifying = subset[subset["TotalReviews"] >= 10]

    if len(qualifying) == 0:
        sensitivity_records.append({"DAUID": dauid, "exp_sentiment_weighted": None})
        continue

    weighted_mean = np.average(
    qualifying["AvgSentiment"],
    weights=np.log1p(qualifying["TotalReviews"])
)

sensitivity_records.append({
    "DAUID": dauid,
    "exp_sentiment_weighted": weighted_mean,
})

da_sensitivity = pd.DataFrame(sensitivity_records)

# Join to divergence output
da_div_s = da_div.merge(da_sensitivity, on="DAUID", how="left")

# Reclassify using weighted sentiment
weighted_med = da_div_s["exp_sentiment_weighted"].median()
da_div_s["experience_hi_weighted"] = (
    da_div_s["exp_sentiment_weighted"] >= weighted_med
).astype(int)

da_div_s["divergence_type_weighted"] = [
    classify_divergence(s, e)
    for s, e in zip(da_div_s["supply_type"], da_div_s["experience_hi_weighted"])
]

# Compare: how many DAs change quadrant?
changed = (da_div_s["divergence_type"] != da_div_s["divergence_type_weighted"]).sum()
total   = da_div_s["divergence_type"].notna().sum()
print(f"DAs changing quadrant: {changed} / {total} ({100*changed/total:.1f}%)")

print(f"\nPrimary distribution:")
print(da_div_s["divergence_type"].value_counts().sort_index())

print(f"\nWeighted sentiment distribution:")
print(da_div_s["divergence_type_weighted"].value_counts().sort_index())

print(f"\nWeighted sentiment median: {weighted_med:.3f}")
print(f"Primary sentiment median:  {sentiment_med:.3f}")






# %%
census = pd.read_csv("/Users/keunpark/Documents/GitHub/social-sentiment-score/data/census/raw/parks_census_CANUE_Apportion_update2.csv")
print(census.shape)
print(census.columns.tolist())

# %%
da_census = pd.read_csv("/Users/keunpark/SynologyDrive/Research projects/park vitality and monitoring/social potentials/R/social_sentiment/census_CANUE_DA_nearVan.csv")
print(da_census.shape)
print(da_census.columns.tolist())
print(da_census.head(3))
# %%
