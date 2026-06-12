"""
05_supply.py  (was 05-quantity.py)
Computes DA-level park quantity (reachable green space supply per capita).

Method:
    Single-source Dijkstra per DB centroid (cutoff = REACH_PRIMARY).
    Union of reachable park_ids per DA.
    quantity = total_unique_area / DA_pop * 1000  (ha per 1,000 residents).
    Area capped at AREA_CAP_MAIN (primary) and AREA_CAP_SENS (sensitivity).

Inputs:
    data/osm/{CITY}_walk.graphml
    data/parks/processed/{CITY}_park_entrances.shp
    data/parks/processed/{CITY}_parks_merged.shp
    data/census/processed/{CITY}_db_centroids.gpkg
    data/processed/{CITY}_da_reachability.gpkg

Outputs:
    data/processed/{CITY}_da_park_sets.json
    data/processed/{CITY}_da_quantity.csv
    data/processed/{CITY}_da_supply.gpkg
    outputs/figures/{CITY}_da_quantity_check2.png
    outputs/figures/{CITY}_da_supply_2x2.png
    outputs/figures/{CITY}_da_supply_typology.png

Notes:
    - This step takes 30–60 min. da_park_sets.json is cached;
      delete only if network or entrance data has changed.
"""

import os
import sys
import json

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

CITY = config.CITY

GRAPH_PATH    = f'data/osm/{CITY}_walk.graphml'
ENT_PATH      = f'data/parks/processed/{CITY}_park_entrances.shp'
PARKS_PATH    = f'data/parks/processed/{CITY}_parks_merged.shp'
DB_PATH       = f'data/census/processed/{CITY}_db_centroids.gpkg'
REACH_PATH    = f'data/processed/{CITY}_da_reachability.gpkg'
SETS_PATH     = f'data/processed/{CITY}_da_park_sets.json'
OUT_DIR       = 'data/processed'
FIG_DIR       = 'outputs/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLD = config.REACH_PRIMARY
CAP_MAIN  = config.AREA_CAP_MAIN
CAP_SENS  = config.AREA_CAP_SENS

# %% Step 1: Load inputs
print("Step 1: Loading inputs...")
G            = ox.load_graphml(GRAPH_PATH)
entrances    = gpd.read_file(ENT_PATH)
parks        = gpd.read_file(PARKS_PATH)
db_centroids = gpd.read_file(DB_PATH)
print(f"  Graph nodes:    {G.number_of_nodes():,}")
print(f"  Park entrances: {len(entrances)}")
print(f"  Parks:          {len(parks)}")
print(f"  DB centroids:   {len(db_centroids)}")

# %% Step 2: Build entrance node → park lookup
print("\nStep 2: Building entrance node → park lookup...")
node_col = 'nearest_no' if 'nearest_no' in entrances.columns else 'nearest_node'
node_to_parks = {}
for _, row in entrances.iterrows():
    node = row[node_col]
    pid  = row['park_id']
    if pd.notna(node):
        node = int(node)
        node_to_parks.setdefault(node, set()).add(pid)

park_area         = parks.set_index('park_id')['area_ha'].to_dict()
entrance_node_set = set(node_to_parks.keys())
print(f"  Unique entrance nodes mapped: {len(node_to_parks)}")

# %% Step 3: DB loop (cached — skip if JSON exists)
if os.path.exists(SETS_PATH):
    print(f"\nStep 3: Loading cached DA park sets from {SETS_PATH}...")
    with open(SETS_PATH, 'r') as f:
        da_park_sets = {k: set(v) for k, v in json.load(f).items()}
    print(f"  Loaded. DAs with reachable parks: {len(da_park_sets)}")
else:
    print(f"\nStep 3: Computing DB-level reachable park sets (cutoff={THRESHOLD}m)...")
    print(f"  Processing {len(db_centroids)} DBs — 30–60 min...")

    da_park_sets = {}
    da_pop_total = {}
    da_pop_valid = {}
    n_processed  = 0

    for _, db_row in db_centroids.iterrows():
        dauid  = db_row['DAUID']
        db_pop = int(db_row['db_pop'])
        da_pop_total[dauid] = da_pop_total.get(dauid, 0) + db_pop

        if db_pop == 0 or pd.isna(db_row['nearest_node']):
            n_processed += 1
            continue

        da_pop_valid[dauid] = da_pop_valid.get(dauid, 0) + db_pop
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

    print(f"  Done. DAs with reachable parks: {len(da_park_sets)}")

    with open(SETS_PATH, 'w') as f:
        json.dump({str(k): sorted(list(v)) for k, v in da_park_sets.items()}, f, indent=2)
    print(f"  Saved: {SETS_PATH}")

# %% Step 4: Aggregate to DA level
print("\nStep 4: Computing DA-level quantity...")

db_centroids_fresh = gpd.read_file(DB_PATH)
da_pop_total = db_centroids_fresh.groupby('DAUID')['db_pop'].sum().to_dict()
da_pop_valid = (
    db_centroids_fresh[db_centroids_fresh['db_pop'] > 0]
    .groupby('DAUID')['db_pop'].sum().to_dict()
)

da_records = []
for dauid, pop_total in da_pop_total.items():
    park_ids  = da_park_sets.get(dauid, set())
    n_parks   = len(park_ids)
    pop_valid = da_pop_valid.get(dauid, 0)

    if pop_valid == 0:
        da_records.append({
            'DAUID': dauid, 'db_pop_total': pop_total, 'db_pop_valid': pop_valid,
            'n_unique_parks': n_parks,
            'area_raw': None, 'area_cap20': None, 'area_cap10': None,
            'qty_raw': None,  'qty_cap20': None,  'qty_cap10': None,
        })
    elif n_parks == 0:
        da_records.append({
            'DAUID': dauid, 'db_pop_total': pop_total, 'db_pop_valid': pop_valid,
            'n_unique_parks': 0,
            'area_raw': 0,    'area_cap20': 0,    'area_cap10': 0,
            'qty_raw': 0,     'qty_cap20': 0,     'qty_cap10': 0,
        })
    else:
        area_raw   = sum(park_area.get(pid, 0) for pid in park_ids)
        area_cap20 = sum(min(park_area.get(pid, 0), CAP_MAIN) for pid in park_ids)
        area_cap10 = sum(min(park_area.get(pid, 0), CAP_SENS) for pid in park_ids)
        da_records.append({
            'DAUID': dauid, 'db_pop_total': pop_total, 'db_pop_valid': pop_valid,
            'n_unique_parks': n_parks,
            'area_raw':   round(area_raw,   2),
            'area_cap20': round(area_cap20, 2),
            'area_cap10': round(area_cap10, 2),
            'qty_raw':    round(area_raw   / pop_valid * 1000, 4),
            'qty_cap20':  round(area_cap20 / pop_valid * 1000, 4),
            'qty_cap10':  round(area_cap10 / pop_valid * 1000, 4),
        })

da_quantity = pd.DataFrame(da_records)
print(f"  DAs in output: {len(da_quantity)}")
print(da_quantity['qty_cap20'].describe().round(3).to_string())

# %% Step 5: Save quantity CSV and supply GeoPackage
print("\nStep 5: Saving quantity CSV...")
csv_path = os.path.join(OUT_DIR, f'{CITY}_da_quantity.csv')
da_quantity.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

print("\nStep 6: Merging with reachability and saving supply GeoPackage...")
da_reach  = gpd.read_file(REACH_PATH)
da_supply = da_reach.merge(
    da_quantity[['DAUID', 'n_unique_parks', 'area_raw', 'area_cap20',
                 'area_cap10', 'qty_raw', 'qty_cap20', 'qty_cap10']],
    on='DAUID', how='left'
)
gpkg_path = os.path.join(OUT_DIR, f'{CITY}_da_supply.gpkg')
da_supply.to_file(gpkg_path, driver='GPKG')
print(f"  Saved: {gpkg_path}")

# %% Step 7: Visualisation
print("\nStep 7: Generating validation maps...")
cmap = LinearSegmentedColormap.from_list('quantity', ['#d73027', '#fee08b', '#1a9850'])

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, col, title in zip(
    axes,
    ['qty_cap20', 'qty_cap10'],
    [f'Quantity — capped at {CAP_MAIN} ha (main)',
     f'Quantity — capped at {CAP_SENS} ha (sensitivity)']
):
    da_supply.plot(
        ax=ax, column=col, cmap=cmap, legend=True,
        missing_kwds={'color': '#cccccc', 'label': 'No data'},
        legend_kwds={'label': 'Ha per 1,000 residents', 'shrink': 0.5}
    )
    parks.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5, alpha=0.7, zorder=2)
    ax.set_title(f'DA-Level Park Quantity — {CITY.title()}\n{title}')
    ax.set_axis_off()
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, f'{CITY}_da_quantity_check2.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  Saved: {fig_path}")

# Supply typology map
REACH_THRESH = config.SUPPLY_REACH_THRESH
qty_med      = da_supply['qty_cap20'].median()

da_supply['reach_cat'] = (da_supply['DA_reach_400'] >= REACH_THRESH).astype(int)
da_supply['qty_cat']   = (da_supply['qty_cap20']    >= qty_med).astype(int)

def classify_supply(r, q):
    if pd.isna(r) or pd.isna(q): return 'No data'
    if r==1 and q==1: return 'HH — Broad coverage, high area'
    if r==1 and q==0: return 'HL — Broad coverage, low area'
    if r==0 and q==1: return 'LH — Limited coverage, high area'
    return 'LL — Limited coverage, low area'

da_supply['supply_type'] = [
    classify_supply(r, q)
    for r, q in zip(da_supply['reach_cat'], da_supply['qty_cat'])
]
counts = da_supply['supply_type'].value_counts()
print(f"\nSupply typology counts:")
print(counts)
print(f"Thresholds — reachability: {REACH_THRESH}, quantity median: {qty_med:.1f} ha/1,000")
if (counts.max() / len(da_supply)) > 0.5:
    print("WARNING: >50% of DAs in one supply quadrant — check thresholds.")

colours = {
    'HH — Broad coverage, high area':   '#4b3f44',
    'HL — Broad coverage, low area':    '#7ea6c2',
    'LH — Limited coverage, high area': '#d0a07a',
    'LL — Limited coverage, low area':  '#e9e2d8',
    'No data':                          '#cccccc',
}

fig, ax = plt.subplots(figsize=(12, 10))
for stype, colour in colours.items():
    subset = da_supply[da_supply['supply_type'] == stype]
    if len(subset) > 0:
        subset.plot(ax=ax, color=colour, edgecolor='white', linewidth=0.2)
parks.plot(ax=ax, facecolor='none', edgecolor='#2d6a2d', linewidth=1.0, zorder=2)
patches = [mpatches.Patch(color=c, label=f"{t} (n={counts.get(t, 0)})")
           for t, c in colours.items() if t != 'No data']
ax.legend(handles=patches, loc='lower left', fontsize=9, framealpha=0.9)
ax.set_title(
    f'Park Supply Typology — {CITY.title()} DAs\n'
    f'Thresholds: reachability ≥ {REACH_THRESH}, quantity ≥ median ({qty_med:.1f} ha/1,000)',
    fontsize=11)
ax.set_axis_off()
plt.tight_layout()
fig_path2 = os.path.join(FIG_DIR, f'{CITY}_da_supply_typology.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path2}")

print("\nDone.")
