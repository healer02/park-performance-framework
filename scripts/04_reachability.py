"""
04_reachability.py  (was 04-reachability.py)
Computes network distances from DB centroids to park entrances,
then aggregates to DA-level reachability.

Method:
    Multi-source Dijkstra from all entrance nodes (cutoff = REACH_SENSITIVITY).
    Each DB looks up its nearest_node in the distance dict.
    DA reachability = sum(DB_pop where reachable) / sum(DB_pop).

Inputs:
    data/osm/{CITY}_walk.graphml
    data/parks/processed/{CITY}_park_entrances.shp
    data/census/processed/{CITY}_db_centroids.gpkg
    data/census/raw/lda_000b21a_e/lda_000b21a_e.shp

Outputs:
    data/processed/{CITY}_db_reachability.csv
    data/processed/{CITY}_da_reachability.csv
    data/processed/{CITY}_da_reachability.gpkg
    outputs/figures/{CITY}_da_reachability_check.png
    outputs/figures/{CITY}_da_reachability_check_800m.png
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CITY = config.CITY

GRAPH_PATH = f'data/osm/{CITY}_walk.graphml'
ENT_PATH   = f'data/parks/processed/{CITY}_park_entrances.shp'
DB_PATH    = f'data/census/processed/{CITY}_db_centroids.gpkg'
OUT_DIR    = 'data/processed'
FIG_DIR    = 'outputs/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLD_PRIMARY     = config.REACH_PRIMARY
THRESHOLD_SENSITIVITY = config.REACH_SENSITIVITY

# ── Step 1: Load inputs ───────────────────────────────────────────────────────

print("Step 1: Loading inputs...")
G            = ox.load_graphml(GRAPH_PATH)
entrances    = gpd.read_file(ENT_PATH)
db_centroids = gpd.read_file(DB_PATH)
print(f"  Graph nodes: {G.number_of_nodes():,}")
print(f"  Park entrances: {len(entrances)}")
print(f"  DB centroids:   {len(db_centroids)}")

# ── Step 2: Collect entrance nodes ────────────────────────────────────────────

print("\nStep 2: Collecting entrance nodes...")
node_col = 'nearest_no' if 'nearest_no' in entrances.columns else 'nearest_node'
entrance_nodes = set(entrances[node_col].dropna().astype(int).unique())
invalid_ent = entrance_nodes - set(G.nodes())
if invalid_ent:
    print(f"  WARNING: {len(invalid_ent)} entrance nodes not in graph — removing")
    entrance_nodes = entrance_nodes - invalid_ent
print(f"  Valid entrance nodes: {len(entrance_nodes)}")

# ── Step 3: Multi-source Dijkstra ─────────────────────────────────────────────

print(f"\nStep 3: Running multi-source Dijkstra (cutoff={THRESHOLD_SENSITIVITY}m)...")
dist_to_nearest = nx.multi_source_dijkstra_path_length(
    G, sources=entrance_nodes, cutoff=THRESHOLD_SENSITIVITY, weight='length'
)
print(f"  Nodes reached within {THRESHOLD_SENSITIVITY}m: {len(dist_to_nearest):,}")

# ── Step 4: Look up distance for each DB ──────────────────────────────────────

print("\nStep 4: Looking up distance for each DB centroid...")
n_before = len(db_centroids)
db_centroids = db_centroids.dropna(subset=['nearest_node']).copy()
db_centroids['nearest_node'] = db_centroids['nearest_node'].astype(int)
if len(db_centroids) < n_before:
    print(f"  WARNING: dropped {n_before - len(db_centroids)} DBs with missing nearest_node")

db_centroids['dist_nearest_entrance'] = db_centroids['nearest_node'].map(dist_to_nearest)
d = db_centroids['dist_nearest_entrance']
db_centroids['reachable_400'] = ((d.notna()) & (d <= THRESHOLD_PRIMARY)).astype(int)
db_centroids['reachable_800'] = ((d.notna()) & (d <= THRESHOLD_SENSITIVITY)).astype(int)

print(f"  DBs reachable within {THRESHOLD_PRIMARY}m: {db_centroids['reachable_400'].sum()}")
print(f"  DBs reachable within {THRESHOLD_SENSITIVITY}m: {db_centroids['reachable_800'].sum()}")
print(f"  DBs unreachable: {db_centroids['dist_nearest_entrance'].isna().sum()}")

# ── Step 5: Save DB-level output ──────────────────────────────────────────────

print("\nStep 5: Saving DB-level reachability...")
db_out = db_centroids[['DBUID', 'DAUID', 'db_pop', 'nearest_node',
                         'dist_nearest_entrance', 'reachable_400', 'reachable_800']].copy()
if 'snap_flag' in db_centroids.columns:
    db_out['snap_flag'] = db_centroids['snap_flag']
db_out['dist_nearest_entrance'] = db_out['dist_nearest_entrance'].round(1)
db_csv = os.path.join(OUT_DIR, f'{CITY}_db_reachability.csv')
db_out.to_csv(db_csv, index=False)
print(f"  Saved: {db_csv}")

# ── Step 6: Aggregate to DA level ─────────────────────────────────────────────

print("\nStep 6: Aggregating to DA level...")

def aggregate_da(g):
    pop_total = g['db_pop'].sum()
    pop_400   = g.loc[g['reachable_400'] == 1, 'db_pop'].sum()
    pop_800   = g.loc[g['reachable_800'] == 1, 'db_pop'].sum()
    return pd.Series({
        'db_count':         len(g),
        'db_pop_total':     pop_total,
        'db_pop_reach_400': pop_400,
        'db_pop_reach_800': pop_800,
        'DA_reach_400':     pop_400 / pop_total if pop_total > 0 else None,
        'DA_reach_800':     pop_800 / pop_total if pop_total > 0 else None,
    })

da_reach = db_centroids.groupby('DAUID').apply(aggregate_da).reset_index()
print(f"  DAs in output: {len(da_reach)}")
print(da_reach['DA_reach_400'].describe().round(3).to_string())
print(f"  DAs with 100% reachability (400m): {(da_reach['DA_reach_400'] == 1).sum()}")
print(f"  DAs with 0% reachability (400m):   {(da_reach['DA_reach_400'] == 0).sum()}")

# ── Step 7: Save DA-level CSVs and GeoPackage ────────────────────────────────

print("\nStep 7: Saving DA-level outputs...")
da_csv = os.path.join(OUT_DIR, f'{CITY}_da_reachability.csv')
da_reach.to_csv(da_csv, index=False)
print(f"  Saved: {da_csv}")

da_boundaries = gpd.read_file(config.DA_SHP).to_crs(config.CRS)
da_van        = da_boundaries[da_boundaries['DAUID'].isin(da_reach['DAUID'])].copy()
da_van        = da_van.merge(da_reach, on='DAUID', how='left')
gpkg_path     = os.path.join(OUT_DIR, f'{CITY}_da_reachability.gpkg')
da_van.to_file(gpkg_path, driver='GPKG')
print(f"  Saved: {gpkg_path}")

# ── Step 8: Validation maps ───────────────────────────────────────────────────

print("\nStep 8: Generating validation maps...")
cmap = LinearSegmentedColormap.from_list('reach', ['#d73027', '#fee08b', '#1a9850'])

for col, suffix, label in [
    ('DA_reach_400', '',     f'DA reachability ({THRESHOLD_PRIMARY}m walking)'),
    ('DA_reach_800', '_800m', f'DA reachability ({THRESHOLD_SENSITIVITY}m walking)'),
]:
    da_map = da_van.copy()
    fig, ax = plt.subplots(figsize=(10, 10))
    da_map.plot(
        ax=ax, column=col, cmap=cmap, vmin=0, vmax=1, legend=True,
        missing_kwds={'color': '#cccccc', 'label': 'No population'},
        legend_kwds={'label': label, 'shrink': 0.5}
    )
    ax.set_title(f'DA-Level Park Reachability — {CITY.title()}\n'
                 f'Proportion of DB population within {THRESHOLD_PRIMARY}m of a park entrance')
    ax.set_axis_off()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f'{CITY}_da_reachability_check{suffix}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

print("\nDone.")
