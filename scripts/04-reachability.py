"""
04-reachability.py
Computes network distances from DB centroids to park entrances,
then aggregates to DA-level reachability.

Method:
    Part A — Network distances (multi-source Dijkstra)
        1. Collect all unique entrance nodes as source set
        2. Run multi_source_dijkstra_path_length (cutoff=800m)
           → distance from every network node to nearest park entrance
        3. Look up each DB's nearest_node → distance to nearest entrance
        4. Assign reachable_400 and reachable_800 flags

    Part B — DA reachability aggregation
        5. DA_reachability = sum(DB_pop where reachable) / sum(DB_pop)
        6. Computed for both 400m and 800m thresholds

Inputs:
    - data/osm/Vancouver_walk.graphml                        (OSM network graph)
    - data/parks/processed/vancouver_park_entrances.shp      (park entrances with nearest_node)
    - data/census/processed/vancouver_db_centroids.gpkg      (DB centroids with nearest_node)

Outputs:
    - data/processed/vancouver_db_reachability.csv           (DB-level distances + flags)
    - data/processed/vancouver_da_reachability.csv           (DA-level reachability proportions)
    - outputs/figures/vancouver_da_reachability_check.png    (visual validation)

Notes:
    - Multi-source Dijkstra runs once from all entrance nodes simultaneously (efficient)
    - DBs whose nearest_node is not reached within 800m get distance = NaN,
      reachable_400 = 0, reachable_800 = 0
    - DBs with snap_flag = 1 (>200m snap) noted in output but not excluded
    - CRS: EPSG:3005 (BC Albers) throughout
"""

import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────

GRAPH_PATH    = 'data/osm/Vancouver_walk.graphml'
ENT_PATH      = 'data/parks/processed/vancouver_park_entrances.shp'
DB_PATH       = 'data/census/processed/vancouver_db_centroids.gpkg'
DA_PATH       = 'data/census/raw/lda_000b21a_e/lda_000b21a_e.shp'
OUT_DIR       = 'data/processed'
FIG_DIR       = 'outputs/figures'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLD_PRIMARY     = 400   # metres — main reachability threshold
THRESHOLD_SENSITIVITY = 800   # metres — sensitivity analysis threshold

# ── Step 1: Load inputs ───────────────────────────────────────────────────────

print("Step 1: Loading inputs...")
G           = ox.load_graphml(GRAPH_PATH)
entrances   = gpd.read_file(ENT_PATH)
db_centroids = gpd.read_file(DB_PATH)

print(f"  Graph nodes: {G.number_of_nodes():,}")
print(f"  Graph edges: {G.number_of_edges():,}")
print(f"  Park entrances: {len(entrances)}")
print(f"  DB centroids:   {len(db_centroids)}")

# ── Step 2: Collect entrance nodes ────────────────────────────────────────────

print("\nStep 2: Collecting entrance nodes...")
entrance_nodes = set(entrances['nearest_no'].dropna().astype(int).unique())
print(f"  Unique entrance nodes: {len(entrance_nodes)}")

# Validate all entrance nodes exist in graph
invalid_ent = entrance_nodes - set(G.nodes())
if invalid_ent:
    print(f"  WARNING: {len(invalid_ent)} entrance nodes not in graph — removing")
    entrance_nodes = entrance_nodes - invalid_ent
print(f"  Valid entrance nodes: {len(entrance_nodes)}")

# ── Step 3: Multi-source Dijkstra ─────────────────────────────────────────────

print(f"\nStep 3: Running multi-source Dijkstra (cutoff={THRESHOLD_SENSITIVITY}m)...")

dist_to_nearest = nx.multi_source_dijkstra_path_length(
    G,
    sources=entrance_nodes,
    cutoff=THRESHOLD_SENSITIVITY,
    weight='length'
)

print(f"  Nodes reached within {THRESHOLD_SENSITIVITY}m: {len(dist_to_nearest):,}")
print(f"  Total graph nodes: {G.number_of_nodes():,}")
print(f"  Nodes unreachable within {THRESHOLD_SENSITIVITY}m: "
      f"{G.number_of_nodes() - len(dist_to_nearest):,}")

# ── Step 4: Look up distance for each DB ──────────────────────────────────────

print("\nStep 4: Looking up distance for each DB centroid...")

# Drop any DBs with missing nearest_node (should be 0 based on script 03 validation)
n_before = len(db_centroids)
db_centroids = db_centroids.dropna(subset=['nearest_node']).copy()
db_centroids['nearest_node'] = db_centroids['nearest_node'].astype(int)
if len(db_centroids) < n_before:
    print(f"  WARNING: dropped {n_before - len(db_centroids)} DBs with missing nearest_node")

db_centroids['dist_nearest_entrance'] = db_centroids['nearest_node'].map(dist_to_nearest)

# Assign reachability flags explicitly (NaN distance → not reachable)
d = db_centroids['dist_nearest_entrance']
db_centroids['reachable_400'] = ((d.notna()) & (d <= THRESHOLD_PRIMARY)).astype(int)
db_centroids['reachable_800'] = ((d.notna()) & (d <= THRESHOLD_SENSITIVITY)).astype(int)

# DBs with no path within cutoff
n_unreached = db_centroids['dist_nearest_entrance'].isna().sum()
print(f"  DBs with distance found: {db_centroids['dist_nearest_entrance'].notna().sum()}")
print(f"  DBs unreachable within {THRESHOLD_SENSITIVITY}m: {n_unreached}")
print(f"  DBs reachable within {THRESHOLD_PRIMARY}m: {db_centroids['reachable_400'].sum()}")
print(f"  DBs reachable within {THRESHOLD_SENSITIVITY}m: {db_centroids['reachable_800'].sum()}")

# ── Step 5: Save DB-level output ──────────────────────────────────────────────

print("\nStep 5: Saving DB-level reachability...")
db_out = db_centroids[['DBUID', 'DAUID', 'db_pop', 'nearest_node',
                         'dist_nearest_entrance', 'reachable_400',
                         'reachable_800']].copy()
if 'snap_flag' in db_centroids.columns:
    db_out['snap_flag'] = db_centroids['snap_flag']

db_out['dist_nearest_entrance'] = db_out['dist_nearest_entrance'].round(1)

db_csv = os.path.join(OUT_DIR, 'vancouver_db_reachability.csv')
db_out.to_csv(db_csv, index=False)
print(f"  Saved: {db_csv}")

# ── Step 6: Aggregate to DA level ─────────────────────────────────────────────

print("\nStep 6: Aggregating to DA level...")

def aggregate_da(g):
    pop_total = g['db_pop'].sum()
    pop_400   = g.loc[g['reachable_400'] == 1, 'db_pop'].sum()
    pop_800   = g.loc[g['reachable_800'] == 1, 'db_pop'].sum()
    return pd.Series({
        'db_count':          len(g),
        'db_pop_total':      pop_total,
        'db_pop_reach_400':  pop_400,
        'db_pop_reach_800':  pop_800,
        'DA_reach_400':      pop_400 / pop_total if pop_total > 0 else None,
        'DA_reach_800':      pop_800 / pop_total if pop_total > 0 else None,
    })

da_reach = db_centroids.groupby('DAUID').apply(aggregate_da).reset_index()

print(f"  DAs in output: {len(da_reach)}")
print(f"\n  DA_reachability_400 summary:")
print(da_reach['DA_reach_400'].describe().round(3).to_string())
print(f"\n  DAs with 100% reachability (400m): {(da_reach['DA_reach_400'] == 1).sum()}")
print(f"  DAs with 0% reachability (400m):   {(da_reach['DA_reach_400'] == 0).sum()}")
print(f"  DAs with null reachability:         {da_reach['DA_reach_400'].isna().sum()}")

# ── Step 7: Save DA-level output ──────────────────────────────────────────────

print("\nStep 7: Saving DA-level reachability...")
da_csv = os.path.join(OUT_DIR, 'vancouver_da_reachability.csv')
da_reach.to_csv(da_csv, index=False)
print(f"  Saved: {da_csv}")

# ── Step 8: Visual validation ─────────────────────────────────────────────────

print("\nStep 8: Generating visual check...")

da_boundaries = gpd.read_file(DA_PATH).to_crs('EPSG:3005')
da_van = da_boundaries[da_boundaries['DAUID'].isin(da_reach['DAUID'])]
da_map = da_van.merge(da_reach[['DAUID', 'DA_reach_400']], on='DAUID', how='left')

fig, ax = plt.subplots(figsize=(10, 10))

# Custom colormap: red (0) → yellow → green (1)
cmap = LinearSegmentedColormap.from_list('reach', ['#d73027', '#fee08b', '#1a9850'])

da_map.plot(
    ax=ax, column='DA_reach_400', cmap=cmap,
    vmin=0, vmax=1, legend=True,
    missing_kwds={'color': '#cccccc', 'label': 'No population'},
    legend_kwds={'label': 'DA reachability (400m walking)', 'shrink': 0.5}
)

ax.set_title(f'DA-Level Park Reachability — Vancouver\n'
             f'Proportion of DB population within 400m of a park entrance')
ax.set_axis_off()
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, 'vancouver_da_reachability_check.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  Saved: {fig_path}")


#sensitivity check for 800m reachability
da_map = da_van.merge(da_reach[['DAUID', 'DA_reach_800']], on='DAUID', how='left')
fig, ax = plt.subplots(figsize=(10, 10))

# Custom colormap: red (0) → yellow → green (1)
cmap = LinearSegmentedColormap.from_list('reach', ['#d73027', '#fee08b', '#1a9850'])

da_map.plot(
    ax=ax, column='DA_reach_800', cmap=cmap,
    vmin=0, vmax=1, legend=True,
    missing_kwds={'color': '#cccccc', 'label': 'No population'},
    legend_kwds={'label': 'DA reachability (800m walking)', 'shrink': 0.5}
)

ax.set_title(f'DA-Level Park Reachability — Vancouver\n'
             f'Proportion of DB population within 800m of a park entrance')
ax.set_axis_off()
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, 'vancouver_da_reachability_check_800m.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  Saved: {fig_path}")

print("\nDone.")


zero_reach = da_reach[da_reach['DA_reach_400'] == 0]
print(zero_reach[['DAUID', 'db_count', 'db_pop_total']].to_string())

zero_reach = da_reach[da_reach['DA_reach_800'] == 0]
print(zero_reach[['DAUID', 'db_count', 'db_pop_total']].to_string())



parks = gpd.read_file('data/parks/processed/vancouver_parks_merged.shp')
da_map = da_van.merge(da_reach[['DAUID', 'DA_reach_400']], on='DAUID', how='left')
fig, ax = plt.subplots(figsize=(10, 10))

da_map.plot(
    ax=ax, column='DA_reach_400', cmap=cmap,
    vmin=0, vmax=1, legend=True,
    missing_kwds={'color': '#cccccc'},
    legend_kwds={'label': 'DA reachability (400m walking)', 'shrink': 0.5}
)

parks.plot(ax=ax, color='white', edgecolor='darkgreen',
           linewidth=0.5, alpha=0.6, zorder=2)

ax.set_title('DA-Level Park Reachability — Vancouver\n'
             'Proportion of DB population within 400m | white = park polygons')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('outputs/figures/vancouver_da_reachability_with_parks.png', dpi=150)
plt.close()




# export a DA shapefile with park reachability for potential use in ArcGIS or QGIS
import pandas as pd
import geopandas as gpd

# Load DA boundaries and reachability CSV
da_boundaries = gpd.read_file('data/census/raw/lda_000b21a_e/lda_000b21a_e.shp').to_crs('EPSG:3005')
da_reach = pd.read_csv('data/processed/vancouver_da_reachability.csv', dtype={'DAUID': str})

# Filter to Vancouver DAs
van_daids = da_reach['DAUID'].unique()
da_van = da_boundaries[da_boundaries['DAUID'].isin(van_daids)].copy()

# Join reachability variables
da_van = da_van.merge(da_reach, on='DAUID', how='left')

print(f"DAs in output: {len(da_van)}")
print(f"Columns: {list(da_van.columns)}")

# Save
import os
os.makedirs('data/processed', exist_ok=True)
da_van.to_file('data/processed/vancouver_da_reachability.gpkg', driver='GPKG')
print("Saved: data/processed/vancouver_da_reachability.gpkg")