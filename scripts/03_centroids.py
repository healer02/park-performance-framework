"""
03_centroids.py  (was 03-db-centroids.py)
Extracts DB-level representative centroids, joins population, snaps to OSM nodes.

Inputs:
    data/census/raw/ldb_000b21a_e/ldb_000b21a_e.shp   (StatCan DB boundaries)
    data/census/raw/2021_92-151_x.csv                  (StatCan GAF)
    data/osm/{CITY}_walk.graphml

Outputs:
    data/census/processed/{CITY}_db_centroids.gpkg
    outputs/figures/{CITY}_db_centroids_check.png

Notes:
    - Zero-population DBs retained and flagged
    - DBs with snap_dist_m > 200m flagged (likely water/industrial/network gaps)
    - nearest_node stored as int64 — GeoPackage avoids shapefile truncation
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
import matplotlib.pyplot as plt

CITY = config.CITY

GRAPH_PATH = f'data/osm/{CITY}_walk.graphml'
OUT_DIR    = 'data/census/processed'
FIG_DIR    = 'outputs/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Step 1: Load GAF ──────────────────────────────────────────────────────────

print("Step 1: Loading Geographic Attribute File...")
gaf = pd.read_csv(
    config.GAF_CSV,
    dtype=str,
    encoding='latin-1',
    usecols=['DAUID_ADIDU', 'DBUID_IDIDU', 'DBPOP2021_IDPOP2021', 'CSDUID_SDRIDU']
)
gaf_city = gaf[gaf['CSDUID_SDRIDU'] == config.STUDY_CSD].copy()
gaf_city['db_pop'] = pd.to_numeric(gaf_city['DBPOP2021_IDPOP2021'], errors='coerce').fillna(0).astype(int)

print(f"  {CITY.title()} DB rows in GAF: {len(gaf_city)}")
print(f"  Unique DBUIDs: {gaf_city['DBUID_IDIDU'].nunique()}")
print(f"  Total population: {gaf_city['db_pop'].sum():,}")
print(f"  Zero-population DBs: {(gaf_city['db_pop'] == 0).sum()}")

# ── Step 2: Load DB boundary polygons ─────────────────────────────────────────

print("\nStep 2: Loading DB boundary polygons...")
db_all = gpd.read_file(config.DB_SHP)
print(f"  Total DB polygons (Canada): {len(db_all)}")

city_dbuids = set(gaf_city['DBUID_IDIDU'].unique())
dbuid_cols  = [c for c in db_all.columns if 'DBUID' in c.upper()]
assert len(dbuid_cols) == 1, f"Expected 1 DBUID column, found: {dbuid_cols}"
dbuid_col = dbuid_cols[0]

db_city = db_all[db_all[dbuid_col].isin(city_dbuids)].copy()
print(f"  {CITY.title()} DB polygons: {len(db_city)}")

# ── Step 3: Reproject and compute centroids ───────────────────────────────────

print("\nStep 3: Computing DB centroids...")
db_city = db_city.to_crs(config.CRS)
db_city['geometry'] = db_city.geometry.centroid
db_city = db_city.rename(columns={dbuid_col: 'DBUID'})

# ── Step 4: Join population from GAF ─────────────────────────────────────────

print("\nStep 4: Joining DB population from GAF...")
pop_lookup = gaf_city[['DBUID_IDIDU', 'DAUID_ADIDU', 'db_pop']].rename(
    columns={'DBUID_IDIDU': 'DBUID', 'DAUID_ADIDU': 'DAUID'}
)
db_city = db_city.merge(pop_lookup, on='DBUID', how='left')
db_city['db_pop'] = db_city['db_pop'].fillna(0).astype(int)
print(f"  DBs with population joined: {db_city['db_pop'].notna().sum()}")
print(f"  Total population after join: {db_city['db_pop'].sum():,.0f}")

# ── Step 5: Snap DB centroids to nearest OSM node ─────────────────────────────

print("\nStep 5: Loading OSM graph and snapping DB centroids to nearest node...")
G = ox.load_graphml(GRAPH_PATH)

db_4326 = db_city.to_crs('EPSG:4326')
nearest_node_ids = ox.distance.nearest_nodes(
    G, db_4326.geometry.x.values, db_4326.geometry.y.values
)
db_city['nearest_node'] = nearest_node_ids

nodes_gdf = ox.graph_to_gdfs(G, edges=False)[['geometry']]
if nodes_gdf.crs != db_city.crs:
    nodes_gdf = nodes_gdf.to_crs(db_city.crs)
nodes_gdf = nodes_gdf.reset_index().set_index('osmid')

node_geoms = nodes_gdf.loc[db_city['nearest_node'].values, 'geometry'].values
db_city['snap_dist_m'] = [p1.distance(p2) for p1, p2 in zip(db_city.geometry.values, node_geoms)]

print(f"  Snap distance — mean:   {db_city['snap_dist_m'].mean():.1f}m")
print(f"  Snap distance — median: {db_city['snap_dist_m'].median():.1f}m")
print(f"  Snap distance — max:    {db_city['snap_dist_m'].max():.1f}m")
db_city['snap_flag'] = (db_city['snap_dist_m'] > 200).astype(int)
n_flagged = db_city['snap_flag'].sum()
if n_flagged > 0:
    print(f"  WARNING: {n_flagged} DBs snap >200m — likely water/industrial/network gaps")
    print(db_city[db_city['snap_flag'] == 1][['DBUID', 'DAUID', 'db_pop', 'snap_dist_m']].to_string())

# ── Step 6: Validation checks ─────────────────────────────────────────────────

print("\nStep 6: Validation checks...")
print(f"  DAs represented: {db_city['DAUID'].nunique()}")
print(f"  Total population: {db_city['db_pop'].sum():,}")
print(f"  Zero-pop DBs: {(db_city['db_pop'] == 0).sum()}")
valid_nodes = set(G.nodes())
invalid = db_city[~db_city['nearest_node'].isin(valid_nodes)]
print(f"  DBs with invalid nearest_node: {len(invalid)}")
assert db_city.crs.to_epsg() == int(config.CRS.split(':')[1]), f"CRS mismatch: {db_city.crs}"
print(f"  CRS: {config.CRS} ✓")

# ── Step 7: Save output ───────────────────────────────────────────────────────

print("\nStep 7: Saving DB centroids...")
out_cols = ['DBUID', 'DAUID', 'db_pop', 'nearest_node', 'snap_dist_m', 'snap_flag', 'geometry']
out_path = os.path.join(OUT_DIR, f'{CITY}_db_centroids.gpkg')
db_city[out_cols].to_file(out_path, driver='GPKG')
print(f"  Saved: {out_path} ({len(db_city)} features)")

# ── Step 8: Visual validation ─────────────────────────────────────────────────

print("\nStep 8: Generating visual check...")
da_all  = gpd.read_file(config.DA_SHP).to_crs(config.CRS)
da_city = da_all[da_all['DAUID'].isin(db_city['DAUID'].unique())]

fig, ax = plt.subplots(figsize=(10, 10))
da_city.plot(ax=ax, color='#f0f0f0', edgecolor='#aaaaaa', linewidth=0.4)
db_city.plot(
    ax=ax, column='db_pop', cmap='OrRd', markersize=3, alpha=0.85,
    vmin=db_city['db_pop'].quantile(0.05),
    vmax=db_city['db_pop'].quantile(0.95),
    legend=True,
    legend_kwds={'label': 'DB population (2021)', 'shrink': 0.5}
)
ax.set_title(f'DB Centroids — {CITY.title()}\n'
             f'n={len(db_city):,} DBs, coloured by 2021 population')
ax.set_axis_off()
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, f'{CITY}_db_centroids_check.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  Saved: {fig_path}")

print("\nDone.")
