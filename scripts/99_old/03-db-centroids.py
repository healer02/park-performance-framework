"""
03-db-centroids.py
Extracts DB-level representative centroids (geometric) for Vancouver study area.

Method:
    1. Load DB boundary polygons (ldb_000b21a_e.shp)
    2. Filter to Vancouver DAs (CSDUID 5915022)
    3. Compute geometric centroids from DB polygons
       Note: geometric centroid used as representative point;
       DBs are small enough (~1 city block) that centroid approximation
       introduces negligible positional error for 400m network analysis
    4. Join DB population from GAF (DBUID → DBPOP2021)
    5. Snap each DB centroid to nearest OSM node
    6. Save as GeoPackage (avoids shapefile integer truncation for osmid)

Inputs:
    - data/census/raw/ldb_000b21a_e/ldb_000b21a_e.shp   (StatCan DB boundaries)
    - data/census/raw/2021_92-151_x.csv                  (StatCan GAF)
    - data/osm/Vancouver_walk.graphml                    (OSM network graph)

Outputs:
    - data/census/processed/vancouver_db_centroids.gpkg  (DB centroids, EPSG:3005)
    - outputs/figures/vancouver_db_centroids_check.png   (visual validation)

Notes:
    - nearest_node stored as int64 — use GeoPackage not shapefile to avoid truncation
    - Zero-population DBs retained (institutional/commercial blocks); flagged in output
    - DBs with snap_dist_m > 200m flagged: likely water bodies, industrial zones,
      or network gaps — review before reachability analysis
    - CRS: EPSG:3005 (BC Albers) throughout
"""

import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────

DB_PATH    = 'data/census/raw/ldb_000b21a_e/ldb_000b21a_e.shp'
GAF_PATH   = 'data/census/raw/2021_92-151_x.csv'
GRAPH_PATH = 'data/osm/Vancouver_walk.graphml'
OUT_DIR    = 'data/census/processed'
FIG_DIR    = 'outputs/figures'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

STUDY_CSD = '5915022'  # City of Vancouver

# ── Step 1: Load GAF — get Vancouver DBUIDs and populations ───────────────────

print("Step 1: Loading Geographic Attribute File...")
gaf = pd.read_csv(
    GAF_PATH,
    dtype=str,
    encoding='latin-1',
    usecols=['DAUID_ADIDU', 'DBUID_IDIDU', 'DBPOP2021_IDPOP2021', 'CSDUID_SDRIDU']
)
gaf_van = gaf[gaf['CSDUID_SDRIDU'] == STUDY_CSD].copy()
gaf_van['db_pop'] = pd.to_numeric(gaf_van['DBPOP2021_IDPOP2021'], errors='coerce').fillna(0).astype(int)

print(f"  Vancouver DB rows in GAF: {len(gaf_van)}")
print(f"  Unique DBUIDs: {gaf_van['DBUID_IDIDU'].nunique()}")
print(f"  Total population: {gaf_van['db_pop'].sum():,}")
print(f"  Zero-population DBs: {(gaf_van['db_pop'] == 0).sum()}")

# ── Step 2: Load DB boundary polygons ─────────────────────────────────────────

print("\nStep 2: Loading DB boundary polygons...")
db_all = gpd.read_file(DB_PATH)
print(f"  Total DB polygons (Canada): {len(db_all)}")
print(f"  Columns: {list(db_all.columns)}")

# Filter to Vancouver DBs using DBUID
van_dbuids = set(gaf_van['DBUID_IDIDU'].unique())

# DB boundary file uses DBUID column — confirm column name
dbuid_cols = [c for c in db_all.columns if 'DBUID' in c.upper()]
assert len(dbuid_cols) == 1, f"Expected 1 DBUID column, found: {dbuid_cols}"
dbuid_col = dbuid_cols[0]
print(f"  DBUID column in shapefile: {dbuid_col}")

db_van = db_all[db_all[dbuid_col].isin(van_dbuids)].copy()
print(f"  Vancouver DB polygons: {len(db_van)}")
print(f"  Source CRS: {db_van.crs}")

# ── Step 3: Reproject and compute centroids ───────────────────────────────────

print("\nStep 3: Computing DB centroids...")
db_van = db_van.to_crs('EPSG:3005')
db_van['geometry'] = db_van.geometry.centroid
db_van = db_van.rename(columns={dbuid_col: 'DBUID'})
print(f"  Centroids computed: {len(db_van)}")

# ── Step 4: Join population from GAF ─────────────────────────────────────────

print("\nStep 4: Joining DB population from GAF...")
pop_lookup = gaf_van[['DBUID_IDIDU', 'DAUID_ADIDU', 'db_pop']].rename(
    columns={'DBUID_IDIDU': 'DBUID', 'DAUID_ADIDU': 'DAUID'}
)
db_van = db_van.merge(pop_lookup, on='DBUID', how='left')

# Check join quality
missing_pop = db_van['db_pop'].isna().sum()
print(f"  DBs with population joined: {db_van['db_pop'].notna().sum()}")
print(f"  DBs missing population: {missing_pop}")
print(f"  Total population after join: {db_van['db_pop'].sum():,.0f}")

db_van['db_pop'] = db_van['db_pop'].fillna(0).astype(int)

# ── Step 5: Snap DB centroids to nearest OSM node ─────────────────────────────

print("\nStep 5: Loading OSM graph and snapping DB centroids to nearest node...")
G = ox.load_graphml(GRAPH_PATH)

# Convert to EPSG:4326 for ox.distance.nearest_nodes
db_4326 = db_van.to_crs('EPSG:4326')
xs = db_4326.geometry.x.values
ys = db_4326.geometry.y.values

nearest_node_ids = ox.distance.nearest_nodes(G, xs, ys)
db_van['nearest_node'] = nearest_node_ids
print(f"  Snapped {len(db_van)} DB centroids to OSM nodes")

# Compute snap distances — vectorised
# Explicitly set index to osmid to ensure correct lookup
nodes_gdf = ox.graph_to_gdfs(G, edges=False)[['geometry']]
if nodes_gdf.crs != db_van.crs:
    nodes_gdf = nodes_gdf.to_crs(db_van.crs)
nodes_gdf = nodes_gdf.reset_index().set_index('osmid')

node_geoms = nodes_gdf.loc[db_van['nearest_node'].values, 'geometry'].values
db_geoms   = db_van.geometry.values
db_van['snap_dist_m'] = [p1.distance(p2) for p1, p2 in zip(db_geoms, node_geoms)]

print(f"  Snap distance — mean:   {db_van['snap_dist_m'].mean():.1f}m")
print(f"  Snap distance — median: {db_van['snap_dist_m'].median():.1f}m")
print(f"  Snap distance — max:    {db_van['snap_dist_m'].max():.1f}m")
db_van['snap_flag'] = (db_van['snap_dist_m'] > 200).astype(int)
n_flagged = db_van['snap_flag'].sum()
if n_flagged > 0:
    print(f"  WARNING: {n_flagged} DBs snap >200m — likely water/industrial/network gaps")
    print(db_van[db_van['snap_flag'] == 1][['DBUID', 'DAUID', 'db_pop', 'snap_dist_m']].to_string())

# ── Step 6: Validation checks ─────────────────────────────────────────────────

print("\nStep 6: Validation checks...")

# DA coverage
da_count = db_van['DAUID'].nunique()
print(f"  DAs represented: {da_count}")

# Population check
print(f"  Total population: {db_van['db_pop'].sum():,}")
print(f"  Zero-pop DBs: {(db_van['db_pop'] == 0).sum()}")

# Node validity
valid_nodes = set(G.nodes())
invalid = db_van[~db_van['nearest_node'].isin(valid_nodes)]
print(f"  DBs with invalid nearest_node: {len(invalid)}")

assert db_van.crs.to_epsg() == 3005, "CRS is not EPSG:3005"
print(f"  CRS: EPSG:{db_van.crs.to_epsg()} ✓")

# ── Step 7: Save output ───────────────────────────────────────────────────────

print("\nStep 7: Saving DB centroids...")
out_cols = ['DBUID', 'DAUID', 'db_pop', 'nearest_node', 'snap_dist_m', 'snap_flag', 'geometry']
out_path = os.path.join(OUT_DIR, 'vancouver_db_centroids.gpkg')
db_van[out_cols].to_file(out_path, driver='GPKG')
print(f"  Saved: {out_path}")
print(f"  Features: {len(db_van)}")

# ── Step 8: Visual validation ─────────────────────────────────────────────────

print("\nStep 8: Generating visual check...")

# Load DA boundaries for context
da_path = 'data/census/raw/lda_000b21a_e/lda_000b21a_e.shp'
da_all  = gpd.read_file(da_path).to_crs('EPSG:3005')
da_van  = da_all[da_all['DAUID'].isin(db_van['DAUID'].unique())]

fig, ax = plt.subplots(figsize=(10, 10))
da_van.plot(ax=ax, color='#f0f0f0', edgecolor='#aaaaaa', linewidth=0.4)
db_van.plot(
    ax=ax, column='db_pop', cmap='OrRd', markersize=3, alpha=0.85,
    vmin=db_van['db_pop'].quantile(0.05),
    vmax=db_van['db_pop'].quantile(0.95),
    legend=True,
    legend_kwds={'label': 'DB population (2021)', 'shrink': 0.5}
)
ax.set_title(f'DB Centroids — Vancouver\n'
             f'n={len(db_van):,} DBs, coloured by 2021 population')
ax.set_axis_off()
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, 'vancouver_db_centroids_check.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  Saved: {fig_path}")

print("\nDone.")
