"""
01_network.py  (was 01-get-osm-network.py)
Downloads OSM pedestrian walk network for the study area.

Inputs:
    data/census/raw/2021_92-151_x.csv        (StatCan Geographic Attribute File)
    data/census/raw/lda_000b21a_e/lda_000b21a_e.shp  (DA boundaries)

Outputs:
    data/osm/{CITY}_walk.graphml
    data/osm/{CITY}_osm_nodes.shp
    data/osm/{CITY}_osm_edges.shp
    data/osm/{CITY}_study_area_boundary.shp
    outputs/figures/{CITY}_osm_network_check.png
    outputs/figures/{CITY}_da_points_check.png
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

OUTPUT_DIR = 'data/osm'
FIG_DIR    = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_CSDS = set(config.CSD_CODES)

# ── Step 1: Load Geographic Attribute File ────────────────────────────────────

print("Step 1: Loading Geographic Attribute File...")
gaf = pd.read_csv(
    config.GAF_CSV,
    dtype=str,
    encoding='latin-1',
    usecols=['DAUID_ADIDU', 'CSDUID_SDRIDU', 'CSDNAME_SDRNOM']
)
print(f"  GAF rows loaded: {len(gaf):,}")

# ── Step 2: Get target DAUIDs ─────────────────────────────────────────────────

target_daids = gaf[gaf['CSDUID_SDRIDU'].isin(TARGET_CSDS)]['DAUID_ADIDU'].unique()
print(f"  Target DAs identified: {len(target_daids)}")

# ── Step 3: Load and filter DA boundaries ─────────────────────────────────────

print("Step 3: Loading DA boundaries...")
da = gpd.read_file(config.DA_SHP)
study_area = da[da['DAUID'].isin(target_daids)].copy()
print(f"  DAs in study area: {len(study_area)}")
print(f"  Source CRS: {study_area.crs}")

# ── Step 4: Build merged boundary polygon ─────────────────────────────────────

study_area_4326 = study_area.to_crs('EPSG:4326')
boundary        = study_area_4326.unary_union
print(f"  Boundary bounds (lon/lat): {boundary.bounds}")

study_area_4326.dissolve().to_file(
    os.path.join(OUTPUT_DIR, f'{CITY}_study_area_boundary.shp')
)

# ── Step 5: Download OSM walk network ─────────────────────────────────────────

print("Step 5: Downloading OSM pedestrian network (1–3 minutes)...")
G = ox.graph_from_polygon(boundary, network_type='walk')

nodes, edges = ox.graph_to_gdfs(G)
print(f"  Nodes: {len(nodes):,}")
print(f"  Edges: {len(edges):,}")

# ── Step 6: Save outputs ──────────────────────────────────────────────────────

print("Step 6: Saving outputs...")

ox.save_graphml(G, os.path.join(OUTPUT_DIR, f'{CITY}_walk.graphml'))

nodes_3005 = nodes.to_crs(config.CRS)
edges_3005 = edges.to_crs(config.CRS)
nodes_3005.to_file(os.path.join(OUTPUT_DIR, f'{CITY}_osm_nodes.shp'))
edges_3005.to_file(os.path.join(OUTPUT_DIR, f'{CITY}_osm_edges.shp'))

print(f"  {CITY}_walk.graphml")
print(f"  {CITY}_osm_nodes.shp")
print(f"  {CITY}_osm_edges.shp")
print(f"  {CITY}_study_area_boundary.shp")

# ── Step 7: Quick network validation map ─────────────────────────────────────

import networkx as nx
boundary_gdf = gpd.read_file(os.path.join(OUTPUT_DIR, f'{CITY}_study_area_boundary.shp'))
print(f"\nStep 7: Validation checks...")
print(f"  Graph nodes: {G.number_of_nodes():,}")
print(f"  Graph edges: {G.number_of_edges():,}")
print(f"  Weakly connected: {nx.is_weakly_connected(G)}")

fig, ax = plt.subplots(figsize=(10, 10))
boundary_gdf.boundary.plot(ax=ax, color='red', linewidth=1)
edges_3005.plot(ax=ax, color='grey', linewidth=0.3, alpha=0.5)
nodes_3005.plot(ax=ax, color='blue', markersize=0.5, alpha=0.3)
ax.set_title(f'{CITY.title()} Walk Network')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f'{CITY}_osm_network_check.png'), dpi=150)
plt.close()

# ── Step 8: DA representative points map ─────────────────────────────────────

print("\nStep 8: DA representative points map...")
gaf_full = pd.read_csv(
    config.GAF_CSV,
    dtype=str,
    encoding='latin-1',
    usecols=['DAUID_ADIDU', 'DBUID_IDIDU', 'DBPOP2021_IDPOP2021',
             'DARPLAT_ADLAT', 'DARPLONG_ADLONG', 'CSDUID_SDRIDU']
)
gaf_primary = gaf_full[gaf_full['CSDUID_SDRIDU'] == config.STUDY_CSD].copy()
gaf_primary['lat']    = pd.to_numeric(gaf_primary['DARPLAT_ADLAT'],      errors='coerce')
gaf_primary['lon']    = pd.to_numeric(gaf_primary['DARPLONG_ADLONG'],     errors='coerce')
gaf_primary['db_pop'] = pd.to_numeric(gaf_primary['DBPOP2021_IDPOP2021'], errors='coerce')

da_pop = (
    gaf_primary.groupby('DAUID_ADIDU')
    .agg(da_pop=('db_pop', 'sum'), lat=('lat', 'first'), lon=('lon', 'first'))
    .reset_index()
)

da_points = gpd.GeoDataFrame(
    da_pop,
    geometry=gpd.points_from_xy(da_pop['lon'], da_pop['lat']),
    crs='EPSG:4326'
).to_crs(config.CRS)

boundary_3005 = boundary_gdf.to_crs(config.CRS)
da_all        = gpd.read_file(config.DA_SHP).to_crs(config.CRS)
primary_daids = gaf_primary['DAUID_ADIDU'].unique()
da_primary    = da_all[da_all['DAUID'].isin(primary_daids)]
da_points_clip = gpd.clip(da_points, boundary_3005)

fig, ax = plt.subplots(figsize=(10, 10))
da_primary.plot(ax=ax, color='#f0f0f0', edgecolor='#aaaaaa', linewidth=0.4)
da_points_clip.plot(
    ax=ax, column='da_pop', cmap='OrRd', markersize=6, alpha=0.9,
    vmin=da_points_clip['da_pop'].quantile(0.05),
    vmax=da_points_clip['da_pop'].quantile(0.95),
    legend=True,
    legend_kwds={'label': 'DA population (2021)', 'shrink': 0.5}
)
boundary_3005.boundary.plot(ax=ax, color='#333333', linewidth=1.2, linestyle='--')
ax.set_title(f'DA Representative Points — {CITY.title()}\n'
             f'n={len(da_points_clip):,} DAs, coloured by 2021 population')
ax.set_axis_off()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f'{CITY}_da_points_check.png'), dpi=150)
plt.close()

print("\nDone. All steps complete.")
