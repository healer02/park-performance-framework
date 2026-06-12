"""
01-get-osm-network.py
Downloads OSM pedestrian walk network for Vancouver study area.

Study area: City of Vancouver (5915022), Burnaby (5915025),
            Metro Vancouver A / Electoral Area A - UBC (5915020)

Inputs:
    - data/census/raw/2021_92-151_x.csv        (StatCan Geographic Attribute File)
    - data/census/raw/lda_000b21a_e/lda_000b21a_e.shp  (DA boundaries)

Outputs:
    - data/osm/vancouver_walk.graphml           (OSM network graph)
    - data/osm/osm_nodes.shp                   (network nodes)
    - data/osm/osm_edges.shp                   (network edges)
    - data/osm/study_area_boundary.shp         (merged study area polygon)

Notes:
    - OSM data downloaded: [date]
    - CRS for analysis: EPSG:3005 (BC Albers, metre-based)
"""

import os
import pandas as pd
import geopandas as gpd
import osmnx as ox

# ── Paths ─────────────────────────────────────────────────────────────────────

GAF_PATH   = 'data/census/raw/2021_92-151_x.csv'
DA_PATH    = 'data/census/raw/lda_000b21a_e/lda_000b21a_e.shp'
OUTPUT_DIR = 'data/osm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Target CSDs ───────────────────────────────────────────────────────────────

TARGET_CSDS = {
    '5915022': 'Vancouver',
    '5915025': 'Burnaby',
    '5915020': 'Metro Vancouver A (UBC/Electoral Area A)',
}

# ── Step 1: Load Geographic Attribute File ────────────────────────────────────

print("Loading Geographic Attribute File...")
gaf = pd.read_csv(
    GAF_PATH,
    dtype=str,
    encoding='latin-1',
    usecols=['DAUID_ADIDU', 'CSDUID_SDRIDU', 'CSDNAME_SDRNOM']
)
print(f"  GAF rows loaded: {len(gaf):,}")

# ── Step 2: Get target DAUIDs ─────────────────────────────────────────────────

target_daids = gaf[gaf['CSDUID_SDRIDU'].isin(TARGET_CSDS)]['DAUID_ADIDU'].unique()
print(f"  Target DAs identified: {len(target_daids)}")

# ── Step 3: Load and filter DA boundaries ─────────────────────────────────────

print("Loading DA boundaries...")
da = gpd.read_file(DA_PATH)
study_area = da[da['DAUID'].isin(target_daids)].copy()
print(f"  DAs in study area: {len(study_area)}")
print(f"  Source CRS: {study_area.crs}")

# ── Step 4: Build merged boundary polygon ─────────────────────────────────────

study_area_4326 = study_area.to_crs('EPSG:4326')
boundary = study_area_4326.unary_union

# Sanity check bounds (expect roughly -123.3, 49.0, -122.6, 49.4)
print(f"  Boundary bounds (lon/lat): {boundary.bounds}")

# Save study area boundary for reference
study_area_4326.dissolve().to_file(
    os.path.join(OUTPUT_DIR, 'study_area_boundary.shp')
)

# ── Step 5: Download OSM walk network ─────────────────────────────────────────

print("Downloading OSM pedestrian network (this may take 1–3 minutes)...")
G = ox.graph_from_polygon(boundary, network_type='walk')

nodes, edges = ox.graph_to_gdfs(G)
print(f"  Nodes: {len(nodes):,}")
print(f"  Edges: {len(edges):,}")

# ── Step 6: Save outputs ──────────────────────────────────────────────────────

print("Saving outputs...")

# Graph (for network analysis)
ox.save_graphml(G, os.path.join(OUTPUT_DIR, 'vancouver_walk.graphml'))

# Reproject to BC Albers (EPSG:3005) for metre-based distance analysis
nodes_3005 = nodes.to_crs('EPSG:3005')
edges_3005 = edges.to_crs('EPSG:3005')

nodes_3005.to_file(os.path.join(OUTPUT_DIR, 'osm_nodes.shp'))
edges_3005.to_file(os.path.join(OUTPUT_DIR, 'osm_edges.shp'))

print("Done. Outputs saved to data/osm/")
print("  - vancouver_walk.graphml")
print("  - osm_nodes.shp")
print("  - osm_edges.shp")
print("  - study_area_boundary.shp")



import geopandas as gpd
import osmnx as ox
import networkx as nx

# Check study area boundary
boundary = gpd.read_file('data/osm/study_area_boundary.shp')
print(f"Study area CRS: {boundary.crs}")
print(f"Study area bounds: {boundary.total_bounds}")
# Expected: roughly xmin=-123.3, ymin=49.0, xmax=-122.6, ymax=49.4

# Check nodes and edges
nodes = gpd.read_file('data/osm/osm_nodes.shp')
edges = gpd.read_file('data/osm/osm_edges.shp')
print(f"Nodes: {len(nodes):,}, CRS: {nodes.crs}")
print(f"Edges: {len(edges):,}, CRS: {edges.crs}")
# Nodes expect ~150,000-250,000 for Vancouver area
# Edges expect ~300,000-500,000

# Check graph
G = ox.load_graphml('data/osm/vancouver_walk.graphml')
print(f"Graph nodes: {G.number_of_nodes():,}")
print(f"Graph edges: {G.number_of_edges():,}")
print(f"Graph connected: {nx.is_weakly_connected(G)}")


import geopandas as gpd
import matplotlib.pyplot as plt

nodes = gpd.read_file('data/osm/osm_nodes.shp')
edges = gpd.read_file('data/osm/osm_edges.shp')
boundary = gpd.read_file('data/osm/study_area_boundary.shp')

fig, ax = plt.subplots(figsize=(10, 10))
boundary.boundary.plot(ax=ax, color='red', linewidth=1)
edges.plot(ax=ax, color='grey', linewidth=0.3, alpha=0.5)
nodes.plot(ax=ax, color='blue', markersize=0.5, alpha=0.3)
ax.set_title('Vancouver Walk Network')
plt.tight_layout()
os.makedirs('outputs/figures', exist_ok=True)
plt.savefig('outputs/figures/osm_network_check.png', dpi=150)
plt.show()




# ── Step 9: Check and visualise DA-level representative points ────────────────

print("\nStep 9: Checking DA-level representative points...")

STUDY_CSD = '5915022'  # City of Vancouver

gaf_full = pd.read_csv(
    GAF_PATH,
    dtype=str,
    encoding='latin-1',
    usecols=['DAUID_ADIDU', 'DBUID_IDIDU', 'DBPOP2021_IDPOP2021',
             'DARPLAT_ADLAT', 'DARPLONG_ADLONG', 'CSDUID_SDRIDU']
)

# Filter to Vancouver only
gaf_van = gaf_full[gaf_full['CSDUID_SDRIDU'] == STUDY_CSD].copy()
print(f"  Vancouver DB rows in GAF: {len(gaf_van)}")

# Convert to numeric
gaf_van['lat']    = pd.to_numeric(gaf_van['DARPLAT_ADLAT'],        errors='coerce')
gaf_van['lon']    = pd.to_numeric(gaf_van['DARPLONG_ADLONG'],       errors='coerce')
gaf_van['db_pop'] = pd.to_numeric(gaf_van['DBPOP2021_IDPOP2021'],   errors='coerce')

# Aggregate to DA level: sum DB populations per DA
da_pop = (
    gaf_van.groupby('DAUID_ADIDU')
    .agg(
        da_pop=('db_pop', 'sum'),
        lat=('lat', 'first'),
        lon=('lon', 'first')
    )
    .reset_index()
)
print(f"  Vancouver DAs: {len(da_pop)}")
print(f"  DA population — total: {da_pop['da_pop'].sum():,.0f}")
print(f"  DA population — zeros: {(da_pop['da_pop'] == 0).sum()}")

# Convert to GeoDataFrame
da_points = gpd.GeoDataFrame(
    da_pop,
    geometry=gpd.points_from_xy(da_pop['lon'], da_pop['lat']),
    crs='EPSG:4326'
).to_crs('EPSG:3005')

# Load DA boundaries and study boundary
boundary_3005 = gpd.read_file(
    os.path.join(OUTPUT_DIR, 'Vancouver_study_area_boundary.shp')
).to_crs('EPSG:3005')

da_all = gpd.read_file(DA_PATH).to_crs('EPSG:3005')
target_daids_van = gaf_van['DAUID_ADIDU'].unique()
da_van = da_all[da_all['DAUID'].isin(target_daids_van)].copy()

# Clip points to Vancouver boundary
da_points_van = gpd.clip(da_points, boundary_3005)
print(f"  DA points within Vancouver boundary: {len(da_points_van)}")

# ── Plot ──────────────────────────────────────────────────────────────────────

print("\n  Generating DA representative points map...")

fig, ax = plt.subplots(figsize=(10, 10))

# DA boundaries as base layer
da_van.plot(ax=ax, color='#f0f0f0', edgecolor='#aaaaaa', linewidth=0.4)

# DA points coloured by total DA population
da_points_van.plot(
    ax=ax, column='da_pop', cmap='OrRd',
    markersize=6, alpha=0.9,
    vmin=da_points_van['da_pop'].quantile(0.05),  # clip low end for contrast
    vmax=da_points_van['da_pop'].quantile(0.95),  # clip high end for contrast
    legend=True,
    legend_kwds={'label': 'DA population (2021)', 'shrink': 0.5}
)

# Study boundary outline
boundary_3005.boundary.plot(
    ax=ax, color='#333333', linewidth=1.2, linestyle='--'
)

ax.set_title(f'DA-Level Representative Points — Vancouver\n'
             f'n={len(da_points_van):,} DAs, coloured by total 2021 population')
ax.set_axis_off()
plt.tight_layout()

FIG_DIR = 'outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)
fig_path = os.path.join(FIG_DIR, 'vancouver_da_points_check.png')
plt.savefig(fig_path, dpi=150)
plt.close()  # close instead of show to prevent empty second figure
print(f"  Saved: {fig_path}")

print("\nAll steps complete.")

