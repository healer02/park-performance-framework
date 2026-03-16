"""
02-merge-parks-entrances.py
Merges park polygons from three sources, clips to study boundary,
and extracts park entrances via OSM road intersection method.

Sources:
    1. City of Vancouver        data/parks/raw/Vancouver/parks-polygon-representation/
    2. Burnaby                  data/parks/raw/Burnaby/Park_Inventory.shp
    3. Metro Vancouver Regional data/parks/raw/Metro Vancouver Regional Parks/

Entrance extraction method (T2P paper):
    1. Buffer OSM road centerlines by 30.5m
    2. Intersect buffered roads with park polygon boundaries
    3. Centroid of each intersection segment = entrance point
    4. Deduplicate: one entrance per road edge per park

Inputs:
    - data/parks/raw/Vancouver/parks-polygon-representation/    (Vancouver park polygons)
    - data/parks/raw/Burnaby/Park_Inventory.shp                 (Burnaby park polygons)
    - data/parks/raw/Metro Vancouver Regional Parks/            (Metro Van park polygons)
    - data/osm/study_area_boundary.shp                          (Vancouver + 1km buffer)
    - data/osm/osm_edges.shp                                    (OSM walk edges, EPSG:3005)

Outputs:
    - data/parks/processed/vancouver_parks_merged.shp       (merged + clipped polygons, EPSG:3005)
    - data/parks/processed/vancouver_park_entrances.shp     (entrance points, EPSG:3005)
    - outputs/figures/vancouver_parks_merged_check.png      (visual validation)
    - outputs/figures/vancouver_entrances_check.png         (visual validation — sample parks)

Notes:
    - Area filter (>=0.5 ha) applied BEFORE clipping to avoid removing border parks
      whose clipped fragment would fall below threshold (e.g. Pacific Spirit Park)
    - park_id format: {source}_{index} for stable, traceable IDs across reruns
    - Parks with zero entrances flagged for manual review
    - CRS: EPSG:3005 (BC Albers) throughout
    - OSM data downloaded: [update date when run]
"""

import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox

# -- Paths ---------------------------------------------------------------------

VAN_PARKS_DIR = 'data/parks/raw/Vancouver/parks-polygon-representation'
BURNABY_PATH  = 'data/parks/raw/Burnaby/Park_Inventory.shp'
METRO_DIR     = 'data/parks/raw/Metro Vancouver Regional Parks'
BOUNDARY_PATH = 'data/osm/Vancouver_study_area_boundary.shp'
EDGES_PATH    = 'data/osm/Vancouver_osm_edges.shp'
OUTPUT_DIR    = 'data/parks/processed'
FIG_DIR       = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MIN_AREA_HA = 0.1   # minimum park size (methods threshold)
# BUFFER_M and DEDUP_M defined in Part B

# ==============================================================================
# PART A: MERGE PARK POLYGONS
# ==============================================================================

# -- Step 1: Load Vancouver park polygons --------------------------------------

print("Step 1: Loading Vancouver park polygons...")

shp_files  = glob.glob(os.path.join(VAN_PARKS_DIR, '*.shp'))
json_files = glob.glob(os.path.join(VAN_PARKS_DIR, '*.geojson')) + \
             glob.glob(os.path.join(VAN_PARKS_DIR, '*.json'))

if shp_files:
    van = gpd.read_file(shp_files[0])
elif json_files:
    van = gpd.read_file(json_files[0])
else:
    raise FileNotFoundError(f"No shapefile or GeoJSON found in {VAN_PARKS_DIR}")

print(f"  Vancouver raw: {len(van)} features, CRS: {van.crs}")
print(f"  Columns: {list(van.columns)}")

van = van.to_crs('EPSG:3005')
van['source'] = 'Vancouver'
van['park_name'] = van['PARK_NAME']
van = van[['geometry', 'source', 'park_name']].copy()

# -- Step 2: Load Burnaby park polygons ----------------------------------------

print("\nStep 2: Loading Burnaby park polygons...")
burnaby = gpd.read_file(BURNABY_PATH)
print(f"  Burnaby raw: {len(burnaby)} features, CRS: {burnaby.crs}")
print(f"  Columns: {list(burnaby.columns)}")

burnaby = burnaby.to_crs('EPSG:3005')
burnaby['source'] = 'Burnaby'
burnaby['park_name'] = burnaby['NAME']
burnaby = burnaby[['geometry', 'source', 'park_name']].copy()

# Dissolve fragmented Burnaby polygons by park name
# (Burnaby shapefile stores multi-part parks as separate rows)
burnaby = burnaby.dissolve(by='park_name', aggfunc='first').reset_index()
print(f"  Burnaby after dissolve by name: {len(burnaby)} parks")

# -- Step 3: Load Metro Vancouver Regional Parks -------------------------------

print("\nStep 3: Loading Metro Vancouver Regional Parks polygons...")
metro_shp = glob.glob(os.path.join(METRO_DIR, '*.shp'))
if not metro_shp:
    raise FileNotFoundError(f"No shapefile found in {METRO_DIR}")

metro = gpd.read_file(metro_shp[0])
print(f"  Metro Van raw: {len(metro)} features, CRS: {metro.crs}")
print(f"  Columns: {list(metro.columns)}")

metro = metro.to_crs('EPSG:3005')
metro['source'] = 'MetroVancouver'
metro['park_name'] = metro['parkname']
metro = metro[['geometry', 'source', 'park_name']].copy()

# -- Step 4: Load study boundary -----------------------------------------------

print("\nStep 4: Loading study boundary...")
boundary = gpd.read_file(BOUNDARY_PATH).to_crs('EPSG:3005')
print(f"  Boundary CRS: {boundary.crs}")

# -- Step 5: Merge, filter by area THEN clip -----------------------------------
# Area filter applied BEFORE clipping to avoid removing border parks
# whose clipped fragment would fall below threshold (e.g. Pacific Spirit Park)

print("\nStep 5: Merging, filtering by area, then clipping...")
parks_all = gpd.GeoDataFrame(
    pd.concat([van, burnaby, metro], ignore_index=True),
    crs='EPSG:3005'
)
print(f"  Total after merge: {len(parks_all)}")

# Keep valid geometries only
parks_all = parks_all[parks_all.geometry.is_valid & ~parks_all.geometry.is_empty].copy()

# Check geometry types -- expect Polygon and/or MultiPolygon only
print("\n  Geometry types:")
print(parks_all.geometry.geom_type.value_counts().to_string())
# WARNING: if GeometryCollection appears, investigate before continuing

# Filter by area FIRST using pre-clip original geometries
parks_all['area_ha'] = parks_all.geometry.area / 10_000
parks_filtered_preclip = parks_all[parks_all['area_ha'] >= MIN_AREA_HA].copy()
print(f"\n  After area filter (>={MIN_AREA_HA} ha): {len(parks_filtered_preclip)}")

# Clip to study boundary
parks_clipped = gpd.clip(parks_filtered_preclip, boundary)
parks_clipped = parks_clipped.reset_index(drop=True)
print(f"  After clip to study boundary: {len(parks_clipped)}")

# Recompute area after clip
parks_clipped['area_ha'] = parks_clipped.geometry.area / 10_000

# Second area filter: remove fragments created by clipping at study boundary
# (e.g. Burnaby parks that straddle the boundary and shrink below threshold)
parks_clipped = parks_clipped[parks_clipped['area_ha'] >= MIN_AREA_HA].copy()
parks_clipped = parks_clipped.reset_index(drop=True)
print(f"  After post-clip area filter: {len(parks_clipped)}")
print(f"\n  Source breakdown:\n{parks_clipped['source'].value_counts().to_string()}")

# exclude two parks with no entrances via OSM walk network 
EXCLUDE_PARKS = ['Iona Beach Regional Park']
parks_clipped = parks_clipped[~parks_clipped['park_name'].isin(EXCLUDE_PARKS)].copy()
print(f"  After excluding two parks: {len(parks_clipped)}")
print(f"\n  Source breakdown:\n{parks_clipped['source'].value_counts().to_string()}")


# Stable, traceable park IDs: {source}_{index}
parks_clipped['park_id'] = (
    parks_clipped['source'] + '_' + parks_clipped.index.astype(str)
)
print(f"\n  Source breakdown:\n{parks_clipped['source'].value_counts().to_string()}")
print(f"\n  Park names sample:\n{parks_clipped[['park_id', 'park_name', 'source', 'area_ha']].to_string()}")

# -- Step 6: Overlap diagnostic ------------------------------------------------

print("\nStep 6: Checking for overlapping parks across sources...")
overlap = gpd.overlay(parks_clipped, parks_clipped, how='intersection')
overlap = overlap[overlap['park_id_1'] != overlap['park_id_2']]
print(f"  Potential overlapping park pairs: {len(overlap)}")
if len(overlap) > 0:
    print("  NOTE: overlaps kept as-is; review if count is unexpectedly high")

# -- Step 7: Save merged parks -------------------------------------------------

print("\nStep 7: Saving merged parks...")
parks_path = os.path.join(OUTPUT_DIR, 'vancouver_parks_merged.shp')
parks_clipped.to_file(parks_path)
print(f"  Saved: {parks_path}")

# -- Step 8: Visual validation — merged parks ----------------------------------

print("\nStep 8: Generating merged parks visual check...")

colours = {'Vancouver': '#2ca25f', 'Burnaby': '#2b8cbe', 'MetroVancouver': '#d95f0e'}

fig, ax = plt.subplots(figsize=(12, 10))
boundary.boundary.plot(ax=ax, color='black', linewidth=0.8, linestyle='--')

for source, colour in colours.items():
    subset = parks_clipped[parks_clipped['source'] == source]
    if len(subset) > 0:
        subset.plot(ax=ax, color=colour, alpha=0.6, edgecolor='white', linewidth=0.3)

patches = [mpatches.Patch(color=c, label=s, alpha=0.6) for s, c in colours.items()]
patches.append(mpatches.Patch(fill=False, edgecolor='black', linestyle='--',
                               label='Study boundary (1km buffer)'))
ax.legend(handles=patches, loc='upper left')
ax.set_title(f'Merged Park Polygons — Vancouver Study Area\n'
             f'n={len(parks_clipped)}, >={MIN_AREA_HA} ha, clipped to 1km buffer')
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, 'vancouver_parks_merged_check.png')
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"  Saved: {fig_path}")

# ==============================================================================
# PART B: EXTRACT PARK ENTRANCES
# ==============================================================================
# Method (T2P paper):
#   1. Buffer park boundaries by BUFFER_M
#   2. Intersect buffered park polygons with OSM walk edge centrelines
#      → intersection points where roads enter the park buffer zone
#   3. Deduplicate: merge entrance points within DEDUP_M of each other per park
#   4. Snap entrance points to nearest OSM node via ox.distance.nearest_nodes()

BUFFER_M = 10    # park boundary buffer distance (metres)
DEDUP_M  = 25    # merge entrances within this distance (metres)

# -- Step 9: Load OSM graph and edges ------------------------------------------

print("\nStep 9: Loading OSM walk network...")
G     = ox.load_graphml('data/osm/Vancouver_walk.graphml')
edges = gpd.read_file(EDGES_PATH)
print(f"  OSM edges: {len(edges)}, CRS: {edges.crs}")
assert parks_clipped.crs == edges.crs, \
    f"CRS mismatch: parks={parks_clipped.crs}, edges={edges.crs}"

# -- Step 10: Buffer park boundaries -------------------------------------------

print(f"\nStep 10: Buffering park boundaries by {BUFFER_M}m...")
parks_buffered = parks_clipped.copy()
parks_buffered['geometry'] = parks_clipped.geometry.boundary.buffer(BUFFER_M)
print(f"  Buffered park boundaries: {len(parks_buffered)}")

# -- Step 11: Intersect buffered parks with OSM edge centrelines ---------------

print("\nStep 11: Intersecting buffered park boundaries with OSM edge centrelines...")

# Spatial join: which edges fall within each buffered park boundary
joined = gpd.sjoin(
    edges[['geometry']].reset_index().rename(columns={'index': 'edge_idx'}),
    parks_buffered[['park_id', 'park_name', 'source', 'area_ha', 'geometry']],
    how='inner',
    predicate='intersects'
)
print(f"  Edge-park intersections: {len(joined)}")

# Compute actual intersection points (road centreline x buffered park boundary)
entrance_records = []

parks_buffered_idx = parks_buffered.set_index('park_id')

for _, row in joined.iterrows():
    edge_geom = edges.iloc[row['edge_idx']].geometry
    park_geom = parks_buffered_idx.loc[row['park_id'], 'geometry']

    intersection = edge_geom.intersection(park_geom)

    if intersection.is_empty:
        continue

    # Extract point(s) from intersection result
    if intersection.geom_type == 'Point':
        pts = [intersection]
    elif intersection.geom_type == 'MultiPoint':
        pts = list(intersection.geoms)
    elif intersection.geom_type in ('LineString', 'MultiLineString'):
        # Road runs along boundary — use midpoint
        pts = [intersection.interpolate(0.5, normalized=True)]
    else:
        pts = [intersection.centroid]

    for pt in pts:
        entrance_records.append({
            'park_id':   row['park_id'],
            'park_name': row['park_name'],
            'source':    row['source'],
            'area_ha':   row['area_ha'],
            'geometry':  pt
        })

entrances_raw = gpd.GeoDataFrame(entrance_records, crs=parks_clipped.crs)
print(f"  Raw entrance points: {len(entrances_raw)}")

print(entrances_raw.columns.tolist())

# -- Step 12: Deduplicate entrances within DEDUP_M per park --------------------

print(f"\nStep 12: Deduplicating entrances within {DEDUP_M}m per park...")

deduped = []

for park_id, group in entrances_raw.groupby('park_id'):
    group = group.copy().reset_index(drop=True)
    used  = [False] * len(group)

    for i in range(len(group)):
        if used[i]:
            continue
        # Find all points within DEDUP_M of point i
        cluster = [i]
        for j in range(i + 1, len(group)):
            if not used[j]:
                dist = group.geometry.iloc[i].distance(group.geometry.iloc[j])
                if dist <= DEDUP_M:
                    cluster.append(j)
                    used[j] = True
        used[i] = True

        # Representative point = centroid of cluster
        cluster_geom = group.geometry.iloc[cluster]
        rep_point    = cluster_geom.union_all().centroid \
                       if hasattr(cluster_geom, 'union_all') \
                       else cluster_geom.unary_union.centroid

        deduped.append({
            'park_id':   park_id,
            'park_name': group['park_name'].iloc[0],
            'source':    group['source'].iloc[0],
            'area_ha':   group['area_ha'].iloc[0],
            'geometry':  rep_point
        })

entrances = gpd.GeoDataFrame(deduped, crs=parks_clipped.crs)
entrances = entrances.reset_index(drop=True)
print(f"  Entrances after dedup: {len(entrances)}")
print(f"  Reduction: {len(entrances_raw)} → {len(entrances)} "
      f"({100*(1 - len(entrances)/len(entrances_raw)):.1f}% removed)")

entrances['area_ha'] = entrances['park_id'].map(parks_clipped.set_index('park_id')['area_ha'])
print(entrances.columns.tolist())

# -- Step 13: Snap entrances to nearest OSM node -------------------------------

print("\nStep 13: Snapping entrances to nearest OSM node...")

# ox.nearest_nodes expects EPSG:4326 coordinates
entrances_4326 = entrances.to_crs('EPSG:4326')
xs = entrances_4326.geometry.x.values
ys = entrances_4326.geometry.y.values

nearest_node_ids = ox.distance.nearest_nodes(G, xs, ys)
entrances['nearest_node'] = nearest_node_ids

# Compute snap distance in metres (EPSG:3005)
# Build nodes lookup directly from the projected graph
nodes_gdf = ox.graph_to_gdfs(G, edges=False)[['geometry']].to_crs('EPSG:3005')
# nodes_gdf is already indexed by osmid
snap_dists = []
for _, row in entrances.iterrows():
    node_geom = nodes_gdf.loc[row['nearest_node'], 'geometry']
    snap_dists.append(row.geometry.distance(node_geom))

entrances['snap_dist_m'] = snap_dists

print(f"  Snap distance — mean:  {entrances['snap_dist_m'].mean():.1f}m")
print(f"  Snap distance — median:{entrances['snap_dist_m'].median():.1f}m")
print(f"  Snap distance — max:   {entrances['snap_dist_m'].max():.1f}m")
if entrances['snap_dist_m'].max() > 50:
    print("  WARNING: some entrances snap >50m — review those parks manually")

# -- Step 14: Assign entrance IDs ----------------------------------------------

entrances['entrance_id'] = entrances.index + 1

# -- Step 15: Flag parks with no entrances -------------------------------------

print("\nStep 15: Checking for parks with no entrances...")
parks_with_entrances = entrances['park_id'].unique()
parks_no_entrance    = parks_clipped[~parks_clipped['park_id'].isin(parks_with_entrances)]
print(f"  Parks with >=1 entrance: {len(parks_with_entrances)}")
print(f"  Parks with 0 entrances:  {len(parks_no_entrance)} <- review manually")
if len(parks_no_entrance) > 0:
    print(parks_no_entrance[['park_id', 'park_name', 'source', 'area_ha']].to_string())

# Dedup Shaughnessy entrances
s_gdf = gpd.GeoDataFrame(records_s, crs=parks_clipped.crs)
s_gdf = s_gdf.reset_index(drop=True)

# Simple distance-based dedup
used = [False] * len(s_gdf)
s_deduped = []
for i in range(len(s_gdf)):
    if used[i]: continue
    cluster = [i]
    for j in range(i+1, len(s_gdf)):
        if not used[j] and s_gdf.geometry.iloc[i].distance(s_gdf.geometry.iloc[j]) <= DEDUP_M:
            cluster.append(j)
            used[j] = True
    used[i] = True
    s_deduped.append({
        'park_id':   s_gdf['park_id'].iloc[0],
        'park_name': s_gdf['park_name'].iloc[0],
        'source':    s_gdf['source'].iloc[0],
        'area_ha':   s_gdf['area_ha'].iloc[0],
        'geometry':  s_gdf.geometry.iloc[cluster].unary_union.centroid
    })

s_entrances = gpd.GeoDataFrame(s_deduped, crs=parks_clipped.crs)

# Snap to nearest node
s_4326 = s_entrances.to_crs('EPSG:4326')
s_entrances['nearest_node'] = ox.distance.nearest_nodes(
    G, s_4326.geometry.x.values, s_4326.geometry.y.values)
s_entrances['snap_dist_m'] = [
    s_entrances.geometry.iloc[i].distance(nodes_gdf.loc[s_entrances['nearest_node'].iloc[i], 'geometry'])
    for i in range(len(s_entrances))
]

# Concat and reassign entrance_ids
entrances = pd.concat([entrances, s_entrances], ignore_index=True)
entrances['entrance_id'] = entrances.index + 1
print(f"  Shaughnessy entrances after dedup: {len(s_entrances)}")
print(f"  Total entrances now: {len(entrances)}")
print(entrances.columns.tolist())

# -- Step 16: Sanity checks ----------------------------------------------------

print("\nStep 16: Sanity checks...")

# Entrances per park
ent_per_park = entrances.groupby('park_id').size()
print(f"  Entrances per park — mean:   {ent_per_park.mean():.1f}")
print(f"  Entrances per park — median: {ent_per_park.median():.1f}")
print(f"  Entrances per park — max:    {ent_per_park.max()}")
print(f"  Parks with 1 entrance: {(ent_per_park == 1).sum()}")
print(f"  Parks with >20 entrances: {(ent_per_park > 20).sum()} <- may need review")

# CRS check
assert entrances.crs.to_epsg() == 3005, "Entrances CRS is not EPSG:3005"
print(f"  CRS: EPSG:{entrances.crs.to_epsg()} ✓")

# All nearest_node values are valid graph nodes
valid_nodes = set(G.nodes())
invalid     = entrances[~entrances['nearest_node'].isin(valid_nodes)]
print(f"  Entrances with invalid nearest_node: {len(invalid)}")

print(entrances.nlargest(5, 'snap_dist_m')[['park_id', 'park_name', 'snap_dist_m']])
print(ent_per_park.nlargest(10))


# -- Step 17: Save entrances ---------------------------------------------------

print("\nStep 17: Saving entrance points...")
ent_path = os.path.join(OUTPUT_DIR, 'vancouver_park_entrances.shp')
entrances[['entrance_id', 'park_id', 'park_name', 'source',
           'area_ha', 'nearest_node', 'snap_dist_m', 'geometry']].to_file(ent_path)
print(f"  Saved: {ent_path}")

# -- Step 18: Visual validation — sample parks ---------------------------------
# run 02b-entrance-review.py for detailed entrance review maps (for parks with 20+ entrances or 1 entrances)