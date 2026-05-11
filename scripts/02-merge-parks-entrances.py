"""
02-merge-parks-entrances.py
Merges park polygons from three sources, clips to study boundary,
and extracts park entrances via OSM road intersection method.

Sources:
    1. City of Vancouver        data/parks/raw/Vancouver/parks-polygon-representation/
    2. Burnaby                  data/parks/raw/Burnaby/Park_Inventory.shp
    3. Metro Vancouver Regional data/parks/raw/Metro Vancouver Regional Parks/

Outputs:
    - data/parks/processed/vancouver_parks_merged.shp
    - data/parks/processed/vancouver_park_entrances.shp
"""

# %% 1. IMPORTS AND PATHS
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox

VAN_PARKS_DIR = 'data/parks/raw/Vancouver/parks-polygon-representation'
BURNABY_PATH  = 'data/parks/raw/Burnaby/Park_Inventory.shp'
METRO_DIR     = 'data/parks/raw/Metro Vancouver Regional Parks'
BOUNDARY_PATH = 'data/osm/Vancouver_study_area_boundary.shp'
EDGES_PATH    = 'data/osm/Vancouver_osm_edges.shp'
OUTPUT_DIR    = 'data/parks/processed'
FIG_DIR       = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MIN_AREA_HA = 0.1  # default size threshold

# Parks below 0.1 ha explicitly included based on manual review:
# criteria: named park with >=50 reviews OR verified distinct recreational infrastructure
EXPLICIT_INCLUDE = [
    'Choklit Park',           # 0.067 ha, 90 reviews
    'Major Matthews Park',    # 0.074 ha, 37 reviews, gazebo + playground
    'Willow Park',            # 0.071 ha, 75 reviews
    'Street End - Wall St @ Nanaimo',  # Meditation Park, 0.06 ha, 46 reviews
]

EXCLUDE_PARKS = ['Iona Beach Regional Park']

BUFFER_M = 10   # park boundary buffer for entrance extraction
DEDUP_M  = 25   # merge entrances within this distance

print("Ready.")


# %% 2. LOAD AND MERGE PARK POLYGONS
print("Loading Vancouver parks...")
van = gpd.read_file(
    'data/parks/raw/Vancouver/parks-polygon-representation/parks_polygon_SpanishBanksMerged.gpkg'
).to_crs('EPSG:3005')
van['source']    = 'Vancouver'
van['park_name'] = van['PARK_NAME']
van = van[['geometry', 'source', 'park_name']].copy()
print(f"  Vancouver: {len(van)}")

print("Loading Burnaby parks...")
burnaby = gpd.read_file(BURNABY_PATH).to_crs('EPSG:3005')
burnaby['source']    = 'Burnaby'
burnaby['park_name'] = burnaby['NAME']
burnaby = burnaby[['geometry', 'source', 'park_name']].copy()
burnaby = burnaby.dissolve(by='park_name', aggfunc='first').reset_index()
print(f"  Burnaby after dissolve: {len(burnaby)}")

print("Loading Metro Vancouver parks...")
metro_shp = glob.glob(os.path.join(METRO_DIR, '*.shp'))
metro = gpd.read_file(metro_shp[0]).to_crs('EPSG:3005')
metro['source']    = 'MetroVancouver'
metro['park_name'] = metro['parkname']
metro = metro[['geometry', 'source', 'park_name']].copy()
print(f"  Metro Vancouver: {len(metro)}")

parks_all = gpd.GeoDataFrame(
    pd.concat([van, burnaby, metro], ignore_index=True),
    crs='EPSG:3005'
)
parks_all = parks_all[parks_all.geometry.is_valid & ~parks_all.geometry.is_empty].copy()
parks_all['area_ha'] = parks_all.geometry.area / 10_000

print(f"\nTotal merged: {len(parks_all)}")
print(f"Geometry types:\n{parks_all.geometry.geom_type.value_counts().to_string()}")


# %% 3. FILTER, CLIP, AND ASSIGN IDs
print("Loading study boundary...")
boundary = gpd.read_file(BOUNDARY_PATH).to_crs('EPSG:3005')

# Parks below 0.1 ha explicitly included based on manual review
EXPLICIT_INCLUDE = [
    'Choklit Park',
    'Major Matthews Park',
    'Willow Park',
    'Street End - Wall St @ Nanaimo',
]

# Non-parks, no-review parks, or polygons already merged into another entry
EXCLUDE_PARKS = [
    'Iona Beach Regional Park',
    'BOUNDARY CREEK RAVINE PARK',
    'STILL CREEK CONSERVATION AREA',
    'Mont Royal Square',
    'Nat Bailey Stadium Park',
    'Shannon Mews Park',
    "Gibby's Field",
    'Downtown Skateboard Plaza',
    'West End minipark - GILFORD ST @ HARO ST',
    'Spanish Banks Extension',
    'Roundhouse Turntable Plaza',
    'Empire Fields - Hastings Park',
    'Slidey Slides',
    'Locarno Park',
    'Helmcken Park',
    'RIVERWAY GOLF COURSE',
    'Vanier Park (Cultural Harmony Grove)'
]

# Size filter: keep parks >= MIN_AREA_HA OR in explicit include list
# Applied BEFORE clipping to avoid removing border parks like Pacific Spirit
parks_filtered = parks_all[
    (parks_all['area_ha'] >= MIN_AREA_HA) |
    (parks_all['park_name'].isin(EXPLICIT_INCLUDE))
].copy()
print(f"After size filter + explicit includes: {len(parks_filtered)}")

# Clip to study boundary
parks_clipped = gpd.clip(parks_filtered, boundary).reset_index(drop=True)
parks_clipped['area_ha'] = parks_clipped.geometry.area / 10_000
print(f"After clip: {len(parks_clipped)}")

# Post-clip size filter: remove boundary fragments, keep explicit includes
parks_clipped = parks_clipped[
    (parks_clipped['area_ha'] >= MIN_AREA_HA) |
    (parks_clipped['park_name'].isin(EXPLICIT_INCLUDE))
].copy().reset_index(drop=True)
print(f"After post-clip filter: {len(parks_clipped)}")

# Exclude non-parks, no-review parks, and already-merged polygons
parks_clipped = parks_clipped[parks_clipped['park_name'].notna()].copy()
parks_clipped = parks_clipped[
    ~parks_clipped['park_name'].isin(EXCLUDE_PARKS)
].copy().reset_index(drop=True)

# Stable park IDs
parks_clipped['park_id'] = (
    parks_clipped['source'] + '_' + parks_clipped.index.astype(str)
)

print(f"\nFinal park count: {len(parks_clipped)}")  # should be ~238
print(f"Source breakdown:\n{parks_clipped['source'].value_counts().to_string()}")
print(f"\nExplicit includes confirmed:")
print(parks_clipped[parks_clipped['park_name'].isin(EXPLICIT_INCLUDE)][['park_id', 'park_name', 'area_ha']])

# Overlap check
overlap = gpd.overlay(parks_clipped, parks_clipped, how='intersection')
overlap = overlap[overlap['park_id_1'] != overlap['park_id_2']]
print(f"\nOverlapping park pairs: {len(overlap)}")

# Save
parks_clipped.to_file(f"{OUTPUT_DIR}/vancouver_parks_merged.shp")
print("Saved vancouver_parks_merged.shp")

# Add to end of cell 3, after saving the shapefile
parks_clipped[['park_id', 'source', 'park_name', 'area_ha']].to_csv(
    f"{OUTPUT_DIR}/vancouver_parks_merged.csv", index=False
)
print("Saved vancouver_parks_merged.csv")

# %% 4. EXTRACT AND DEDUPLICATE ENTRANCES
print("Loading OSM network and edges...")
import osmnx as ox
G     = ox.load_graphml('data/osm/Vancouver_walk.graphml')
edges = gpd.read_file(EDGES_PATH)
assert parks_clipped.crs == edges.crs, "CRS mismatch"


# Buffer park boundaries (park-specific overrides supported)
PARK_BUFFERS = {
    'Shaughnessy Park': 30,
}

parks_buffered = parks_clipped.copy()
parks_buffered['geometry'] = parks_clipped.apply(
    lambda row: row.geometry.boundary.buffer(
        PARK_BUFFERS.get(row['park_name'], BUFFER_M)
    ), axis=1
)
# Spatial join edges to buffered parks
joined = gpd.sjoin(
    edges[['geometry']].reset_index().rename(columns={'index': 'edge_idx'}),
    parks_buffered[['park_id', 'park_name', 'source', 'area_ha', 'geometry']],
    how='inner',
    predicate='intersects'
)
print(f"Edge-park intersections: {len(joined)}")

# Extract entrance points
entrance_records = []
parks_buffered_idx = parks_buffered.set_index('park_id')

for _, row in joined.iterrows():
    edge_geom = edges.iloc[row['edge_idx']].geometry
    park_geom = parks_buffered_idx.loc[row['park_id'], 'geometry']
    intersection = edge_geom.intersection(park_geom)

    if intersection.is_empty:
        continue

    if intersection.geom_type == 'Point':
        pts = [intersection]
    elif intersection.geom_type == 'MultiPoint':
        pts = list(intersection.geoms)
    elif intersection.geom_type in ('LineString', 'MultiLineString'):
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
print(f"Raw entrance points: {len(entrances_raw)}")

# Deduplicate within DEDUP_M per park
deduped = []
for park_id, group in entrances_raw.groupby('park_id'):
    group = group.copy().reset_index(drop=True)
    used  = [False] * len(group)

    for i in range(len(group)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(group)):
            if not used[j]:
                if group.geometry.iloc[i].distance(group.geometry.iloc[j]) <= DEDUP_M:
                    cluster.append(j)
                    used[j] = True
        used[i] = True

        rep_point = group.geometry.iloc[cluster].union_all().centroid
        deduped.append({
            'park_id':   park_id,
            'park_name': group['park_name'].iloc[0],
            'source':    group['source'].iloc[0],
            'area_ha':   group['area_ha'].iloc[0],
            'geometry':  rep_point
        })

entrances = gpd.GeoDataFrame(deduped, crs=parks_clipped.crs).reset_index(drop=True)
print(f"Entrances after dedup: {len(entrances_raw)} → {len(entrances)} "
      f"({100*(1 - len(entrances)/len(entrances_raw)):.1f}% removed)")

# Flag parks with no entrances
parks_with_entrances = entrances['park_id'].unique()
parks_no_entrance = parks_clipped[~parks_clipped['park_id'].isin(parks_with_entrances)]
print(f"\nParks with 0 entrances: {len(parks_no_entrance)} <- review manually")
if len(parks_no_entrance) > 0:
    print(parks_no_entrance[['park_id', 'park_name', 'area_ha']].to_string())


# %% 5. SNAP TO OSM NODES AND SAVE
print("Snapping entrances to nearest OSM node...")
nodes_gdf = ox.graph_to_gdfs(G, edges=False)[['geometry']].to_crs('EPSG:3005')

entrances_4326 = entrances.to_crs('EPSG:4326')
entrances['nearest_node'] = ox.distance.nearest_nodes(
    G,
    entrances_4326.geometry.x.values,
    entrances_4326.geometry.y.values
)

entrances['snap_dist_m'] = [
    entrances.geometry.iloc[i].distance(
        nodes_gdf.loc[entrances['nearest_node'].iloc[i], 'geometry']
    )
    for i in range(len(entrances))
]

print(f"Snap distance — mean: {entrances['snap_dist_m'].mean():.1f}m, "
      f"median: {entrances['snap_dist_m'].median():.1f}m, "
      f"max: {entrances['snap_dist_m'].max():.1f}m")
if entrances['snap_dist_m'].max() > 50:
    print("WARNING: some entrances snap >50m — review:")
    print(entrances.nlargest(5, 'snap_dist_m')[['park_name', 'snap_dist_m']])

# Sanity checks
ent_per_park = entrances.groupby('park_id').size()
print(f"\nEntrances per park — mean: {ent_per_park.mean():.1f}, "
      f"median: {ent_per_park.median():.1f}, max: {ent_per_park.max()}")
print(f"Parks with >20 entrances: {(ent_per_park > 20).sum()}")

valid_nodes = set(G.nodes())
invalid = entrances[~entrances['nearest_node'].isin(valid_nodes)]
print(f"Entrances with invalid nearest_node: {len(invalid)}")

entrances['entrance_id'] = entrances.index + 1
entrances[['entrance_id', 'park_id', 'park_name', 'source',
           'area_ha', 'nearest_node', 'snap_dist_m', 'geometry']].to_file(
    f"{OUTPUT_DIR}/vancouver_park_entrances.shp"
)
print("Saved vancouver_park_entrances.shp")


# %% 6. VALIDATION MAPS
import contextily as ctx

colours = {'Vancouver': '#2ca25f', 'Burnaby': '#2b8cbe', 'MetroVancouver': '#d95f0e'}

# Map 1: Merged parks by source
fig, ax = plt.subplots(figsize=(12, 10))
boundary_3857 = boundary.to_crs(epsg=3857)
parks_3857 = parks_clipped.to_crs(epsg=3857)
boundary_3857.boundary.plot(ax=ax, color='black', linewidth=0.8, linestyle='--')
for source, colour in colours.items():
    subset = parks_3857[parks_3857['source'] == source]
    if len(subset):
        subset.plot(ax=ax, color=colour, alpha=0.6, edgecolor='white', linewidth=0.3)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
patches = [mpatches.Patch(color=c, label=s, alpha=0.6) for s, c in colours.items()]
ax.legend(handles=patches, loc='upper left')
ax.set_title(f'Merged Park Polygons — Vancouver Study Area\n'
             f'n={len(parks_clipped)}, area >={MIN_AREA_HA} ha')
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_parks_merged_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved parks map.")

# Map 2: Entrances
fig, ax = plt.subplots(figsize=(12, 10))
entrances_3857 = entrances.to_crs(epsg=3857)
parks_3857.plot(ax=ax, color='#e0ede0', edgecolor='#2ca25f', linewidth=0.5)
entrances_3857.plot(ax=ax, color='#d95f0e', markersize=2, alpha=0.6)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title(f'Park Entrances — Vancouver\nn={len(entrances)} entrances, {len(parks_clipped)} parks')
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_entrances_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved entrances map.")
# For detailed per-park entrance review: run 02b-entrance-review.py
# %%
old_master = pd.read_csv("data/parks/processed/06-master-park-placeids.csv")
new_parks = parks_clipped[~parks_clipped['park_name'].isin(old_master['park_name'])]
print(f"New parks not in old master: {len(new_parks)}")
print(new_parks[['park_name', 'source', 'area_ha']])
# %%
# Check the NaN park
parks_clipped[parks_clipped['park_name'].isna()][['park_id', 'area_ha', 'geometry']]
# %%
