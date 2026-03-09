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
    - data/osm/Vancouver_study_area_boundary.shp                          (Vancouver + 1km buffer)
    - data/osm/Vancouver_osm_edges.shp                                    (OSM walk edges, EPSG:3005)

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
BUFFER_M    = 30.5  # road buffer for entrance extraction (T2P paper)

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
van = van[['geometry', 'source']].copy()

# -- Step 2: Load Burnaby park polygons ----------------------------------------

print("\nStep 2: Loading Burnaby park polygons...")
burnaby = gpd.read_file(BURNABY_PATH)
print(f"  Burnaby raw: {len(burnaby)} features, CRS: {burnaby.crs}")
print(f"  Columns: {list(burnaby.columns)}")

burnaby = burnaby.to_crs('EPSG:3005')
burnaby['source'] = 'Burnaby'
burnaby = burnaby[['geometry', 'source']].copy()

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
metro = metro[['geometry', 'source']].copy()


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
print(parks_all['area_ha'].describe())
print((parks_all['area_ha'] < 0.1).sum())
print((parks_all['area_ha'] < 0.25).sum())
print((parks_all['area_ha'] < 0.5).sum())


parks_filtered_preclip = parks_all[parks_all['area_ha'] >= MIN_AREA_HA].copy()
print(f"\n  After area filter (>={MIN_AREA_HA} ha): {len(parks_filtered_preclip)}")

# Clip to study boundary
parks_clipped = gpd.clip(parks_filtered_preclip, boundary)
parks_clipped = parks_clipped.reset_index(drop=True)
print(f"  After clip to study boundary: {len(parks_clipped)}")

# Recompute area after clip
parks_clipped['area_ha'] = parks_clipped.geometry.area / 10_000

# Stable, traceable park IDs: {source}_{index}
parks_clipped['park_id'] = (
    parks_clipped['source'] + '_' + parks_clipped.index.astype(str)
)
print(f"\n  Source breakdown:\n{parks_clipped['source'].value_counts().to_string()}")

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

# -- Step 9: Load OSM edges ----------------------------------------------------

print("\nStep 9: Loading OSM walk edges...")
edges = gpd.read_file(EDGES_PATH)
print(f"  OSM edges: {len(edges)}, CRS: {edges.crs}")
assert parks_clipped.crs == edges.crs, \
    f"CRS mismatch: parks={parks_clipped.crs}, edges={edges.crs}"

# -- Step 10: Buffer OSM road edges --------------------------------------------

print(f"\nStep 10: Buffering OSM edges by {BUFFER_M}m...")
edges_buffered = edges.copy()
edges_buffered['geometry'] = edges.geometry.buffer(BUFFER_M)
edges_buffered = edges_buffered.reset_index().rename(columns={'index': 'edge_id'})

# -- Step 11: Extract park boundary lines --------------------------------------

print("\nStep 11: Extracting park boundary lines...")
parks_boundary = parks_clipped.copy()
parks_boundary['geometry'] = parks_clipped.geometry.boundary
parks_boundary = parks_boundary[~parks_boundary.geometry.is_empty]
print(f"  Park boundaries: {len(parks_boundary)}")

# -- Step 12: Spatial join: buffered roads x park boundaries -------------------

print("\nStep 12: Intersecting buffered roads with park boundaries...")
joined = gpd.sjoin(
    parks_boundary[['park_id', 'source', 'area_ha', 'geometry']],
    edges_buffered[['edge_id', 'geometry']],
    how='inner',
    predicate='intersects'
)
print(f"  Park-edge intersections: {len(joined)}")

# -- Step 13: Extract entrance centroids ---------------------------------------

print("\nStep 13: Extracting entrance centroids...")

entrance_records = []

for _, row in joined.iterrows():
    park_geom = parks_clipped.loc[
        parks_clipped['park_id'] == row['park_id'], 'geometry'
    ].iloc[0]
    edge_geom = edges_buffered.loc[
        edges_buffered['edge_id'] == row['edge_id'], 'geometry'
    ].iloc[0]

    intersection = park_geom.boundary.intersection(edge_geom)

    if intersection.is_empty:
        continue

    entrance_records.append({
        'park_id':  row['park_id'],
        'source':   row['source'],
        'area_ha':  row['area_ha'],
        'edge_id':  row['edge_id'],
        'geometry': intersection.centroid
    })

entrances_raw = gpd.GeoDataFrame(entrance_records, crs=parks_clipped.crs)
print(f"  Raw entrance points: {len(entrances_raw)}")

# -- Step 14: Deduplicate ------------------------------------------------------

print("\nStep 14: Deduplicating (one entrance per road edge per park)...")
entrances = entrances_raw.drop_duplicates(subset=['park_id', 'edge_id']).copy()
entrances = entrances.reset_index(drop=True)
entrances['entrance_id'] = entrances.index + 1
print(f"  Entrances after dedup: {len(entrances)}")

# -- Step 15: Flag parks with no entrances -------------------------------------

print("\nStep 15: Checking for parks with no entrances...")
parks_with_entrances = entrances['park_id'].unique()
parks_no_entrance = parks_clipped[~parks_clipped['park_id'].isin(parks_with_entrances)]
print(f"  Parks with >=1 entrance: {len(parks_with_entrances)}")
print(f"  Parks with 0 entrances:  {len(parks_no_entrance)} <- review manually")
if len(parks_no_entrance) > 0:
    print(parks_no_entrance[['park_id', 'source', 'area_ha']].to_string())

# -- Step 16: Save entrances ---------------------------------------------------

print("\nStep 16: Saving entrance points...")
ent_path = os.path.join(OUTPUT_DIR, 'vancouver_park_entrances.shp')
entrances[['entrance_id', 'park_id', 'source', 'area_ha', 'geometry']].to_file(ent_path)
print(f"  Saved: {ent_path}")
print(f"  Entrances per park -- mean:   {len(entrances) / len(parks_clipped):.1f}")
print(f"  Entrances per park -- median: {entrances.groupby('park_id').size().median():.1f}")

# -- Step 17: Visual validation — sample parks ---------------------------------

print("\nStep 17: Generating entrance visual check (sample parks)...")

sample_parks = parks_clipped.sample(min(6, len(parks_clipped)), random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (_, park_row) in enumerate(sample_parks.iterrows()):
    ax = axes[i]
    pid = park_row['park_id']
    bbox = park_row.geometry.buffer(100).bounds
    park_entrances = entrances[entrances['park_id'] == pid]

    gpd.GeoDataFrame([park_row], crs=parks_clipped.crs).plot(
        ax=ax, color='#2ca25f', alpha=0.5, edgecolor='darkgreen', linewidth=1
    )
    if len(park_entrances) > 0:
        park_entrances.plot(ax=ax, color='red', markersize=20, zorder=5)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(
        f"{pid} | {park_row['source']}\n"
        f"{park_row['area_ha']:.1f} ha | {len(park_entrances)} entrances",
        fontsize=8
    )
    ax.set_axis_off()

plt.suptitle('Park Entrance Extraction — Sample Check (red = entrances)', fontsize=12)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, 'vancouver_entrances_check.png')
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"  Saved: {fig_path}")

print("\nDone.")