"""
03-extract-entrances.py
Extracts park entrances as points where OSM road edges intersect park boundaries.

Method (T2P paper approach):
    1. Buffer OSM road centerlines by 30.5m
    2. Intersect buffered roads with park polygon boundaries
    3. Extract centroids of intersection segments as candidate entrance points
    4. Retain one entrance per road segment per park (deduplication)

This produces entrance points suitable for 400m network-based reachability
analysis in script 04.

Inputs:
    - data/parks/processed/parks_merged.shp     (merged park polygons, EPSG:3005)
    - data/osm/osm_edges.shp                    (OSM walk network edges, EPSG:3005)

Outputs:
    - data/parks/processed/park_entrances.shp   (entrance points, EPSG:3005)
    - outputs/figures/entrances_check.png       (visual validation — sample parks)

Notes:
    - 30.5m buffer threshold from T2P paper (captures road-park adjacency)
    - Only pedestrian-accessible OSM edges used (already filtered in script 01)
    - Parks with zero entrances flagged for manual review
    - CRS: EPSG:3005 throughout (metre-based)
"""

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────

PARKS_PATH    = 'data/parks/processed/parks_merged.shp'
EDGES_PATH    = 'data/osm/osm_edges.shp'
OUTPUT_DIR    = 'data/parks/processed'
FIG_DIR       = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BUFFER_M = 30.5  # road buffer distance (T2P paper method)

# ── Step 1: Load inputs ───────────────────────────────────────────────────────

print("Step 1: Loading inputs...")
parks = gpd.read_file(PARKS_PATH)
edges = gpd.read_file(EDGES_PATH)

print(f"  Parks: {len(parks)}, CRS: {parks.crs}")
print(f"  OSM edges: {len(edges)}, CRS: {edges.crs}")

# Confirm CRS match
assert parks.crs == edges.crs, f"CRS mismatch: parks={parks.crs}, edges={edges.crs}"

# ── Step 2: Buffer OSM road edges ─────────────────────────────────────────────

print(f"\nStep 2: Buffering OSM edges by {BUFFER_M}m...")
edges_buffered = edges.copy()
edges_buffered['geometry'] = edges.geometry.buffer(BUFFER_M)
print(f"  Buffered edges: {len(edges_buffered)}")

# ── Step 3: Get park boundaries (exterior rings) ──────────────────────────────

print("\nStep 3: Extracting park boundary lines...")
parks_boundary = parks.copy()
parks_boundary['geometry'] = parks.geometry.boundary
# Drop any empty boundaries
parks_boundary = parks_boundary[~parks_boundary.geometry.is_empty]
print(f"  Park boundaries: {len(parks_boundary)}")

# ── Step 4: Intersect buffered roads with park boundaries ─────────────────────

print("\nStep 4: Intersecting buffered roads with park boundaries...")
# Spatial join: which edges (buffered) overlap with park boundaries
joined = gpd.sjoin(
    parks_boundary[['park_id', 'source', 'area_ha', 'geometry']],
    edges_buffered[['geometry']].reset_index().rename(columns={'index': 'edge_id'}),
    how='inner',
    predicate='intersects'
)
print(f"  Park-edge intersections: {len(joined)}")

# ── Step 5: Extract entrance centroids ────────────────────────────────────────

print("\nStep 5: Extracting entrance centroids...")

entrance_records = []

for _, row in joined.iterrows():
    park_geom = parks.loc[parks['park_id'] == row['park_id'], 'geometry'].iloc[0]
    edge_geom = edges_buffered.loc[
        edges_buffered.index == row['edge_id'], 'geometry'
    ].iloc[0]

    # Intersection of park boundary with buffered road
    intersection = park_geom.boundary.intersection(edge_geom)

    if intersection.is_empty:
        continue

    # Use centroid of intersection segment as entrance point
    entrance_point = intersection.centroid

    entrance_records.append({
        'park_id': row['park_id'],
        'source':  row['source'],
        'area_ha': row['area_ha'],
        'edge_id': row['edge_id'],
        'geometry': entrance_point
    })

entrances_raw = gpd.GeoDataFrame(entrance_records, crs=parks.crs)
print(f"  Raw entrance points: {len(entrances_raw)}")

# ── Step 6: Deduplicate ────────────────────────────────────────────────────────

print("\nStep 6: Deduplicating (one entrance per road edge per park)...")
# Already one per park-edge pair from the loop above, but drop exact duplicates
entrances = entrances_raw.drop_duplicates(subset=['park_id', 'edge_id']).copy()
entrances = entrances.reset_index(drop=True)
entrances['entrance_id'] = entrances.index + 1
print(f"  Entrances after dedup: {len(entrances)}")

# ── Step 7: Flag parks with no entrances ──────────────────────────────────────

print("\nStep 7: Checking for parks with no entrances...")
parks_with_entrances = entrances['park_id'].unique()
parks_no_entrance = parks[~parks['park_id'].isin(parks_with_entrances)]
print(f"  Parks with ≥1 entrance: {len(parks_with_entrances)}")
print(f"  Parks with 0 entrances: {len(parks_no_entrance)} ← review manually")

if len(parks_no_entrance) > 0:
    print(parks_no_entrance[['park_id', 'source', 'area_ha']].to_string())

# ── Step 8: Save output ───────────────────────────────────────────────────────

print("\nStep 8: Saving output...")
out_path = os.path.join(OUTPUT_DIR, 'park_entrances.shp')
entrances[['entrance_id', 'park_id', 'source', 'area_ha', 'geometry']].to_file(out_path)
print(f"  Saved: {out_path}")
print(f"  Entrances per park (mean): {len(entrances)/len(parks):.1f}")
print(f"  Entrances per park (median): {entrances.groupby('park_id').size().median():.1f}")

# ── Step 9: Visual validation (sample 6 parks) ────────────────────────────────

print("\nStep 9: Generating visual check (sample parks)...")

sample_parks = parks.sample(min(6, len(parks)), random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (_, park_row) in enumerate(sample_parks.iterrows()):
    ax = axes[i]
    pid = park_row['park_id']

    # Buffer for context
    bbox = park_row.geometry.buffer(100).bounds
    park_entrances = entrances[entrances['park_id'] == pid]

    gpd.GeoDataFrame([park_row], crs=parks.crs).plot(
        ax=ax, color='#2ca25f', alpha=0.5, edgecolor='darkgreen', linewidth=1
    )

    if len(park_entrances) > 0:
        park_entrances.plot(ax=ax, color='red', markersize=20, zorder=5)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(
        f"park_id={pid} | {park_row['source']}\n"
        f"{park_row['area_ha']:.1f} ha | {len(park_entrances)} entrances",
        fontsize=8
    )
    ax.set_axis_off()

plt.suptitle('Park Entrance Extraction — Sample Check (red = entrances)', fontsize=12)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, 'entrances_check.png')
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"  Saved: {fig_path}")

print("\nDone.")