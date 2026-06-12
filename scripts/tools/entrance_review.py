"""
tools/entrance_review.py  (was 02b-entrance-review.py)
QA helper — generates one PNG per flagged park for manual entrance review.

NOT a numbered pipeline step. Run after 02_parks.py and before 03_centroids.py
when you want to visually inspect auto-detected park entrances.

Flags:
    - Parks with >20 entrances (possible over-extraction)
    - Parks with <=1 entrance  (possible under-extraction)

Inputs:
    data/parks/processed/{CITY}_parks_merged.shp
    data/parks/processed/{CITY}_park_entrances.shp
    data/osm/{CITY}_osm_edges.shp
    data/osm/{CITY}_osm_nodes.shp

Outputs:
    outputs/figures/entrance_review/{park_id}_{park_name}.png
"""

import os
import sys

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import geopandas as gpd
import matplotlib.pyplot as plt

CITY = config.CITY

PARKS_PATH = f'data/parks/processed/{CITY}_parks_merged.shp'
ENT_PATH   = f'data/parks/processed/{CITY}_park_entrances.shp'
EDGES_PATH = f'data/osm/{CITY}_osm_edges.shp'
NODES_PATH = f'data/osm/{CITY}_osm_nodes.shp'
FIG_DIR    = 'outputs/figures/entrance_review'

os.makedirs(FIG_DIR, exist_ok=True)

print("Loading data...")
parks     = gpd.read_file(PARKS_PATH)
entrances = gpd.read_file(ENT_PATH)
edges     = gpd.read_file(EDGES_PATH)
nodes     = gpd.read_file(NODES_PATH)
print(f"  Parks: {len(parks)}, Entrances: {len(entrances)}")

ent_per_park  = entrances.groupby('park_id').size()
all_park_ids  = set(parks['park_id'])
parks_no_ent  = all_park_ids - set(ent_per_park.index)

flagged_over  = list(ent_per_park[ent_per_park > 20].index)
flagged_under = list(ent_per_park[ent_per_park <= 1].index) + list(parks_no_ent)
flagged_ids   = flagged_over + flagged_under

print(f"  Parks with >20 entrances: {len(flagged_over)}")
print(f"  Parks with <=1 entrance:  {len(flagged_under)}")
print(f"  Total flagged:            {len(flagged_ids)}")

print("\nGenerating review maps...")

for park_id in flagged_ids:
    park_rows = parks[parks['park_id'] == park_id]
    if len(park_rows) == 0:
        print(f"  WARNING: {park_id} not found in parks layer, skipping")
        continue

    park_row       = park_rows.iloc[0]
    park_entrances = entrances[entrances['park_id'] == park_id]
    n_ent          = len(park_entrances)
    flag           = f">20 ({n_ent})" if n_ent > 20 else f"<=1 ({n_ent})"

    buf       = max(150, park_row.geometry.area ** 0.5 * 0.3)
    bbox      = park_row.geometry.buffer(buf).bounds
    clip_geom = park_row.geometry.buffer(buf)

    edges_clip = gpd.clip(edges, clip_geom)
    nodes_clip = gpd.clip(nodes, clip_geom)

    fig, ax = plt.subplots(figsize=(8, 8))
    gpd.GeoDataFrame([park_row], crs=parks.crs).plot(
        ax=ax, color='#c8e6c9', edgecolor='darkgreen', linewidth=1.5, alpha=0.6
    )
    if len(edges_clip) > 0:
        edges_clip.plot(ax=ax, color='#888888', linewidth=0.8, alpha=0.7)
    if len(nodes_clip) > 0:
        nodes_clip.plot(ax=ax, color='steelblue', markersize=8, alpha=0.6, zorder=3)
    if n_ent > 0:
        park_entrances.plot(ax=ax, color='red', markersize=40, zorder=5, marker='*')

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(
        f"{park_row['park_name']} [{park_id}]\n"
        f"{park_row['area_ha']:.1f} ha | {park_row['source']} | flag: {flag}",
        fontsize=9
    )
    ax.set_axis_off()
    plt.tight_layout()

    safe_name = park_row['park_name'].replace(' ', '_').replace('/', '_')[:30]
    fname     = f"{park_id}_{safe_name}.png"
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()

print(f"\nDone. {len(flagged_ids)} maps saved to {FIG_DIR}/")
