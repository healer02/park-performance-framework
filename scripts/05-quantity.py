"""
05-quantity.py
Computes DA-level park quantity (reachable green space supply per capita).

Method:
    For each DB centroid:
        1. Run single_source_dijkstra from DB's nearest_node (cutoff=400m)
        2. Find all park entrance nodes within 400m
        3. Collect unique reachable park_ids for this DB

    Aggregate to DA level:
        4. Union of reachable park_ids across all DBs in DA
        5. Sum area_ha for unique parks (not double-counted)
        6. quantity = total_unique_area / DA_pop * 1000 (ha per 1,000 residents)

Area cap variants:
    - Main:        min(area_ha, 20) — caps large parks at 20 ha
    - Sensitivity: uncapped area_ha
    - Sensitivity: min(area_ha, 10)

Inputs:
    - data/osm/Vancouver_walk.graphml
    - data/parks/processed/vancouver_park_entrances.shp
    - data/parks/processed/vancouver_parks_merged.shp
    - data/census/processed/vancouver_db_centroids.gpkg
    - data/processed/vancouver_da_reachability.gpkg

Outputs:
    - data/processed/vancouver_da_quantity.csv
    - data/processed/vancouver_da_supply.gpkg
    - outputs/figures/vancouver_da_quantity_check.png

Notes:
    - DA-level union approach: each park counted once per DA regardless of
      how many DBs can reach it (avoids double-counting)
    - Zero-population DBs excluded from computation
    - CRS: EPSG:3005 throughout
"""
# %%  Imports and setup
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────

GRAPH_PATH  = 'data/osm/Vancouver_walk.graphml'
ENT_PATH    = 'data/parks/processed/vancouver_park_entrances.shp'
PARKS_PATH  = 'data/parks/processed/vancouver_parks_merged.shp'
DB_PATH     = 'data/census/processed/vancouver_db_centroids.gpkg'
REACH_PATH  = 'data/processed/vancouver_da_reachability.gpkg'
OUT_DIR     = 'data/processed'
FIG_DIR     = 'outputs/figures'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLD = 400   # metres
CAP_MAIN  = 20    # ha — main area cap
CAP_SENS  = 10    # ha — sensitivity area cap

# %% Step 1-2: Load inputs and build lookup
# ── Step 1: Load inputs ───────────────────────────────────────────────────────

print("Step 1: Loading inputs...")
G            = ox.load_graphml(GRAPH_PATH)
entrances    = gpd.read_file(ENT_PATH)
parks        = gpd.read_file(PARKS_PATH)
db_centroids = gpd.read_file(DB_PATH)

print(f"  Graph nodes:    {G.number_of_nodes():,}")
print(f"  Park entrances: {len(entrances)}")
print(f"  Parks:          {len(parks)}")
print(f"  DB centroids:   {len(db_centroids)}")

# ── Step 2: Build entrance node → park_id lookup ──────────────────────────────

print("\nStep 2: Building entrance node → park lookup...")

node_col = 'nearest_no' if 'nearest_no' in entrances.columns else 'nearest_node'
print(f"  Using entrance node column: {node_col}")

node_to_parks = {}
for _, row in entrances.iterrows():
    node = row[node_col]
    pid  = row['park_id']
    if pd.notna(node):
        node = int(node)
        node_to_parks.setdefault(node, set()).add(pid)

print(f"  Unique entrance nodes mapped: {len(node_to_parks)}")

# Build park area lookup
park_area = parks.set_index('park_id')['area_ha'].to_dict()

# One-time check for missing park_ids in area lookup
all_pids = set(pid for pids in node_to_parks.values() for pid in pids)
missing_pids = all_pids - set(park_area.keys())
if missing_pids:
    print(f"  WARNING: {len(missing_pids)} park_ids in entrances not found in parks layer")

entrance_node_set = set(node_to_parks.keys())

# %% Step 3: DB loop (run once, takes 5-15 min)
# ── Step 3: DB-level reachable park sets ──────────────────────────────────────

print(f"\nStep 3: Computing DB-level reachable park sets (cutoff={THRESHOLD}m)...")
print(f"  Processing {len(db_centroids)} DBs — may take 5–15 minutes...")

# DA → set of unique reachable park_ids (union across all DBs in DA)
da_park_sets  = {}
da_pop_total  = {}  # all DB population (for reachability denominator)
da_pop_valid  = {}  # only DBs with valid nearest_node (for quantity denominator)

n_processed = 0

for _, db_row in db_centroids.iterrows():
    dauid  = db_row['DAUID']
    db_pop = int(db_row['db_pop'])

    # Accumulate total DA population (all DBs)
    da_pop_total[dauid] = da_pop_total.get(dauid, 0) + db_pop

    # Skip zero-pop DBs
    if db_pop == 0:
        n_processed += 1
        continue

    # Skip DBs with missing nearest_node — exclude from valid pop denominator
    if pd.isna(db_row['nearest_node']):
        n_processed += 1
        continue

    # This DB has valid node — count toward quantity denominator
    da_pop_valid[dauid] = da_pop_valid.get(dauid, 0) + db_pop

    db_node = int(db_row['nearest_node'])

    # Shortest-path distances from this DB node
    try:
        distances = nx.single_source_dijkstra_path_length(
            G, db_node, cutoff=THRESHOLD, weight='length'
        )
    except nx.NodeNotFound:
        n_processed += 1
        continue

    # Reachable entrance nodes → unique park_ids
    reachable_entrance_nodes = set(distances.keys()) & entrance_node_set
    reachable_parks = set()
    for node in reachable_entrance_nodes:
        reachable_parks.update(node_to_parks[node])

    # Union into DA park set
    if dauid not in da_park_sets:
        da_park_sets[dauid] = set()
    da_park_sets[dauid].update(reachable_parks)

    n_processed += 1
    if n_processed % 500 == 0:
        print(f"  Processed {n_processed}/{len(db_centroids)} DBs...")

print(f"  Done. DAs with reachable parks: {len(da_park_sets)}")

# %% Step 4-5: Aggregate and save
# ── Step 4: Aggregate to DA level ─────────────────────────────────────────────

print("\nStep 4: Computing DA-level quantity...")

da_records = []

for dauid, pop_total in da_pop_total.items():
    park_ids  = da_park_sets.get(dauid, set())
    n_parks   = len(park_ids)
    pop_valid = da_pop_valid.get(dauid, 0)  # DBs with valid nearest_node only

    if pop_valid == 0:
        # No valid DBs — truly missing, exclude from analysis
        da_records.append({
            'DAUID':          dauid,
            'db_pop_total':   pop_total,
            'db_pop_valid':   pop_valid,
            'n_unique_parks': n_parks,
            'area_raw':       None,
            'area_cap20':     None,
            'area_cap10':     None,
            'qty_raw':        None,
            'qty_cap20':      None,
            'qty_cap10':      None,
        })
        continue
    elif n_parks == 0:
        # Has population but no reachable parks — genuine access gap, quantity = 0
        da_records.append({
            'DAUID':          dauid,
            'db_pop_total':   pop_total,
            'db_pop_valid':   pop_valid,
            'n_unique_parks': 0,
            'area_raw':       0,
            'area_cap20':     0,
            'area_cap10':     0,
            'qty_raw':        0,
            'qty_cap20':      0,
            'qty_cap10':      0,
        })
        continue
    else:
        area_raw   = sum(park_area.get(pid, 0) for pid in park_ids)
        area_cap20 = sum(min(park_area.get(pid, 0), CAP_MAIN) for pid in park_ids)
        area_cap10 = sum(min(park_area.get(pid, 0), CAP_SENS) for pid in park_ids)

        da_records.append({
            'DAUID':          dauid,
            'db_pop_total':   pop_total,
            'db_pop_valid':   pop_valid,
            'n_unique_parks': n_parks,
            'area_raw':       round(area_raw,   2),
            'area_cap20':     round(area_cap20, 2),
            'area_cap10':     round(area_cap10, 2),
            'qty_raw':        round(area_raw   / pop_valid * 1000, 4),
            'qty_cap20':      round(area_cap20 / pop_valid * 1000, 4),
            'qty_cap10':      round(area_cap10 / pop_valid * 1000, 4),
        })

da_quantity = pd.DataFrame(da_records)

print(f"  DAs in output: {len(da_quantity)}")
print(f"\n  qty_cap20 (main, ha per 1,000 residents) summary:")
print(da_quantity['qty_cap20'].describe().round(3).to_string())
print(f"\n  DAs with 0 quantity:    {(da_quantity['qty_cap20'] == 0).sum()}")
print(f"  DAs with null quantity: {da_quantity['qty_cap20'].isna().sum()}")

# ── Step 5: Save quantity CSV ─────────────────────────────────────────────────

print("\nStep 5: Saving DA quantity CSV...")
csv_path = os.path.join(OUT_DIR, 'vancouver_da_quantity.csv')
da_quantity.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# ── Step 6: Merge with reachability and save combined supply GeoPackage ───────

print("\nStep 6: Merging with reachability and saving supply GeoPackage...")
da_reach = gpd.read_file(REACH_PATH)
da_supply = da_reach.merge(
    da_quantity[['DAUID', 'n_unique_parks', 'area_raw', 'area_cap20',
                 'area_cap10', 'qty_raw', 'qty_cap20', 'qty_cap10']],
    on='DAUID', how='left'
)
gpkg_path = os.path.join(OUT_DIR, 'vancouver_da_supply.gpkg')
da_supply.to_file(gpkg_path, driver='GPKG')
print(f"  Saved: {gpkg_path}")

# %% Step 7: Visualization (re-run this cell to tweak plots)
# ── Step 7: Visual validation ─────────────────────────────────────────────────

print("\nStep 7: Generating visual check...")

cmap = LinearSegmentedColormap.from_list('quantity', ['#d73027', '#fee08b', '#1a9850'])

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

parks = gpd.read_file(PARKS_PATH)

for ax, col, title in zip(
    axes,
    ['qty_cap20', 'qty_cap10'],
    [f'Quantity — capped at {CAP_MAIN} ha (main)',
     f'Quantity — capped at {CAP_SENS} ha (sensitivity)']
):
    da_supply.plot(
        ax=ax, column=col, cmap=cmap, legend=True,
        missing_kwds={'color': '#cccccc', 'label': 'No data'},
        legend_kwds={'label': 'Ha per 1,000 residents', 'shrink': 0.5}
    )
    parks.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5, alpha=0.7, zorder=2)
    ax.set_title(f'DA-Level Park Quantity — Vancouver\n{title}')
    ax.set_axis_off()

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'vancouver_da_quantity_check2.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  Saved: {fig_path}")

print("\nDone.")

# %% step 8: 2x2 supply map
import pandas as pd
da_supply = gpd.read_file('data/processed/vancouver_da_supply.gpkg')

# Median split both variables
da_supply['reach_hi'] = (da_supply['DA_reach_400'] >= da_supply['DA_reach_400'].median()).astype(int)
da_supply['qty_hi']   = (da_supply['qty_cap20'] >= da_supply['qty_cap20'].median()).astype(int)

print(pd.crosstab(da_supply['reach_hi'], da_supply['qty_hi'],
                  rownames=['Reachability (hi=1)'],
                  colnames=['Quantity (hi=1)'],
                  margins=True))


# %% Supply side-by-side map
import numpy as np

da_supply = gpd.read_file('data/processed/vancouver_da_supply.gpkg')
parks     = gpd.read_file('data/parks/processed/vancouver_parks_merged.shp')

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Panel 1: Reachability — YlGn
da_supply.plot(ax=axes[0], column='DA_reach_400', cmap='YlGn',
    vmin=0, vmax=1, legend=True,
    missing_kwds={'color': '#cccccc'},
    legend_kwds={'label': 'Proportion within 400m', 'shrink': 0.5})
parks.plot(ax=axes[0], color='none', edgecolor='grey', linewidth=0.5, alpha=0.7, zorder=2)
axes[0].set_title('Coverage\nProportion of DA population within 400m of a park entrance')
axes[0].set_axis_off()

# Panel 2: Quantity — YlGnBu to distinguish
qty_p95 = da_supply['qty_cap20'].quantile(0.95)
da_supply.plot(ax=axes[1], column='qty_cap20', cmap='YlGnBu',
    vmin=0, vmax=qty_p95, legend=True,
    missing_kwds={'color': '#cccccc'},
    legend_kwds={'label': f'Ha per 1,000 residents\n(per-park cap=20ha; display max={qty_p95:.0f}ha)', 'shrink': 0.5})
parks.plot(ax=axes[1], color='none', edgecolor='grey', linewidth=0.5, alpha=0.7, zorder=2)
axes[1].set_title('Intensity\nReachable park area per 1,000 residents (per-park cap = 20 ha)')
axes[1].set_axis_off()

plt.suptitle('Park Supply Dimensions — Coverage vs. Intensity\nVancouver Dissemination Areas (2021)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('outputs/figures/vancouver_da_supply_2x2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved. Quantity display capped at 95th percentile: {qty_p95:.1f} ha/1,000')


# %% Supply typology map v2
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

da_supply = gpd.read_file('data/processed/vancouver_da_supply.gpkg')
parks     = gpd.read_file('data/parks/processed/vancouver_parks_merged.shp')

# Thresholds — 0.8 for reachability (avoids brittle 1.00 split), median for quantity
REACH_THRESH = 0.8
qty_med      = da_supply['qty_cap20'].median()

da_supply['reach_cat'] = (da_supply['DA_reach_400'] >= REACH_THRESH).astype(int)
da_supply['qty_cat']   = (da_supply['qty_cap20']    >= qty_med).astype(int)

def classify_supply(r, q):
    if pd.isna(r) or pd.isna(q): return 'No data'
    if r==1 and q==1: return 'HH — Well-served'
    if r==1 and q==0: return 'HL — High access, small area'
    if r==0 and q==1: return 'LH — High area, partial access'
    return 'LL — Underserved'

da_supply['supply_type'] = [
    classify_supply(r, q)
    for r, q in zip(da_supply['reach_cat'], da_supply['qty_cat'])
]

counts = da_supply['supply_type'].value_counts()
print(counts)
mismatch_pct = 100 * (counts.get('HL — Access without area', 0) +
                      counts.get('LH — Area without access', 0)) / len(da_supply)
print(f"\nMismatch DAs (HL+LH): {mismatch_pct:.0f}%")
print(f"Thresholds — Reachability: {REACH_THRESH}, Quantity median: {qty_med:.1f} ha/1,000")

# Colours — HL more saturated to distinguish from LH
colours = {
    'HH — Well-served':               '#3b2f3a',  # deeper, more saturated dark (clear peak)
    'HL — High access, small area':   '#4e8bab',  # blue (keep)
    'LH — High area, partial access': '#c27d4f',  # warmer, clearer orange
    'LL — Underserved':               '#f2eadf',  # lighter, more neutral base
}

fig, ax = plt.subplots(figsize=(12, 10))

for stype, colour in colours.items():
    subset = da_supply[da_supply['supply_type'] == stype]
    if len(subset) > 0:
        subset.plot(ax=ax, color=colour, edgecolor='white', linewidth=0.2)

parks.plot(ax=ax, facecolor='none', edgecolor='#2d6a2d', linewidth=1.0, zorder=2)

patches = [mpatches.Patch(color=c, label=f"{t} (n={counts.get(t, 0)})")
           for t, c in colours.items() if t != 'No data']
patches.append(mpatches.Patch(facecolor='none', edgecolor='#2d6a2d',
                               linewidth=1.0, label='Park boundaries'))
ax.legend(handles=patches, loc='lower left', fontsize=9, framealpha=0.9)

ax.set_title(
    f'Park Supply Typology — Vancouver DAs\n'
    f'Substantial mismatch between coverage and intensity (~{mismatch_pct:.0f}% of DAs)\n'
    f'Thresholds: reachability ≥ {REACH_THRESH}, quantity ≥ median ({qty_med:.1f} ha/1,000)',
    fontsize=11)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('outputs/figures/vancouver_da_supply_typology.png', dpi=150, bbox_inches='tight')
plt.close()

# add asymmetry note to print output
hl_n = counts.get('HL — Access without area', 0)
lh_n = counts.get('LH — Area with partial access', 0)
print(f"Mismatch asymmetry: HL={hl_n} vs LH={lh_n} — "
      f"small-park coverage more common than insufficient area")

print('Saved.')
# %%
