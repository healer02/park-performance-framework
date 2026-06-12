"""
vancouver/v01_place_ids.py  (was 06_merge_park_placeids.py)
Build the master park list with Google Place IDs for Vancouver.

Run this before v02_retrieve_reviews.py and v03_sentiment.py.
This script is Vancouver-specific — replication cities skip this sub-track.

Inputs:
    data/parks/processed/vancouver_parks_merged.csv
    data/parks/processed/parkperformance_placeIDs-AVERY.csv

Outputs:
    data/parks/processed/06-master-park-placeids.csv
"""

import os
import sys

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd

BASE_PATH   = "data/parks/processed/vancouver_parks_merged.csv"
AVERY_PATH  = "data/parks/processed/parkperformance_placeIDs-AVERY.csv"
OUTPUT_PATH = "data/parks/processed/06-master-park-placeids.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)


# %% 2. LOAD AND MERGE
base  = pd.read_csv(BASE_PATH)
avery = pd.read_csv(AVERY_PATH)

print(f"Base park list:  {len(base)} parks")
print(f"Avery place IDs: {len(avery)} parks")

for df in [base, avery]:
    df["_join_name"]   = df["park_name"].str.strip().str.lower()
    df["_join_source"] = df["source"].str.strip().str.lower()

avery["place_id"] = avery["place_id_proposed"].fillna(avery["place_id_old"])

avery_slim = avery[["_join_name", "_join_source", "place_id", "match_flag", "notes"]].copy()

merged = base.merge(avery_slim, on=["_join_name", "_join_source"], how="left")
merged = merged.drop(columns=["_join_name", "_join_source"])

no_id_mask = merged["place_id"].isna()
merged.loc[no_id_mask, "match_flag"] = "MISSING"

print(f"\nAfter merge: {len(merged)} parks | "
      f"With place_id: {(~no_id_mask).sum()} | Missing: {no_id_mask.sum()}")


# %% 3. PATCH MANUALLY FOUND PLACE IDs + REMOVE NON-PARKS
MANUAL_IDS = {
    "Stanley Park":                   "ChIJo-QmrYxxhlQRFuIJtJ1jSjY",
    "Queen Elizabeth Park":           "ChIJIcZrTvVzhlQRiKTnD03vt7Q",
    "Vandusen Botanical Garden":      "ChIJwW3HeIZzhlQRVxJgWI8VjAg",
    "Vanier Park":                    "ChIJcUir1MxzhlQRBUXcDCRBgoI",
    "Spanish Banks Extension":        "ChIJ7XKZJqdzhlQRioKAQM6UiTs",
    "Hastings Park - Italian Garden": "ChIJoQ4kvd9whlQRmGkWeuxCzpI",
    "Hastings Park - Sanctuary":      "ChIJM8cCYt9whlQRY02U31n5pbs",
    "Victory Square":                "ChIJgRB-c3lxhlQRxAU0p5Nk128",
    "Barclay Heritage Square":        "ChIJuRmp2ylyhlQRRRIBDobG380",
    "Marina Square":                  "ChIJO-T6lIhxhlQRCWwZrAhZ7vA",
    "Kinross Corridor – South":        "ChIJDdlYSlV1hlQR1hzKJVB7Ckc",
    "Kinross Corridor - Middle":       "ChIJ97K1cFF1hlQRZbJi19XTdwA",
    "sθәqәlxenәm ts'exwts'áxwi7 (Rainbow)": "ChIJ91Xo0aRxhlQRRaOE88piwQ4",
    "Cathedral Square":               "ChIJz2Z0KHlxhlQRLtddSsnlI6U",
    "Park Site on Point Grey at šxʷməθkʷəy̓əmasəm (Musq": "ChIJd7CrglFyhlQRhCfsNwzHzYE",
    "Sun Hop Park":                   "ChIJPSghNvtzhlQRiFdqmVl152A",
    "Lilian To":                      "ChIJ5RWQau9zhlQR7QbpfAPnPME",
    "Yaletown Park":                  "ChIJ2UOPGdZzhlQRwvt3gDG7RIY",
    "Choklit Park":                   "ChIJmYd6Q8RzhlQRnQcUS9mQFVo",
    "Willow Park":                    "ChIJI6DcWsNzhlQRkRpRrXP9q0w",
    "Street End - Wall St @ Nanaimo": "ChIJV6yYZh1xhlQRJb41JgTpR54",
    "Major Matthews Park":            "ChIJ2Zs7K-FzhlQRUrWRcS8n4jE",
    "Street End - Wall St @ Kamloops":"ChIJ0Ry2keJwhlQRx1QZL7yHxGg",
    "Street End - Wall St @ Penticton":"ChIJCSGA_eJwhlQRJo-DM9T0oVw",
    "Street End - Wall St @ Slocan":  "ChIJXayKduNwhlQRCIpah-P90nQ",
    "6th and Fir":                    "ChIJQ1sOFchzhlQRoXMYDrzBCi8",
    "5th and Pine":                   "ChIJ__-vMMhzhlQRvgmDj4xwh40",
    "Art Phillips Park":              "ChIJuSjYgoFxhlQRkNmzBbPyqU4",
    "Portal Park":                    "ChIJK5QMvYNxhlQRDW_gXLRmz9I",
    "Nat Bailey Stadium Park":        "REMOVE",
    "Empire Fields - Hastings Park":  "REMOVE",
    "Slidey Slides":                  "REMOVE",
    "BOUNDARY CREEK RAVINE PARK":     "REMOVE",
    "STILL CREEK CONSERVATION AREA":  "REMOVE",
    "Spanish Banks Beach Park":       "ChIJLQnMgZJyhlQRnRaqpqnAhgg, ChIJXRC8b41yhlQRLSiIqsnw5m0",
    "Spanish Banks Extension":        "REMOVE",
    "Locarno Park":                   "REMOVE",
    "Roundhouse Turntable Plaza":     "REMOVE",
    "Shannon Mews Park":              "REMOVE",
    "Downtown Skateboard Plaza":      "REMOVE",
    "Mont Royal Square":              "REMOVE",
    "Helmcken Park":                  "REMOVE",
    "West End minipark - GILFORD ST @ HARO ST": "REMOVE",
    "Gibby's Field":                  "REMOVE",
    "Sunset Beach Park":              "ChIJAWo0tC1yhlQRAL6Iz7Cs6G4, ChIJ0UCTbi1yhlQRQCKD7zPvHBk",
    "Moberly Park":                   "ChIJ7Rw__kV0hlQRQ_CXlmJWnFU",
    "Vanier Park (Cultural Harmony Grove)": "REMOVE",
    "East Fraserlands Neighbourhood Park South": "",
    "Park Site on Point Grey at Trafalgar": "ChIJqeRyDUxyhlQRhqdvgElWoRo",
    "Park Site On Trafalgar Street":  "ChIJ3StweQBzhlQROdIhlxaHYNU",
}

patched = 0
for park_name, place_id in MANUAL_IDS.items():
    mask = merged["park_name"] == park_name
    if mask.sum() == 0:
        print(f"  WARNING: '{park_name}' not found in base list")
        continue
    merged.loc[mask, "place_id"]   = place_id
    merged.loc[mask, "match_flag"] = "manual" if place_id != "REMOVE" else "removed"
    if place_id not in ("REMOVE",):
        patched += 1

n_before = len(merged)
merged   = merged[merged["place_id"] != "REMOVE"].copy()
merged   = merged[merged["park_name"].notna()].copy()

print(f"Removed {n_before - len(merged)} non-park entries")
print(f"Patched {patched} parks with manually found place IDs")
print(f"Still missing place_id: {merged['place_id'].isna().sum()} parks")

still_missing = merged[merged["place_id"].isna()][["source", "park_name", "area_ha"]]
if len(still_missing):
    print("\nParks still missing place IDs:")
    print(still_missing.sort_values("area_ha", ascending=False).to_string(index=False))


# %% 4. SUMMARY AND EXPORT
merged = merged[["park_id", "source", "park_name", "area_ha", "place_id", "match_flag", "notes"]]

print("\n=== FINAL SUMMARY ===")
print(f"Total parks:       {len(merged)}")
print(f"With place_id:     {merged['place_id'].notna().sum()}")
print(f"Missing place_id:  {merged['place_id'].isna().sum()}")
print(f"\nMatch flag breakdown:")
print(merged["match_flag"].value_counts())

merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
