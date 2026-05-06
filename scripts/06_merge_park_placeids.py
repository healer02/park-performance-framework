# =============================================================================
# 06_merge_park_placeids.py
# Purpose: Build master park list with place IDs for park performance study
#
# Inputs:
#   - data/parks/processed/vancouver_parks_merged.csv        (249 parks, base)
#   - data/parks/processed/parkperformance_placeIDs-AVERY.csv
#
# Output:
#   - data/parks/processed/06-master-park-placeids.csv
# =============================================================================

# %% -------------------------------------------------------------------------
# 1. IMPORTS AND PATHS
# ----------------------------------------------------------------------------
import pandas as pd

BASE_PATH   = "../data/parks/processed/vancouver_parks_merged.csv"
AVERY_PATH  = "../data/parks/processed/parkperformance_placeIDs-AVERY.csv"
OUTPUT_PATH = "../data/parks/processed/06-master-park-placeids.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)


# %% -------------------------------------------------------------------------
# 2. LOAD AND MERGE
# Joins Avery's place IDs onto the base park list.
# Rule: prefer place_id_proposed (Avery's correction) over place_id_old.
# ----------------------------------------------------------------------------
base  = pd.read_csv(BASE_PATH)
avery = pd.read_csv(AVERY_PATH)

print(f"Base park list:  {len(base)} parks")
print(f"Avery place IDs: {len(avery)} parks")

# Normalise join keys
for df in [base, avery]:
    df["_join_name"]   = df["park_name"].str.strip().str.lower()
    df["_join_source"] = df["source"].str.strip().str.lower()

# Resolve place_id: prefer proposed correction over old
avery["place_id"] = avery["place_id_proposed"].fillna(avery["place_id_old"])

avery_slim = avery[["_join_name", "_join_source", "place_id", "match_flag", "notes"]].copy()

merged = base.merge(avery_slim, on=["_join_name", "_join_source"], how="left")
merged = merged.drop(columns=["_join_name", "_join_source"])

no_id_mask = merged["place_id"].isna()
merged.loc[no_id_mask, "match_flag"] = "MISSING"

print(f"\nAfter merge: {len(merged)} parks | With place_id: {(~no_id_mask).sum()} | Missing: {no_id_mask.sum()}")

# %% -------------------------------------------------------------------------
# 3. PATCH MANUALLY FOUND PLACE IDs + REMOVE NON-PARKS
# ----------------------------------------------------------------------------
MANUAL_IDS = {
    # confirmed place IDs for destination parks 
    "Stanley Park":                   "ChIJo-QmrYxxhlQRFuIJtJ1jSjY",
    "Queen Elizabeth Park":           "ChIJIcZrTvVzhlQRiKTnD03vt7Q",
    "Vandusen Botanical Garden":      "ChIJwW3HeIZzhlQRVxJgWI8VjAg",
    "Vanier Park":                    "ChIJcUir1MxzhlQRBUXcDCRBgoI",
    "Spanish Banks Extension":        "ChIJ7XKZJqdzhlQRioKAQM6UiTs",
    "Hastings Park - Italian Garden": "ChIJoQ4kvd9whlQRmGkWeuxCzpI",
    "Hastings Park - Sanctuary":      "ChIJM8cCYt9whlQRY02U31n5pbs",
    # confirmed place IDs for small parks
    "Victory Square":                "ChIJgRB-c3lxhlQRxAU0p5Nk128",
    "Barclay Heritage Square"  : "ChIJuRmp2ylyhlQRRRIBDobG380",
    "Marina Square":                "ChIJO-T6lIhxhlQRCWwZrAhZ7vA",
    "Kinross Corridor – South"        : "ChIJDdlYSlV1hlQR1hzKJVB7Ckc",
    "Kinross Corridor - Middle"        : "ChIJ97K1cFF1hlQRZbJi19XTdwA",
    "sθәqәlxenәm ts'exwts'áxwi7 (Rainbow)": "ChIJ91Xo0aRxhlQRRaOE88piwQ4",
    "Cathedral Square":     "ChIJz2Z0KHlxhlQRLtddSsnlI6U",
    "Park Site on Point Grey at šxʷməθkʷəy̓əmasəm (Musq": "ChIJd7CrglFyhlQRhCfsNwzHzYE",
    "Sun Hop Park": "ChIJPSghNvtzhlQRiFdqmVl152A",
    "Lilian To": "ChIJ5RWQau9zhlQR7QbpfAPnPME",
    "Yaletown Park": "ChIJ2UOPGdZzhlQRwvt3gDG7RIY",
    "Street End - Wall St @ Kamloops": "ChIJ0Ry2keJwhlQRx1QZL7yHxGg", #Portside View Park
    "Street End - Wall St @ Penticton": "ChIJCSGA_eJwhlQRJo-DM9T0oVw", #Harbour View Park
    "Street End - Wall St @ Slocan" : "ChIJXayKduNwhlQRCIpah-P90nQ", #Commissioner Park
    "6th and Fir": "ChIJQ1sOFchzhlQRoXMYDrzBCi8", #Burrard Slopes Park
    "5th and Pine": "ChIJ__-vMMhzhlQRvgmDj4xwh40", #Pop-Up Park at 5th and Pine
    "Art Phillips Park": "ChIJuSjYgoFxhlQRkNmzBbPyqU4",
    "Portal Park": "ChIJK5QMvYNxhlQRDW_gXLRmz9I",

    # remove: stadiums, unfindable, or duplicates
    "Nat Bailey Stadium Park":        "REMOVE",
    "Empire Fields - Hastings Park":  "REMOVE",
    "Slidey Slides":                  "REMOVE",
    "BOUNDARY CREEK RAVINE PARK":     "REMOVE",
    "STILL CREEK CONSERVATION AREA":  "REMOVE",
    "Spanish Banks Beach Park": "ChIJLQnMgZJyhlQRnRaqpqnAhgg, ChIJXRC8b41yhlQRLSiIqsnw5m0",  # merged: Spanish Banks Beach Park + Spanish Banks Beach
    "Spanish Banks Extension":  "REMOVE",  # merged into the main polygon
    "Locarno Park":                   "REMOVE",  # duplicate of Locarno Beach Park
    "Roundhouse Turntable Plaza": "REMOVE", #not a park
    "Shannon Mews Park": "REMOVE", # not a park
    "Downtown Skateboard Plaza": "REMOVE", # not a park
    "Mont Royal Square": "REMOVE", # no review
    "Helmcken Park": "REMOVE", # not a park
    "West End minipark - GILFORD ST @ HARO ST": "REMOVE", # not a park
    "Gibby's Field": "REMOVE", # not a park
}

patched = 0
for park_name, place_id in MANUAL_IDS.items():
    mask = merged["park_name"] == park_name
    if mask.sum() == 0:
        print(f"⚠️  '{park_name}' not found in base list — check spelling")
        continue
    merged.loc[mask, "place_id"]   = place_id
    merged.loc[mask, "match_flag"] = "manual" if place_id != "REMOVE" else "removed"
    if place_id not in ("REMOVE",):
        patched += 1

# Drop removed parks
n_before = len(merged)
merged = merged[merged["place_id"] != "REMOVE"].copy()
# remove rows with no park name
merged = merged[merged["park_name"].notna()].copy()

print(f"Removed {n_before - len(merged)} non-park entries")
print(f"Patched {patched} parks with manually found place IDs")
print(f"Still missing place_id: {merged['place_id'].isna().sum()} parks")

still_missing = merged[merged["place_id"].isna()][["source", "park_name", "area_ha"]]
if len(still_missing):
    print("\nParks still missing place IDs:")
    print(still_missing.sort_values("area_ha", ascending=False).to_string(index=False))

# %% -------------------------------------------------------------------------
# 4. SUMMARY AND EXPORT
# ----------------------------------------------------------------------------
merged = merged[["park_id", "source", "park_name", "area_ha", "place_id", "match_flag", "notes"]]

print("\n=== FINAL SUMMARY ===")
print(f"Total parks:       {len(merged)}")
print(f"With place_id:     {merged['place_id'].notna().sum()}")
print(f"Missing place_id:  {merged['place_id'].isna().sum()}")
print(f"\nMatch flag breakdown:")
print(merged["match_flag"].value_counts())

merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved to: {OUTPUT_PATH}")


# %%
