# =============================================================================
# 07-retrieve-missing-reviews.py
# Purpose: Extract Google Reviews for parks missing from file 06
#
# Inputs:
#   - data/processed/11-park-coverage-audit.csv
#   - data/google-reviews/raw/from_social_sentiment_study/05-merged-all-reviews.csv
#
# Outputs:
#   - data/google-reviews/raw/07-missing-parks-raw/   (checkpoints)
#   - data/google-reviews/raw/07-missing-parks-reviews.csv
#   - data/google-reviews/raw/07-destination-parks-reviews.csv
#   - data/google-reviews/processed/07-all-reviews-complete.csv
# =============================================================================

# %%
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import time
from datetime import datetime
from apify_client import ApifyClient

# Load API key from local file (not committed to git)
with open('scripts/apify_key.txt', 'r') as f:
    API_KEY = f.read().strip()

ACTOR_ID      = "Xb8osYTtOjlsgI6k9"
AUDIT_PATH    = "data/processed/11-park-coverage-audit.csv"
EXISTING_PATH = "data/google-reviews/raw/from_social_sentiment_study/05-merged-all-reviews.csv"
CHECKPOINT_DIR= "data/google-reviews/raw/07-missing-parks-raw"
RAW_OUT       = "data/google-reviews/raw/07-missing-parks-reviews.csv"
DEST_OUT      = "data/google-reviews/raw/07-destination-parks-reviews.csv"
MERGED_OUT    = "data/google-reviews/processed/07-all-reviews-complete.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# %% 1. BUILD PULL LISTS
audit = pd.read_csv(AUDIT_PATH)

def clean_placeids(val):
    if pd.isna(val) or val == "":
        return []
    return [x.strip() for x in str(val).split(",")
            if x.strip().startswith("ChI") and len(x.strip()) >= 25]

# --- Destination parks: run individually with no review cap ---
# These have tens of thousands of reviews and need chunk_size=1
DESTINATION_IDS = [
    "ChIJo-QmrYxxhlQRFuIJtJ1jSjY",  # Stanley Park
    "ChIJIcZrTvVzhlQRiKTnD03vt7Q",  # Queen Elizabeth Park
    "ChIJwW3HeIZzhlQRVxJgWI8VjAg",  # VanDusen Botanical Garden
    "ChIJEaF3zSVzhlQRoHELsXvwmDM",  # Pacific Spirit Regional Park
    "ChIJAWo0tC1yhlQRAL6Iz7Cs6G4",  # Sunset Beach (ID 1)
    "ChIJ0UCTbi1yhlQRQCKD7zPvHBk",  # Sunset Beach (ID 2)
    "ChIJ7WHSBi9yhlQRdLXmpczA6wo",  # English Bay (ID 1)
    "ChIJ86zkaC9yhlQR4GhBA8iTiy4",  # English Bay (ID 2)
    "ChIJM8cCYt9whlQRY02U31n5pbs",  # Hastings Park - Sanctuary
    "ChIJoQ4kvd9whlQRmGkWeuxCzpI",  # Hastings Park - Italian Garden
    "ChIJcUir1MxzhlQRBUXcDCRBgoI",  # Vanier Park
]

# --- Remaining missing parks: exclude destination IDs and 0-review parks ---
ZERO_REVIEW_IDS = [
    "ChIJ____gOx2hlQRFFnP-o_4Quo",  # Price Park
    "ChIJDzRKSAB3hlQRQ6KtnnNUH9c",  # Wesburn Park
    "ChIJKS61DfV3hlQRIvEUMTi7k8E",  # Discovery Place Conservation Area
]

missing = audit[
    audit["in_neither"] == True
].copy()
missing = missing[missing["place_id"].notna() & (missing["place_id"] != "")].copy()
missing["place_id_list"] = missing["place_id"].apply(clean_placeids)
all_missing_ids = missing.explode("place_id_list")["place_id_list"].dropna().unique().tolist()

# Remaining = all missing minus destination and zero-review parks
remaining_ids = [
    pid for pid in all_missing_ids
    if pid not in DESTINATION_IDS and pid not in ZERO_REVIEW_IDS
]

print(f"Destination parks to pull (individually): {len(DESTINATION_IDS)}")
print(f"Remaining parks to pull:                  {len(remaining_ids)}")
print(f"Skipping (confirmed 0 reviews):           {len(ZERO_REVIEW_IDS)}")


# %% 2. APIFY EXTRACTION FUNCTION
os.environ["APIFY_TOKEN"] = API_KEY
client = ApifyClient(os.getenv("APIFY_TOKEN"))
print("Authenticated with Apify.")

def run_apify_in_chunks(place_ids, chunk_size=10, pause_sec=5,
                        max_reviews=99999, label="batch"):
    all_results = []
    total_chunks = (len(place_ids) + chunk_size - 1) // chunk_size

    for i in range(0, len(place_ids), chunk_size):
        chunk = place_ids[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"\n[{label}] Chunk {chunk_num}/{total_chunks} ({len(chunk)} PlaceIDs)...")
        print(f"  {chunk}")

        actor_input = {
            "placeIds":              chunk,
            "language":              "en",
            "maxReviews":            max_reviews,
            "reviewsSort":           "newest",
            "maxConcurrency":        3,
            "maxRequestRetries":     2,
            "maxRequestConcurrency": 2,
            "saveHtml":              False,
            "includeImages":         False,
        }

        try:
            run        = client.actor(ACTOR_ID).call(run_input=actor_input)
            dataset_id = run["defaultDatasetId"]
            items      = client.dataset(dataset_id).list_items().items
            print(f"  Retrieved {len(items)} reviews.")

            all_results.extend(items)

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"{label}_chunk_{chunk_num:02d}.csv"
            )
            pd.DataFrame(items).to_csv(checkpoint_path, index=False)
            print(f"  Saved checkpoint: {checkpoint_path}")

        except Exception as e:
            print(f"  ERROR on chunk {chunk_num}: {e}")
            continue

        if chunk_num < total_chunks:
            print(f"  Pausing {pause_sec}s...")
            time.sleep(pause_sec)

    print(f"\nDone [{label}]. Total reviews: {len(all_results)}")
    return pd.DataFrame(all_results)


# %% 3A. PULL DESTINATION PARKS (one at a time, no cap)
# WARNING: Stanley Park alone has ~49k reviews -- this will take 1-2 hours
print("=== PULLING DESTINATION PARKS (no review cap, chunk_size=1) ===")
dest_df = run_apify_in_chunks(
    DESTINATION_IDS,
    chunk_size=1,
    pause_sec=10,
    max_reviews=99999,
    label="dest"
)
dest_df.to_csv(DEST_OUT, index=False)
print(f"Saved destination parks raw: {DEST_OUT}")

# Quick sanity check
for name, pid in {
    "Stanley Park":    "ChIJo-QmrYxxhlQRFuIJtJ1jSjY",
    "Queen Elizabeth": "ChIJIcZrTvVzhlQRiKTnD03vt7Q",
    "VanDusen":        "ChIJwW3HeIZzhlQRVxJgWI8VjAg",
    "Sunset Beach 1":  "ChIJAWo0tC1yhlQRAL6Iz7Cs6G4",
    "English Bay 2":   "ChIJ86zkaC9yhlQR4GhBA8iTiy4",
}.items():
    count = len(dest_df[dest_df["placeId"] == pid]) if "placeId" in dest_df.columns else 0
    google_total = dest_df[dest_df["placeId"] == pid]["reviewsCount"].iloc[0] \
                   if count > 0 else "N/A"
    print(f"  {name}: {count} scraped / {google_total} on Google")


# %% 3B. PULL REMAINING PARKS (with cap, chunk_size=6)
print("\n=== PULLING REMAINING PARKS ===")
remaining_df = run_apify_in_chunks(
    remaining_ids,
    chunk_size=6,
    pause_sec=5,
    max_reviews=99999,
    label="remaining"
)

# Append to existing raw output (from first run)
existing_raw = pd.read_csv(RAW_OUT)
combined_raw = pd.concat([existing_raw, remaining_df, dest_df], ignore_index=True)
combined_raw = combined_raw.drop_duplicates(subset=["reviewId"], keep="first") \
              if "reviewId" in combined_raw.columns else combined_raw
combined_raw.to_csv(RAW_OUT, index=False)
print(f"Updated raw output: {RAW_OUT} ({len(combined_raw)} total rows)")

# %% 3C. PULL MISSED PLACE IDS (targeted re-pull)
print("\n=== PULLING MISSED PLACE IDS ===")

MISSED_IDS = [
    "ChIJXRC8b41yhlQRLSiIqsnw5m0",  # Spanish Banks Beach (956 reviews, missed in earlier pull)
]

missed_df = run_apify_in_chunks(
    MISSED_IDS,
    chunk_size=1,
    pause_sec=5,
    max_reviews=99999,
    label="missed"
)

# Append to existing raw output
existing_raw = pd.read_csv(RAW_OUT)
combined_raw = pd.concat([existing_raw, missed_df], ignore_index=True)
combined_raw = combined_raw.drop_duplicates(subset=["reviewId"], keep="first") \
              if "reviewId" in combined_raw.columns else combined_raw
combined_raw.to_csv(RAW_OUT, index=False)
print(f"Updated raw output: {RAW_OUT} ({len(combined_raw)} total rows)")

# Also append to complete merged file
existing_complete = pd.read_csv(MERGED_OUT, low_memory=False)
missed_std = missed_df.rename(columns={
    "placeId":  "PlaceID",
    "reviewId": "ReviewID",
    "stars":    "Rating",
})
missed_std["Review"] = (
    missed_std.get("textTranslated", pd.Series(dtype=str))
    .fillna(missed_std.get("text", pd.Series(dtype=str)))
)
common_cols = [c for c in existing_complete.columns if c in missed_std.columns]
merged_complete = pd.concat([existing_complete, missed_std[common_cols]], ignore_index=True)
merged_complete = merged_complete.drop_duplicates(subset=["ReviewID"], keep="first") \
                 if "ReviewID" in merged_complete.columns else merged_complete
merged_complete.to_csv(MERGED_OUT, index=False)
print(f"Updated complete file: {MERGED_OUT} ({len(merged_complete)} rows, "
      f"{merged_complete['PlaceID'].nunique()} PlaceIDs)")

# %% 4. COVERAGE DIAGNOSTIC
print("\n=== COVERAGE CHECK ===")
all_scraped_ids = set(combined_raw["placeId"].dropna().astype(str).str.strip().unique()) \
                  if "placeId" in combined_raw.columns else set()

for name, pid in {
    "Stanley Park":          "ChIJo-QmrYxxhlQRFuIJtJ1jSjY",
    "Queen Elizabeth":       "ChIJIcZrTvVzhlQRiKTnD03vt7Q",
    "VanDusen":              "ChIJwW3HeIZzhlQRVxJgWI8VjAg",
    "Pacific Spirit":        "ChIJEaF3zSVzhlQRoHELsXvwmDM",
    "Sunset Beach 1":        "ChIJAWo0tC1yhlQRAL6Iz7Cs6G4",
    "English Bay 2":         "ChIJ86zkaC9yhlQR4GhBA8iTiy4",
    "Hastings Sanctuary":    "ChIJM8cCYt9whlQRY02U31n5pbs",
    "Vanier Park":           "ChIJcUir1MxzhlQRBUXcDCRBgoI",
    "Central Park Burnaby":  "ChIJc6RIWYp2hlQRn1enR_bqb2A",
}.items():
    count = len(combined_raw[combined_raw["placeId"] == pid]) \
            if "placeId" in combined_raw.columns else 0
    total = combined_raw[combined_raw["placeId"] == pid]["reviewsCount"].iloc[0] \
            if count > 0 else "N/A"
    print(f"  {name}: {count} scraped / {total} on Google")


# %% 5. MERGE WITH EXISTING FILE 06
# Reload combined raw (already pulled, skip re-running Apify)
combined_raw = pd.read_csv(RAW_OUT)
print(f"Loaded combined raw: {len(combined_raw)} rows")

existing = pd.read_csv(EXISTING_PATH, low_memory=False)
print(f"\nExisting file 06 reviews: {len(existing)}")

# Standardise columns
scraped_std = combined_raw.rename(columns={
    "placeId":  "PlaceID",
    "reviewId": "ReviewID",
    "stars":    "Rating",
})

scraped_std["Review"] = (
    scraped_std.get("textTranslated", pd.Series(dtype=str))
    .fillna(scraped_std.get("text", pd.Series(dtype=str)))
)

common_cols = [c for c in existing.columns if c in scraped_std.columns]
scraped_std = scraped_std[common_cols].copy()

merged = pd.concat([existing, scraped_std], ignore_index=True)
print(f"Combined before dedup: {len(merged)}")

if "ReviewID" in merged.columns:
    merged = merged.drop_duplicates(subset=["ReviewID"], keep="first")
else:
    merged = merged.drop_duplicates(subset=["PlaceID", "Review"], keep="first")

print(f"After dedup: {len(merged)} rows")
print(f"Unique PlaceIDs: {merged['PlaceID'].nunique()}")

merged.to_csv(MERGED_OUT, index=False)
print(f"Saved: {MERGED_OUT}")


# %% 6. FINAL COVERAGE CHECK
print("\n=== FINAL COVERAGE CHECK ===")
audit = pd.read_csv(AUDIT_PATH)
merged_pids = set(merged["PlaceID"].dropna().unique())
covered = sum(
    any(pid in merged_pids for pid in clean_placeids(row["place_id"]))
    for _, row in audit.iterrows()
    if pd.notna(row["place_id"]) and row["place_id"] != ""
)
print(f"Parks covered: {covered} / {len(audit[audit['place_id'].notna()])}")
# %%
import pandas as pd

audit = pd.read_csv('data/processed/11-park-coverage-audit.csv')
complete = pd.read_csv('data/google-reviews/processed/07-all-reviews-complete.csv', low_memory=False)
master = pd.read_csv('data/parks/processed/06-master-park-placeids.csv')

complete_pids = set(complete['PlaceID'].dropna().unique())

def park_covered(place_id_str, pid_set):
    if pd.isna(place_id_str) or place_id_str == '':
        return False
    ids = [x.strip() for x in str(place_id_str).split(',')]
    return any(pid in pid_set for pid in ids)

master['now_covered'] = master['place_id'].apply(lambda x: park_covered(x, complete_pids))
print(master[~master['now_covered']][['park_name', 'place_id']])
# %%
import pandas as pd

complete = pd.read_csv('data/google-reviews/processed/07-all-reviews-complete.csv', low_memory=False)

dual_pid_parks = {
    'Jericho Beach':      ['ChIJ5xH-_PZyhlQR2oyzoLe8M-c', 'ChIJBdOUdVhyhlQRf5X1qM6aus8'],
    'Locarno Beach':      ['ChIJwzH1iPNyhlQRnNtrx--1dRo', 'ChIJU85SfvNyhlQROqo2GnhKkPY'],
    'Kitsilano Beach':    ['ChIJnVFsezVyhlQRf5O-Pe5uQ04', 'ChIJHUYfoDVyhlQRrGFzpe5fhaQ'],
    'Spanish Banks':      ['ChIJLQnMgZJyhlQRnRaqpqnAhgg', 'ChIJXRC8b41yhlQRLSiIqsnw5m0'],
}

for park, pids in dual_pid_parks.items():
    for pid in pids:
        count = len(complete[complete['PlaceID'] == pid])
        print(f"{park} | {pid}: {count} reviews")
    print()
# %%
complete = pd.read_csv('data/google-reviews/processed/07-all-reviews-complete.csv', low_memory=False)
has_text = complete['text'].notna() & (complete['text'].str.strip() != '')
print(f"Total: {len(complete)}")
print(f"With text: {has_text.sum()}")
print(f"Without text: {(~has_text).sum()}")
# %%
