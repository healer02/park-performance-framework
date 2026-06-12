"""
vancouver/v02_retrieve_reviews.py  (was 06c_retrieve_reviews.py)
Scrape Google Reviews for all Vancouver parks via Apify and merge with
the prior-study review file.

This script is Vancouver-specific — replication cities skip this sub-track.
WARNING: Running the full pull for Stanley Park (~49k reviews) takes 1–2 hours.

Inputs:
    data/processed/11-park-coverage-audit.csv
    data/google-reviews/raw/from_social_sentiment_study/05-merged-all-reviews.csv
    data/parks/processed/06-master-park-placeids.csv
    scripts/apify_key.txt   (gitignored)

Outputs:
    data/google-reviews/raw/07-missing-parks-raw/    (checkpoints)
    data/google-reviews/raw/07-missing-parks-reviews.csv
    data/google-reviews/raw/07-destination-parks-reviews.csv
    data/google-reviews/processed/07-all-reviews-complete.csv
"""

import os
import sys

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd
import time
from apify_client import ApifyClient

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

ZERO_REVIEW_IDS = [
    "ChIJ____gOx2hlQRFFnP-o_4Quo",  # Price Park
    "ChIJDzRKSAB3hlQRQ6KtnnNUH9c",  # Wesburn Park
    "ChIJKS61DfV3hlQRIvEUMTi7k8E",  # Discovery Place Conservation Area
]

missing = audit[audit["in_neither"] == True].copy()
missing = missing[missing["place_id"].notna() & (missing["place_id"] != "")].copy()
missing["place_id_list"] = missing["place_id"].apply(clean_placeids)
all_missing_ids = missing.explode("place_id_list")["place_id_list"].dropna().unique().tolist()

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
    all_results  = []
    total_chunks = (len(place_ids) + chunk_size - 1) // chunk_size

    for i in range(0, len(place_ids), chunk_size):
        chunk     = place_ids[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"\n[{label}] Chunk {chunk_num}/{total_chunks} ({len(chunk)} PlaceIDs)...")

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
print("=== PULLING DESTINATION PARKS (no review cap, chunk_size=1) ===")
dest_df = run_apify_in_chunks(
    DESTINATION_IDS, chunk_size=1, pause_sec=10, max_reviews=99999, label="dest"
)
dest_df.to_csv(DEST_OUT, index=False)
print(f"Saved destination parks raw: {DEST_OUT}")


# %% 3B. PULL REMAINING PARKS
print("\n=== PULLING REMAINING PARKS ===")
remaining_df = run_apify_in_chunks(
    remaining_ids, chunk_size=6, pause_sec=5, max_reviews=99999, label="remaining"
)

existing_raw  = pd.read_csv(RAW_OUT)
combined_raw  = pd.concat([existing_raw, remaining_df, dest_df], ignore_index=True)
combined_raw  = combined_raw.drop_duplicates(subset=["reviewId"], keep="first") \
               if "reviewId" in combined_raw.columns else combined_raw
combined_raw.to_csv(RAW_OUT, index=False)
print(f"Updated raw output: {RAW_OUT} ({len(combined_raw)} rows)")


# %% 4. MERGE WITH EXISTING FILE 06
combined_raw = pd.read_csv(RAW_OUT)
existing     = pd.read_csv(EXISTING_PATH, low_memory=False)
print(f"\nExisting file 06 reviews: {len(existing)}")

scraped_std          = combined_raw.rename(columns={"placeId": "PlaceID", "reviewId": "ReviewID", "stars": "Rating"})
scraped_std["Review"] = (
    scraped_std.get("textTranslated", pd.Series(dtype=str))
    .fillna(scraped_std.get("text", pd.Series(dtype=str)))
)
common_cols  = [c for c in existing.columns if c in scraped_std.columns]
scraped_std  = scraped_std[common_cols].copy()

merged_out = pd.concat([existing, scraped_std], ignore_index=True)
if "ReviewID" in merged_out.columns:
    merged_out = merged_out.drop_duplicates(subset=["ReviewID"], keep="first")
else:
    merged_out = merged_out.drop_duplicates(subset=["PlaceID", "Review"], keep="first")

print(f"After dedup: {len(merged_out)} rows, {merged_out['PlaceID'].nunique()} PlaceIDs")

merged_out.to_csv(MERGED_OUT, index=False)
print(f"Saved: {MERGED_OUT}")

print("\nDone.")
