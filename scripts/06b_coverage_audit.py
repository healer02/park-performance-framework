# =============================================================================
# 06b-coverage-audit.py
# Purpose: Audit review coverage against 236-park master
#
# Inputs:
#   - data/parks/processed/06-master-park-placeids.csv
#   - data/google-reviews/raw/from_social_sentiment_study/06-all-text-reviews-predicted.csv
#
# Outputs:
#   - data/processed/11-park-coverage-audit.csv
#   - data/processed/11-review-master.csv
#   - data/processed/11-park-metrics.csv
# =============================================================================

# %%
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import numpy as np

# %% 1. LOAD FILES
master  = pd.read_csv("data/parks/processed/06-master-park-placeids.csv")
reviews = pd.read_csv("data/google-reviews/raw/from_social_sentiment_study/06-all-text-reviews-predicted.csv",
                      low_memory=False)

print(f"Master parks:       {len(master)}")
print(f"File 06 reviews:    {len(reviews)}")
print(f"File 06 PlaceIDs:   {reviews['PlaceID'].nunique()}")

# %% 2. EXPLODE MASTER ON PLACE ID (handles dual PlaceID parks)
master["place_id_list"] = master["place_id"].apply(
    lambda x: [i.strip() for i in x.split(",")] if pd.notna(x) and x != "" else []
)
master_exploded = master.explode("place_id_list").rename(
    columns={"place_id_list": "PlaceID"}
)
master_exploded = master_exploded[master_exploded["PlaceID"] != ""].copy()

print(f"\nMaster rows after explode: {len(master_exploded)}")
print(f"Unique PlaceIDs in master: {master_exploded['PlaceID'].nunique()}")


# %% 3. COVERAGE AUDIT
place_ids_master = set(master_exploded["PlaceID"].dropna())
place_ids_file06 = set(reviews["PlaceID"].dropna())

def park_covered(place_id_str, pid_set):
    if pd.isna(place_id_str):
        return False
    ids = [x.strip() for x in place_id_str.split(",")]
    return any(pid in pid_set for pid in ids)

master["in_file06"]  = master["place_id"].apply(lambda x: park_covered(x, place_ids_file06))
master["in_neither"] = ~master["in_file06"]

print(f"\n--- Park-level coverage (n={len(master)}) ---")
print(f"In file 06 (reviews):  {master['in_file06'].sum()}")
print(f"Missing from file 06:  {master['in_neither'].sum()}")

print(f"\nParks missing from file 06 (Apify pull list):")
print(master[~master['in_file06']][['park_id', 'park_name', 'place_id']].to_string())

master.to_csv("data/processed/11-park-coverage-audit.csv", index=False)
print("\nSaved coverage audit.")


# %% 4. BUILD REVIEW-LEVEL MASTER (file 06 as base)
review_master = reviews.merge(
    master_exploded[["PlaceID", "park_id", "park_name", "area_ha"]],
    on="PlaceID",
    how="left"
)

unmatched = review_master["park_id"].isna().sum()
print(f"\nReviews with no park_id match: {unmatched}")
print(f"Review-level master rows: {len(review_master)}")

review_master.to_csv("data/processed/11-review-master.csv", index=False)
print("Saved review-level master.")


# %% 5. BUILD PARK-LEVEL METRICS (rating only -- sentiment added after Apify + RoBERTa)
place_metrics = reviews.groupby("PlaceID").agg(
    totalScore    = ("totalScore",   "first"),
    reviewsCount  = ("reviewsCount", "first"),
    n_text_reviews= ("ReviewID",     "count"),
).reset_index()

joined = master_exploded.merge(place_metrics, on="PlaceID", how="left")

def park_agg(g):
    rat_mask = g["totalScore"].notna() & g["reviewsCount"].notna()
    avg_rating = (
        np.average(g.loc[rat_mask, "totalScore"],
                   weights=g.loc[rat_mask, "reviewsCount"])
        if rat_mask.any() else None
    )
    return pd.Series({
        "AvgRating":      round(avg_rating, 4) if avg_rating is not None else None,
        "TotalReviews":   g["reviewsCount"].sum(),
        "n_text_reviews": g["n_text_reviews"].sum(),
        "has_rating":     g["totalScore"].notna().any(),
    })

park_metrics = joined.groupby("park_id").apply(park_agg).reset_index()
park_metrics = park_metrics.merge(
    master[["park_id", "park_name", "area_ha"]], on="park_id", how="left"
)

print(f"\n--- Park-level metrics summary (rating only) ---")
print(f"Parks with AvgRating: {park_metrics['has_rating'].sum()}")
print(f"Parks without rating: {(~park_metrics['has_rating']).sum()}")
print(f"\nAvgRating summary:")
print(park_metrics["AvgRating"].describe().round(3))

park_metrics.to_csv("data/processed/11-park-metrics.csv", index=False)
print("\nSaved park-level metrics.")


# %% 6. UNMATCHED REVIEW DIAGNOSTIC
unmatched_pids = review_master[review_master["park_id"].isna()]["PlaceID"].unique()
print(f"Unmatched PlaceIDs: {len(unmatched_pids)}")

unmatched_info = reviews[reviews["PlaceID"].isin(unmatched_pids)][
    ["PlaceID", "title", "reviewsCount"]
].drop_duplicates("PlaceID")
print(unmatched_info.to_string())
# %%
