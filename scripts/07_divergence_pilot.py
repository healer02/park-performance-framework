# %% 1. LOAD
import pandas as pd

MASTER_PATH    = "../data/parks/processed/06-master-park-placeids.csv"
SENTIMENT_PATH = "../data/google-reviews/processed/10-avgsentiment-avgrating.csv"
METADATA_PATH  = "../data/google-reviews/processed/09a-all-place-metadata.csv"

master   = pd.read_csv(MASTER_PATH)
sentiment = pd.read_csv(SENTIMENT_PATH)
metadata  = pd.read_csv(METADATA_PATH)

print(f"Master parks: {len(master)}")
print(f"Sentiment file columns: {sentiment.columns.tolist()}")
print(f"Metadata file columns: {metadata.columns.tolist()}")
print(sentiment.head(3))


# %% 2. JOIN TO MASTER
# Master has comma-separated PlaceIDs for some parks -- explode to join, then collapse back

master["place_id_list"] = master["place_id"].str.split(",").apply(
    lambda ids: [x.strip() for x in ids]
)

# Explode so each PlaceID gets its own row
master_exploded = master.explode("place_id_list").rename(columns={"place_id_list": "PlaceID"})

# Join sentiment and metadata
joined = master_exploded.merge(sentiment, on="PlaceID", how="left")
joined = joined.merge(metadata[["PlaceID", "TotalReviews_All"]], on="PlaceID", how="left")

# Collapse back to park level (average across multiple PlaceIDs)
park_metrics = joined.groupby("park_id").agg(
    source        = ("source", "first"),
    park_name     = ("park_name", "first"),
    area_ha       = ("area_ha", "first"),
    AvgSentiment  = ("AvgSentiment", "mean"),
    AvgRating     = ("AvgRating", "mean"),
    TotalReviews  = ("TotalReviews_All", "sum"),
).reset_index()

print(f"Parks with sentiment data: {park_metrics['AvgSentiment'].notna().sum()}")
print(f"Parks with review counts:  {park_metrics['TotalReviews'].notna().sum()}")
print(f"Parks with no data:        {park_metrics['AvgSentiment'].isna().sum()}")
print(park_metrics.sort_values("TotalReviews", ascending=False).head(10).to_string(index=False))


# %% 3. DIVERGENCE MATRIX (PILOT - VANCOUVER ONLY, EXPERIENCE SIDE)
import numpy as np

van = park_metrics[park_metrics["source"] == "Vancouver"].copy()
van_exp = van[van["AvgSentiment"].notna()].copy()
van_exp = van_exp[van_exp["TotalReviews"] > 0].copy()

# Log-transform review counts to compress skew from destination parks
van_exp["LogReviews"] = np.log1p(van_exp["TotalReviews"])

# Define high/low thresholds at city median
rating_median  = van_exp["AvgRating"].median()
log_reviews_median = van_exp["LogReviews"].median()

van_exp["satisfaction"] = (van_exp["AvgRating"] >= rating_median).map({True: "high", False: "low"})
van_exp["salience"]     = (van_exp["LogReviews"] >= log_reviews_median).map({True: "high", False: "low"})
van_exp["experience_type"] = van_exp["salience"] + "_salience_" + van_exp["satisfaction"] + "_satisfaction"

print(f"Rating median:       {rating_median:.3f}")
print(f"Log-reviews median:  {log_reviews_median:.3f} (= ~{int(np.expm1(log_reviews_median))} raw reviews)")
print(f"\nExperience quadrant distribution:")
print(van_exp["experience_type"].value_counts())

print(f"\nSample high salience + high satisfaction parks:")
hh = van_exp[van_exp["experience_type"] == "high_salience_high_satisfaction"]
print(hh[["park_name", "area_ha", "AvgRating", "TotalReviews"]].sort_values("TotalReviews", ascending=False).head(5).to_string(index=False))

print(f"\nSample low salience + low satisfaction parks:")
ll = van_exp[van_exp["experience_type"] == "low_salience_low_satisfaction"]
print(ll[["park_name", "area_ha", "AvgRating", "TotalReviews"]].sort_values("TotalReviews", ascending=False).head(5).to_string(index=False))

# %% 4. PEEK AT DIVERGENCE QUADRANTS
print("High salience + LOW satisfaction (performing below expectations):")
hl = van_exp[van_exp["experience_type"] == "high_salience_low_satisfaction"]
print(hl[["park_name", "area_ha", "AvgRating", "TotalReviews"]].sort_values("TotalReviews", ascending=False).to_string(index=False))

print("\nLow salience + HIGH satisfaction (hidden gems):")
lh = van_exp[van_exp["experience_type"] == "low_salience_high_satisfaction"]
print(lh[["park_name", "area_ha", "AvgRating", "TotalReviews"]].sort_values("TotalReviews", ascending=False).to_string(index=False))

# %%
