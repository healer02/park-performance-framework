"""
vancouver/v03_sentiment.py  (was 06d-sentiment-analysis.py)
Run RoBERTa sentiment on all text reviews; aggregate to PlaceID and park level.

GPU strongly recommended. CPU takes ~20–30 min.
Model must be pre-downloaded: cardiffnlp/twitter-roberta-base-sentiment-latest

This script is Vancouver-specific — replication cities skip this sub-track.

Inputs:
    data/google-reviews/processed/07-all-reviews-complete.csv
    data/parks/processed/06-master-park-placeids.csv

Outputs:
    data/google-reviews/processed/08a-text-reviews-with-sentiment.csv
    data/google-reviews/processed/08b-placeid-metrics.csv
    data/google-reviews/processed/08c-park-metrics.csv
"""

import os
import sys

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import config
os.chdir(config.REPO_DIR)

import pandas as pd
import numpy as np
from tqdm import tqdm

REVIEWS_PATH = "data/google-reviews/processed/07-all-reviews-complete.csv"
MASTER_PATH  = "data/parks/processed/06-master-park-placeids.csv"
OUT_A        = "data/google-reviews/processed/08a-text-reviews-with-sentiment.csv"
OUT_B        = "data/google-reviews/processed/08b-placeid-metrics.csv"
OUT_C        = "data/google-reviews/processed/08c-park-metrics.csv"

print("Ready.")


# %% 1. LOAD FULL REVIEW FILE
reviews = pd.read_csv(REVIEWS_PATH, low_memory=False)
print(f"Total reviews loaded: {len(reviews)}")
print(f"Unique PlaceIDs:      {reviews['PlaceID'].nunique()}")


# %% 2. PLACEID-LEVEL METADATA (all reviews, before text filter)
placeid_meta = reviews.groupby("PlaceID").agg(
    TotalReviews=("reviewsCount", "max"),
    AvgRating   =("totalScore",   "max"),
).reset_index()

print(f"\nPlaceID-level metadata:")
print(f"  With TotalReviews: {placeid_meta['TotalReviews'].notna().sum()}")
print(f"  With AvgRating:    {placeid_meta['AvgRating'].notna().sum()}")


# %% 3. FILTER TEXT REVIEWS + LANGUAGE QA
has_text     = reviews['text'].notna() & (reviews['text'].str.strip() != '')
text_reviews = reviews[has_text].copy()
text_reviews['Review'] = text_reviews['textTranslated'].fillna(text_reviews['text'])

print(f"\n--- Language QA ---")
print(f"Total reviews:     {len(reviews)}")
print(f"With text:         {len(text_reviews)} ({100*len(text_reviews)/len(reviews):.1f}%)")

has_translated = (
    text_reviews['textTranslated'].notna() &
    (text_reviews['textTranslated'].str.strip() != '')
)
print(f"Translated:        {has_translated.sum()} ({100*has_translated.mean():.1f}%)")
if 'originalLanguage' in text_reviews.columns:
    print(f"\nTop languages:")
    print(text_reviews['originalLanguage'].value_counts().head(10).to_string())

empty_review = text_reviews['Review'].isna() | (text_reviews['Review'].str.strip() == '')
text_reviews = text_reviews[~empty_review].copy()
print(f"\nText reviews for RoBERTa: {len(text_reviews)}")


# %% 4. LOAD ROBERTA SENTIMENT MODEL
import torch
print(f"\nTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available:  {torch.backends.mps.is_available()}")

os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("\nLoading RoBERTa sentiment model...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer  = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name, local_files_only=True
)
model.eval()

device = "cpu"  # Force CPU — MPS causes initialization hangs with large transformers
model.to(device)
print(f"Model loaded. Device: {device}")


# %% 5. RUN SENTIMENT SCORING
# Score = P(positive) - P(negative), range -1 to +1
texts      = text_reviews['Review'].fillna("").tolist()
batch_size = 32
scores     = []

for i in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiment"):
    batch  = texts[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = F.softmax(outputs.logits, dim=1)
    scores.extend((probs[:, 2] - probs[:, 0]).cpu().tolist())

text_reviews['Sentiment'] = scores
print(f"\nSentiment summary:")
print(text_reviews['Sentiment'].describe().round(3))

text_reviews.to_csv(OUT_A, index=False)
print(f"Saved: {OUT_A}")


# %% 6. PLACEID-LEVEL METRICS
placeid_sentiment = text_reviews.groupby("PlaceID").agg(
    MeanSentiment =("Sentiment", "mean"),
    n_text_reviews=("Sentiment", "count"),
).reset_index()
placeid_sentiment['MeanSentiment']       = placeid_sentiment['MeanSentiment'].round(4)
placeid_sentiment['has_valid_sentiment'] = placeid_sentiment['n_text_reviews'] >= config.MIN_REVIEWS

placeid_metrics = placeid_meta.merge(placeid_sentiment, on="PlaceID", how="left")
placeid_metrics['n_text_reviews']      = placeid_metrics['n_text_reviews'].fillna(0).astype(int)
placeid_metrics['has_valid_sentiment'] = placeid_metrics['has_valid_sentiment'].fillna(False)
placeid_metrics['text_review_ratio']   = (
    placeid_metrics['n_text_reviews'] / placeid_metrics['TotalReviews']
).round(3)

print(f"\n--- PlaceID-level metrics ---")
print(f"Total PlaceIDs:                           {len(placeid_metrics)}")
print(f"With valid sentiment (>={config.MIN_REVIEWS} text reviews): "
      f"{placeid_metrics['has_valid_sentiment'].sum()}")

placeid_metrics.to_csv(OUT_B, index=False)
print(f"Saved: {OUT_B}")


# %% 7. PARK-LEVEL METRICS (weighted aggregation across PlaceIDs)
master = pd.read_csv(MASTER_PATH)
master["place_id_list"] = master["place_id"].apply(
    lambda x: [i.strip() for i in x.split(",")] if pd.notna(x) and x != "" else []
)
master_exploded = master.explode("place_id_list").rename(columns={"place_id_list": "PlaceID"})
master_exploded = master_exploded[
    master_exploded["PlaceID"].notna() & (master_exploded["PlaceID"] != "")
].copy()

dup_placeids = master_exploded.groupby("PlaceID")["park_id"].nunique()
dup_placeids = dup_placeids[dup_placeids > 1]
if len(dup_placeids) > 0:
    print(f"\nWARNING: {len(dup_placeids)} PlaceIDs linked to multiple parks:")
    print(master_exploded[master_exploded["PlaceID"].isin(dup_placeids.index)][
        ["PlaceID", "park_id", "park_name"]
    ].sort_values("PlaceID").to_string())
else:
    print("\nQA: No PlaceIDs linked to multiple parks — clean.")

joined = master_exploded.merge(placeid_metrics, on="PlaceID", how="left")

def park_agg(g):
    total_reviews = g["TotalReviews"].sum()
    rat_mask      = g["AvgRating"].notna() & g["TotalReviews"].notna() & (g["TotalReviews"] > 0)
    avg_rating    = (
        np.average(g.loc[rat_mask, "AvgRating"], weights=g.loc[rat_mask, "TotalReviews"])
        if rat_mask.any() else np.nan
    )
    sent_mask      = g["has_valid_sentiment"] == True
    total_text     = g.loc[sent_mask, "n_text_reviews"].sum()
    mean_sentiment = (
        np.average(
            g.loc[sent_mask, "MeanSentiment"],
            weights=g.loc[sent_mask, "n_text_reviews"]
        )
        if sent_mask.any() and total_text > 0 else np.nan
    )
    return pd.Series({
        "TotalReviews":        total_reviews,
        "AvgRating":           round(avg_rating,     4) if not np.isnan(avg_rating)     else np.nan,
        "MeanSentiment":       round(mean_sentiment,  4) if not np.isnan(mean_sentiment)  else np.nan,
        "n_text_reviews":      int(g["n_text_reviews"].sum()),
        "has_valid_sentiment": not np.isnan(mean_sentiment),
    })

park_metrics = joined.groupby("park_id").apply(park_agg).reset_index()
park_metrics = park_metrics.merge(
    master[["park_id", "park_name", "area_ha"]], on="park_id", how="left"
)
park_metrics["text_review_ratio"] = (
    park_metrics["n_text_reviews"] / park_metrics["TotalReviews"]
).round(3)

print(f"\n--- Park-level metrics ---")
print(f"Total parks:                              {len(park_metrics)}")
print(f"With valid sentiment (>={config.MIN_REVIEWS} text reviews): "
      f"{park_metrics['has_valid_sentiment'].sum()}")
print(f"\nMeanSentiment summary:")
print(park_metrics['MeanSentiment'].describe().round(3))
print(f"\nAvgRating summary:")
print(park_metrics['AvgRating'].describe().round(3))

park_metrics.to_csv(OUT_C, index=False)
print(f"Saved: {OUT_C}")

print("\nDone.")
