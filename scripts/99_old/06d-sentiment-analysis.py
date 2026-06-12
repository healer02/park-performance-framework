# =============================================================================
# 06d-sentiment-analysis.py
# Purpose: Run RoBERTa sentiment analysis on all text reviews
#          and aggregate to PlaceID and park level
#
# Inputs:
#   - data/google-reviews/processed/07-all-reviews-complete.csv
#   - data/parks/processed/06-master-park-placeids.csv
#
# Outputs:
#   - data/google-reviews/processed/08a-text-reviews-with-sentiment.csv
#   - data/google-reviews/processed/08b-placeid-metrics.csv
#   - data/google-reviews/processed/08c-park-metrics.csv
# =============================================================================

# %%
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

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
# Use max for safety -- Google metadata occasionally varies across scraping times
placeid_meta = reviews.groupby("PlaceID").agg(
    TotalReviews=("reviewsCount", "max"),
    AvgRating   =("totalScore",   "max"),
).reset_index()

print(f"\nPlaceID-level metadata:")
print(f"  PlaceIDs with TotalReviews: {placeid_meta['TotalReviews'].notna().sum()}")
print(f"  PlaceIDs with AvgRating:    {placeid_meta['AvgRating'].notna().sum()}")
print(placeid_meta[['TotalReviews', 'AvgRating']].describe().round(2))


# %% 3. FILTER TEXT REVIEWS + LANGUAGE QA
has_text = reviews['text'].notna() & (reviews['text'].str.strip() != '')
text_reviews = reviews[has_text].copy()

# Build Review column: prefer translated text
text_reviews['Review'] = text_reviews['textTranslated'].fillna(text_reviews['text'])

# Language QA summary
print(f"\n--- Language QA ---")
print(f"Total reviews:        {len(reviews)}")
print(f"With text:            {len(text_reviews)} ({100*len(text_reviews)/len(reviews):.1f}%)")

has_translated = (
    text_reviews['textTranslated'].notna() &
    (text_reviews['textTranslated'].str.strip() != '')
)
print(f"Translated:           {has_translated.sum()} ({100*has_translated.mean():.1f}%)")
print(f"Original only:        {(~has_translated).sum()} ({100*(~has_translated).mean():.1f}%)")

if 'originalLanguage' in text_reviews.columns:
    print(f"\nTop languages (original):")
    print(text_reviews['originalLanguage'].value_counts().head(10).to_string())

# Drop rows where Review is still empty after fill
empty_review = text_reviews['Review'].isna() | (text_reviews['Review'].str.strip() == '')
print(f"\nEmpty Review after fill: {empty_review.sum()}")
text_reviews = text_reviews[~empty_review].copy()
print(f"Text reviews for RoBERTa: {len(text_reviews)}")

# %%
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# %% 4. LOAD ROBERTA SENTIMENT MODEL
## will take a few miutes (up to 10?) on first run 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("\nLoading RoBERTa sentiment model...")

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    model_name, local_files_only=True
)
model.eval()

# Force CPU -- MPS causes initialization hangs with large transformer models
device = "cpu"
model.to(device)

print(f"Model loaded. Device: {device}")



# %% 5. RUN SENTIMENT SCORING
# will take ~20-30 minutes depending on hardware -- runs in batches with progress bar
# Score = P(positive) - P(negative), range -1 to +1
texts      = text_reviews['Review'].fillna("").tolist()
batch_size = 32
scores     = []

for i in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiment"):
    batch  = texts[i:i + batch_size]
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = F.softmax(outputs.logits, dim=1)
    batch_scores = (probs[:, 2] - probs[:, 0]).cpu().tolist()
    scores.extend(batch_scores)

text_reviews['Sentiment'] = scores

print(f"\nSentiment summary:")
print(text_reviews['Sentiment'].describe().round(3))

text_reviews.to_csv(OUT_A, index=False)
print(f"Saved: {OUT_A}")


# %% 6. PLACEID-LEVEL METRICS
# MeanSentiment: simple mean across text reviews for that PlaceID
# has_valid_sentiment: >= 10 text reviews (stability threshold)
placeid_sentiment = text_reviews.groupby("PlaceID").agg(
    MeanSentiment =("Sentiment", "mean"),
    n_text_reviews=("Sentiment", "count"),
).reset_index()
placeid_sentiment['MeanSentiment']      = placeid_sentiment['MeanSentiment'].round(4)
placeid_sentiment['has_valid_sentiment'] = placeid_sentiment['n_text_reviews'] >= 10

# Merge with metadata (TotalReviews, AvgRating from all reviews)
placeid_metrics = placeid_meta.merge(placeid_sentiment, on="PlaceID", how="left")
placeid_metrics['n_text_reviews']       = placeid_metrics['n_text_reviews'].fillna(0).astype(int)
placeid_metrics['has_valid_sentiment']  = placeid_metrics['has_valid_sentiment'].fillna(False)

# Text review ratio per PlaceID
placeid_metrics['text_review_ratio'] = (
    placeid_metrics['n_text_reviews'] / placeid_metrics['TotalReviews']
).round(3)

print(f"\n--- PlaceID-level metrics ---")
print(f"Total PlaceIDs:                      {len(placeid_metrics)}")
print(f"With valid sentiment (>=10 text):    {placeid_metrics['has_valid_sentiment'].sum()}")
print(f"Without valid sentiment (<10 text):  {(~placeid_metrics['has_valid_sentiment']).sum()}")
print(f"\nMeanSentiment summary:")
print(placeid_metrics['MeanSentiment'].describe().round(3))
print(f"\nText review ratio summary:")
print(placeid_metrics['text_review_ratio'].describe().round(3))

placeid_metrics.to_csv(OUT_B, index=False)
print(f"Saved: {OUT_B}")


# %% 7. PARK-LEVEL METRICS (weighted aggregation across PlaceIDs)
master = pd.read_csv(MASTER_PATH)
master["place_id_list"] = master["place_id"].apply(
    lambda x: [i.strip() for i in x.split(",")] if pd.notna(x) and x != "" else []
)
master_exploded = master.explode("place_id_list").rename(
    columns={"place_id_list": "PlaceID"}
)
master_exploded = master_exploded[
    master_exploded["PlaceID"].notna() &
    (master_exploded["PlaceID"] != "")
].copy()

# QA: PlaceIDs linked to multiple parks
dup_placeids = master_exploded.groupby("PlaceID")["park_id"].nunique()
dup_placeids = dup_placeids[dup_placeids > 1]
print(f"\n--- QA: PlaceIDs linked to multiple parks ---")
if len(dup_placeids) > 0:
    print(dup_placeids)
    print(master_exploded[master_exploded["PlaceID"].isin(dup_placeids.index)][
        ["PlaceID", "park_id", "park_name"]
    ].sort_values("PlaceID").to_string())
else:
    print("None -- clean.")

# Join PlaceID metrics to park_id
joined = master_exploded.merge(placeid_metrics, on="PlaceID", how="left")

def park_agg(g):
    # TotalReviews: sum across PlaceIDs
    total_reviews = g["TotalReviews"].sum()

    # AvgRating: weighted by TotalReviews
    rat_mask = g["AvgRating"].notna() & g["TotalReviews"].notna() & (g["TotalReviews"] > 0)
    if rat_mask.any():
        avg_rating = np.average(
            g.loc[rat_mask, "AvgRating"],
            weights=g.loc[rat_mask, "TotalReviews"]
        )
    else:
        avg_rating = np.nan

    # MeanSentiment: weighted by n_text_reviews, only valid PlaceIDs (>=10 text reviews)
    sent_mask  = g["has_valid_sentiment"] == True
    total_text = g.loc[sent_mask, "n_text_reviews"].sum()
    if sent_mask.any() and total_text > 0:
        mean_sentiment = np.average(
            g.loc[sent_mask, "MeanSentiment"],
            weights=g.loc[sent_mask, "n_text_reviews"]
        )
    else:
        mean_sentiment = np.nan

    n_text_total = int(g["n_text_reviews"].sum())

    return pd.Series({
        "TotalReviews":        total_reviews,
        "AvgRating":           round(avg_rating,     4) if not np.isnan(avg_rating)     else np.nan,
        "MeanSentiment":       round(mean_sentiment,  4) if not np.isnan(mean_sentiment)  else np.nan,
        "n_text_reviews":      n_text_total,
        "has_valid_sentiment": not np.isnan(mean_sentiment),
    })

park_metrics = joined.groupby("park_id").apply(park_agg).reset_index()
park_metrics  = park_metrics.merge(
    master[["park_id", "park_name", "area_ha"]], on="park_id", how="left"
)

# Text review ratio at park level
park_metrics["text_review_ratio"] = (
    park_metrics["n_text_reviews"] / park_metrics["TotalReviews"]
).round(3)

print(f"\n--- Park-level metrics ---")
print(f"Total parks:                         {len(park_metrics)}")
print(f"With valid sentiment (>=10 text):    {park_metrics['has_valid_sentiment'].sum()}")
print(f"Without valid sentiment:             {(~park_metrics['has_valid_sentiment']).sum()}")
print(f"\nMeanSentiment summary:")
print(park_metrics['MeanSentiment'].describe().round(3))
print(f"\nAvgRating summary:")
print(park_metrics['AvgRating'].describe().round(3))
print(f"\nTotalReviews summary:")
print(park_metrics['TotalReviews'].describe().round(0))
print(f"\nText review ratio (text/total):")
print(park_metrics['text_review_ratio'].describe().round(3))

park_metrics.to_csv(OUT_C, index=False)
print(f"Saved: {OUT_C}")


# %% 8. QA CHECKS
print("\n--- QA: Parks missing valid sentiment ---")
no_sent = park_metrics[~park_metrics['has_valid_sentiment']][
    ['park_id', 'park_name', 'n_text_reviews', 'TotalReviews']
].sort_values('TotalReviews', ascending=False)
print(no_sent.to_string())

print("\n--- QA: Destination parks ---")
destination = [
    'Stanley Park', 'Queen Elizabeth Park', 'Vandusen Botanical Garden',
    'Pacific Spirit Regional Park', 'Sunset Beach Park', 'English Bay Beach Park',
    'Hastings Park - Sanctuary', 'Vanier Park',
]
print(park_metrics[park_metrics['park_name'].isin(destination)][
    ['park_name', 'n_text_reviews', 'MeanSentiment', 'AvgRating',
     'TotalReviews', 'text_review_ratio']
].to_string())

print("\n--- QA: Extreme sentiment (top/bottom 5) ---")
print("Highest MeanSentiment:")
print(park_metrics.nlargest(5, 'MeanSentiment')[
    ['park_name', 'MeanSentiment', 'n_text_reviews']
].to_string())
print("\nLowest MeanSentiment:")
print(park_metrics.nsmallest(5, 'MeanSentiment')[
    ['park_name', 'MeanSentiment', 'n_text_reviews']
].to_string())
