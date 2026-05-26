# =============================================================================
# 10-usability.py
# Purpose: Extract amenity mentions from Google Reviews via keyword matching,
#          aggregate to park and DA level, and validate against official inventory
#
# Inputs:
#   - data/google-reviews/processed/08a-text-reviews-with-sentiment.csv
#   - data/parks/processed/06-master-park-placeids.csv
#   - data/processed/vancouver_da_park_sets.json
#   - data/census/processed/vancouver_db_centroids.gpkg
#   - data/parks/raw/Vancouver/parks-facilities.csv
#   - data/parks/raw/Vancouver/public-washrooms.csv
#   - data/processed/vancouver_da_equity.csv      (for divergence + chi-square)
#
# Outputs:
#   - data/processed/vancouver_park_amenities.csv
#   - data/processed/vancouver_da_usability.csv
#   - outputs/figures/vancouver_amenity_by_quadrant.png   (main figure)
#   - outputs/figures/vancouver_amenity_heatmap_appendix.png
#   - outputs/tables/vancouver_amenity_kappa.csv
#   - outputs/tables/vancouver_amenity_quadrant_chi2.csv
#   - outputs/tables/vancouver_amenity_sentiment_correlation.csv
# =============================================================================

# %%
import os
os.chdir('/Users/keunpark/Documents/GitHub/park-performance-framework')

import pandas as pd
import numpy as np
import json
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import cohen_kappa_score

REVIEWS_PATH  = "data/google-reviews/processed/08a-text-reviews-with-sentiment.csv"
MASTER_PATH   = "data/parks/processed/06-master-park-placeids.csv"
DA_SETS_PATH  = "data/processed/vancouver_da_park_sets.json"
DB_PATH       = "data/census/processed/vancouver_db_centroids.gpkg"
FAC_PATH      = "data/parks/raw/Vancouver/parks-facilities.csv"
WC_PATH       = "data/parks/raw/Vancouver/public-washrooms.csv"
EQUITY_PATH   = "data/processed/vancouver_da_equity.csv"
PARK_MET_PATH = "data/google-reviews/processed/08c-park-metrics.csv"

OUT_DIR = "data/processed"
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"
os.makedirs(TAB_DIR, exist_ok=True)

print("Ready.")


# %% 1. DEFINE AMENITY TAXONOMY
# Frozen after Vancouver -- applied unchanged to Coquitlam/New Westminster
TAXONOMY = {
    "playground":       ["playground", "playgrounds", "play structure"],
    "sports_fields":    ["soccer field", "football field", "baseball diamond",
                         "baseball field", "sports field"],
    "courts":           ["basketball court", "tennis court", "pickleball court",
                         "sports court"],
    "trails":           ["walking trail", "hiking trail", "walking path",
                         "bike path", "forest trail"],
    "dog_offleash":     ["off leash area", "off-leash area"],
    "water_play":       ["spray park", "splash pad", "wading pool"],
    "beach_waterfront": ["beach", "shoreline", "waterfront"],
    "picnic":           ["picnic area", "picnic table"],
    "washroom":         ["washroom", "washrooms", "restroom", "restrooms"],
    "community_garden": ["community garden", "allotment", "garden plot"],
    "seating_shelter":  ["park bench", "benches", "picnic shelter",
                         "covered shelter"],
}

AMENITY_LABELS = {
    "playground":       "Playground",
    "sports_fields":    "Sports fields",
    "courts":           "Courts",
    "trails":           "Trails",
    "dog_offleash":     "Dog off-leash",
    "water_play":       "Water play",
    "beach_waterfront": "Beach/waterfront",
    "picnic":           "Picnic area",
    "washroom":         "Washroom",
    "community_garden": "Community garden",
    "seating_shelter":  "Seating & shelter",
}

print(f"Amenity categories: {len(TAXONOMY)}")


# %% 2. LOAD TEXT REVIEWS AND JOIN TO PARK
reviews = pd.read_csv(REVIEWS_PATH, low_memory=False)
master  = pd.read_csv(MASTER_PATH)

reviews = reviews[
    reviews["Review"].notna() & (reviews["Review"].str.strip() != "")
].copy()
reviews["Review_lower"] = reviews["Review"].str.lower()
print(f"Text reviews loaded: {len(reviews)}")

master["place_id_list"] = master["place_id"].apply(
    lambda x: [i.strip() for i in x.split(",")] if pd.notna(x) and x != "" else []
)
master_exploded = master.explode("place_id_list").rename(
    columns={"place_id_list": "PlaceID"}
)
master_exploded = master_exploded[
    master_exploded["PlaceID"].notna() & (master_exploded["PlaceID"] != "")
].copy()

reviews_joined = reviews.merge(
    master_exploded[["PlaceID", "park_id", "park_name"]],
    on="PlaceID", how="left"
)
reviews_joined = reviews_joined[reviews_joined["park_id"].notna()].copy()
print(f"Reviews matched to parks: {len(reviews_joined)}")


# %% 3. KEYWORD MATCHING -- REVIEW LEVEL
def match_amenities(text, taxonomy):
    results = {}
    for category, keywords in taxonomy.items():
        pattern = r"\b(" + "|".join([re.escape(kw) for kw in keywords]) + r")\b"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        positive = 0
        for match in matches:
            start      = match.start()
            preceding  = text[max(0, start-30):start].lower()
            if re.search(r"\b(no|without|lack|missing|needs?|need a|no public)\s*$",
                         preceding):
                pass  # negation -- skip
            else:
                positive += 1
        results[category] = 1 if positive > 0 else 0
    return results

print("Running keyword matching...")
amenity_flags  = reviews_joined["Review_lower"].apply(
    lambda t: match_amenities(t, TAXONOMY)
)
amenity_df     = pd.DataFrame(list(amenity_flags))
reviews_amenity = pd.concat(
    [reviews_joined[["park_id", "park_name", "PlaceID"]].reset_index(drop=True),
     amenity_df.reset_index(drop=True)],
    axis=1
)

print(f"\nAmenity mention rates (% of reviews mentioning each category):")
for cat, rate in (amenity_df.mean() * 100).items():
    print(f"  {AMENITY_LABELS[cat]:20s}: {rate:.1f}%")


# %% 4. AGGREGATE TO PARK LEVEL (>=2 mentions threshold)
park_counts = reviews_amenity.groupby("park_id")[list(TAXONOMY.keys())].sum()
park_totals = reviews_amenity.groupby("park_id").size().rename("n_reviews")

park_amenities_binary = (park_counts >= 2).astype(int)
park_amenities = park_amenities_binary.reset_index()
park_amenities = park_amenities.merge(park_totals, on="park_id", how="left")
park_amenities = park_amenities.merge(
    master[["park_id", "park_name", "area_ha"]], on="park_id", how="left"
)
park_amenities["amenity_type_count"] = park_amenities[list(TAXONOMY.keys())].sum(axis=1)

print(f"\n--- Park-level amenity summary ---")
print(f"Parks with amenity data: {len(park_amenities)}")
print(f"\nAmenity type count per park:")
print(park_amenities["amenity_type_count"].describe().round(1))
print(f"\nPrevalence across parks (% of parks with each type):")
for cat in TAXONOMY.keys():
    pct = park_amenities[cat].mean() * 100
    print(f"  {AMENITY_LABELS[cat]:20s}: {pct:.1f}%")

park_amenities.to_csv(f"{OUT_DIR}/vancouver_park_amenities.csv", index=False)
print(f"\nSaved: {OUT_DIR}/vancouver_park_amenities.csv")


# %% 5. DA-LEVEL USABILITY
with open(DA_SETS_PATH, "r") as f:
    da_park_sets = {k: set(v) for k, v in json.load(f).items()}

db_centroids  = gpd.read_file(DB_PATH)
da_pop_lookup = db_centroids.groupby("DAUID")["db_pop"].sum().to_dict()
park_index    = park_amenities.set_index("park_id")

usability_records = []
for dauid, park_ids in da_park_sets.items():
    subset = park_index[park_index.index.isin(park_ids)]

    if len(subset) == 0:
        usability_records.append({
            "DAUID":               dauid,
            "amenity_type_count":  np.nan,
            "n_parks_usability":   0,
            **{cat: np.nan for cat in TAXONOMY.keys()}
        })
        continue

    amenity_union       = subset[list(TAXONOMY.keys())].max()
    amenity_type_count  = amenity_union.sum()
    mean_types_per_park = subset["amenity_type_count"].mean()

    record = {
        "DAUID":               dauid,
        "amenity_type_count":  amenity_type_count,
        "mean_types_per_park": round(mean_types_per_park, 2),
        "n_parks_usability":   len(subset),
    }
    record.update(amenity_union.to_dict())
    usability_records.append(record)

da_usability = pd.DataFrame(usability_records)
da_usability["DAUID"] = da_usability["DAUID"].astype(str)
da_usability.to_csv(f"{OUT_DIR}/vancouver_da_usability.csv", index=False)

print(f"\n--- DA-level usability summary ---")
print(f"DAs with usability data: {da_usability['amenity_type_count'].notna().sum()}")
print(f"\nAmenity type count across reachable parks:")
print(da_usability["amenity_type_count"].describe().round(1))


# %% 6. VALIDATION -- COHEN'S KAPPA
FAC_MAP = {
    "Playgrounds":                 "playground",
    "Soccer Fields":               "sports_fields",
    "Football Fields":             "sports_fields",
    "Baseball Diamonds":           "sports_fields",
    "Softball":                    "sports_fields",
    "Ultimate Fields":             "sports_fields",
    "Rugby Fields":                "sports_fields",
    "Field Hockey":                "sports_fields",
    "Basketball Courts":           "courts",
    "Tennis Courts":               "courts",
    "Pickleball":                  "courts",
    "Ball Hockey":                 "courts",
    "Outdoor Roller Hockey Rinks": "courts",
    "Dogs Off-Leash Areas":        "dog_offleash",
    "Water/Spray Parks":           "water_play",
    "Wading Pool":                 "water_play",
    "Beaches":                     "beach_waterfront",
    "Picnic Sites":                "picnic",
    "Jogging Trails":              "trails",
    "Swimming Pools":              "water_play",
}

fac = pd.read_csv(FAC_PATH, sep=";", encoding="utf-8-sig")
fac.columns = fac.columns.str.strip()
fac["amenity_cat"] = fac["FacilityType"].map(FAC_MAP)
fac_valid = fac[fac["amenity_cat"].notna()].copy()

park_name_lookup = fac[["ParkID", "Name"]].drop_duplicates(subset="ParkID")
park_name_lookup["name_lower"] = park_name_lookup["Name"].str.lower().str.strip()
name_fixes = {
    "hastings park": "hastings park - sanctuary",
    "locarno park":  "locarno beach park",
}
park_name_lookup["name_lower"] = park_name_lookup["name_lower"].replace(name_fixes)

fac_valid["present"] = 1
official_wide = fac_valid.pivot_table(
    index="ParkID", columns="amenity_cat",
    values="present", aggfunc="max"
).fillna(0).astype(int).reset_index()
official_wide.columns.name = None

official_wide = official_wide.merge(park_name_lookup, on="ParkID", how="left")

amenity_cats_in_official = [c for c in official_wide.columns if c in list(TAXONOMY.keys())]
official_wide = official_wide.rename(
    columns={c: f"{c}_official" for c in amenity_cats_in_official}
)

master_van = master[master["source"] == "Vancouver"].copy()
master_van["park_name_lower"] = master_van["park_name"].str.lower().str.strip()
official_wide = official_wide.merge(
    master_van[["park_name_lower", "park_id"]],
    left_on="name_lower", right_on="park_name_lower", how="inner"
)

wc = pd.read_csv(WC_PATH, sep=";", encoding="utf-8-sig")
wc.columns = wc.columns.str.strip()
wc_parks = set(wc["Park Name"].str.lower().str.strip().unique())
official_wide["washroom_official"] = (
    official_wide["Name"].str.lower().str.strip().isin(wc_parks)
).astype(int)

print(f"Parks matched for validation: {len(official_wide)}")

validation = official_wide.merge(
    park_amenities[["park_id"] + list(TAXONOMY.keys())],
    on="park_id", how="inner"
)
print(f"Parks in final validation set: {len(validation)}")


# %% 7. KAPPA COMPUTATION
cats_to_validate = [
    "playground", "sports_fields", "courts", "dog_offleash",
    "water_play", "picnic", "washroom"
]

kappa_results = []
for cat in cats_to_validate:
    off_col = f"{cat}_official"
    kw_col  = cat

    if off_col not in validation.columns or kw_col not in validation.columns:
        print(f"  Skipping {cat} -- column not found")
        continue

    y_true = validation[off_col].fillna(0).astype(int)
    y_pred = validation[kw_col].fillna(0).astype(int)

    if y_true.nunique() < 2 and y_pred.nunique() < 2:
        print(f"  Skipping {cat} -- no variation")
        continue

    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except Exception:
        kappa = np.nan

    agree = (y_true == y_pred).mean() * 100
    kappa_results.append({
        "Category":      AMENITY_LABELS.get(cat, cat),
        "Official (n)":  int(y_true.sum()),
        "Keyword (n)":   int(y_pred.sum()),
        "Agreement (%)": round(agree, 1),
        "Cohen's kappa": round(kappa, 3) if not np.isnan(kappa) else np.nan,
    })
    print(f"  {AMENITY_LABELS.get(cat, cat):20s}: κ={kappa:.3f}, "
          f"official={y_true.sum()}, keyword={y_pred.sum()}, agree={agree:.1f}%")

kappa_df = pd.DataFrame(kappa_results)
kappa_df.to_csv(f"{TAB_DIR}/vancouver_amenity_kappa.csv", index=False)
mean_kappa = kappa_df["Cohen's kappa"].mean()
print(f"\nMean Cohen's kappa: {mean_kappa:.3f}")
print(f"Saved: {TAB_DIR}/vancouver_amenity_kappa.csv")


# %% 8. PARK-LEVEL AMENITY-SENTIMENT CORRELATION
park_metrics = pd.read_csv(PARK_MET_PATH)

park_corr = park_amenities.merge(
    park_metrics[["park_id", "MeanSentiment", "AvgRating", "n_text_reviews"]],
    on="park_id", how="inner"
)
park_corr = park_corr[park_corr["MeanSentiment"].notna()].copy()
print(f"Parks in correlation analysis: {len(park_corr)}")

r, p = spearmanr(park_corr["amenity_type_count"], park_corr["MeanSentiment"])
print(f"\nAmenity type count vs MeanSentiment: r={r:.3f}, p={p:.4f}")
r2, p2 = spearmanr(park_corr["amenity_type_count"], park_corr["AvgRating"])
print(f"Amenity type count vs AvgRating:    r={r2:.3f}, p={p2:.4f}")

print(f"\n--- Individual amenity type vs MeanSentiment ---")
cat_results = []
for cat in TAXONOMY.keys():
    r_cat, p_cat = spearmanr(park_corr[cat], park_corr["MeanSentiment"])
    cat_results.append({
        "Category": AMENITY_LABELS[cat],
        "r":        round(r_cat, 3),
        "p":        round(p_cat, 4),
        "sig":      "***" if p_cat < 0.001 else "**" if p_cat < 0.01
                    else "*" if p_cat < 0.05 else "ns"
    })
    print(f"  {AMENITY_LABELS[cat]:20s}: r={r_cat:.3f}, p={p_cat:.4f}")

cat_df = pd.DataFrame(cat_results).sort_values("r", ascending=False)
cat_df.to_csv(f"{TAB_DIR}/vancouver_amenity_sentiment_correlation.csv", index=False)
print(f"\nSaved: {TAB_DIR}/vancouver_amenity_sentiment_correlation.csv")


# %% 9. CHI-SQUARE: AMENITY PRESENCE BY DIVERGENCE QUADRANT
# Load equity file which has corrected divergence_2x2 (NaN = No data, not LL)
da_equity = pd.read_csv(EQUITY_PATH, dtype={"DAUID": str})

# Merge usability onto equity (classified DAs only)
quad_order = ["HH", "LH", "HL", "LL"]
da_equity_class = da_equity[da_equity["divergence_2x2"].isin(quad_order)].copy()
da_chi = da_equity_class.merge(da_usability, on="DAUID", how="left")

print(f"DAs in chi-square analysis: {len(da_chi)}")
print(f"\n--- Chi-square: Amenity presence by divergence quadrant ---")

chi2_results = []
for cat in TAXONOMY.keys():
    if cat not in da_chi.columns:
        print(f"  Skipping {AMENITY_LABELS[cat]} -- column not found")
        continue
    ct = pd.crosstab(da_chi["divergence_2x2"], da_chi[cat])
    if ct.shape[1] < 2:
        print(f"  Skipping {AMENITY_LABELS[cat]} -- no variation")
        continue
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {AMENITY_LABELS[cat]:20s}: χ²={chi2:.1f}, df={dof}, p={p:.4f} {sig}")
    chi2_results.append({
        "Amenity": AMENITY_LABELS[cat],
        "chi2":    round(chi2, 1),
        "df":      dof,
        "p":       round(p, 4),
        "sig":     sig,
    })

chi2_df = pd.DataFrame(chi2_results).sort_values("chi2", ascending=False)
chi2_df.to_csv(f"{TAB_DIR}/vancouver_amenity_quadrant_chi2.csv", index=False)
print(f"\nSaved: {TAB_DIR}/vancouver_amenity_quadrant_chi2.csv")


# %% 10. MAIN FIGURE: AMENITY HEATMAP BY DIVERGENCE QUADRANT
# Use da_chi which has both amenity and divergence data (classified DAs only)
amenity_cols = list(TAXONOMY.keys())

quad_order = ["HH", "LH", "HL", "LL"]

quad_amenity = (
    da_chi.groupby("divergence_2x2")[amenity_cols]
    .mean() * 100
).reindex(quad_order)

# Add "All DAs" column (classified only)
all_da = da_chi[amenity_cols].mean() * 100
quad_amenity.loc["All"] = all_da

# Transpose: amenities as rows, quadrants + All as columns
quad_amenity_T = quad_amenity.T
quad_amenity_T.index = [AMENITY_LABELS[c] for c in amenity_cols]

col_labels = [
    f"High - High \n(n={da_chi['divergence_2x2'].eq('HH').sum()})",
    f"Low Supply \n High Exp.\n(n={da_chi['divergence_2x2'].eq('LH').sum()})",
    f"High Supply \n Low Exp.\n(n={da_chi['divergence_2x2'].eq('HL').sum()})",
    f"Low - Low\n(n={da_chi['divergence_2x2'].eq('LL').sum()})",
    f"All DAs\n(n={len(da_chi)})",
]

colours_2x2 = {
    "HH": "#01665e",   # dark teal
    "LH":  "#80cdc1",   # light teal
    "HL":  "#8c510a",   # dark brown
    "LL":   "#dfc27d",   # light brown
    "No data":                           "#cccccc",
}

quad_keys = ["HH", "LH", "HL", "LL", "All"]

header_colors = [
    colours_2x2["HH"],
    colours_2x2["LH"],
    colours_2x2["HL"],
    colours_2x2["LL"],
    "#666666"  # All DAs
]

# Build y-axis labels with significance markers
sig_map = chi2_df.set_index("Amenity")["sig"].to_dict()
ylabels = []
for cat in amenity_cols:
    label = AMENITY_LABELS[cat]
    sig   = sig_map.get(label, "")
    star  = " ***" if sig == "***" else " **" if sig == "**" \
            else " *" if sig == "*" else ""
    ylabels.append(f"{label}{star}")

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(quad_amenity_T.values, cmap="YlGn", aspect="auto", vmin=0, vmax=100)

for i in range(len(amenity_cols)):
    for j in range(len(quad_amenity_T.columns)):
        val = quad_amenity_T.values[i, j]
        if np.isnan(val):
            continue
        text_col = "white" if val > 60 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=9, color=text_col, fontweight="bold")

ax.set_xticks(range(len(quad_amenity_T.columns)))
ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold")

ax.set_yticks(range(len(amenity_cols)))
ax.set_yticklabels(ylabels, fontsize=9)

ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Re-fetch labels AFTER moving ticks to top
xt = ax.get_xticklabels()

# Color-code quadrant headers
for label, bg in zip(xt, header_colors):
    label.set_color("white")
    label.set_bbox(dict(
        facecolor=bg,
        edgecolor="none",
        boxstyle="round,pad=0.5"
    ))
ax.set_title(
    "Socially Perceived Recreational Affordances by Divergence Quadrant — Vancouver\n"
    "% of DAs with ≥1 reachable park mentioning each amenity type (≥2 review mentions)",
    fontsize=10, pad=12
)
plt.colorbar(im, ax=ax, shrink=0.6, label="% of DAs")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{FIG_DIR}/vancouver_amenity_by_quadrant.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}/vancouver_amenity_by_quadrant.png")


# %% 11. APPENDIX: PARK-LEVEL AMENITY HEATMAP (top 40 parks)
top_parks = park_amenities.nlargest(40, "amenity_type_count")[
    ["park_name"] + list(TAXONOMY.keys())
].set_index("park_name")
top_parks = top_parks.rename(columns=AMENITY_LABELS)

fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(top_parks.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(top_parks.columns)))
ax.set_xticklabels(top_parks.columns, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(top_parks)))
ax.set_yticklabels(top_parks.index, fontsize=8)
ax.set_title(
    "Amenity Type Presence by Park (Top 40) — Vancouver\n"
    "Based on keyword matching of Google Reviews (≥2 review mentions per park)",
    fontsize=11, pad=12
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vancouver_amenity_heatmap_appendix.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}/vancouver_amenity_heatmap_appendix.png")


# %% 12. QA: DISCREPANCY ANALYSIS
print("\n--- Official amenities not mentioned in reviews ---")
for cat in cats_to_validate:
    off_col = f"{cat}_official"
    kw_col  = cat
    if off_col not in validation.columns or kw_col not in validation.columns:
        continue
    missed = validation[
        (validation[off_col]==1) & (validation[kw_col]==0)
    ]["Name"].tolist()
    if missed:
        print(f"\n  {AMENITY_LABELS.get(cat, cat)} ({len(missed)} parks):")
        print(f"    {', '.join(missed[:5])}{'...' if len(missed)>5 else ''}")

print("\n--- User-mentioned amenities absent from official inventory ---")
for cat in cats_to_validate:
    off_col = f"{cat}_official"
    kw_col  = cat
    if off_col not in validation.columns or kw_col not in validation.columns:
        continue
    extra = validation[
        (validation[off_col]==0) & (validation[kw_col]==1)
    ]["Name"].tolist()
    if extra:
        print(f"\n  {AMENITY_LABELS.get(cat, cat)} ({len(extra)} parks):")
        print(f"    {', '.join(extra[:5])}{'...' if len(extra)>5 else ''}")
# %%
