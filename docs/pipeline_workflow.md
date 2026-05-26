# Pipeline workflow
# Vancouver

## 0. Setup — inputs

- Vancouver parks + parks-facilities + parks-special-features + drinking-fountains + public-washrooms (City of Vancouver open data)
- Metro Vancouver Regional Parks (Metro Van open data)
- Burnaby parks (Burnaby open data)
- DA boundaries (`lda_000b21a_e.shp`) & DB boundaries (`ldb_000b21a_e.shp`) (StatCan 2021)
- DA/DB population-weighted representative points and GAF (`2021_92-151_x.csv`) (StatCan 2021)
- OSM pedestrian network → script 01
- Google Reviews → Apify extraction (scripts 06c), stored in `data/google-reviews/`
- DA census profile (`census_CANUE_DA_nearVan.csv`) — for equity analysis

---

## 1. Park entrance extraction (script 02) [done]

```
1a. Extract park boundary lines from merged park polygons
1b. Buffer park boundaries by 10m
1c. Intersect buffered park polygons with OSM walk edge centrelines
     → points where roads enter the park buffer zone
1d. Deduplicate entrances within 25m of each other per park
1e. Snap entrance points to nearest OSM node via ox.distance.nearest_nodes()
1f. Parks with zero entrances flagged for manual review (script 02b — now deleted;
     review done manually)
     Special case: Shaughnessy Park uses 35m buffer
     (park polygon ~28m inside road centrelines — digitisation offset)
1g. Output: vancouver_park_entrances.shp
     (park_id, park_name, entrance_id, nearest_node, snap_dist_m, geometry)
     Note: nearest_node truncated to nearest_no in shapefile (10-char limit)
```

---

## 2. DB centroid extraction (script 03) [done]

```
2a. Load DB boundary polygons (ldb_000b21a_e.shp)
2b. Filter to Vancouver DAs (CSDUID 5915022) → 4,561 DBs, 1,016 DAs
2c. Compute geometric centroids from DB polygons
2d. Join DB population from GAF (DBUID → DBPOP2021)
     Total population: 662,248; zero-pop DBs: 498 (retained)
2e. Snap each DB centroid to nearest OSM node
     Mean snap distance: 50.5m; 2 DBs flagged >200m (snap_flag=1)
2f. Output: vancouver_db_centroids.gpkg
     (DAUID, DBUID, db_pop, nearest_node, snap_dist_m, snap_flag, geometry)
     Note: GeoPackage used to avoid int64 truncation of osmid
```

---

## 3. Network distance: entrances → all nodes (script 04, Part A) [done]

```
3a. Load OSM graph, DB centroids, park entrances
3b. Collect all unique entrance nearest_nodes → 2,648 unique nodes
3c. Run multi-source Dijkstra (cutoff=800m):
     nx.multi_source_dijkstra_path_length(G, entrance_nodes, cutoff=800, weight='length')
     → distance from every network node to nearest entrance
     Coverage: 58,961 / 60,400 nodes reached (97.6%)
3d. For each DB: look up distance via nearest_node
3e. Assign reachable_400 = 1 if distance ≤ 400m (3,085 DBs)
         reachable_800 = 1 if distance ≤ 800m (4,451 DBs)
3f. Output: vancouver_db_reachability.csv
     (DBUID, DAUID, db_pop, nearest_node,
      dist_nearest_entrance, reachable_400, reachable_800, snap_flag)
```

---

## 4. DA reachability (script 04, Part B) [done]

```
4a. DA_reachability = sum(DB_pop where reachable_400=1) / sum(DB_pop)
4b. Sensitivity: repeat with reachable_800
4c. Results: mean=0.715, median=1.0; 528 DAs fully covered;
     127 DAs with 0% reachability (genuine access gaps, populated)
4d. Output: vancouver_da_reachability.csv + vancouver_da_reachability.gpkg
     (DAUID, DA_reach_400, DA_reach_800, db_count, db_pop_total)
```

---

## 5. DA park quantity (script 05) [done]

```
5a. For each DB: run single_source_dijkstra_path_length(G, db_node, cutoff=400m)
5b. Intersect result nodes with entrance node set → reachable entrances
5c. Deduplicate by park_id → reachable park set per DB
5d. Aggregate to DA using union of park sets across all DBs
     (avoids double-counting parks reachable by multiple DBs)
5e. Area cap: main = min(area_ha, 20); sensitivity = min(area_ha, 10); uncapped
5f. Denominator: db_pop_valid (DBs with valid nearest_node only)
5g. DAs with no reachable parks → qty = 0 (genuine access gaps, not null)
5h. Output: vancouver_da_quantity.csv + vancouver_da_supply.gpkg
     (DAUID, n_unique_parks, area_raw, area_cap20, area_cap10,
      qty_raw, qty_cap20, qty_cap10, db_pop_total, db_pop_valid)
```

---

## 5b. Supply typology (script 05) [done]

```
5b-a. Median split on DA_reach_400 (threshold=0.8) and qty_cap20 (median=5.0 ha/1,000)
       Note: 0.8 threshold used instead of median (median=1.0 is too brittle)
5b-b. Four supply types:
       HH — Well-served (n=354): high reachability + high quantity
       HL — High access, small area (n=239): high reachability + low quantity
       LH — High area, partial access (n=154): low reachability + high quantity
       LL — Underserved (n=269): low reachability + low quantity
5b-c. Output: vancouver_da_supply.gpkg (supply_type field added)
       outputs/figures/vancouver_da_supply_typology.png
```

---

## 6. Google reviews pipeline (scripts 06, 06c, 06d) [done]

```
6a. Build master park list with Google Place IDs (script 06)
     Input: vancouver_parks_merged.csv + parkperformance_placeIDs-AVERY.csv
     Output: data/parks/processed/06-master-park-placeids.csv

6b. Retrieve Google Reviews via Apify (script 06c)
     Actor: Xb8osYTtOjlsgI6k9; API key in scripts/apify_key.txt (gitignored)
     Output: data/google-reviews/raw/07-all-reviews-complete.csv

6c. RoBERTa sentiment analysis (script 06d)
     Filter: reviews ≥10 characters, non-empty text
     Aggregate to PlaceID level → park level
     Park qualifies for satisfaction if has_valid_sentiment=True (≥10 text reviews)
     Outputs:
       data/google-reviews/processed/08a-text-reviews-with-sentiment.csv
       data/google-reviews/processed/08b-placeid-metrics.csv
       data/google-reviews/processed/08c-park-metrics.csv
         (park_id, AvgSentiment, AvgRating, TotalReviews, has_valid_sentiment)
```

---

## 7. DA-level experience (script 08) [done]

```
7a. Load or rebuild da_park_sets (reachable park set per DA)
     Cached to: data/processed/vancouver_da_park_sets.json
     (Rebuilding takes 5–15 min via single_source_dijkstra per DB, cutoff=400m)

7b. SALIENCE: total reviews across reachable parks / DA_pop × 1,000

7c. SATISFACTION (primary): unweighted mean of AvgSentiment across
     qualifying parks (has_valid_sentiment=True)
     SATISFACTION (sensitivity): same using log-weighted mean by TotalReviews

7d. SATISFACTION (validation): mean AvgRating across qualifying parks

7e. Outputs:
     data/processed/vancouver_da_experience.csv
       (DAUID, salience, satisfaction_sentiment, satisfaction_star,
        coverage_pct, n_reachable_parks, n_qualifying_parks)
```

---

## 8. Supply–experience divergence matrix (script 08) [done]

```
8a. Supply binary: supply_type == "HH" → supply_binary=1, else 0
8b. Experience binary: satisfaction_sentiment ≥ median → experience_hi=1, else 0
8c. 2×2 divergence classification:
     HH — high supply, high experience
     HL — high supply, low experience (experience deficit)
     LH — low supply,  high experience (overperforming)
     LL — low supply,  low experience  (compounded disadvantage)
8d. Full 4×2 divergence_type (supply_type × experience): also stored for exploratory use
8e. Outputs:
     data/processed/vancouver_da_divergence.gpkg
       (supply_type, supply_binary, experience_hi, divergence_type, divergence_2x2, geometry)
     outputs/figures/vancouver_da_divergence_2x2.png      (main map)
     outputs/figures/vancouver_da_divergence_prototype.png (4×2 low-experience highlight)
```

---

## 9. Equity analysis (script 09) [done]

```
9a. Recompute supply typology + 2×2 divergence from vancouver_da_divergence.gpkg
     (mirrors script 08 thresholds exactly)

9b. Join StatCan/CANUE census variables (census_CANUE_DA_nearVan.csv)
     Derive proportions: pct_visible_minority, pct_age_65plus, pct_LIM_AT,
                         pct_immigrant, pct_bachelor_plus

9c. Define 7 equity strata (tertile or threshold-based):
     inc_stratum      — household income (low/mid/high by ±40% of city median)
     vm_stratum       — visible minority % (0–20 / 20–50 / >50)
     age_stratum      — 65+ share (0–10 / 10–20 / >20)
     limat_stratum    — LIM-AT poverty rate (0–20 / 20–35 / >35)
     immigrant_stratum — immigrant share (0–30 / 30–50 / >50)
     edu_stratum      — bachelor+ share (0–30 / 30–50 / >50)
     ale_stratum      — Active Living Environment index (tertile split)

9d. For each stratum: chi-square test + Cramér's V against divergence_2x2

9e. Multinomial logistic regression (reference = HH):
     Outcome: divergence_2x2 ∈ {HH, LH, HL, LL}
     Predictors: pct_visible_minority, pct_age_65plus, pct_LIM_AT,
                 pct_bachelor_plus, ale16_08 (all standardised)
     Reports odds ratios + 95% CI for LH vs HH, HL vs HH, LL vs HH
     Model fit: McFadden R², Cox-Snell R², Nagelkerke R², VIF check

9f. Binary logistic models (supplementary): LL vs others, HL vs others, LH vs others

9g. Outputs:
     data/processed/vancouver_da_equity.csv
     outputs/figures/vancouver_equity_stacked_bar.png   (5 strata, 3×2 grid)
     outputs/figures/vancouver_equity_heatmap.png       (high-stratum DAs × quadrant)
     outputs/figures/vancouver_equity_socioeconomic.png (income, LIM-AT, education)
     outputs/figures/vancouver_equity_demographic_builtenv.png (VM, age, immigrant, ALE)
     outputs/tables/vancouver_multinomial_logit.csv
     outputs/tables/vancouver_binary_logit.csv
```

---

## 10. Usability: amenity extraction (script 10) [done]

```
10a. Keyword matching against 11-category TAXONOMY (frozen after Vancouver calibration)
      Negation detection: checks 4 words preceding each match
      Park threshold: ≥2 review mentions to flag amenity as present

10b. DA-level usability: union of amenity types across reachable parks (from da_park_sets)

10c. Validation: Cohen's kappa vs official park facilities inventory
      (City of Vancouver parks-facilities.csv + public-washrooms.csv)
      7 categories validated (trails, community_garden, seating_shelter excluded —
      not in official inventory)

10d. Chi-square test: amenity presence by divergence quadrant (4 groups)

10e. Amenity–sentiment correlation: Spearman r per amenity category vs MeanSentiment

10f. Outputs:
      data/processed/vancouver_park_amenities.csv
      data/processed/vancouver_da_usability.csv
        (DAUID, amenity_type_count, mean_types_per_park, n_parks_usability, [11 binary flags])
      outputs/figures/vancouver_amenity_by_quadrant.png   (main figure, sig markers)
      outputs/figures/vancouver_amenity_heatmap_appendix.png (top-40 parks)
      outputs/tables/vancouver_amenity_kappa.csv
      outputs/tables/vancouver_amenity_quadrant_chi2.csv
      outputs/tables/vancouver_amenity_sentiment_correlation.csv
```

---

## 11. Validation (script 11) [done]

```
11a. Park-level: Pearson r + Spearman r (RoBERTa sentiment vs star rating)
11b. DA-level: Pearson r + Spearman r (satisfaction_sentiment vs satisfaction_star)
11c. DA quadrant agreement: Cohen's kappa on high/low binary classification
      (sentiment median split vs star median split)
11d. Outputs:
      outputs/figures/vancouver_validation_scatter.png
      outputs/tables/vancouver_validation_summary.csv
```

---

## 12. Descriptive statistics (script 07)

```
Inputs: vancouver_da_supply.gpkg, vancouver_da_experience.csv,
        08c-park-metrics.csv, vancouver_da_usability.csv,
        outputs/tables/vancouver_amenity_kappa.csv
Output: outputs/tables/table1-descriptive-stats.csv
```
