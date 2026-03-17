# Pipeline workflow
# Vancouver as an example

0. Setup
inputs:
- [done: 2026/3/6] Vancouver parks + parks-facilities + parks-special-features + drinking-fountains + public-washrooms (City of Vancouver open data: https://opendata.vancouver.ca/pages/home/)
- [done: 2026/3/9] Metro Vancouver Regional Parks (Metro Van open data: https://open-data-portal-metrovancouver.hub.arcgis.com/)
- [done: 2026/3/9] Burnaby parks (Burnaby open data: https://data.burnaby.ca/)
- [done: 2026/3/6] DA boundaries (lda_000b21a_e.shp) & DB boundaries (ldb_000b21a_e.shp) (StatCan 2021: https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21)
- [done: 2026/3/6] DA points and DB population (StatCan 2021, population-weighted representative points: https://www150.statcan.gc.ca/n1/en/catalogue/92-151-X)
- [done: 2026/3/9] run scripts (to get OSM network + DA-level data) > 01-get-osm-network.py
- [for Experience dimension] google_reviews.csv (Apify, park_id, rating, text, date)
- [for analysis 2] DA census profile 

```markdown
1. Park entrance extraction: Python (script 02, Part B) [done: 2026/3/16]
1a. Extract park boundary lines from merged park polygons
1b. Buffer park boundaries by 10m
1c. Intersect buffered park polygons with OSM walk edge centrelines
     → result is points where roads enter the park buffer zone
1d. Deduplicate entrances within 25m of each other per park
1e. Snap entrance points to nearest OSM node
     using ox.distance.nearest_nodes(G, x, y)
1f. Parks with zero entrances flagged for manual review (script 02b)
     Special case: Shaughnessy Park handled with 35m buffer
     (park polygon ~28m inside road centrelines — digitisation offset)
1g. Output: vancouver_park_entrances.shp
     (park_id, park_name, entrance_id, nearest_node, snap_dist_m, geometry)
     Note: nearest_node truncated to nearest_no in shapefile (10-char limit)

2. DB centroid extraction (script 03) [done: 2026/3/17]
2a. Load DB boundary polygons (ldb_000b21a_e.shp)
2b. Filter to Vancouver DAs (CSDUID 5915022) → 4,561 DBs, 1,016 DAs
2c. Compute geometric centroids from DB polygons
     Note: geometric centroid used; DB scale small enough that
     positional error is negligible for 400m network analysis
2d. Join DB population from GAF (DBUID → DBPOP2021)
     Total population: 662,248; zero-pop DBs: 498 (retained)
2e. Snap each DB centroid to nearest OSM node
     using ox.distance.nearest_nodes(G, x, y)
     Mean snap distance: 50.5m; 2 DBs flagged >200m (snap_flag=1)
2f. Output: vancouver_db_centroids.gpkg
     (DAUID, DBUID, db_pop, nearest_node, snap_dist_m, snap_flag, geometry)
     Note: GeoPackage used to avoid int64 truncation of osmid

3. Network distance: entrances → all nodes (script 04, Part A) [done: 2026/3/17]
3a. Load OSM graph, DB centroids, park entrances
3b. Collect all unique entrance nearest_nodes → 2,648 unique nodes
3c. Run multi-source Dijkstra (cutoff=800m):
     nx.multi_source_dijkstra_path_length(G, entrance_nodes,
                                          cutoff=800, weight='length')
     → distance from every network node to nearest entrance
     Coverage: 58,961 / 60,400 nodes reached (97.6%)
3d. For each DB: look up distance via nearest_node
3e. Assign reachable_400 = 1 if distance ≤ 400m (3,085 DBs)
         reachable_800 = 1 if distance ≤ 800m (4,451 DBs)
3f. Output: vancouver_db_reachability.csv
     (DBUID, DAUID, db_pop, nearest_node,
      dist_nearest_entrance, reachable_400, reachable_800, snap_flag)

4. DA reachability (script 04, Part B) [done: 2026/3/17]
4a. DA_reachability = sum(DB_pop where reachable_400=1) / sum(DB_pop)
4b. Sensitivity: repeat with reachable_800
4c. Results: mean=0.715, median=1.0; 528 DAs fully covered;
     127 DAs with 0% reachability (genuine access gaps, populated)
4d. Output: vancouver_da_reachability.csv + vancouver_da_reachability.gpkg
     (DAUID, DA_reach_400, DA_reach_800, db_count, db_pop_total)

5. DA park quantity (script 05) [done: 2026/3/17]
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

6. Supply typology (visualization)(script 05) [done: 2026/3/17]
6a. Median split on DA_reach_400 (threshold=0.8) and qty_cap20 (median=5.0 ha/1,000)
     Note: 0.8 threshold used instead of median (median=1.0 is too brittle)
6b. Four supply types:
     HH — Well-served (n=354): high reachability + high quantity
     HL — High access, small area (n=239): high reachability + low quantity
     LH — High area, partial access (n=154): low reachability + high quantity
     LL — Underserved (n=269): low reachability + low quantity
6c. Bivariate colour scheme (matrix logic):
     HH=#3b2f3a (dark), HL=#4e8bab (blue), LH=#c27d4f (orange), LL=#f2eadf (light)
6d. ~39% of DAs show divergence between coverage and intensity (HL+LH)
     Asymmetry: HL (239) > LH (154) — access-without-area more common
6e. Output: vancouver_da_supply_typology.png + vancouver_da_supply.gpkg




5. Google review extraction (done for City of Vancouver, except for Destination Parks, in Dec 2025)
- add Google place ID to park list
- for vancouver, we still need to add several destination parks (e.g., Stanley Park, Queen Elizabeth Park)
- for large parks with multiple place ids, we should decide whether we keep all or only the major one (e.g., Stanely Park (49k), Aquarium (11k), 10+ other POIs within Stanley Park -> may only keep Stanley Park)
- 

5. Park-level sentiment + star aggregation
5a. Load google_reviews.csv
5b. Filter: parks ≥10 reviews, reviews ≥10 characters, remove spam
5c. Run RoBERTa sentiment model on each review text
     → sentiment_score (continuous, e.g., -1 to 1 or 0 to 1)
5d. Per park:
     - mean_sentiment = mean(sentiment_score) [park-balanced input]
     - mean_star = mean(star_rating)
     - review_count = n reviews
5e. Output: park_sentiment.csv (park_id, mean_sentiment, mean_star, review_count)

Validation: hand-code 50-100 reviews → Spearman correlation with RoBERTa scores

6. DA-level salience + satisfaction
6a. Join db_park_distances (reachable parks per DA) to park_sentiment
6b. For each DA:

     SALIENCE:
     - sum all reviews across reachable qualifying parks
     - Salience(DA) = total_reviews / DA_pop × 1000

     SATISFACTION (primary, park-balanced):
     - mean(mean_sentiment) across reachable parks ≥10 reviews
     - Satisfaction_parkmean(DA) = mean of park-level sentiment scores

     SATISFACTION (validation):
     - mean(mean_star) across reachable parks ≥10 reviews

     COVERAGE descriptor:
     - % of reachable parks meeting ≥10 review threshold per DA

6c. Output: da_experience.csv (da_id, salience, satisfaction_sentiment,
     satisfaction_star, coverage_pct)

7. Usability (amenity diversity)
<!-- How robust is amenity extraction from reviews? -->
input: Merge the attribute/facility CSVs (parks, facilities, special features, fountains, washrooms) → parks_inventory.csv

7a. For each review, run keyword matching against taxonomy (11 categories)
     → binary flags: playground=1, trails=1, etc.
7b. Per park: identify which amenity categories are mentioned
     → park_amenities.csv (park_id, amenity_type, present=1/0)
7c. Per DA:
     - get union of amenity types across all reachable parks
     - Usability_diversity(DA) = count of unique amenity categories present
7d. Output: da_usability.csv (da_id, usability_diversity)

Validation (Vancouver): compare park_amenities against vancouver_park_inventory
     → Cohen's kappa per amenity category

8. Supply + experience composites → divergence matrix
8a. Join: da_reachability + da_quantity + da_experience + da_usability

8b. SUPPLY 2×2 (city median splits):
     - reachability_high = reachability > median
     - quantity_high = quantity > median
     - supply_group = HH / HL / LH / LL

8c. EXPERIENCE 2×2 (city median splits):
     - salience_high = salience > median
     - satisfaction_high = satisfaction > median
     - experience_group = HH / HL / LH / LL

8d. COLLAPSE TO BINARY:
     - supply_high = 1 if supply_group == HH, else 0
     - experience_high = 1 if experience_group == HH, else 0

8e. DIVERGENCE MATRIX:
     - Q1 (HH): high supply, high experience → well-served
     - Q2 (HL): high supply, low experience → experience deficit
     - Q3 (LH): low supply, high experience → overperforming
     - Q4 (LL): low supply, low experience → compounded disadvantage

8f. Output: da_divergence.csv (da_id, supply_group, experience_group,
     supply_binary, experience_binary, divergence_quadrant)

Sanity checks before full run:

Steps 1–4: run for 2 DAs, verify reachability proportions are plausible
Step 5: run for 2–3 parks, read outputs manually
Step 8: check quadrant distribution isn't heavily skewed to one cell