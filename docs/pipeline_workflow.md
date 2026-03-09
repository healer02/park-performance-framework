# Pipeline workflow
# Vancouver as an example

0. Setup
inputs:
- [done: 2026/3/6] Vancouver parks + parks-facilities + parks-special-features + drinking-fountains + public-washrooms (City of Vancouver open data: https://opendata.vancouver.ca/pages/home/)
- [done: 2026/3/9] Metro Vancouver Regional Parks (Metro Van open data: https://open-data-portal-metrovancouver.hub.arcgis.com/)
- [done: 2026/3/9] Burnaby parks (Burnaby open data: https://data.burnaby.ca/)
- [done: 2026/3/6] DA boundaries: lda_000b21a_e.shp (StatCan 2021: https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21)
- [done: 2026/3/6] DB points and population (StatCan 2021, population-weighted representative points: https://www150.statcan.gc.ca/n1/en/catalogue/92-151-X)
- [done: 2026/3/9] run scripts > 01-get-osm-network.py
- [for Experience dimension] google_reviews.csv (Apify, park_id, rating, text, date)
- [for analysis 2] DA census profile 

1. Park entrance extraction: ArcGIS or Python
1a. Buffer park boundaries by 30m
1b. Intersect buffered park polygons with OSM walkable streets
     (keep: footway, path, pedestrian, residential, service)
1c. Extract intersection points as candidate entrances
1d. Snap entrance points to nearest OSM node
    if no entrance detected → use nearest boundary point
1e. Output: park_entrances.shp (park_id, entrance_id, geometry)

Validation: spot-check 10 entrances against Google Maps satellite

2. DB → nearest entrance distance 
2a. Load OSM network graph (NetworkX or ArcGIS Network Analyst)
2b. For each DB representative point:
     - find nearest network node
     - run Closest Facility to all park entrances
     - record: db_id, park_id, distance_m, reachable (1 if ≤400m)
2c. Output: db_park_distances.csv (db_id, park_id, distance_m, reachable)

Note: record park_id for all reachable parks per DB, not just nearest (two cases: 400m vs. 800m for sensitivity analysis)

3. DA reachability
3a. Join db_park_distances to db_population
3b. For each DA:
     - sum DB pop where reachable = 1 (any park within 400m)
     - divide by total DA population
3c. DA_reachability = sum(DB_pop_reachable) / sum(DB_pop_total)
3d. Output: da_reachability.csv (da_id, reachability_proportion)

4. Quantity (per-capita reachable park area)
<!-- Why union parks at DA level for quantity? -->
4a. From db_park_distances, get unique park_ids reachable from any DB in each DA
4b. Join to park_boundaries to get park area (ha)
4c. Dissolve to DA level: sum unique park areas
4d. Quantity(DA) = sum(unique_park_area_ha) / DA_pop × 1000
4e. Output: da_quantity.csv (da_id, quantity_ha_per_1000)

Sensitivity: rerun with 800m threshold, no population division


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