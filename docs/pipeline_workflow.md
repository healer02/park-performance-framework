# Pipeline workflow (draft)

## Supply
1. Park polygons (City of Vancouver)
2. Park entrances (park boundary buffer + intersect walkable OSM)
3. DB points + population (StatCan)
4. DB → nearest entrance network distance
5. DA reachability = reachable DB pop / total DB pop
6. Quantity (ha per 1,000) = sum reachable park area / DA pop * 1,000

## Experience
7. Google Reviews (Apify)
8. Park-level: review count, mean stars, sentiment
9. DA-level: salience = reachable reviews / DA pop * 1,000
10. DA-level: satisfaction = park-balanced sentiment (parks ≥10 reviews)

## Usability
11. Amenity extraction via keyword taxonomy
12. DA usability = unique amenity categories reachable

## Divergence
13. 2×2: supply high/low × experience high/low (median splits)