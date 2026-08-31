# Methodology

This analysis applies the study design in the project proposal to six Metro
Vancouver municipalities: Burnaby, Coquitlam, New Westminster, Richmond,
Surrey, and Vancouver. Results are produced for each municipality and for a
combined six-city study area. The combined result is labelled `metro` in the
pipeline but does not represent every municipality in Metro Vancouver.

Google star ratings are the shared experience measure for the six-city
transferability analysis. Park-level sentiment is the primary Vancouver paper
measure and is also analysed alongside ratings for comparison.

## Spatial units and study context

The dissemination area (DA) is the neighbourhood unit used for reporting,
classification, equity analysis, and regression. Dissemination blocks (DBs)
represent the distribution of residents within each DA during the park-access
calculation.

The pedestrian network and park inventory extend beyond the six study
municipalities to reduce boundary effects. A park outside a municipality can
therefore serve residents near its boundary. Statistical outputs include only
the DAs belonging to the selected municipality or the combined six-city area.
Stage 04 reports the number of DBs and DAs reaching parks from outside the
analysis municipality inventory as a reproducible boundary-effect check.

All distance and area calculations use BC Albers (`EPSG:3005`).

## Park inventory

The park inventory combines 2022 municipal park polygons with the Metro
Vancouver Regional Parks boundary layer. Invalid geometries are repaired, and
records with the same source municipality and park name are dissolved. Parks
smaller than 0.1 ha are excluded unless a documented exception confirms a
public recreational function or a valid Google park listing with sufficient
reviews.

Names and source attributes provide an initial eligibility screen. Ambiguous
records are reviewed through `config/overrides/park_eligibility.csv`. This
separates research decisions from generated spatial files.

Some municipal and regional records are separate or overlapping representations
of one physical park. Confirmed components are assigned a shared
`park_entity_id` and their geometries are unioned. Grouping requires evidence
of a physical or administrative relationship; proximity or a shared Google
Place ID alone is not sufficient. The union prevents overlapping components
from contributing area more than once and ensures that the park-area cap is
applied once per physical park.

## Park entrances and pedestrian network

The walking network is derived from OpenStreetMap and treated as undirected.
Entrance candidates are created where the network intersects a 10 m park
boundary zone and from network-edge endpoints touching a park. Candidates
within 25 m are clustered and snapped to network nodes.

If no boundary entrance is found, the nearest network edge can provide a
fallback entrance when it is within 100 m. Parks without a plausible network
connection remain in the quality-assurance output but are excluded from the
reachability calculation.

## Park access

Each DB is represented by the official Statistics Canada coordinates
`DBRPLAMX` and `DBRPLAMY`. These points are snapped to the pedestrian network;
the snap distance is recorded for quality assurance but is not added to the
access threshold.

A park is accessible from a DB when a validated entrance can be reached within
400 m along the pedestrian network. Every unique reachable park is retained,
not only the nearest park.

DA population coverage is calculated as:

```text
population in DBs with at least one reachable park / total DA population
```

A DA has high coverage when at least 80% of its residents have access. DBs with
zero population remain in quality-assurance tables but do not add parks to a
DA’s resident-based supply or experience measures.

## Accessible park supply

For each DA, the analysis identifies all unique physical park entities
reachable from its populated DBs. Each entity is counted once, even if several
DBs or several municipal components can reach it.

The primary area measure caps each reachable park entity at 20 ha before the
areas are summed. Accessible park supply is:

```text
sum of capped reachable park areas / DA population × 1,000
```

A populated DA with no reachable park receives zero supply. A DA without
residents is marked as having insufficient population.

High supply requires both:

- population coverage of at least 80%; and
- accessible park area per 1,000 residents at or above the analysis-area
  median.

Municipal analyses use their own medians. The combined analysis uses one median
across all populated DAs in the six-city study area.

## Google matching and park experience

Park records are matched to Google Places using normalized name similarity,
Google place type, and distance from the Google marker to the park polygon.
Ratings and review counts are not used to choose the match. Clear candidates
can be accepted by the matching rules; ambiguous candidates are reviewed and
recorded in the override tables.

Rule-based matching is used because its evidence can be inspected directly and
the available reviewed matches do not justify a supervised machine-learning
model. Indoor facilities are not substituted for outdoor park experience.

A matched Google entity is eligible for the primary experience measure when it
has a rating and at least 10 reviews. A reviewed Place ID shared by park records
is counted once within a DA.

DA experience is the unweighted mean rating of all unique eligible Google
entities reachable from its populated DBs. Ratings are not weighted by review
count in the primary measure. A DA with no eligible reachable rating retains a
missing experience value rather than receiving zero.

The 10-review cutoff is therefore applied at the Google-place level, not as a
minimum number of parks per DA. A populated DA is classified when it reaches at
least one eligible rated Google entity; otherwise it is labelled insufficient
experience data.

Digital salience is reported separately as the number of reviews associated
with validated reachable Google entities per 1,000 residents. It describes the
amount of online attention and is not used to classify satisfaction.

## Supply-experience divergence

Experience is high when the DA’s mean rating is at or above the median for the
applicable analysis area. Supply and experience are combined into four groups:

| Class | Supply | Experience |
|---|---|---|
| HH | High | High |
| HL | High | Low |
| LH | Low | High |
| LL | Low | Low |

DAs without a usable rating are labelled as having insufficient experience,
and DAs without residents are labelled as having insufficient population.
Missing ratings are not treated as low ratings.

Each city uses its own supply and rating medians. The combined analysis uses
one pair of medians across the six-city area. A DA can therefore have one class
in its city analysis and another in the combined analysis: the first describes
its position within the city, while the second describes its position within
the regional study area.

Values equal to a median are assigned to the high group. A small numerical
tolerance prevents equivalent decimal ratings from being separated by
floating-point representation.

## Equity indicators

The equity analysis uses seven variables from the 2021 Census Profile and the
2021 Can-ALE index:

| Indicator | Source |
|---|---|
| Median household income | Census characteristic 243 |
| Bachelor’s degree or higher | Census characteristic 2008 |
| Population aged 0 to 14 | Census characteristic 9 |
| Population aged 65 or older | Census characteristic 24 |
| Prevalence of LIM-AT | Census characteristic 345 |
| Immigrant population | Census characteristic 1529 |
| Visible-minority population | Census characteristic 1684 |
| Active living environment | Can-ALE `ALE_index` |

Published Census percentages are used rather than recalculating rates with
different denominators. Missing values are retained and omitted only from the
analysis requiring that indicator.

The equity profile includes DAs classified as HH, HL, LH, or LL. Each indicator
is standardized within the analysis area, and the mean z-score is reported for
each divergence class. Variables are ordered by the difference between
higher-experience classes (HH and LH) and lower-experience classes (HL and LL).

For the continuous equity-profile dot plot, differences across HH, LH, HL, and
LL are tested using Kruskal–Wallis tests. Eta-squared is reported as an effect
size, and the figure uses `*` for p<.05, `**` for p<.01, and `***` for p<.001.

For the separate categorized-strata analysis, the six Census percentages use
fixed low, middle, and high bands. Median household income uses thresholds at
60% and 140% of the analysis-area median. Can-ALE uses analysis-area tertiles.
Strata are defined before restricting the data to DAs with a divergence class.

## Statistical analysis

Descriptive tables report sample size, missing values, mean, standard
deviation, minimum, quartiles, and maximum for the main supply, experience,
salience, income, and equity measures.

Spearman correlations assess:

- population coverage and rating;
- accessible park area and rating;
- population coverage and digital salience;
- accessible park area and digital salience; and
- rating and digital salience.

Global Moran’s I tests spatial autocorrelation in population coverage,
accessible park area, ratings, and membership in each divergence class.
Neighbourhoods are defined by first-order Queen contiguity, and weights are
row-standardized. Significance is estimated with 999 reproducible permutations.
Each divergence class is tested as a separate binary indicator rather than
assigning an artificial numeric order to the four classes.

Associations between categorized equity strata and divergence class are tested
with Pearson’s chi-square and summarized with Cramér’s V. The asymptotic p-value is
used when all expected counts are at least one and at least 80% are five or
greater. Otherwise, the p-value is estimated using 4,999 reproducible label
permutations. Figure labels use `*` for p<.05, `**` for p<.01, and `***` for
p<.001.

Multinomial logistic regression uses LL as the reference class. Predictors are
children aged 0 to 14, visible-minority share, age 65 or older, LIM-AT,
bachelor’s-plus education, and Can-ALE, standardized within the analysis area.
The combined six-city model is the primary regional model. A city model is
fitted only when at least 100
complete DAs are available and every divergence class contains at least 20
complete DAs. Models that do not meet these conditions are reported as not run.

For Vancouver, the descriptive statistics, correlations, Moran’s I tests,
equity tests, and regression are repeated using mean sentiment and the
sentiment-based divergence classes. Sentiment outputs use a `_sentiment`
filename suffix and do not replace the rating results.

## Sensitivity analyses

The primary definitions are accompanied by one-at-a-time alternatives:

- 10 ha rather than 20 ha park-area cap;
- uncapped accessible park area;
- 50% rather than 80% population coverage;
- review-count-weighted rating; and
- DB-population-weighted rating.

Each alternative experience measure uses its own analysis-area median.

The Vancouver sentiment score also generates parallel divergence, equity, and
statistical results. The primary score is linked from Keun's park-level metrics
to the rebuilt park inventory by source and park name; it is never combined
with the rating score. Google Place IDs remain the linkage for review-level
validation and perceived-amenity analyses.

## Figures

All figures use a shared teal-brown palette for HH, LH, HL, and LL. Grey denotes
insufficient experience, and light grey denotes insufficient population.

The divergence map displays DA classes and park outlines. The equity profile
shows mean standardized values by class. The supply-experience figure uses a
hexagonal density display for park coverage and a logarithmic area axis for
accessible park area. Stacked equity bars show the class distribution within
each equity stratum. Figure titles are omitted where the information is better
placed in the manuscript caption.

## Limitations

Google users are not a representative sample of all park visitors, and parks
with few or no reviews have missing experience data. Ratings may reflect
features outside municipal park boundaries or change after the retrieval date.
The 2021 Census, 2021 Can-ALE data, 2022 park inventory, and current Google data
do not describe exactly the same moment in time.

The analysis is ecological: DA-level associations do not establish individual
experiences or causal effects. The combined study area contains six
municipalities rather than the full Metro Vancouver region. A comparable
multi-city park-amenity dataset was not available, so perceived amenities are
analysed for Vancouver only and are not included in the six-city comparison.

## Vancouver sentiment validation and perceived usability

For Vancouver, review-level sentiment is compared with star ratings at the
review, Google Place ID, and DA levels using Pearson and Spearman correlations.
Median-based high/low experience classifications are compared using percentage
agreement and Cohen's kappa. A sensitivity analysis weights each reachable
park's sentiment by the natural log of its total Google reviews;
the primary measure remains the unweighted mean so that popular destination
parks do not dominate.

Perceived usability is derived from eleven predefined amenity categories in
review text. An amenity is considered present for a Google park entity when at
least two independent reviews mention one of its keywords. Matches preceded by
a negation within four words are ignored. Within each DA, amenity indicators
are the union across reachable validated Google entities.

Amenity prevalence is compared across the sentiment-based HH, LH, HL, and LL
classes using chi-square tests. Keyword-derived indicators are compared with
official Vancouver facilities and washroom records using Cohen's kappa for the
seven categories with comparable records. The amenity-adjusted multinomial
model retains the six neighbourhood predictors from the main model and adds
eight discriminatory amenity indicators; playgrounds, washrooms, and seating
or shelter are excluded from that model as baseline infrastructure.
