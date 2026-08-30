# Workflow

This guide explains how to configure and run the park performance analysis.
The study covers Burnaby, Coquitlam, New Westminster, Richmond, Surrey, and
Vancouver. The `metro` analysis combines these six municipalities; it does not
represent the entire Metro Vancouver region.

## 1. Set up Python

Python 3.11 or newer is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If Python is installed but `python` is not recognized, use the full path to the
interpreter in the commands below.

## 2. Configure the data folder

Source data and generated outputs are not stored in Git. Copy
`config/local.example.yaml` to `config/local.yaml` and set `local.data_root` to
the folder containing the project data:

```yaml
local:
  data_root: "C:/path/to/project/data"
```

`config/local.yaml` is ignored by Git. The data root can also be supplied for a
terminal session through `PARK_PERFORMANCE_DATA_ROOT`.

Shared settings, field names, thresholds, and output styles are defined in
`config/settings.yaml`. The files under `config/cities/` identify each
municipality, and `config/metro.yaml` defines the combined six-city analysis.

## 3. Required inputs

The default paths below are relative to the configured data root.

| Input | Default location | Purpose |
|---|---|---|
| 2021 Census geographic attribute file | `raw/census/2021_92-151_X.csv` | DB population and geographic identifiers |
| 2021 DA boundaries | `raw/census/lda_000b21a_e/lda_000b21a_e.shp` | Neighbourhood boundaries |
| 2021 DB boundaries | `raw/census/ldb_000b21a_e/ldb_000b21a_e.shp` | Within-DA population locations |
| 2021 Census Profile | `raw/census/98-401-X2021006_BC_CB_eng_CSV/98-401-X2021006_English_CSV_data_BritishColumbia.csv` | Household and demographic indicators |
| 2021 Can-ALE | `raw/census/CanALE_2021.csv` | Active living environment indicator |
| Municipal park polygons | `raw/parks/all parks in Metro Vancouver (2022)/merged.shp` | Park inventory |
| Validated Google ratings | `processed/google/metro_google_ratings_validated.csv` | Park ratings and review counts |
| Vancouver park sentiment | `raw/google/08c-park-metrics.csv` | Primary park-level sentiment branch |

Run an input check before beginning:

```powershell
python scripts/run_pipeline.py --area all --check-only
python scripts/run_pipeline.py --area metro --check-only
```

The validation report identifies missing files, layers, and required fields.

## 4. Pipeline stages

| Stage | Script | Output |
|---|---|---|
| 01 | `01_network.py` | Shared pedestrian network |
| 02 | `02_parks.py` | Cleaned parks and park entrances |
| 03 | `03_population.py` | DA and DB population data |
| 04 | `04_reachability.py` | DB-to-park reachability within 400 m |
| 05 | `05_supply.py` | DA park coverage and accessible area |
| 06 | `06_experience.py` | DA ratings and digital salience |
| 07 | `07_divergence.py` | Supply-experience classes |
| 08 | `08_equity.py` | Equity indicators and class profiles |
| 09 | `09_statistics.py` | Descriptive, association, spatial, and regression tables |
| 10 | `10_figures.py` | Maps and publication figures |
| 11 | `11_vancouver_extension.py` | Vancouver sentiment validation and perceived-usability analysis |

Stages 01 and 02 create shared regional inputs. Stages 03–10 run separately for
each municipality and for the combined six-city analysis. Stage 11 runs only
for Vancouver when sentiment and review-text inputs are available.

Use a dry run to review the commands without executing them:

```powershell
python scripts/run_pipeline.py --area all --dry-run
python scripts/run_pipeline.py --area metro --dry-run
```

## 5. First complete run

Build the shared network and park inventory:

```powershell
python scripts/run_pipeline.py --area all --stages 01 02 --force
```

Prepare population, reachability, and supply for the cities and combined area:

```powershell
python scripts/run_pipeline.py --area all --stages 03 04 05 --force
python scripts/run_pipeline.py --area metro --stages 03 04 05 --force
```

If the validated Google rating table already exists, continue with Stages
06–10:

```powershell
python scripts/run_pipeline.py --area all --stages 06 07 08 09 10 --force
python scripts/run_pipeline.py --area metro --stages 06 07 08 09 10 --force
```

`--area all` runs the six municipalities. The combined result always requires
the separate `--area metro` command.

## 6. Preparing Google ratings

Google matching is prepared once for the shared park inventory. Existing
candidates can be rescored without making API calls:

```powershell
python scripts/prepare_google_ratings.py --area metro --candidate-source fresh
```

To retrieve fresh candidates, place the API key in the current terminal only:

```powershell
$secureApiKey = Read-Host "Google Maps API key" -AsSecureString
$env:GOOGLE_MAPS_API_KEY = [System.Net.NetworkCredential]::new("", $secureApiKey).Password
```

First print the request plan without making billable calls:

```powershell
python scripts/prepare_google_ratings.py --area metro --candidate-source fresh --api-scope all
```

Then execute it with a positive safety limit. Replace `REQUEST_LIMIT` with the
number of requests you intend to allow:

```powershell
python scripts/prepare_google_ratings.py --area metro --candidate-source fresh --api-scope all --execute-api --max-api-requests REQUEST_LIMIT
```

Review ambiguous matches in the generated review table and record permanent
decisions under `config/overrides/`. Rerun the preparation script after editing
an override. Never store an API key in a configuration file or commit it to
Git.

## 7. Rerunning part of the analysis

Select only the stages affected by a change. For example, after changing a
rating match:

```powershell
python scripts/run_pipeline.py --area all --stages 06 07 08 09 10 --force
python scripts/run_pipeline.py --area metro --stages 06 07 08 09 10 --force
```

After changing figure code only:

```powershell
python scripts/run_pipeline.py --area all --stages 10 --force
python scripts/run_pipeline.py --area metro --stages 10 --force
```

Without `--force`, completed stages validate and reuse their existing outputs
where supported. Use `--force` when an output must be rebuilt.

## 8. Manual decisions

Permanent corrections belong in `config/overrides/`, not in a generated CSV or
GeoPackage.

| File | Decision recorded |
|---|---|
| `park_eligibility.csv` | Include or exclude an ambiguous park record |
| `park_small_exceptions.csv` | Retain a justified park below 0.1 ha |
| `park_entities.csv` | Group municipal or regional records belonging to one physical park |
| `park_entrances.csv` | Add, replace, or remove an entrance |
| `google_matches.csv` | Accept or reject a Google candidate |
| `google_place_ids.csv` | Select a specific Google Place ID |
| `google_shared_entities.csv` | Approve a Place ID shared by park records |

Each override should include enough information for another researcher to
understand the decision.

## 9. Outputs

Each analysis area has the same output structure:

```text
outputs/<area>/
├── figures/
├── maps/
├── tables/
└── logs/
```

Inspectable stage data are written under `data/interim/<stage>/<area>/`.
Canonical shared datasets, including cleaned parks and validated Google
ratings, are written under `data/processed/`.

Stage 10 produces a divergence map, an equity profile, a two-panel
supply-experience figure, and an equity-strata stacked-bar figure. Captions,
source fingerprints, thresholds, and dimensions are recorded in the Stage 10
summary table.

Running Stage 10 for `metro` also produces the manuscript transferability
outputs from the six city-specific rating classifications:

- `outputs/metro/tables/six_city_transferability_summary.csv` (Table 3); and
- `outputs/metro/maps/six_city_divergence_comparison.png` (Figure 8).

The Figure 8 panels all have the same physical size, use the shared divergence
palette and one legend, and omit the individual-map 2x2 count matrix.

## 10. Vancouver sentiment analysis

The shared transferability workflow uses ratings for all six cities. The
Vancouver primary-analysis branch adds review text and sentiment to the same
spatial core. Its primary input is Keun's park-level `08c-park-metrics.csv`,
containing `park_id`, `park_name`, `MeanSentiment`, `TotalReviews`, and
`has_valid_sentiment`. Stage 06 maps these records to the rebuilt inventory by
source and park name. The Place ID-level sentiment table remains an input to
Stage 11 validation and perceived-amenity analyses.

The sentiment result is generated alongside Vancouver's rating result. The two
scores are not combined, and the rest of the cities remain rating-based. If the local
sentiment file is unavailable, set `experience.sentiment.enabled: false` in
`config/cities/vancouver.yaml` before running Vancouver.

To reproduce the proposal's sentiment validation and perceived-usability
analyses, also provide the review-level sentiment file and the official
Vancouver facilities and washroom tables configured in
`config/cities/vancouver.yaml`, then run:

```powershell
python scripts/run_pipeline.py --area vancouver --stages 11 --force
```

Stage 11 reports sentiment-versus-rating agreement at review, park,
and DA levels; the six-panel pairwise supply-and-experience figure;
review-weighted sentiment sensitivity; review-derived amenity profiles and
official-inventory agreement; and the amenity-adjusted
multinomial model. Its figures and tables are written beside the other
Vancouver outputs.
