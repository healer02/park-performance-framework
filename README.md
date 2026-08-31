# Park Performance Analysis

Reproducible analysis of neighbourhood park supply, Google-rating experience,
and supply-experience divergence across six Metro Vancouver municipalities.

## Study area

- Burnaby
- Coquitlam
- New Westminster
- Richmond
- Surrey
- Vancouver
- Combined six-city Metro Vancouver study area

The combined analysis represents these six municipalities, not the entire Metro
Vancouver region. A wider pedestrian network and park context may be used to
reduce boundary effects.

## Project status

The repository structure, configuration loader, input validation, and pipeline
command are established. Stages 01-08 now reproducibly build and validate the
shared network, reviewed parks and entrances, population, 400 m reachability,
accessible park supply, and Google-rating park experience for every city and
the combined study area, including supply-experience classification and seven
equity indicators. Stage 09 creates descriptive, association, spatial
autocorrelation, diagnostic, and regression tables. Stage 10 creates the
divergence map, standardized equity profile, supply-experience relationships,
and equity-strata appendix figure for every analysis area.
For Vancouver, Stage 11 completes the proposal-specific sentiment validation,
six-panel pairwise relationships figure, review-weighted sensitivity analysis,
perceived-amenity profiles, official inventory agreement, and amenity-adjusted
regression.

## Analysis overview

1. Prepare the shared pedestrian network.
2. Clean park polygons and derive entrances.
3. Prepare Census dissemination blocks (DBs) and dissemination areas (DAs).
4. Calculate 400 m network-based park reachability.
5. Reconcile confirmed physical park components and calculate accessible park
   area per 1,000 residents with one area cap per park entity.
6. Aggregate eligible Google star ratings to each DA.
7. Classify DAs as HH, HL, LH, LL, or insufficient experience data.
8. Join median household income, six Census equity rates, and Can-ALE to each DA.
9. Run the statistical analyses.
10. Produce city and combined-study-area maps, Table 3, Figure 8, and other
    manuscript figures.
11. For Vancouver, validate sentiment and analyse perceived park amenities.

Google star ratings form the shared six-city transferability workflow.
Vancouver sentiment is the primary paper branch for Sections 4.1-4.3 and is
linked by Google Place ID. It uses the same
Stage 06–10 scripts and writes separate `_sentiment` outputs. The single
Vancouver-only Stage 11 contains the review-level validation and amenity work
that cannot be run for the other cities without comparable text-review data.

## Repository guide

- [`docs/workflow.md`](docs/workflow.md): how to set up and run the project.
- [`docs/methodology.md`](docs/methodology.md): what the indicators calculate and why.
- [`config/settings.yaml`](config/settings.yaml): shared paths, thresholds, and plot style.
- `config/cities/`: one small configuration file per municipality.
- [`config/metro.yaml`](config/metro.yaml): combined six-city analysis scope.
- [`data/README.md`](data/README.md): expected input data and Git rules.

## Project origin and development

This repository extends the original Vancouver park-performance framework
developed by Keun Park:

https://github.com/healer02/park-performance-framework

The original framework was adapted into a reproducible, configuration-driven
pipeline for six Metro Vancouver municipalities. Major extensions include
multi-city processing, shared network preparation, Google-rating-based park
experience, combined-area analysis, equity indicators, statistical analysis,
and standardized manuscript outputs.

## Environment

Python 3.11 or newer is recommended. The foundation has been tested locally with
Python 3.14.4. Provisional dependencies are listed in `requirements.txt`; exact
versions will be locked after the first successful end-to-end validation.

Large raw data, intermediate files, API credentials, and caches are excluded
from Git. Selected publication-ready outputs are retained so the reported
results can be reviewed without rerunning the spatial pipeline.

## Foundation commands

After installing the dependencies:

```powershell
python scripts/run_pipeline.py --list-areas
python scripts/run_pipeline.py --area vancouver --check-only
python scripts/run_pipeline.py --area all --check-only
python scripts/run_pipeline.py --area metro --check-only
python scripts/run_pipeline.py --area metro --stages 01 02
python scripts/run_pipeline.py --area metro --stages 02 --force
python scripts/run_pipeline.py --area all --stages 03 04 05 06 07
python scripts/run_pipeline.py --area metro --stages 03 04 05 06 07
python scripts/prepare_google_ratings.py --area metro
python scripts/run_pipeline.py --area all --stages 08 --force
python scripts/run_pipeline.py --area metro --stages 08 --force
python scripts/run_pipeline.py --area all --stages 09 10 --force
python scripts/run_pipeline.py --area metro --stages 09 10 --force
python scripts/run_pipeline.py --area vancouver --stages 11 --force
```
