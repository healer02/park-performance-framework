"""
config.py
City-specific parameters for the park performance pipeline.

To run the pipeline on a new city, update only this file.
All scripts import from here instead of hardcoding values.
"""

import os

# ── Repo root ─────────────────────────────────────────────────────────────────
# Update this path when running on a different machine.
REPO_DIR = '/Users/keunpark/Documents/GitHub/park-performance-framework'

# ── City identity ─────────────────────────────────────────────────────────────
CITY      = 'vancouver'         # used as prefix on all output filenames
CSD_CODES = ['5915022',         # City of Vancouver
             '5915025',         # Burnaby
             '5915020']         # Metro Vancouver A / Electoral Area A (UBC)
STUDY_CSD = '5915022'           # primary CSD for DB centroid extraction
CRS       = 'EPSG:3005'         # BC Albers, metre-based

# ── Census file paths (national — no need to change per city) ─────────────────
GAF_CSV = 'data/census/raw/2021_92-151_x.csv'
DA_SHP  = 'data/census/raw/lda_000b21a_e/lda_000b21a_e.shp'
DB_SHP  = 'data/census/raw/ldb_000b21a_e/ldb_000b21a_e.shp'

# ── Park polygon sources ──────────────────────────────────────────────────────
# Parallel lists: one entry per source.
# PATH can be a file or a directory (for glob *.shp).
PARK_SOURCES = [
    {
        'path':      'data/parks/raw/Vancouver/parks-polygon-representation/parks_polygon_SpanishBanksMerged.gpkg',
        'name':      'Vancouver',
        'name_col':  'PARK_NAME',
        'dissolve':  False,
    },
    {
        'path':      'data/parks/raw/Burnaby/Park_Inventory.shp',
        'name':      'Burnaby',
        'name_col':  'NAME',
        'dissolve':  True,   # dissolve by park_name before merging
    },
    {
        'path':      'data/parks/raw/Metro Vancouver Regional Parks',  # directory
        'name':      'MetroVancouver',
        'name_col':  'parkname',
        'dissolve':  False,
    },
]

# ── Park filtering ────────────────────────────────────────────────────────────
MIN_AREA_HA = 0.1

# Parks below MIN_AREA_HA explicitly kept (named parks with ≥50 reviews or
# verified distinct recreational infrastructure).
EXPLICIT_INCLUDE = [
    'Choklit Park',
    'Major Matthews Park',
    'Willow Park',
    'Street End - Wall St @ Nanaimo',
]

# Non-parks, no-review parks, or polygons already merged into another entry.
EXCLUDE_PARKS = [
    'Iona Beach Regional Park',
    'BOUNDARY CREEK RAVINE PARK',
    'STILL CREEK CONSERVATION AREA',
    'Mont Royal Square',
    'Nat Bailey Stadium Park',
    'Shannon Mews Park',
    "Gibby's Field",
    'Downtown Skateboard Plaza',
    'West End minipark - GILFORD ST @ HARO ST',
    'Spanish Banks Extension',
    'Roundhouse Turntable Plaza',
    'Empire Fields - Hastings Park',
    'Slidey Slides',
    'Locarno Park',
    'Helmcken Park',
    'RIVERWAY GOLF COURSE',
    'Vanier Park (Cultural Harmony Grove)',
]

# Per-park entrance extraction buffer overrides (metres).
# Default buffer is BUFFER_M below. Add entries here for edge-case parks.
PARK_BUFFERS = {
    'Shaughnessy Park': 30,   # polygon digitised ~28m inside road centreline
}

# ── Network parameters ────────────────────────────────────────────────────────
BUFFER_M          = 10    # entrance extraction: park boundary buffer
DEDUP_M           = 25    # entrance deduplication: merge within this distance
REACH_PRIMARY     = 400   # metres — primary reachability threshold
REACH_SENSITIVITY = 800   # metres — sensitivity analysis threshold
AREA_CAP_MAIN     = 20    # ha — primary per-park area cap
AREA_CAP_SENS     = 10    # ha — sensitivity per-park area cap
SUPPLY_REACH_THRESH = 0.8 # reachability proportion for "high supply" binary

# ── Experience source ─────────────────────────────────────────────────────────
# "sentiment"   — use RoBERTa sentiment scores (requires vancouver/ sub-track)
# "star_rating" — use Google star ratings directly (replication cities)
#
# When EXPERIENCE_SOURCE = "star_rating":
#   • scripts 08_usability.py and 09_validation.py self-exit with a message
#   • the vancouver/ sub-track (v01–v03) does not need to be run
EXPERIENCE_SOURCE = 'sentiment'

MIN_REVIEWS = 10  # minimum text reviews for a park to contribute to satisfaction

# ── Equity / CANUE ────────────────────────────────────────────────────────────
# Set CANUE_AVAILABLE = False for replication cities until CANUE data is received.
#
# NOTE: census DA/DB *boundaries* and population are national (StatCan GAF +
# shapefiles) and always available. The neighbourhood sociodemographic variables
# used in 07_equity.py — visible minority share, age composition, LIM-AT
# low-income rate, education, household income, immigrant share, and the
# Can-ALE walkability index — all come from CANUE and must be requested
# separately before running 07_equity.py on any replication city.
#
# Submit a CANUE data request covering the relevant CSDs:
#   5915022  City of Vancouver
#   5915029  New Westminster
#   5915034  Coquitlam
#   5915025  Burnaby
#   5915031  Richmond
CANUE_AVAILABLE = True
CANUE_CSV       = 'data/census/raw/census_CANUE_DA_nearVan.csv'

# ── Official park facilities inventory ───────────────────────────────────────
# Set HAS_OFFICIAL_INVENTORY = False for cities without a machine-readable
# park facilities inventory. This skips the Cohen's kappa validation section
# in 08_usability.py without affecting amenity extraction.
HAS_OFFICIAL_INVENTORY = True
FACILITIES_CSV = 'data/parks/raw/Vancouver/parks-facilities.csv'
WASHROOMS_CSV  = 'data/parks/raw/Vancouver/public-washrooms.csv'

# ── Divergence colour scheme ──────────────────────────────────────────────────
COLOURS_2X2 = {
    'HH': '#01665e',   # high supply / high experience — dark teal
    'LH': '#80cdc1',   # low supply  / high experience — light teal
    'HL': '#8c510a',   # high supply / low experience  — dark brown
    'LL': '#dfc27d',   # low supply  / low experience  — light tan
}
