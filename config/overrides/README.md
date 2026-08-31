# Manual overrides

These tracked tables preserve reviewed decisions separately from generated data.

- `park_eligibility.csv`: include/exclude decisions for ambiguous park records.
- `park_small_exceptions.csv`: documented sub-0.1 ha parks retained under the
  proposal's Google-presence or recreational-infrastructure exception.
- `park_entities.csv`: municipal or regional component records confirmed to form one
  physical park for supply-area union and per-park area caps.
- `google_matches.csv`: accepted or rejected automated Google matches.
- `google_place_ids.csv`: explicitly selected Google Place IDs.
- `google_shared_entities.csv`: reviewed Google entities legitimately shared by
  multiple park polygons; these are counted once per DA.
- `park_entrances.csv`: future manual entrance additions, replacements, or removals.

Do not edit processed GeoPackages to make a permanent correction. Record the
correction here and rerun the applicable stage so it remains reproducible.
