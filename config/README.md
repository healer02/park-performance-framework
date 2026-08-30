# Configuration

`settings.yaml` contains values shared by every analysis area: input paths,
field names, thresholds, coordinate systems, optional sentiment settings, and
the official plot palette.

Each file in `cities/` defines only a municipality name, output slug, and Census
subdivision code. `metro.yaml` defines the combined six-city analysis.

At runtime, the configuration loader will merge `settings.yaml` with one analysis
area file. Analysis-area values take precedence if a future city requires an
explicit exception.

An optional, untracked `local.yaml` can set the local data location without
changing shared configuration. Copy `local.example.yaml` to `local.yaml` and
edit `local.data_root`. The environment variable `PARK_PERFORMANCE_DATA_ROOT`
can override both values when needed.

The wider network boundary in `settings.yaml` is routing context. It must not be
confused with the six CSD codes included in the combined statistical analysis.

Google ratings are the shared primary experience source. Vancouver's city file
defines the additional sentiment input contract. Google `PlaceID` links the
sentiment table to the current park inventory even when internal park IDs
differ. Sentiment adds parallel Stage 06–10 results and never replaces or
combines with the rating score.

The Vancouver city file also identifies the review-level sentiment data and
official facilities and washroom tables used by Stage 11. These large source
files remain in the configured data folder and are not committed to Git.

The shared plot palette contains the four divergence colours plus separate grey
values for missing experience and insufficient population. Scripts should read
these values from configuration instead of redefining colours locally.
