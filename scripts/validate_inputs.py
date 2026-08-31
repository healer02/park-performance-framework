"""Validate project configuration, input presence, and required input fields."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project_config import ConfigError, data_path, load_config


@dataclass
class CheckResult:
    item: str
    status: str
    detail: str


def _csv_header(path: Path) -> set[str]:
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                return set(next(csv.reader(stream)))
        except (UnicodeDecodeError, StopIteration) as error:
            last_error = error

    raise ValueError(f"Could not read CSV header: {last_error}")


def _spatial_columns(path: Path) -> set[str]:
    try:
        import geopandas as gpd
    except ImportError as error:
        raise RuntimeError(
            "geopandas is not installed; install requirements before spatial validation"
        ) from error

    sample = gpd.read_file(path, rows=1)
    return set(sample.columns)


def _input_specs(config: dict[str, Any]) -> dict[str, set[str]]:
    fields = config["fields"]

    return {
        "census_gaf": {
            fields["gaf_csd_id"],
            fields["gaf_da_id"],
            fields["gaf_db_id"],
            fields["gaf_db_population"],
        },
        "census_profile": {
            "ALT_GEO_CODE",
            "GEO_LEVEL",
            "CHARACTERISTIC_ID",
            "C10_RATE_TOTAL",
        },
        "canale": {"DAUID", "ALE_index"},
        "da_boundaries": {fields["da_id"]},
        "db_boundaries": {
            fields["db_id"],
            fields["db_representative_x"],
            fields["db_representative_y"],
        },
        "parks_source": {
            config["parks_source"]["name_field"],
            config["parks_source"]["city_field"],
        },
        "regional_parks_source": {
            config["regional_parks_source"]["name_field"],
        },
        "vancouver_parks_supplement": {
            config["vancouver_parks_supplement"]["name_field"],
        },
        "parks": {
            fields["park_id"],
            fields["park_name"],
            fields["park_city"],
        },
        "entrances": {fields["park_id"], "nearest_node"},
        "google_ratings": {
            fields["park_id"],
            "google_rating",
            "google_review_count",
            "match_status",
        },
        "google_ratings_validated": {
            fields["park_id"],
            "PlaceID",
            "match_validated",
            "google_rating",
            "google_review_count",
            "rating_eligible",
        },
    }


def validate_config(config: dict[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    area = config.get("analysis_area", {})
    analysis = config.get("analysis", {})
    palette = config.get("plot", {}).get("palette", {})

    codes = [str(value) for value in area.get("analysis_csd_codes", [])]
    if codes and all(len(value) == 7 and value.isdigit() for value in codes):
        results.append(CheckResult("analysis_csd_codes", "OK", ", ".join(codes)))
    else:
        results.append(
            CheckResult("analysis_csd_codes", "ERROR", "Expected one or more 7-digit codes")
        )

    threshold = analysis.get("high_coverage_threshold")
    if isinstance(threshold, (int, float)) and 0 <= threshold <= 1:
        results.append(CheckResult("high_coverage_threshold", "OK", str(threshold)))
    else:
        results.append(CheckResult("high_coverage_threshold", "ERROR", "Expected 0 to 1"))

    required_colours = {
        "HH",
        "LH",
        "HL",
        "LL",
        "insufficient_experience",
        "insufficient_population",
    }
    missing_colours = required_colours - set(palette)
    if missing_colours:
        results.append(
            CheckResult("plot.palette", "ERROR", f"Missing: {', '.join(sorted(missing_colours))}")
        )
    else:
        results.append(CheckResult("plot.palette", "OK", "All divergence classes configured"))

    return results


def validate_inputs(
    config: dict[str, Any],
    input_keys: set[str] | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    specs = _input_specs(config)
    if input_keys is not None:
        unknown = input_keys - set(specs)
        if unknown:
            raise ConfigError(f"Unknown input keys: {', '.join(sorted(unknown))}")
        specs = {key: value for key, value in specs.items() if key in input_keys}

    for key, required_columns in specs.items():
        path = data_path(config, key)

        if not path.exists():
            results.append(CheckResult(key, "MISSING", str(path)))
            continue

        try:
            if path.suffix.lower() == ".csv":
                columns = _csv_header(path)
            else:
                columns = _spatial_columns(path)

            missing = required_columns - columns
            if missing:
                results.append(
                    CheckResult(key, "ERROR", f"Missing columns: {', '.join(sorted(missing))}")
                )
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                results.append(CheckResult(key, "OK", f"{size_mb:,.2f} MB | {path}"))
        except Exception as error:
            results.append(CheckResult(key, "ERROR", str(error)))

    return results


def print_results(config: dict[str, Any], results: list[CheckResult]) -> None:
    area = config["analysis_area"]
    print(f"\nInput validation: {area['name']} ({area['slug']})")
    print(f"Data root: {config['runtime']['data_root']}\n")

    item_width = max(len(result.item) for result in results)
    for result in results:
        print(f"{result.status:<7} {result.item:<{item_width}}  {result.detail}")

    errors = sum(result.status != "OK" for result in results)
    print(f"\n{len(results) - errors} checks passed; {errors} require attention.")


def run_validation(area: str, input_keys: set[str] | None = None) -> bool:
    config = load_config(area)
    results = validate_config(config) + validate_inputs(config, input_keys=input_keys)
    print_results(config, results)
    return all(result.status == "OK" for result in results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    args = parser.parse_args()

    try:
        return 0 if run_validation(args.area) else 1
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
