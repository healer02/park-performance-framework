"""Prepare or validate DA-level supply-experience divergence classes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from project_config import (
    ConfigError,
    available_areas,
    inherited_temp_directory,
    load_config,
    output_root,
    workspace_data_root,
)


PRIMARY_CLASSES = ["HH", "HL", "LH", "LL"]
THRESHOLD_ABSOLUTE_TOLERANCE = 1e-12
ALL_CLASSES = PRIMARY_CLASSES + ["insufficient_experience", "insufficient_population"]
CLASS_LABELS = {
    "HH": "High supply / high experience",
    "HL": "High supply / low experience",
    "LH": "Low supply / high experience",
    "LL": "Low supply / low experience",
    "insufficient_experience": "Insufficient experience data",
    "insufficient_population": "Insufficient population",
}


def divergence_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    root = data_root or workspace_data_root(config)
    base = root / "interim" / "divergence" / slug
    return {
        "da_divergence_csv": base / f"{slug}_da_divergence.csv",
        "da_divergence_gpkg": base / f"{slug}_da_divergence.gpkg",
    }


def experience_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = workspace_data_root(config) / "interim" / "experience" / slug
    return {
        "da_experience_csv": base / f"{slug}_da_experience.csv",
        "da_experience_gpkg": base / f"{slug}_da_experience.gpkg",
    }


def _read_csv(path: Path, required: set[str], dtype: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=dtype)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    return frame


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        examples = frame.loc[frame.duplicated(columns, keep=False), columns].head()
        raise ValueError(
            f"{label} contains duplicate keys {columns}: {examples.to_dict('records')}"
        )


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    text = series.astype("string").str.strip().str.casefold()
    invalid = text.notna() & ~text.isin(["true", "false"])
    if invalid.any():
        values = sorted(text[invalid].unique().tolist())
        raise ValueError(f"{label} contains invalid Boolean values: {values[:5]}")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[text.eq("true").fillna(False)] = True
    result.loc[text.eq("false").fillna(False)] = False
    return result


def _single_threshold(frame: pd.DataFrame, column: str, label: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{label} must contain one finite threshold; found {values}")
    return float(values[0])


def _median_threshold(frame: pd.DataFrame, column: str, label: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Cannot calculate {label}: no usable values")
    threshold = float(values.median())
    if not np.isfinite(threshold):
        raise ValueError(f"Cannot calculate {label}: median is not finite")
    return threshold


def _classify(
    population: pd.Series,
    high_supply: pd.Series,
    experience: pd.Series,
    experience_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    population = pd.to_numeric(population, errors="raise")
    experience = pd.to_numeric(experience, errors="coerce")
    valid_population = population.gt(0)
    valid_experience = valid_population & experience.notna()
    if high_supply.loc[valid_experience].isna().any():
        raise ValueError("A populated DA with experience is missing its supply class")

    # Values that differ from the stored median only by floating-point
    # representation are ties and must follow the documented "at or above"
    # rule. Aggregated 4.3-star ratings, for example, may be represented as
    # either 4.3 or 4.300000000000001.
    median_ties = pd.Series(
        np.isclose(
            experience,
            experience_threshold,
            rtol=0.0,
            atol=THRESHOLD_ABSOLUTE_TOLERANCE,
            equal_nan=False,
        ),
        index=experience.index,
    )
    high_experience = pd.Series(pd.NA, index=experience.index, dtype="boolean")
    high_experience.loc[valid_experience] = (
        experience.loc[valid_experience].gt(experience_threshold)
        | median_ties.loc[valid_experience]
    )

    classes = pd.Series("insufficient_experience", index=experience.index, dtype="string")
    classes.loc[~valid_population] = "insufficient_population"
    high_supply_mask = high_supply.fillna(False).astype(bool)
    high_experience_mask = high_experience.fillna(False).astype(bool)
    classes.loc[valid_experience & high_supply_mask & high_experience_mask] = "HH"
    classes.loc[valid_experience & high_supply_mask & ~high_experience_mask] = "HL"
    classes.loc[valid_experience & ~high_supply_mask & high_experience_mask] = "LH"
    classes.loc[valid_experience & ~high_supply_mask & ~high_experience_mask] = "LL"
    return high_experience, classes


def _classification_changed(primary: pd.Series, sensitivity: pd.Series) -> pd.Series:
    comparable = primary.isin(PRIMARY_CLASSES) & sensitivity.isin(PRIMARY_CLASSES)
    changed = pd.Series(pd.NA, index=primary.index, dtype="boolean")
    changed.loc[comparable] = primary.loc[comparable] != sensitivity.loc[comparable]
    return changed


def build_divergence_frames(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict[str, float | None]]:
    fields = config["fields"]
    da_id = fields["da_id"]
    source_paths = experience_paths(config)
    required = {
        da_id,
        "da_pop",
        "park_access_pop_coverage",
        "supply_ha_per_1000_cap20",
        "supply_median_cap20",
        "supply_median_cap10",
        "supply_median_uncapped",
        "high_supply",
        "supply_type",
        "high_supply_cap10_sensitivity",
        "high_supply_uncapped_sensitivity",
        "high_supply_coverage50_sensitivity",
        "eligible_google_entity_count",
        "experience_rating_mean",
        "experience_rating_review_weighted_sensitivity",
        "experience_rating_mean_population_weighted_sensitivity",
        "has_usable_experience_rating",
        "experience_sentiment_mean",
        "sentiment_enabled",
    }
    frame = _read_csv(
        source_paths["da_experience_csv"], required, {da_id: "string"}
    )
    _assert_unique(frame, [da_id], "Stage 06 DA experience")
    frame["da_pop"] = pd.to_numeric(frame["da_pop"], errors="raise").astype("int64")
    if (frame["da_pop"] < 0).any():
        raise ValueError("DA population cannot be negative")
    for column in [
        "high_supply",
        "high_supply_cap10_sensitivity",
        "high_supply_uncapped_sensitivity",
        "high_supply_coverage50_sensitivity",
        "has_usable_experience_rating",
        "sentiment_enabled",
    ]:
        frame[column] = _bool_series(frame[column], column)

    if config.get("experience", {}).get("primary_source") != "rating":
        raise ValueError("Stage 07 currently requires experience.primary_source: rating")
    if config["analysis"].get("experience_threshold") != "analysis_area_median":
        raise ValueError(
            "Stage 07 requires analysis.experience_threshold: analysis_area_median"
        )
    if config["analysis"].get("supply_area_threshold") != "analysis_area_median":
        raise ValueError(
            "Stage 07 requires analysis.supply_area_threshold: analysis_area_median"
        )

    usable = frame["has_usable_experience_rating"].fillna(False).astype(bool)
    rating = pd.to_numeric(frame["experience_rating_mean"], errors="coerce")
    expected_usable = frame["da_pop"].gt(0) & rating.notna()
    if not np.array_equal(
        usable.to_numpy(dtype=bool), expected_usable.to_numpy(dtype=bool)
    ):
        raise ValueError(
            "Stage 06 usable-rating flags disagree with population and rating values"
        )
    if rating.dropna().between(1, 5).eq(False).any():
        raise ValueError("DA mean Google ratings must lie between 1 and 5")

    supply_threshold = _single_threshold(
        frame.loc[frame["da_pop"] > 0],
        "supply_median_cap20",
        "Stage 05 primary supply median",
    )
    experience_threshold = _median_threshold(
        frame.loc[usable], "experience_rating_mean", "primary experience threshold"
    )
    expected_high_supply = (
        pd.to_numeric(frame["park_access_pop_coverage"], errors="coerce").ge(
            float(config["analysis"]["high_coverage_threshold"])
        )
        & pd.to_numeric(frame["supply_ha_per_1000_cap20"], errors="coerce").ge(
            supply_threshold
        )
    )
    populated = frame["da_pop"].gt(0)
    if not np.array_equal(
        frame.loc[populated, "high_supply"].to_numpy(dtype=bool),
        expected_high_supply.loc[populated].to_numpy(dtype=bool),
    ):
        raise ValueError(
            "Stage 05 high-supply flags disagree with the configured joint rule"
        )
    frame["high_experience_rating"], frame["divergence_class"] = _classify(
        frame["da_pop"], frame["high_supply"], rating, experience_threshold
    )
    frame["divergence_label"] = frame["divergence_class"].map(CLASS_LABELS)
    frame["supply_level"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    valid_population = frame["da_pop"].gt(0)
    frame.loc[valid_population, "supply_level"] = np.where(
        frame.loc[valid_population, "high_supply"].fillna(False), "High", "Low"
    )
    frame["experience_level"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[usable, "experience_level"] = np.where(
        frame.loc[usable, "high_experience_rating"].fillna(False), "High", "Low"
    )
    frame["supply_area_threshold"] = supply_threshold
    frame["experience_rating_threshold"] = experience_threshold
    frame["supply_threshold_method"] = (
        "80pct_population_coverage_and_cap20_area_at_or_above_analysis_area_median"
    )
    frame["experience_threshold_method"] = (
        "analysis_area_median_of_DA_mean_unique_eligible_Google_entities"
    )
    frame["threshold_scope"] = config["analysis_area"]["slug"]

    sensitivity_specs = [
        (
            "review_weighted",
            "high_supply",
            "experience_rating_review_weighted_sensitivity",
        ),
        (
            "population_weighted",
            "high_supply",
            "experience_rating_mean_population_weighted_sensitivity",
        ),
        (
            "cap10_supply",
            "high_supply_cap10_sensitivity",
            "experience_rating_mean",
        ),
        (
            "uncapped_supply",
            "high_supply_uncapped_sensitivity",
            "experience_rating_mean",
        ),
        (
            "coverage50_supply",
            "high_supply_coverage50_sensitivity",
            "experience_rating_mean",
        ),
    ]
    thresholds: dict[str, float | None] = {
        "primary_supply": supply_threshold,
        "primary_experience": experience_threshold,
    }
    for name, supply_column, experience_column in sensitivity_specs:
        sensitivity_experience = pd.to_numeric(
            frame[experience_column], errors="coerce"
        )
        if experience_column == "experience_rating_mean":
            threshold = experience_threshold
        else:
            threshold = _median_threshold(
                frame.loc[frame["da_pop"].gt(0) & sensitivity_experience.notna()],
                experience_column,
                f"{name} experience threshold",
            )
        high_column = f"high_experience_{name}_sensitivity"
        class_column = f"divergence_{name}_sensitivity"
        change_column = f"divergence_changed_{name}_sensitivity"
        frame[high_column], frame[class_column] = _classify(
            frame["da_pop"],
            frame[supply_column],
            sensitivity_experience,
            threshold,
        )
        frame[change_column] = _classification_changed(
            frame["divergence_class"], frame[class_column]
        )
        thresholds[name] = threshold

    sentiment_enabled = frame["sentiment_enabled"].fillna(False).astype(bool)
    configured_sentiment = bool(
        config.get("experience", {}).get("sentiment", {}).get("enabled", False)
    )
    if not sentiment_enabled.eq(configured_sentiment).all():
        raise ValueError("Stage 06 sentiment status differs from the current configuration")
    frame["high_experience_sentiment_sensitivity"] = pd.Series(
        pd.NA, index=frame.index, dtype="boolean"
    )
    frame["divergence_sentiment_sensitivity"] = pd.Series(
        pd.NA, index=frame.index, dtype="string"
    )
    frame["divergence_changed_sentiment_sensitivity"] = pd.Series(
        pd.NA, index=frame.index, dtype="boolean"
    )
    frame["experience_sentiment_threshold"] = np.nan
    sentiment_threshold: float | None = None
    if configured_sentiment:
        sentiment = pd.to_numeric(frame["experience_sentiment_mean"], errors="coerce")
        sentiment_threshold = _median_threshold(
            frame.loc[frame["da_pop"].gt(0) & sentiment.notna()],
            "experience_sentiment_mean",
            "sentiment experience threshold",
        )
        high_sentiment, sentiment_class = _classify(
            frame["da_pop"], frame["high_supply"], sentiment, sentiment_threshold
        )
        frame["high_experience_sentiment_sensitivity"] = high_sentiment
        frame["divergence_sentiment_sensitivity"] = sentiment_class
        frame["divergence_changed_sentiment_sensitivity"] = _classification_changed(
            frame["divergence_class"], sentiment_class
        )
        frame["experience_sentiment_threshold"] = sentiment_threshold
    thresholds["sentiment"] = sentiment_threshold

    geometry = pyogrio.read_dataframe(
        source_paths["da_experience_gpkg"],
        layer="da_experience",
        columns=[da_id, "csd_id"],
    )
    geometry[da_id] = geometry[da_id].astype("string")
    geometry["csd_id"] = geometry["csd_id"].astype("string")
    _assert_unique(geometry, [da_id], "Stage 06 DA experience GeoPackage")
    if set(geometry[da_id]) != set(frame[da_id]):
        raise ValueError("Stage 06 DA experience CSV and GeoPackage IDs differ")
    city_names: dict[str, str] = {}
    for area in available_areas():
        if area == "metro":
            continue
        city_config = load_config(area)
        for code in city_config["analysis_area"]["analysis_csd_codes"]:
            city_names[str(code)] = city_config["analysis_area"]["name"]
    attributes = geometry[[da_id, "csd_id"]].copy()
    attributes["municipality"] = attributes["csd_id"].map(city_names)
    if attributes["municipality"].isna().any():
        unknown = sorted(attributes.loc[attributes["municipality"].isna(), "csd_id"].unique())
        raise ValueError(f"No configured municipality name for CSD IDs: {unknown}")
    frame = frame.merge(attributes, on=da_id, how="left", validate="one_to_one")
    frame = frame.sort_values(da_id).reset_index(drop=True)
    mapped = geometry.drop(columns="csd_id").merge(
        frame, on=da_id, how="left", validate="one_to_one"
    )
    return frame, mapped, thresholds


def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>")


def _compare_numeric(observed: pd.Series, expected: pd.Series, label: str) -> None:
    if not np.allclose(
        pd.to_numeric(observed, errors="coerce"),
        pd.to_numeric(expected, errors="coerce"),
        equal_nan=True,
        atol=1e-9,
    ):
        raise ValueError(f"Stored Stage 07 values are inconsistent: {label}")


def validate_divergence(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 07 artifacts: " + ", ".join(path.name for path in missing)
        )
    da_id = config["fields"]["da_id"]
    required = {
        da_id,
        "csd_id",
        "municipality",
        "da_pop",
        "high_supply",
        "experience_rating_mean",
        "high_experience_rating",
        "divergence_class",
        "divergence_label",
        "supply_area_threshold",
        "experience_rating_threshold",
        "experience_sentiment_threshold",
        "supply_median_cap10",
        "supply_median_uncapped",
        "divergence_review_weighted_sensitivity",
        "divergence_population_weighted_sensitivity",
        "divergence_cap10_supply_sensitivity",
        "divergence_uncapped_supply_sensitivity",
        "divergence_coverage50_supply_sensitivity",
        "divergence_sentiment_sensitivity",
    }
    observed = _read_csv(
        paths["da_divergence_csv"], required, {da_id: "string"}
    )
    _assert_unique(observed, [da_id], "DA divergence table")
    expected, _, thresholds = build_divergence_frames(config)
    check = expected[[da_id]].merge(
        observed[[da_id]],
        on=da_id,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not check["_merge"].eq("both").all():
        raise ValueError("Stage 07 DA IDs differ from the current Stage 06 data")
    expected = expected.set_index(da_id).loc[observed[da_id]].reset_index()

    text_columns = [
        "divergence_class",
        "divergence_label",
        "supply_level",
        "experience_level",
        "supply_threshold_method",
        "experience_threshold_method",
        "threshold_scope",
        "divergence_review_weighted_sensitivity",
        "divergence_population_weighted_sensitivity",
        "divergence_cap10_supply_sensitivity",
        "divergence_uncapped_supply_sensitivity",
        "divergence_coverage50_supply_sensitivity",
        "divergence_sentiment_sensitivity",
    ]
    for column in text_columns:
        if not _normalise_text(observed[column]).equals(_normalise_text(expected[column])):
            raise ValueError(f"Stored Stage 07 values are inconsistent: {column}")
    for column in [
        "supply_area_threshold",
        "experience_rating_threshold",
        "experience_sentiment_threshold",
    ]:
        _compare_numeric(observed[column], expected[column], column)

    observed_classes = set(observed["divergence_class"].dropna())
    invalid_classes = observed_classes - set(ALL_CLASSES)
    if invalid_classes:
        raise ValueError(f"Invalid primary divergence classes: {sorted(invalid_classes)}")
    high_supply = _bool_series(observed["high_supply"], "high_supply").fillna(False)
    high_experience = _bool_series(
        observed["high_experience_rating"], "high_experience_rating"
    ).fillna(False)
    if not high_supply.loc[observed["divergence_class"].isin(["HH", "HL"])].all():
        raise ValueError("A high-supply divergence class contains a low-supply DA")
    if high_supply.loc[observed["divergence_class"].isin(["LH", "LL"])].any():
        raise ValueError("A low-supply divergence class contains a high-supply DA")
    if not high_experience.loc[observed["divergence_class"].isin(["HH", "LH"])].all():
        raise ValueError("A high-experience class contains a low-experience DA")
    if high_experience.loc[observed["divergence_class"].isin(["HL", "LL"])].any():
        raise ValueError("A low-experience class contains a high-experience DA")
    missing_experience = pd.to_numeric(
        observed["experience_rating_mean"], errors="coerce"
    ).isna()
    positive_population = pd.to_numeric(observed["da_pop"], errors="raise").gt(0)
    if not observed.loc[
        missing_experience & positive_population, "divergence_class"
    ].eq("insufficient_experience").all():
        raise ValueError("Missing experience was incorrectly classified as low")
    if not observed.loc[
        ~positive_population, "divergence_class"
    ].eq("insufficient_population").all():
        raise ValueError("A zero-population DA received an analytical class")

    mapped = gpd.read_file(paths["da_divergence_gpkg"], layer="da_divergence")
    mapped[da_id] = mapped[da_id].astype("string")
    _assert_unique(mapped, [da_id], "DA divergence GeoPackage")
    if set(mapped[da_id]) != set(observed[da_id]):
        raise ValueError("DA divergence GeoPackage IDs differ from the CSV")
    if str(mapped.crs) != config["crs"]["projected"]:
        raise ValueError(
            f"DA divergence CRS is {mapped.crs}; expected {config['crs']['projected']}"
        )

    class_counts = observed["divergence_class"].value_counts()
    metrics: dict[str, Any] = {
        "da_count": len(observed),
        "classified_da_count": int(observed["divergence_class"].isin(PRIMARY_CLASSES).sum()),
        "insufficient_experience_da_count": int(
            class_counts.get("insufficient_experience", 0)
        ),
        "insufficient_population_da_count": int(
            class_counts.get("insufficient_population", 0)
        ),
        "high_supply_da_count": int(
            _bool_series(observed["high_supply"], "high_supply").fillna(False).sum()
        ),
        "high_experience_da_count": int(
            _bool_series(
                observed["high_experience_rating"], "high_experience_rating"
            )
            .fillna(False)
            .sum()
        ),
        "supply_area_threshold": float(thresholds["primary_supply"]),
        "supply_area_threshold_cap10": _single_threshold(
            observed.loc[pd.to_numeric(observed["da_pop"]) > 0],
            "supply_median_cap10",
            "10 ha supply median",
        ),
        "supply_area_threshold_uncapped": _single_threshold(
            observed.loc[pd.to_numeric(observed["da_pop"]) > 0],
            "supply_median_uncapped",
            "uncapped supply median",
        ),
        "primary_coverage_threshold": float(
            config["analysis"]["high_coverage_threshold"]
        ),
        "sensitivity_coverage_threshold": float(
            config["analysis"]["coverage_sensitivity_threshold"]
        ),
        "experience_rating_threshold": float(thresholds["primary_experience"]),
        "review_weighted_experience_threshold": float(thresholds["review_weighted"]),
        "population_weighted_experience_threshold": float(
            thresholds["population_weighted"]
        ),
        "sentiment_experience_threshold": thresholds["sentiment"],
        "sentiment_classified_da_count": int(
            observed["divergence_sentiment_sensitivity"]
            .isin(PRIMARY_CLASSES)
            .sum()
        ),
        "crs": config["crs"]["projected"],
    }
    for class_name in ALL_CLASSES:
        metrics[f"{class_name}_da_count"] = int(class_counts.get(class_name, 0))
    for name in [
        "review_weighted",
        "population_weighted",
        "cap10_supply",
        "uncapped_supply",
        "coverage50_supply",
        "sentiment",
    ]:
        change_column = f"divergence_changed_{name}_sensitivity"
        changed = _bool_series(observed[change_column], change_column)
        metrics[f"{name}_classification_changes"] = int(changed.fillna(False).sum())
    return metrics


def build_divergence(
    config: dict[str, Any], final_paths: dict[str, Path]
) -> dict[str, Any]:
    frame, mapped, _ = build_divergence_frames(config)
    slug = config["analysis_area"]["slug"]
    final_paths["da_divergence_csv"].parent.mkdir(parents=True, exist_ok=True)
    temp_parent = final_paths["da_divergence_csv"].parent.parent
    with inherited_temp_directory(temp_parent, f"{slug}_07_") as temp_dir:
        temp_paths = divergence_paths(config, temp_dir)
        temp_paths["da_divergence_csv"].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(temp_paths["da_divergence_csv"], index=False)
        mapped.to_file(
            temp_paths["da_divergence_gpkg"], layer="da_divergence", driver="GPKG"
        )
        metrics = validate_divergence(config, temp_paths)
        for key, target in final_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_paths[key], target)
    return metrics


def write_summaries(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    status: str,
) -> tuple[Path, Path]:
    slug = config["analysis_area"]["slug"]
    table_dir = output_root(config) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = table_dir / f"{slug}_07_divergence_thresholds.csv"
    summary_path = table_dir / f"{slug}_07_divergence_summary.csv"
    pd.DataFrame(
        [
            {
                "analysis_area": slug,
                "status": status,
                "primary_supply_definition": (
                    "80pct coverage and cap20 area at/above analysis-area median"
                ),
                "primary_experience_definition": (
                    "DA mean rating at/above analysis-area median"
                ),
                **metrics,
                **{f"{key}_path": str(value) for key, value in paths.items()},
            }
        ]
    ).to_csv(thresholds_path, index=False)

    divergence = pd.read_csv(
        paths["da_divergence_csv"], dtype={config["fields"]["da_id"]: "string"}
    )
    rows = []
    total_population = pd.to_numeric(divergence["da_pop"], errors="raise").sum()
    for class_name in ALL_CLASSES:
        subset = divergence.loc[divergence["divergence_class"].eq(class_name)]
        population = pd.to_numeric(subset["da_pop"], errors="raise").sum()
        rows.append(
            {
                "analysis_area": slug,
                "status": status,
                "divergence_class": class_name,
                "divergence_label": CLASS_LABELS[class_name],
                "da_count": len(subset),
                "da_share": len(subset) / len(divergence) if len(divergence) else np.nan,
                "population": int(population),
                "population_share": (
                    population / total_population if total_population > 0 else np.nan
                ),
                "mean_supply_ha_per_1000_cap20": pd.to_numeric(
                    subset["supply_ha_per_1000_cap20"], errors="coerce"
                ).mean(),
                "mean_experience_rating": pd.to_numeric(
                    subset["experience_rating_mean"], errors="coerce"
                ).mean(),
            }
        )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return thresholds_path, summary_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = divergence_paths(config)
    slug = config["analysis_area"]["slug"]
    print(f"\nStage 07: supply-experience divergence ({slug})")
    print(
        "Primary high supply: >=80% population coverage and cap-20 area at or "
        "above the analysis-area median."
    )
    print(
        "Primary high experience: mean rating at or above the analysis-area "
        "median; missing ratings remain unclassified."
    )

    if force:
        print("Force enabled: building new artifacts atomically in this repository.")
        metrics = build_divergence(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing artifacts.")
        try:
            metrics = validate_divergence(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Divergence validation failed: {error}")
            print("Build this area's Stage 07 artifacts explicitly with --force.")
            return 1
        status = "reused"

    metrics = validate_divergence(config, paths) | metrics
    thresholds_path, summary_path = write_summaries(
        config, paths, metrics, status
    )
    print(f"Validated DAs: {metrics['da_count']:,}")
    print(f"Classified DAs: {metrics['classified_da_count']:,}")
    print(
        "Primary thresholds: "
        f"{metrics['supply_area_threshold']:.3f} ha/1,000 and "
        f"{metrics['experience_rating_threshold']:.3f} stars"
    )
    print(
        "Classes: "
        + ", ".join(
            f"{name}={metrics[f'{name}_da_count']:,}" for name in PRIMARY_CLASSES
        )
    )
    print(f"Thresholds: {thresholds_path}")
    print(f"Class summary: {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this area's generated Stage 07 artifacts",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
