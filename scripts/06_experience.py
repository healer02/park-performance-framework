"""Prepare or validate DB- and DA-level park-experience measures."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    output_root,
    workspace_data_root,
)


def experience_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    root = data_root or workspace_data_root(config)
    base = root / "interim" / "experience" / slug
    return {
        "db_google_entities_csv": base / f"{slug}_db_google_entities.csv",
        "da_google_entities_csv": base / f"{slug}_da_google_entities.csv",
        "db_experience_csv": base / f"{slug}_db_experience.csv",
        "da_experience_csv": base / f"{slug}_da_experience.csv",
        "da_experience_gpkg": base / f"{slug}_da_experience.gpkg",
    }


def reachability_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = workspace_data_root(config) / "interim" / "reach" / slug
    return {
        "pairs": base / f"{slug}_db_park_reachability.csv",
        "db_summary": base / f"{slug}_db_reachability_summary.csv",
    }


def supply_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = workspace_data_root(config) / "interim" / "supply" / slug
    return {
        "crosswalk": base / f"{slug}_park_entity_crosswalk.csv",
        "db_supply": base / f"{slug}_db_supply.csv",
        "da_supply": base / f"{slug}_da_supply.csv",
    }


def da_boundaries_path(config: dict[str, Any]) -> Path:
    slug = config["analysis_area"]["slug"]
    return (
        workspace_data_root(config)
        / "interim"
        / "census"
        / slug
        / f"{slug}_da_boundaries.gpkg"
    )


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
    invalid = ~text.isin(["true", "false"])
    if invalid.any():
        values = sorted(text[invalid].dropna().unique().tolist())
        raise ValueError(f"{label} contains invalid Boolean values: {values[:5]}")
    return text.eq("true")


def _first_nonblank(values: pd.Series) -> str | pd.NA:
    text = values.astype("string").str.strip().replace("", pd.NA).dropna()
    return text.iloc[0] if not text.empty else pd.NA


def _load_google_ratings(
    config: dict[str, Any], expected_park_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = data_path(config, "google_ratings_validated")
    required = {
        "park_id",
        "park_name",
        "municipality",
        "match_validated",
        "PlaceID",
        "google_name",
        "google_rating",
        "google_review_count",
        "rating_eligible",
    }
    ratings = _read_csv(
        path,
        required,
        {"park_id": "string", "PlaceID": "string", "google_name": "string"},
    )
    ratings["park_id"] = ratings["park_id"].str.strip()
    _assert_unique(ratings, ["park_id"], "Validated Google rating table")
    observed_park_ids = set(ratings["park_id"])
    if observed_park_ids != expected_park_ids:
        missing = sorted(expected_park_ids - observed_park_ids)
        extra = sorted(observed_park_ids - expected_park_ids)
        raise ValueError(
            "Validated Google ratings do not match the current park inventory "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    ratings["PlaceID"] = ratings["PlaceID"].str.strip().replace("", pd.NA)
    ratings["match_validated"] = _bool_series(
        ratings["match_validated"], "match_validated"
    )
    ratings["rating_eligible"] = _bool_series(
        ratings["rating_eligible"], "rating_eligible"
    )
    ratings["google_rating"] = pd.to_numeric(
        ratings["google_rating"], errors="coerce"
    )
    ratings["google_review_count"] = pd.to_numeric(
        ratings["google_review_count"], errors="coerce"
    )
    invalid_reviews = ratings["google_review_count"].notna() & (
        (ratings["google_review_count"] < 0)
        | ~np.isclose(ratings["google_review_count"] % 1, 0)
    )
    if invalid_reviews.any():
        raise ValueError("Google review counts must be non-negative whole numbers")

    minimum_reviews = int(config["analysis"]["minimum_google_reviews"])
    expected_eligibility = (
        ratings["match_validated"]
        & ratings["PlaceID"].notna()
        & ratings["google_rating"].between(1, 5)
        & ratings["google_review_count"].ge(minimum_reviews)
    )
    if not ratings["rating_eligible"].equals(expected_eligibility):
        changed = ratings.loc[
            ratings["rating_eligible"] != expected_eligibility,
            ["park_id", "PlaceID", "rating_eligible"],
        ]
        raise ValueError(
            "Stored rating_eligible values disagree with the configured validation rule: "
            f"{changed.head().to_dict('records')}"
        )
    invalid_validated = ratings["match_validated"] & ratings["PlaceID"].isna()
    if invalid_validated.any():
        raise ValueError("A validated Google match cannot have a blank PlaceID")

    validated = ratings.loc[ratings["match_validated"]].copy()
    for column in ["google_rating", "google_review_count"]:
        conflicts = validated.groupby("PlaceID")[column].nunique(dropna=True)
        if conflicts.gt(1).any():
            raise ValueError(
                f"Validated rows sharing a PlaceID disagree on {column}: "
                f"{conflicts[conflicts.gt(1)].index.tolist()[:5]}"
            )
    eligible_conflicts = validated.groupby("PlaceID")["rating_eligible"].nunique()
    if eligible_conflicts.gt(1).any():
        raise ValueError(
            "Validated rows sharing a PlaceID disagree on rating eligibility"
        )

    google_entities = (
        validated.groupby("PlaceID", as_index=False, dropna=False)
        .agg(
            google_name=("google_name", _first_nonblank),
            google_rating=("google_rating", "first"),
            google_review_count=("google_review_count", "first"),
            rating_eligible=("rating_eligible", "first"),
            matched_park_count=("park_id", "nunique"),
        )
        .sort_values("PlaceID")
        .reset_index(drop=True)
    )
    google_entities["rating_eligible"] = google_entities["rating_eligible"].astype(bool)
    return ratings, google_entities


def _weighted_means(
    frame: pd.DataFrame, group: str, value: str, weight: str, output: str
) -> pd.DataFrame:
    working = frame[[group, value, weight]].copy()
    working[value] = pd.to_numeric(working[value], errors="coerce")
    working[weight] = pd.to_numeric(working[weight], errors="coerce")
    working = working[
        working[value].notna() & working[weight].notna() & working[weight].gt(0)
    ]
    if working.empty:
        return pd.DataFrame(columns=[group, output])
    working["weighted_value"] = working[value] * working[weight]
    totals = working.groupby(group, as_index=False).agg(
        weighted_value=("weighted_value", "sum"), total_weight=(weight, "sum")
    )
    totals[output] = totals["weighted_value"] / totals["total_weight"]
    return totals[[group, output]]


def _rating_aggregates(
    entity_rows: pd.DataFrame, group: str
) -> pd.DataFrame:
    eligible = entity_rows.loc[entity_rows["rating_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                group,
                "eligible_google_entity_count",
                "experience_rating_mean",
                "experience_rating_median_sensitivity",
                "eligible_google_review_count",
                "experience_rating_review_weighted_sensitivity",
            ]
        )
    summary = eligible.groupby(group, as_index=False).agg(
        eligible_google_entity_count=("PlaceID", "nunique"),
        experience_rating_mean=("google_rating", "mean"),
        experience_rating_median_sensitivity=("google_rating", "median"),
        eligible_google_review_count=("google_review_count", "sum"),
    )
    weighted = _weighted_means(
        eligible,
        group,
        "google_rating",
        "google_review_count",
        "experience_rating_review_weighted_sensitivity",
    )
    return summary.merge(weighted, on=group, how="left", validate="one_to_one")


def _sentiment_path(config: dict[str, Any]) -> Path | None:
    sentiment = config.get("experience", {}).get("sentiment", {})
    if not sentiment.get("enabled", False):
        return None
    value = sentiment.get("path")
    if not value:
        raise ValueError("Sentiment is enabled but experience.sentiment.path is blank")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    workspace = workspace_data_root(config) / path
    if workspace.exists():
        return workspace.resolve()
    return (Path(config["runtime"]["data_root"]) / path).resolve()


def _sentiment_setting_path(config: dict[str, Any], key: str) -> Path:
    settings = config["experience"]["sentiment"]
    value = settings.get(key)
    if not value:
        raise ValueError(f"Sentiment is enabled but experience.sentiment.{key} is blank")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    workspace = workspace_data_root(config) / path
    if workspace.exists():
        return workspace.resolve()
    return (Path(config["runtime"]["data_root"]) / path).resolve()


def _make_slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _sentiment_summaries(
    config: dict[str, Any],
    pairs: pd.DataFrame,
    populated_db_ids: set[str],
    ratings: pd.DataFrame,
    park_crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    settings = config.get("experience", {}).get("sentiment", {})
    if not settings.get("enabled", False):
        return (
            pd.DataFrame(
                columns=[db_id, "sentiment_reachable_park_count", "experience_sentiment_mean"]
            ),
            pd.DataFrame(
                columns=[da_id, "sentiment_reachable_park_count", "experience_sentiment_mean"]
            ),
            False,
        )
    linkage = settings.get("primary_linkage", "place_id")
    path = (
        _sentiment_setting_path(config, "park_metrics_path")
        if linkage == "park_metrics"
        else _sentiment_path(config)
    )
    if path is None:
        raise ValueError("Enabled sentiment configuration did not resolve an input path")
    if not path.exists():
        raise FileNotFoundError(f"Enabled sentiment input does not exist: {path}")

    score_field = settings.get("score_field", "AvgSentiment")
    eligibility_field = settings.get("eligibility_field")
    if linkage == "park_metrics":
        legacy_id_field = settings.get("park_metrics_id_field", "park_id")
        park_name_field = settings.get("park_metrics_name_field", "park_name")
        required = {legacy_id_field, park_name_field, score_field}
        if eligibility_field:
            required.add(eligibility_field)
        sentiment = _read_csv(path, required, {legacy_id_field: "string"})
        sentiment = sentiment.rename(
            columns={
                legacy_id_field: "sentiment_park_id",
                park_name_field: "sentiment_park_name",
                score_field: "sentiment_score",
            }
        )
        sentiment["sentiment_park_id"] = (
            sentiment["sentiment_park_id"].str.strip().replace("", pd.NA)
        )
        sentiment["sentiment_score"] = pd.to_numeric(
            sentiment["sentiment_score"], errors="coerce"
        )
        if eligibility_field:
            sentiment = sentiment.loc[
                _bool_series(sentiment[eligibility_field], eligibility_field)
            ].copy()
        sentiment = sentiment.dropna(subset=["sentiment_park_id", "sentiment_score"])
        _assert_unique(sentiment, ["sentiment_park_id"], "Park-level sentiment table")

        source_lookup = settings.get("legacy_source_municipalities", {})
        sentiment["legacy_source"] = sentiment["sentiment_park_id"].str.rsplit(
            "_", n=1
        ).str[0]
        sentiment["municipality"] = sentiment["legacy_source"].map(source_lookup)
        if sentiment["municipality"].isna().any():
            missing_sources = sorted(
                sentiment.loc[sentiment["municipality"].isna(), "legacy_source"].unique()
            )
            raise ValueError(
                "Park-level sentiment contains unmapped legacy sources: "
                f"{missing_sources}"
            )
        sentiment["park_key"] = (
            sentiment["municipality"].map(_make_slug)
            + "__"
            + sentiment["sentiment_park_name"].map(_make_slug)
        )
        key_overrides = settings.get("park_key_overrides", {})
        if key_overrides:
            sentiment["park_key"] = sentiment["sentiment_park_id"].map(
                key_overrides
            ).fillna(sentiment["park_key"])

        current_keys = park_crosswalk[["park_id", "park_key"]].copy()
        _assert_unique(current_keys, ["park_key"], "Stage 05 park-key crosswalk")
        sentiment = sentiment.merge(
            current_keys, on="park_key", how="left", validate="one_to_one"
        )
        if sentiment["park_id"].isna().any():
            missing_parks = sentiment.loc[
                sentiment["park_id"].isna(),
                ["sentiment_park_id", "sentiment_park_name", "park_key"],
            ].to_dict("records")
            print(
                "Valid park-level sentiment records outside the final reachable "
                f"park inventory: {missing_parks[:10]}"
            )
            sentiment = sentiment.dropna(subset=["park_id"]).copy()
        if sentiment.empty:
            raise ValueError(
                "No valid park-level sentiment records map to the current park inventory"
            )
        _assert_unique(sentiment, ["park_id"], "Mapped park-level sentiment table")
        scored = pairs[[db_id, da_id, "park_id"]].merge(
            sentiment[["park_id", "sentiment_park_id", "sentiment_score"]],
            on="park_id",
            how="inner",
            validate="many_to_one",
        )
        sentiment_entity = "sentiment_park_id"
    elif linkage == "place_id":
        entity_field = settings.get("entity_id_field", "PlaceID")
        required = {entity_field, score_field}
        if eligibility_field:
            required.add(eligibility_field)
        sentiment = _read_csv(path, required, {entity_field: "string"})
        sentiment = sentiment.rename(
            columns={entity_field: "PlaceID", score_field: "sentiment_score"}
        )
        sentiment["PlaceID"] = sentiment["PlaceID"].str.strip().replace("", pd.NA)
        sentiment["sentiment_score"] = pd.to_numeric(
            sentiment["sentiment_score"], errors="coerce"
        )
        if eligibility_field:
            sentiment = sentiment.loc[
                _bool_series(sentiment[eligibility_field], eligibility_field)
            ].copy()
        sentiment = sentiment.dropna(subset=["PlaceID", "sentiment_score"])
        conflicts = sentiment.groupby("PlaceID")["sentiment_score"].nunique(dropna=True)
        if conflicts.gt(1).any():
            raise ValueError(
                f"Sentiment rows sharing a PlaceID disagree on {score_field}: "
                f"{conflicts[conflicts.gt(1)].index.tolist()[:5]}"
            )
        sentiment = sentiment.groupby("PlaceID", as_index=False).agg(
            sentiment_score=("sentiment_score", "first")
        )
        validated = ratings.loc[
            ratings["match_validated"] & ratings["PlaceID"].notna(),
            ["park_id", "PlaceID"],
        ]
        if not (set(sentiment["PlaceID"]) & set(validated["PlaceID"])):
            raise ValueError(
                "The enabled sentiment file has no PlaceIDs in the current validated "
                "Google matching table"
            )
        scored = pairs[[db_id, da_id, "park_id"]].merge(
            validated, on="park_id", how="inner", validate="many_to_one"
        ).merge(sentiment, on="PlaceID", how="inner", validate="many_to_one")
        sentiment_entity = "PlaceID"
    else:
        raise ValueError(f"Unsupported experience.sentiment.primary_linkage: {linkage}")

    if scored["sentiment_score"].dropna().between(-1, 1).eq(False).any():
        raise ValueError("Sentiment scores must lie between -1 and 1")
    db = scored.drop_duplicates([db_id, sentiment_entity]).groupby(db_id, as_index=False).agg(
        sentiment_reachable_park_count=(sentiment_entity, "nunique"),
        experience_sentiment_mean=("sentiment_score", "mean"),
    )
    da = (
        scored.loc[
            scored[db_id].isin(populated_db_ids),
            [da_id, sentiment_entity, "sentiment_score"],
        ]
        .drop_duplicates([da_id, sentiment_entity])
        .groupby(da_id, as_index=False)
        .agg(
            sentiment_reachable_park_count=(sentiment_entity, "nunique"),
            experience_sentiment_mean=("sentiment_score", "mean"),
        )
    )
    return db, da, True


def build_experience_frames(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    reach = reachability_paths(config)
    supply = supply_paths(config)

    crosswalk = _read_csv(
        supply["crosswalk"],
        {"park_id", "park_key", "park_entity_id"},
        {"park_id": "string", "park_key": "string", "park_entity_id": "string"},
    )
    _assert_unique(crosswalk, ["park_id"], "Stage 05 park-entity crosswalk")
    ratings, google_entities = _load_google_ratings(config, set(crosswalk["park_id"]))

    pairs = _read_csv(
        reach["pairs"],
        {
            db_id,
            da_id,
            "db_pop",
            "park_id",
            "park_name",
            "park_municipality",
            "network_dist_m",
        },
        {db_id: "string", da_id: "string", "park_id": "string"},
    )
    _assert_unique(pairs, [db_id, "park_id"], "Stage 04 DB-park reachability")
    pairs["db_pop"] = pd.to_numeric(pairs["db_pop"], errors="raise").astype("int64")
    pairs["network_dist_m"] = pd.to_numeric(
        pairs["network_dist_m"], errors="raise"
    )

    db_supply = _read_csv(
        supply["db_supply"],
        {db_id, da_id, "db_pop", "reachable_park_count", "reachable_component_park_count"},
        {db_id: "string", da_id: "string"},
    )
    da_supply = _read_csv(
        supply["da_supply"],
        {da_id, "da_pop", "unique_reachable_park_count"},
        {da_id: "string"},
    )
    _assert_unique(db_supply, [db_id], "Stage 05 DB supply")
    _assert_unique(da_supply, [da_id], "Stage 05 DA supply")
    db_supply["db_pop"] = pd.to_numeric(db_supply["db_pop"], errors="raise").astype("int64")
    da_supply["da_pop"] = pd.to_numeric(da_supply["da_pop"], errors="raise").astype("int64")

    component_google = ratings[
        ["park_id", "match_validated", "PlaceID", "rating_eligible"]
    ]
    matched_pairs = pairs.merge(
        component_google,
        on="park_id",
        how="left",
        validate="many_to_one",
    )
    matched_pairs = matched_pairs.loc[matched_pairs["match_validated"]].copy()
    if matched_pairs["PlaceID"].isna().any():
        raise ValueError("A reachable validated Google match has a blank PlaceID")

    db_mapping_counts = (
        matched_pairs.groupby([db_id, da_id, "PlaceID"], as_index=False)
        .agg(mapped_component_park_count=("park_id", "nunique"))
    )
    nearest = (
        matched_pairs.sort_values([db_id, "PlaceID", "network_dist_m", "park_id"])
        .drop_duplicates([db_id, "PlaceID"])
        .rename(
            columns={
                "park_id": "nearest_component_park_id",
                "park_name": "nearest_component_park_name",
                "park_municipality": "nearest_component_municipality",
            }
        )
    )
    db_google = nearest[
        [
            db_id,
            da_id,
            "db_pop",
            "PlaceID",
            "nearest_component_park_id",
            "nearest_component_park_name",
            "nearest_component_municipality",
            "network_dist_m",
        ]
    ].merge(
        db_mapping_counts,
        on=[db_id, da_id, "PlaceID"],
        how="left",
        validate="one_to_one",
    ).merge(
        google_entities,
        on="PlaceID",
        how="left",
        validate="many_to_one",
    )
    if db_google["rating_eligible"].isna().any():
        raise ValueError("A reachable PlaceID is missing from the Google entity table")
    db_google["rating_eligible"] = db_google["rating_eligible"].astype(bool)
    db_google = db_google.sort_values([db_id, "PlaceID"]).reset_index(drop=True)

    populated_db_ids = set(db_supply.loc[db_supply["db_pop"] > 0, db_id])
    populated_google = db_google.loc[db_google[db_id].isin(populated_db_ids)].copy()
    populated_matched_pairs = matched_pairs.loc[
        matched_pairs[db_id].isin(populated_db_ids)
    ]
    da_mapping_counts = (
        populated_matched_pairs.groupby([da_id, "PlaceID"], as_index=False)
        .agg(
            reachable_db_count=(db_id, "nunique"),
            mapped_component_park_count=("park_id", "nunique"),
        )
    )
    da_nearest = (
        populated_google.sort_values([da_id, "PlaceID", "network_dist_m", db_id])
        .drop_duplicates([da_id, "PlaceID"])
    )
    da_google = da_nearest[
        [
            da_id,
            "PlaceID",
            "google_name",
            "google_rating",
            "google_review_count",
            "rating_eligible",
            "matched_park_count",
            "nearest_component_park_id",
            "nearest_component_park_name",
            "nearest_component_municipality",
            "network_dist_m",
        ]
    ].merge(
        da_mapping_counts,
        on=[da_id, "PlaceID"],
        how="left",
        validate="one_to_one",
    )
    da_google = da_google.sort_values([da_id, "PlaceID"]).reset_index(drop=True)

    enriched_pairs = matched_pairs.merge(
        crosswalk[["park_id", "park_entity_id"]],
        on="park_id",
        how="left",
        validate="many_to_one",
    )
    eligible_physical = enriched_pairs.loc[enriched_pairs["rating_eligible"]]
    db_physical_coverage = (
        eligible_physical[[db_id, "park_entity_id"]]
        .drop_duplicates()
        .groupby(db_id, as_index=False)
        .agg(rating_covered_physical_park_count=("park_entity_id", "nunique"))
    )
    da_physical_coverage = (
        eligible_physical.loc[
            eligible_physical[db_id].isin(populated_db_ids), [da_id, "park_entity_id"]
        ]
        .drop_duplicates()
        .groupby(da_id, as_index=False)
        .agg(rating_covered_physical_park_count=("park_entity_id", "nunique"))
    )

    db_validated = db_google.groupby(db_id, as_index=False).agg(
        validated_google_entity_count=("PlaceID", "nunique"),
        validated_google_review_count=("google_review_count", "sum"),
    )
    db_rating = _rating_aggregates(db_google, db_id)
    db_experience = (
        db_supply.merge(db_validated, on=db_id, how="left", validate="one_to_one")
        .merge(db_rating, on=db_id, how="left", validate="one_to_one")
        .merge(db_physical_coverage, on=db_id, how="left", validate="one_to_one")
    )
    for column in [
        "validated_google_entity_count",
        "validated_google_review_count",
        "eligible_google_entity_count",
        "eligible_google_review_count",
        "rating_covered_physical_park_count",
    ]:
        db_experience[column] = db_experience[column].fillna(0).astype("int64")
    db_experience["rating_coverage_share"] = np.where(
        pd.to_numeric(db_experience["reachable_park_count"]) > 0,
        db_experience["rating_covered_physical_park_count"]
        / pd.to_numeric(db_experience["reachable_park_count"]),
        np.nan,
    )
    db_experience["has_usable_experience_rating"] = db_experience[
        "eligible_google_entity_count"
    ].gt(0)

    da_validated = da_google.groupby(da_id, as_index=False).agg(
        validated_google_entity_count=("PlaceID", "nunique"),
        validated_google_review_count=("google_review_count", "sum"),
    )
    da_rating = _rating_aggregates(da_google, da_id)
    db_population_rating = _weighted_means(
        db_experience.loc[db_experience["has_usable_experience_rating"]],
        da_id,
        "experience_rating_mean",
        "db_pop",
        "experience_rating_mean_population_weighted_sensitivity",
    )
    covered_population = (
        db_experience.loc[db_experience["has_usable_experience_rating"]]
        .groupby(da_id, as_index=False)
        .agg(experience_covered_population=("db_pop", "sum"))
    )
    da_experience = (
        da_supply.merge(da_validated, on=da_id, how="left", validate="one_to_one")
        .merge(da_rating, on=da_id, how="left", validate="one_to_one")
        .merge(db_population_rating, on=da_id, how="left", validate="one_to_one")
        .merge(covered_population, on=da_id, how="left", validate="one_to_one")
        .merge(da_physical_coverage, on=da_id, how="left", validate="one_to_one")
    )
    for column in [
        "validated_google_entity_count",
        "validated_google_review_count",
        "eligible_google_entity_count",
        "eligible_google_review_count",
        "experience_covered_population",
        "rating_covered_physical_park_count",
    ]:
        da_experience[column] = da_experience[column].fillna(0).astype("int64")
    da_experience["rating_coverage_share"] = np.where(
        da_experience["unique_reachable_park_count"] > 0,
        da_experience["rating_covered_physical_park_count"]
        / da_experience["unique_reachable_park_count"],
        np.nan,
    )
    da_experience["experience_population_coverage"] = np.where(
        da_experience["da_pop"] > 0,
        da_experience["experience_covered_population"] / da_experience["da_pop"],
        np.nan,
    )
    da_experience["digital_salience_reviews_per_1000"] = np.where(
        da_experience["da_pop"] > 0,
        da_experience["validated_google_review_count"] / da_experience["da_pop"] * 1000,
        np.nan,
    )
    da_experience["has_usable_experience_rating"] = da_experience[
        "eligible_google_entity_count"
    ].gt(0)

    db_sentiment, da_sentiment, sentiment_enabled = _sentiment_summaries(
        config, pairs, populated_db_ids, ratings, crosswalk
    )
    db_experience = db_experience.merge(
        db_sentiment, on=db_id, how="left", validate="one_to_one"
    )
    da_experience = da_experience.merge(
        da_sentiment, on=da_id, how="left", validate="one_to_one"
    )
    for frame in [db_experience, da_experience]:
        frame["sentiment_reachable_park_count"] = (
            frame["sentiment_reachable_park_count"].fillna(0).astype("int64")
        )
        frame["sentiment_enabled"] = sentiment_enabled

    db_experience = db_experience.sort_values(db_id).reset_index(drop=True)
    da_experience = da_experience.sort_values(da_id).reset_index(drop=True)
    da_boundaries = gpd.read_file(da_boundaries_path(config), layer="da_boundaries")
    da_boundaries[da_id] = da_boundaries[da_id].astype("string")
    da_map = da_boundaries.merge(
        da_experience, on=da_id, how="left", validate="one_to_one"
    )
    return db_google, da_google, db_experience, da_experience, da_map


def _compare_numeric(
    observed: pd.Series, expected: pd.Series, label: str, atol: float = 1e-9
) -> None:
    if not np.allclose(
        pd.to_numeric(observed, errors="coerce"),
        pd.to_numeric(expected, errors="coerce"),
        equal_nan=True,
        atol=atol,
    ):
        raise ValueError(f"Stored Stage 06 values are inconsistent: {label}")


def validate_experience(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 06 artifacts: " + ", ".join(path.name for path in missing)
        )
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    id_types = {db_id: "string", da_id: "string", "PlaceID": "string"}

    db_google = _read_csv(
        paths["db_google_entities_csv"],
        {db_id, da_id, "PlaceID", "rating_eligible", "google_rating"},
        id_types,
    )
    da_google = _read_csv(
        paths["da_google_entities_csv"],
        {da_id, "PlaceID", "rating_eligible", "google_rating"},
        {da_id: "string", "PlaceID": "string"},
    )
    db_experience = _read_csv(
        paths["db_experience_csv"],
        {
            db_id,
            da_id,
            "eligible_google_entity_count",
            "experience_rating_mean",
            "rating_covered_physical_park_count",
            "has_usable_experience_rating",
        },
        {db_id: "string", da_id: "string"},
    )
    da_experience = _read_csv(
        paths["da_experience_csv"],
        {
            da_id,
            "da_pop",
            "unique_reachable_park_count",
            "validated_google_entity_count",
            "eligible_google_entity_count",
            "experience_rating_mean",
            "rating_covered_physical_park_count",
            "digital_salience_reviews_per_1000",
            "has_usable_experience_rating",
            "sentiment_enabled",
        },
        {da_id: "string"},
    )
    _assert_unique(db_google, [db_id, "PlaceID"], "DB-Google entity table")
    _assert_unique(da_google, [da_id, "PlaceID"], "DA-Google entity table")
    _assert_unique(db_experience, [db_id], "DB experience table")
    _assert_unique(da_experience, [da_id], "DA experience table")

    expected = build_experience_frames(config)
    expected_db_google, expected_da_google, expected_db, expected_da, _ = expected
    observed_db_keys = set(zip(db_google[db_id], db_google["PlaceID"], strict=True))
    expected_db_keys = set(
        zip(expected_db_google[db_id], expected_db_google["PlaceID"], strict=True)
    )
    if observed_db_keys != expected_db_keys:
        raise ValueError("Stored DB-Google entity relationships are stale")
    observed_da_keys = set(zip(da_google[da_id], da_google["PlaceID"], strict=True))
    expected_da_keys = set(
        zip(expected_da_google[da_id], expected_da_google["PlaceID"], strict=True)
    )
    if observed_da_keys != expected_da_keys:
        raise ValueError("Stored DA-Google entity relationships are stale")

    db_check = expected_db[[db_id]].merge(
        db_experience,
        on=db_id,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not db_check["_merge"].eq("both").all():
        raise ValueError("Stage 06 DB IDs differ from current Stage 05 supply")
    da_check = expected_da[[da_id]].merge(
        da_experience,
        on=da_id,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not da_check["_merge"].eq("both").all():
        raise ValueError("Stage 06 DA IDs differ from current Stage 05 supply")

    expected_db_indexed = expected_db.set_index(db_id).loc[db_experience[db_id]]
    expected_da_indexed = expected_da.set_index(da_id).loc[da_experience[da_id]]
    for column in [
        "eligible_google_entity_count",
        "experience_rating_mean",
        "experience_rating_review_weighted_sensitivity",
        "rating_covered_physical_park_count",
        "rating_coverage_share",
    ]:
        _compare_numeric(
            db_experience[column], expected_db_indexed[column], f"DB {column}"
        )
    for column in [
        "validated_google_entity_count",
        "eligible_google_entity_count",
        "experience_rating_mean",
        "experience_rating_review_weighted_sensitivity",
        "experience_rating_mean_population_weighted_sensitivity",
        "rating_covered_physical_park_count",
        "rating_coverage_share",
        "experience_population_coverage",
        "digital_salience_reviews_per_1000",
        "experience_sentiment_mean",
    ]:
        _compare_numeric(
            da_experience[column], expected_da_indexed[column], f"DA {column}"
        )

    db_has_score = _bool_series(
        db_experience["has_usable_experience_rating"],
        "DB has_usable_experience_rating",
    )
    da_has_score = _bool_series(
        da_experience["has_usable_experience_rating"],
        "DA has_usable_experience_rating",
    )
    if not np.array_equal(
        db_has_score.to_numpy(dtype=bool),
        pd.to_numeric(db_experience["eligible_google_entity_count"])
        .gt(0)
        .to_numpy(dtype=bool),
    ):
        raise ValueError("DB usable-rating flags disagree with eligible entity counts")
    if not np.array_equal(
        da_has_score.to_numpy(dtype=bool),
        pd.to_numeric(da_experience["eligible_google_entity_count"])
        .gt(0)
        .to_numpy(dtype=bool),
    ):
        raise ValueError("DA usable-rating flags disagree with eligible entity counts")
    if (
        pd.to_numeric(db_experience["rating_covered_physical_park_count"])
        > pd.to_numeric(db_experience["reachable_park_count"])
    ).any() or (
        pd.to_numeric(da_experience["rating_covered_physical_park_count"])
        > pd.to_numeric(da_experience["unique_reachable_park_count"])
    ).any():
        raise ValueError("Rating coverage exceeds reachable physical park counts")

    da_map = gpd.read_file(paths["da_experience_gpkg"], layer="da_experience")
    da_map[da_id] = da_map[da_id].astype("string")
    _assert_unique(da_map, [da_id], "DA experience GeoPackage")
    if set(da_map[da_id]) != set(da_experience[da_id]):
        raise ValueError("DA experience GeoPackage IDs differ from the CSV")
    if str(da_map.crs) != config["crs"]["projected"]:
        raise ValueError(
            f"DA experience CRS is {da_map.crs}; expected {config['crs']['projected']}"
        )

    valid_scores = pd.to_numeric(
        da_experience.loc[da_has_score, "experience_rating_mean"], errors="raise"
    )
    sentiment_enabled = _bool_series(
        da_experience["sentiment_enabled"], "sentiment_enabled"
    )
    configured_sentiment = bool(
        config.get("experience", {}).get("sentiment", {}).get("enabled", False)
    )
    if not sentiment_enabled.eq(configured_sentiment).all():
        raise ValueError("Stored sentiment status differs from configuration")
    return {
        "db_count": len(db_experience),
        "da_count": len(da_experience),
        "db_google_entity_rows": len(db_google),
        "da_google_entity_rows": len(da_google),
        "area_reachable_validated_google_entities": int(
            expected_db_google["PlaceID"].nunique()
        ),
        "area_eligible_google_entities": int(
            da_google.loc[_bool_series(da_google["rating_eligible"], "DA rating_eligible"), "PlaceID"].nunique()
        ),
        "das_with_experience_rating": int(da_has_score.sum()),
        "das_without_experience_rating": int((~da_has_score).sum()),
        "median_da_experience_rating": (
            float(valid_scores.median()) if not valid_scores.empty else np.nan
        ),
        "minimum_da_experience_rating": (
            float(valid_scores.min()) if not valid_scores.empty else np.nan
        ),
        "maximum_da_experience_rating": (
            float(valid_scores.max()) if not valid_scores.empty else np.nan
        ),
        "das_with_sentiment": int(
            pd.to_numeric(da_experience["sentiment_reachable_park_count"]).gt(0).sum()
        ),
        "sentiment_enabled": configured_sentiment,
        "minimum_google_reviews": int(config["analysis"]["minimum_google_reviews"]),
        "crs": config["crs"]["projected"],
    }


def build_experience(
    config: dict[str, Any], final_paths: dict[str, Path]
) -> dict[str, Any]:
    db_google, da_google, db_experience, da_experience, da_map = (
        build_experience_frames(config)
    )
    slug = config["analysis_area"]["slug"]
    final_paths["da_experience_csv"].parent.mkdir(parents=True, exist_ok=True)
    temp_parent = final_paths["da_experience_csv"].parent.parent
    with inherited_temp_directory(temp_parent, f"{slug}_06_") as temp_dir:
        temp_paths = experience_paths(config, temp_dir)
        temp_paths["da_experience_csv"].parent.mkdir(parents=True, exist_ok=True)
        db_google.to_csv(temp_paths["db_google_entities_csv"], index=False)
        da_google.to_csv(temp_paths["da_google_entities_csv"], index=False)
        db_experience.to_csv(temp_paths["db_experience_csv"], index=False)
        da_experience.to_csv(temp_paths["da_experience_csv"], index=False)
        da_map.to_file(
            temp_paths["da_experience_gpkg"], layer="da_experience", driver="GPKG"
        )
        metrics = validate_experience(config, temp_paths)
        for key, target in final_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_paths[key], target)
    return metrics


def write_summary(
    config: dict[str, Any], paths: dict[str, Path], metrics: dict[str, Any], status: str
) -> Path:
    slug = config["analysis_area"]["slug"]
    table_dir = output_root(config) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / f"{slug}_06_experience_summary.csv"
    pd.DataFrame(
        [
            {
                "analysis_area": slug,
                "status": status,
                "primary_experience_source": "Google star rating",
                "primary_aggregation": "unweighted mean of unique reachable eligible PlaceIDs",
                "review_weighting_role": "sensitivity only",
                "population_weighting_role": "sensitivity only",
                **metrics,
                **{f"{key}_path": str(value) for key, value in paths.items()},
            }
        ]
    ).to_csv(summary_path, index=False)
    return summary_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = experience_paths(config)
    slug = config["analysis_area"]["slug"]
    print(f"\nStage 06: park experience ({slug})")
    print(
        "Primary: unweighted mean Google star rating across unique reachable "
        "validated PlaceIDs with at least "
        f"{config['analysis']['minimum_google_reviews']} reviews."
    )
    print("Sensitivity: review-weighted and DB-population-weighted ratings.")
    if config.get("experience", {}).get("sentiment", {}).get("enabled", False):
        print("Optional Vancouver sentiment: enabled as a separate parallel measure.")
    else:
        print("Optional sentiment: disabled; placeholder columns will be retained.")

    if force:
        print("Force enabled: building new artifacts atomically in this repository.")
        metrics = build_experience(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing artifacts.")
        try:
            metrics = validate_experience(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Experience validation failed: {error}")
            print("Build this area's Stage 06 artifacts explicitly with --force.")
            return 1
        status = "reused"

    metrics = validate_experience(config, paths) | metrics
    summary_path = write_summary(config, paths, metrics, status)
    print(f"Validated DAs: {metrics['da_count']:,}")
    print(f"DA-Google entity rows: {metrics['da_google_entity_rows']:,}")
    print(f"DAs with a usable experience score: {metrics['das_with_experience_rating']:,}")
    if np.isfinite(metrics["median_da_experience_rating"]):
        print(f"Median DA experience rating: {metrics['median_da_experience_rating']:.3f}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this area's generated Stage 06 artifacts",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
