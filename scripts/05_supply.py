"""Prepare or validate DA-level accessible park supply."""

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
    data_path,
    inherited_temp_directory,
    load_config,
    override_path,
    output_root,
    workspace_data_root,
)


def supply_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    root = data_root or workspace_data_root(config)
    base = root / "interim" / "supply" / slug
    return {
        "park_entity_crosswalk_csv": base / f"{slug}_park_entity_crosswalk.csv",
        "pairs_with_area_csv": base / f"{slug}_db_park_reachability_with_area.csv",
        "da_park_union_csv": base / f"{slug}_da_reachable_parks.csv",
        "db_supply_csv": base / f"{slug}_db_supply.csv",
        "da_supply_csv": base / f"{slug}_da_supply.csv",
        "da_supply_gpkg": base / f"{slug}_da_supply.gpkg",
        "entity_effects_csv": base / f"{slug}_park_entity_effects.csv",
    }


def reachability_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = workspace_data_root(config) / "interim" / "reach" / slug
    return {
        "pairs": base / f"{slug}_db_park_reachability.csv",
        "db_summary": base / f"{slug}_db_reachability_summary.csv",
        "da_summary": base / f"{slug}_da_reachability_summary.csv",
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
        raise ValueError(f"{label} contains duplicate keys: {columns}")


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold() == "true"


def _classify_supply(coverage_high: pd.Series, area_high: pd.Series) -> pd.Series:
    values = np.select(
        [
            coverage_high & area_high,
            coverage_high & ~area_high,
            ~coverage_high & area_high,
        ],
        ["HH", "HL", "LH"],
        default="LL",
    )
    return pd.Series(values, index=coverage_high.index, dtype="string")


def _park_entity_tables(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    cap20 = float(config["analysis"]["park_area_cap_ha"])
    cap10 = float(config["analysis"]["park_area_sensitivity_cap_ha"])
    parks = gpd.read_file(data_path(config, "parks"), layer="parks_cleaned")
    required = {
        "park_id",
        "park_key",
        "park_name",
        "municipality",
        "area_ha_calc",
        "geometry",
    }
    missing = required - set(parks.columns)
    if missing:
        raise ValueError(f"Cleaned parks are missing fields: {', '.join(sorted(missing))}")
    if parks["park_id"].astype("string").duplicated().any():
        raise ValueError("Cleaned park IDs are not unique")
    if parks["park_key"].astype("string").duplicated().any():
        raise ValueError("Cleaned park keys are not unique")

    overrides_file = override_path(config, "park_entities")
    overrides = pd.read_csv(overrides_file, dtype="string")
    override_required = {
        "park_entity_id",
        "park_entity_name",
        "municipality",
        "park_name",
        "entity_reason",
        "evidence",
    }
    missing = override_required - set(overrides.columns)
    if missing:
        raise ValueError(
            f"{overrides_file.name} is missing columns: {', '.join(sorted(missing))}"
        )
    if overrides[list(override_required)].isna().any().any():
        raise ValueError(f"{overrides_file.name} contains blank required values")
    _assert_unique(
        overrides, ["municipality", "park_name"], "Park-entity overrides"
    )
    group_sizes = overrides.groupby("park_entity_id").size()
    if (group_sizes < 2).any():
        invalid = group_sizes[group_sizes < 2].index.tolist()
        raise ValueError(
            "Park-entity overrides must contain at least two components per entity: "
            f"{invalid[:5]}"
        )
    for column in ["park_entity_name", "entity_reason", "evidence"]:
        conflicts = overrides.groupby("park_entity_id")[column].nunique()
        if (conflicts > 1).any():
            invalid = conflicts[conflicts > 1].index.tolist()
            raise ValueError(
                f"Park-entity override groups disagree on {column}: {invalid[:5]}"
            )

    parks["park_id"] = parks["park_id"].astype("string")
    parks["park_key"] = parks["park_key"].astype("string")
    mapping = parks[
        ["park_id", "park_key", "park_name", "municipality", "area_ha_calc"]
    ].copy()
    mapping = mapping.rename(
        columns={
            "park_name": "component_park_name",
            "municipality": "component_municipality",
            "area_ha_calc": "component_area_ha",
        }
    )
    mapping["park_entity_id"] = mapping["park_key"]
    mapping["park_entity_name"] = mapping["component_park_name"]
    mapping["entity_reason"] = "single municipal park record"
    mapping["evidence"] = "2022 municipal park inventory"

    override_lookup = overrides.merge(
        mapping[["park_id", "component_park_name", "component_municipality"]],
        left_on=["municipality", "park_name"],
        right_on=["component_municipality", "component_park_name"],
        how="left",
        validate="one_to_one",
    )
    if override_lookup["park_id"].isna().any():
        missing_records = override_lookup.loc[
            override_lookup["park_id"].isna(), ["municipality", "park_name"]
        ].to_dict("records")
        raise ValueError(
            "Park-entity overrides do not match cleaned parks: "
            f"{missing_records[:5]}"
        )
    override_lookup = override_lookup.set_index("park_id")
    overridden = mapping["park_id"].isin(override_lookup.index)
    for column in ["park_entity_id", "park_entity_name", "entity_reason", "evidence"]:
        mapping.loc[overridden, column] = mapping.loc[overridden, "park_id"].map(
            override_lookup[column]
        )

    joined = parks.merge(
        mapping[
            [
                "park_id",
                "park_entity_id",
                "park_entity_name",
                "entity_reason",
                "evidence",
            ]
        ],
        on="park_id",
        how="left",
        validate="one_to_one",
    )
    entities = joined.dissolve(
        by="park_entity_id",
        as_index=False,
        aggfunc={
            "park_entity_name": "first",
            "entity_reason": "first",
            "evidence": "first",
        },
    )
    entity_attributes = (
        joined.groupby("park_entity_id", as_index=False)
        .agg(
            component_count=("park_id", "nunique"),
            park_entity_municipality=(
                "municipality",
                lambda values: " | ".join(sorted(set(values))),
            ),
        )
    )
    entities = entities.merge(
        entity_attributes, on="park_entity_id", how="left", validate="one_to_one"
    )
    entities["area_ha_uncapped"] = entities.geometry.area / 10_000
    if entities["area_ha_uncapped"].isna().any() or (
        entities["area_ha_uncapped"] <= 0
    ).any():
        raise ValueError("Cleaned parks contain missing or non-positive area")
    entities["area_ha_cap20"] = entities["area_ha_uncapped"].clip(upper=cap20)
    entities["area_ha_cap10"] = entities["area_ha_uncapped"].clip(upper=cap10)

    mapping = mapping.merge(
        entities[
            [
                "park_entity_id",
                "park_entity_name",
                "park_entity_municipality",
                "component_count",
                "area_ha_uncapped",
                "area_ha_cap20",
                "area_ha_cap10",
            ]
        ],
        on=["park_entity_id", "park_entity_name"],
        how="left",
        validate="many_to_one",
    )
    return mapping, entities


def build_supply_frames(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
]:
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    ids = {db_id: "string", da_id: "string", "park_id": "string"}
    reach_paths = reachability_paths(config)

    pairs = _read_csv(
        reach_paths["pairs"],
        {
            db_id,
            da_id,
            "park_id",
            "park_name",
            "park_municipality",
            "network_dist_m",
        },
        ids,
    )
    db_reach = _read_csv(
        reach_paths["db_summary"],
        {db_id, da_id, "db_pop", "reachable_park_count"},
        {db_id: "string", da_id: "string"},
    )
    da_reach = _read_csv(
        reach_paths["da_summary"],
        {da_id, "da_pop", "db_count", "population_with_access", "park_access_pop_coverage"},
        {da_id: "string"},
    )
    _assert_unique(pairs, [db_id, "park_id"], "Stage 04 DB-park pairs")
    _assert_unique(db_reach, [db_id], "Stage 04 DB summary")
    _assert_unique(da_reach, [da_id], "Stage 04 DA summary")
    db_reach["db_pop"] = pd.to_numeric(db_reach["db_pop"], errors="raise").astype("int64")

    entity_crosswalk, park_entities = _park_entity_tables(config)
    component_pairs = pairs.merge(
        entity_crosswalk[["park_id", "park_entity_id"]],
        on="park_id",
        how="left",
        validate="many_to_one",
    )
    if component_pairs["park_entity_id"].isna().any():
        missing_ids = component_pairs.loc[
            component_pairs["park_entity_id"].isna(), "park_id"
        ].drop_duplicates()
        raise ValueError(
            f"Reachability references parks without an entity: {missing_ids.tolist()[:5]}"
        )

    entity_pair_counts = (
        component_pairs.groupby([db_id, da_id, "park_entity_id"], as_index=False)
        .agg(reachable_component_park_count=("park_id", "nunique"))
    )
    nearest = (
        component_pairs.sort_values(
            [db_id, "park_entity_id", "network_dist_m", "park_id"]
        )
        .drop_duplicates([db_id, "park_entity_id"])
        .rename(
            columns={
                "park_id": "nearest_component_park_id",
                "park_name": "nearest_component_park_name",
                "park_municipality": "nearest_component_municipality",
            }
        )
    )
    pairs_with_area = nearest.merge(
        entity_pair_counts,
        on=[db_id, da_id, "park_entity_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        park_entities.drop(columns="geometry"),
        on="park_entity_id",
        how="left",
        validate="many_to_one",
    )
    if pairs_with_area["area_ha_uncapped"].isna().any():
        missing_ids = pairs_with_area.loc[
            pairs_with_area["area_ha_uncapped"].isna(), "park_entity_id"
        ].drop_duplicates()
        raise ValueError(
            f"Reachability references park entities without area: {missing_ids.tolist()[:5]}"
        )

    area_columns = ["area_ha_uncapped", "area_ha_cap20", "area_ha_cap10"]
    db_area = (
        pairs_with_area.groupby([db_id, da_id], as_index=False)
        .agg(
            reachable_park_count=("park_entity_id", "nunique"),
            accessible_area_ha_uncapped=("area_ha_uncapped", "sum"),
            accessible_area_ha_cap20=("area_ha_cap20", "sum"),
            accessible_area_ha_cap10=("area_ha_cap10", "sum"),
        )
    )
    component_counts = (
        component_pairs.groupby([db_id, da_id], as_index=False)
        .agg(reachable_component_park_count=("park_id", "nunique"))
    )
    db_area = db_area.merge(
        component_counts, on=[db_id, da_id], how="left", validate="one_to_one"
    )
    db_supply = db_reach[[db_id, da_id, "db_pop"]].merge(
        db_area, on=[db_id, da_id], how="left", validate="one_to_one"
    )
    db_supply["reachable_park_count"] = db_supply["reachable_park_count"].fillna(0).astype(int)
    for column in (
        "accessible_area_ha_uncapped",
        "accessible_area_ha_cap20",
        "accessible_area_ha_cap10",
    ):
        db_supply[column] = db_supply[column].fillna(0.0)
        db_supply[f"{column}_per_1000"] = np.where(
            db_supply["db_pop"] > 0,
            db_supply[column] / db_supply["db_pop"] * 1000,
            np.nan,
        )

    # A zero-population DB represents no residents and therefore cannot add a
    # park to the DA-level resident supply union.
    populated_db_ids = set(db_supply.loc[db_supply["db_pop"] > 0, db_id])
    da_component_counts = (
        component_pairs[component_pairs[db_id].isin(populated_db_ids)]
        .groupby([da_id, "park_entity_id"], as_index=False)
        .agg(reachable_component_park_count=("park_id", "nunique"))
    )
    da_park_union = (
        da_component_counts.merge(
            park_entities.drop(columns="geometry"),
            on="park_entity_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values([da_id, "park_entity_id"])
        .reset_index(drop=True)
    )

    union_summary = (
        da_park_union.groupby(da_id, as_index=False)
        .agg(
            unique_reachable_park_count=("park_entity_id", "nunique"),
            unique_reachable_component_park_count=(
                "reachable_component_park_count",
                "sum",
            ),
            accessible_area_ha_uncapped=("area_ha_uncapped", "sum"),
            accessible_area_ha_cap20=("area_ha_cap20", "sum"),
            accessible_area_ha_cap10=("area_ha_cap10", "sum"),
        )
    )
    da_supply = da_reach[
        [
            da_id,
            "da_pop",
            "db_count",
            "dbs_with_access",
            "population_with_access",
            "park_access_pop_coverage",
        ]
    ].merge(union_summary, on=da_id, how="left", validate="one_to_one")
    da_supply["unique_reachable_park_count"] = (
        da_supply["unique_reachable_park_count"].fillna(0).astype(int)
    )
    da_supply["unique_reachable_component_park_count"] = (
        da_supply["unique_reachable_component_park_count"].fillna(0).astype(int)
    )
    for column in (
        "accessible_area_ha_uncapped",
        "accessible_area_ha_cap20",
        "accessible_area_ha_cap10",
    ):
        da_supply[column] = da_supply[column].fillna(0.0)
        suffix = column.removeprefix("accessible_area_ha_")
        da_supply[f"supply_ha_per_1000_{suffix}"] = np.where(
            pd.to_numeric(da_supply["da_pop"]) > 0,
            da_supply[column] / pd.to_numeric(da_supply["da_pop"]) * 1000,
            np.nan,
        )

    valid_population = pd.to_numeric(da_supply["da_pop"]) > 0
    median20 = float(da_supply.loc[valid_population, "supply_ha_per_1000_cap20"].median())
    median10 = float(da_supply.loc[valid_population, "supply_ha_per_1000_cap10"].median())
    median_raw = float(da_supply.loc[valid_population, "supply_ha_per_1000_uncapped"].median())
    coverage_primary = float(config["analysis"]["high_coverage_threshold"])
    coverage_sensitivity = float(config["analysis"]["coverage_sensitivity_threshold"])

    da_supply["supply_median_cap20"] = median20
    da_supply["supply_median_cap10"] = median10
    da_supply["supply_median_uncapped"] = median_raw
    da_supply["high_coverage"] = da_supply["park_access_pop_coverage"] >= coverage_primary
    da_supply["high_area_cap20"] = da_supply["supply_ha_per_1000_cap20"] >= median20
    da_supply["high_supply"] = da_supply["high_coverage"] & da_supply["high_area_cap20"]
    da_supply["supply_type"] = _classify_supply(
        da_supply["high_coverage"], da_supply["high_area_cap20"]
    )

    da_supply["high_area_cap10_sensitivity"] = (
        da_supply["supply_ha_per_1000_cap10"] >= median10
    )
    da_supply["high_supply_cap10_sensitivity"] = (
        da_supply["high_coverage"] & da_supply["high_area_cap10_sensitivity"]
    )
    da_supply["high_area_uncapped_sensitivity"] = (
        da_supply["supply_ha_per_1000_uncapped"] >= median_raw
    )
    da_supply["high_supply_uncapped_sensitivity"] = (
        da_supply["high_coverage"] & da_supply["high_area_uncapped_sensitivity"]
    )
    da_supply["high_coverage_50_sensitivity"] = (
        da_supply["park_access_pop_coverage"] >= coverage_sensitivity
    )
    da_supply["high_supply_coverage50_sensitivity"] = (
        da_supply["high_coverage_50_sensitivity"] & da_supply["high_area_cap20"]
    )

    insufficient = ~valid_population
    for column in [
        "high_coverage",
        "high_area_cap20",
        "high_supply",
        "high_area_cap10_sensitivity",
        "high_supply_cap10_sensitivity",
        "high_area_uncapped_sensitivity",
        "high_supply_uncapped_sensitivity",
        "high_coverage_50_sensitivity",
        "high_supply_coverage50_sensitivity",
    ]:
        da_supply[column] = da_supply[column].astype("boolean")
        da_supply.loc[insufficient, column] = pd.NA
    da_supply.loc[insufficient, "supply_type"] = "insufficient_population"
    da_supply = da_supply.sort_values(da_id).reset_index(drop=True)

    component_areas = entity_crosswalk[["park_id", "component_area_ha"]].copy()
    component_areas["component_area_ha"] = pd.to_numeric(
        component_areas["component_area_ha"], errors="raise"
    )
    component_areas["component_area_ha_cap20"] = component_areas[
        "component_area_ha"
    ].clip(upper=float(config["analysis"]["park_area_cap_ha"]))
    component_areas["component_area_ha_cap10"] = component_areas[
        "component_area_ha"
    ].clip(upper=float(config["analysis"]["park_area_sensitivity_cap_ha"]))
    component_union = (
        component_pairs[component_pairs[db_id].isin(populated_db_ids)][
            [da_id, "park_id"]
        ]
        .drop_duplicates()
        .merge(component_areas, on="park_id", how="left", validate="many_to_one")
    )
    component_summary = (
        component_union.groupby(da_id, as_index=False)
        .agg(
            municipal_record_unique_reachable_park_count=("park_id", "nunique"),
            municipal_record_accessible_area_ha_uncapped=("component_area_ha", "sum"),
            municipal_record_accessible_area_ha_cap20=(
                "component_area_ha_cap20",
                "sum",
            ),
            municipal_record_accessible_area_ha_cap10=(
                "component_area_ha_cap10",
                "sum",
            ),
        )
    )
    component_supply = da_reach[
        [da_id, "da_pop", "park_access_pop_coverage"]
    ].merge(component_summary, on=da_id, how="left", validate="one_to_one")
    component_supply["municipal_record_unique_reachable_park_count"] = (
        component_supply["municipal_record_unique_reachable_park_count"]
        .fillna(0)
        .astype(int)
    )
    for column in [
        "municipal_record_accessible_area_ha_uncapped",
        "municipal_record_accessible_area_ha_cap20",
        "municipal_record_accessible_area_ha_cap10",
    ]:
        component_supply[column] = component_supply[column].fillna(0.0)
        suffix = column.removeprefix("municipal_record_accessible_area_ha_")
        component_supply[f"municipal_record_supply_ha_per_1000_{suffix}"] = np.where(
            pd.to_numeric(component_supply["da_pop"]) > 0,
            component_supply[column]
            / pd.to_numeric(component_supply["da_pop"])
            * 1000,
            np.nan,
        )
    component_valid = pd.to_numeric(component_supply["da_pop"]) > 0
    component_median20 = float(
        component_supply.loc[
            component_valid, "municipal_record_supply_ha_per_1000_cap20"
        ].median()
    )
    component_supply["municipal_record_supply_median_cap20"] = component_median20
    component_supply["municipal_record_high_supply"] = (
        component_supply["park_access_pop_coverage"] >= coverage_primary
    ) & (
        component_supply["municipal_record_supply_ha_per_1000_cap20"]
        >= component_median20
    )
    component_supply["municipal_record_high_supply"] = component_supply[
        "municipal_record_high_supply"
    ].astype("boolean")
    component_supply.loc[
        ~component_valid, "municipal_record_high_supply"
    ] = pd.NA
    entity_effects = component_supply.merge(
        da_supply[
            [
                da_id,
                "unique_reachable_park_count",
                "supply_ha_per_1000_uncapped",
                "supply_ha_per_1000_cap20",
                "supply_ha_per_1000_cap10",
                "supply_median_cap20",
                "high_supply",
            ]
        ],
        on=da_id,
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "unique_reachable_park_count": "physical_entity_unique_reachable_park_count",
            "supply_ha_per_1000_uncapped": "physical_entity_supply_ha_per_1000_uncapped",
            "supply_ha_per_1000_cap20": "physical_entity_supply_ha_per_1000_cap20",
            "supply_ha_per_1000_cap10": "physical_entity_supply_ha_per_1000_cap10",
            "supply_median_cap20": "physical_entity_supply_median_cap20",
            "high_supply": "physical_entity_high_supply",
        }
    )
    for suffix in ["uncapped", "cap20", "cap10"]:
        entity_effects[f"supply_difference_{suffix}"] = (
            entity_effects[f"physical_entity_supply_ha_per_1000_{suffix}"]
            - entity_effects[f"municipal_record_supply_ha_per_1000_{suffix}"]
        )
    entity_effects["high_supply_changed"] = (
        entity_effects["physical_entity_high_supply"].astype("boolean")
        != entity_effects["municipal_record_high_supply"].astype("boolean")
    )
    entity_effects.loc[~component_valid, "high_supply_changed"] = pd.NA

    da_boundaries = gpd.read_file(da_boundaries_path(config), layer="da_boundaries")
    da_boundaries[da_id] = da_boundaries[da_id].astype("string")
    da_map = da_boundaries.merge(da_supply, on=da_id, how="left", validate="one_to_one")
    return (
        entity_crosswalk,
        pairs_with_area,
        da_park_union,
        db_supply,
        da_supply,
        entity_effects,
        da_map,
    )


def validate_supply(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 05 artifacts: " + ", ".join(path.name for path in missing)
        )

    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    ids = {
        db_id: "string",
        da_id: "string",
        "park_entity_id": "string",
        "nearest_component_park_id": "string",
    }
    crosswalk = _read_csv(
        paths["park_entity_crosswalk_csv"],
        {
            "park_id",
            "park_key",
            "park_entity_id",
            "park_entity_name",
            "component_count",
            "area_ha_uncapped",
            "area_ha_cap20",
            "area_ha_cap10",
        },
        {"park_id": "string", "park_key": "string", "park_entity_id": "string"},
    )
    pairs = _read_csv(
        paths["pairs_with_area_csv"],
        {
            db_id,
            da_id,
            "park_entity_id",
            "nearest_component_park_id",
            "area_ha_uncapped",
            "area_ha_cap20",
            "area_ha_cap10",
        },
        ids,
    )
    union = _read_csv(
        paths["da_park_union_csv"],
        {
            da_id,
            "park_entity_id",
            "area_ha_uncapped",
            "area_ha_cap20",
            "area_ha_cap10",
        },
        {da_id: "string", "park_entity_id": "string"},
    )
    db_supply = _read_csv(
        paths["db_supply_csv"],
        {db_id, da_id, "db_pop", "reachable_park_count"},
        {db_id: "string", da_id: "string"},
    )
    da_supply = _read_csv(
        paths["da_supply_csv"],
        {
            da_id,
            "da_pop",
            "park_access_pop_coverage",
            "unique_reachable_park_count",
            "supply_ha_per_1000_cap20",
            "supply_ha_per_1000_cap10",
            "supply_ha_per_1000_uncapped",
            "high_supply",
            "supply_type",
        },
        {da_id: "string"},
    )
    entity_effects = _read_csv(
        paths["entity_effects_csv"],
        {
            da_id,
            "municipal_record_supply_ha_per_1000_cap20",
            "physical_entity_supply_ha_per_1000_cap20",
            "supply_difference_cap20",
            "municipal_record_high_supply",
            "physical_entity_high_supply",
            "high_supply_changed",
        },
        {da_id: "string"},
    )
    _assert_unique(crosswalk, ["park_id"], "Park-entity crosswalk")
    _assert_unique(pairs, [db_id, "park_entity_id"], "Enriched DB-park-entity pairs")
    _assert_unique(union, [da_id, "park_entity_id"], "DA park-entity union")
    _assert_unique(db_supply, [db_id], "DB supply")
    _assert_unique(da_supply, [da_id], "DA supply")
    _assert_unique(entity_effects, [da_id], "Park-entity effect comparison")
    expected_crosswalk, _ = _park_entity_tables(config)
    crosswalk_check = expected_crosswalk[
        [
            "park_id",
            "park_entity_id",
            "park_entity_name",
            "component_count",
            "area_ha_uncapped",
            "area_ha_cap20",
            "area_ha_cap10",
        ]
    ].merge(
        crosswalk[
            [
                "park_id",
                "park_entity_id",
                "park_entity_name",
                "component_count",
                "area_ha_uncapped",
                "area_ha_cap20",
                "area_ha_cap10",
            ]
        ],
        on="park_id",
        how="outer",
        suffixes=("_expected", "_stored"),
        indicator=True,
        validate="one_to_one",
    )
    if not crosswalk_check["_merge"].eq("both").all():
        raise ValueError("Park-entity crosswalk park IDs differ from current configuration")
    for column in ["park_entity_id", "park_entity_name", "component_count"]:
        if not (
            crosswalk_check[f"{column}_expected"].astype("string")
            == crosswalk_check[f"{column}_stored"].astype("string")
        ).all():
            raise ValueError(
                f"Park-entity crosswalk is stale for current configuration: {column}"
            )
    for column in ["area_ha_uncapped", "area_ha_cap20", "area_ha_cap10"]:
        if not np.allclose(
            pd.to_numeric(crosswalk_check[f"{column}_expected"]),
            pd.to_numeric(crosswalk_check[f"{column}_stored"]),
        ):
            raise ValueError(
                f"Park-entity crosswalk is stale for current configuration: {column}"
            )

    cap20 = float(config["analysis"]["park_area_cap_ha"])
    cap10 = float(config["analysis"]["park_area_sensitivity_cap_ha"])
    raw = pd.to_numeric(pairs["area_ha_uncapped"], errors="raise")
    observed20 = pd.to_numeric(pairs["area_ha_cap20"], errors="raise")
    observed10 = pd.to_numeric(pairs["area_ha_cap10"], errors="raise")
    if not np.allclose(observed20, raw.clip(upper=cap20)):
        raise ValueError("20 ha park caps are inconsistent")
    if not np.allclose(observed10, raw.clip(upper=cap10)):
        raise ValueError("10 ha park caps are inconsistent")

    reach = reachability_paths(config)
    source_pairs = pd.read_csv(
        reach["pairs"], dtype={db_id: "string", da_id: "string", "park_id": "string"}
    )
    source_entities = source_pairs.merge(
        crosswalk[["park_id", "park_entity_id"]],
        on="park_id",
        how="left",
        validate="many_to_one",
    )
    if source_entities["park_entity_id"].isna().any():
        raise ValueError("Stage 04 references a park missing from the entity crosswalk")
    expected_pairs = source_entities[[db_id, "park_entity_id"]].drop_duplicates()
    if set(zip(pairs[db_id], pairs["park_entity_id"], strict=True)) != set(
        zip(expected_pairs[db_id], expected_pairs["park_entity_id"], strict=True)
    ):
        raise ValueError("Stage 05 park-entity pair set differs from Stage 04 components")
    source_db = pd.read_csv(reach["db_summary"], dtype={db_id: "string", da_id: "string"})
    source_db["db_pop"] = pd.to_numeric(source_db["db_pop"], errors="raise")
    if set(db_supply[db_id]) != set(source_db[db_id]):
        raise ValueError("Stage 05 DB IDs differ from Stage 04")

    populated = set(source_db.loc[source_db["db_pop"] > 0, db_id])
    expected_union = pairs[pairs[db_id].isin(populated)][
        [da_id, "park_entity_id"]
    ].drop_duplicates()
    if set(zip(union[da_id], union["park_entity_id"], strict=True)) != set(
        zip(expected_union[da_id], expected_union["park_entity_id"], strict=True)
    ):
        raise ValueError("DA park union includes a zero-population-only park or misses a populated-DB park")

    source_da = pd.read_csv(reach["da_summary"], dtype={da_id: "string"})
    if set(da_supply[da_id]) != set(source_da[da_id]):
        raise ValueError("Stage 05 DA IDs differ from Stage 04")
    if set(entity_effects[da_id]) != set(da_supply[da_id]):
        raise ValueError("Park-entity effect comparison has inconsistent DA IDs")
    effect_check = entity_effects.merge(
        da_supply[[da_id, "supply_ha_per_1000_cap20", "high_supply"]],
        on=da_id,
        how="left",
        suffixes=("", "_stored_supply"),
        validate="one_to_one",
    )
    if not np.allclose(
        pd.to_numeric(effect_check["physical_entity_supply_ha_per_1000_cap20"]),
        pd.to_numeric(effect_check["supply_ha_per_1000_cap20"]),
        equal_nan=True,
    ):
        raise ValueError("Park-entity effect comparison disagrees with DA supply")
    expected_difference = (
        pd.to_numeric(effect_check["physical_entity_supply_ha_per_1000_cap20"])
        - pd.to_numeric(effect_check["municipal_record_supply_ha_per_1000_cap20"])
    )
    if not np.allclose(
        expected_difference,
        pd.to_numeric(effect_check["supply_difference_cap20"]),
        equal_nan=True,
    ):
        raise ValueError("Park-entity effect comparison has inconsistent differences")
    comparison = da_supply.merge(
        source_da[[da_id, "da_pop", "park_access_pop_coverage"]],
        on=da_id,
        suffixes=("", "_source"),
        validate="one_to_one",
    )
    if not np.allclose(
        pd.to_numeric(comparison["da_pop"]),
        pd.to_numeric(comparison["da_pop_source"]),
    ) or not np.allclose(
        pd.to_numeric(comparison["park_access_pop_coverage"]),
        pd.to_numeric(comparison["park_access_pop_coverage_source"]),
        equal_nan=True,
    ):
        raise ValueError("Stage 05 population or coverage differs from Stage 04")

    union_sums = union.groupby(da_id).agg(
        expected_count=("park_entity_id", "nunique"),
        raw=("area_ha_uncapped", "sum"),
        cap20=("area_ha_cap20", "sum"),
        cap10=("area_ha_cap10", "sum"),
    )
    check = da_supply.set_index(da_id).join(union_sums).fillna(
        {"expected_count": 0, "raw": 0, "cap20": 0, "cap10": 0}
    )
    if not np.array_equal(
        pd.to_numeric(check["unique_reachable_park_count"]).to_numpy(dtype=int),
        pd.to_numeric(check["expected_count"]).to_numpy(dtype=int),
    ):
        raise ValueError("DA unique park counts do not match the union")
    population = pd.to_numeric(check["da_pop"])
    for output_column, area_column in (
        ("supply_ha_per_1000_uncapped", "raw"),
        ("supply_ha_per_1000_cap20", "cap20"),
        ("supply_ha_per_1000_cap10", "cap10"),
    ):
        expected = pd.to_numeric(check[area_column]) / population.replace(0, np.nan) * 1000
        if not np.allclose(
            pd.to_numeric(check[output_column]), expected, equal_nan=True
        ):
            raise ValueError(f"DA supply normalization is inconsistent: {output_column}")

    median20 = float(check.loc[population > 0, "supply_ha_per_1000_cap20"].median())
    if not np.allclose(pd.to_numeric(check["supply_median_cap20"]), median20):
        raise ValueError("Stored primary supply median is inconsistent")
    expected_high = (
        pd.to_numeric(check["park_access_pop_coverage"])
        >= float(config["analysis"]["high_coverage_threshold"])
    ) & (pd.to_numeric(check["supply_ha_per_1000_cap20"]) >= median20)
    observed_high = _bool_series(check["high_supply"])
    if not np.array_equal(expected_high[population > 0], observed_high[population > 0]):
        raise ValueError("Primary high-supply classification is inconsistent")

    info = pyogrio.read_info(paths["da_supply_gpkg"], force_feature_count=True)
    if int(info["features"]) != len(da_supply):
        raise ValueError("DA supply CSV and GeoPackage feature counts differ")
    if str(info["crs"]).upper() != config["crs"]["projected"].upper():
        raise ValueError("DA supply GeoPackage has the wrong CRS")

    high_supply = _bool_series(da_supply["high_supply"])
    changed_supply = _bool_series(entity_effects["high_supply_changed"])
    cap20_difference = pd.to_numeric(
        entity_effects["supply_difference_cap20"], errors="raise"
    )
    component_median20 = float(
        pd.to_numeric(
            entity_effects["municipal_record_supply_ha_per_1000_cap20"],
            errors="raise",
        ).median()
    )
    merged_entities = crosswalk.groupby("park_entity_id")["park_id"].nunique().gt(1)
    return {
        "db_count": len(db_supply),
        "da_count": len(da_supply),
        "municipal_park_component_count": int(crosswalk["park_id"].nunique()),
        "physical_park_entity_count": int(crosswalk["park_entity_id"].nunique()),
        "merged_park_entity_count": int(merged_entities.sum()),
        "db_park_component_pair_count": len(source_pairs),
        "db_park_entity_pair_count": len(pairs),
        "da_park_entity_union_count": len(union),
        "total_population": int(pd.to_numeric(da_supply["da_pop"]).sum()),
        "median_supply_cap20": median20,
        "municipal_record_median_supply_cap20": component_median20,
        "median_supply_cap10": float(
            check.loc[population > 0, "supply_ha_per_1000_cap10"].median()
        ),
        "median_supply_uncapped": float(
            check.loc[population > 0, "supply_ha_per_1000_uncapped"].median()
        ),
        "high_supply_da_count": int(high_supply.sum()),
        "high_supply_da_classification_changes": int(changed_supply.sum()),
        "das_with_cap20_supply_change": int(
            (~np.isclose(cap20_difference.fillna(0), 0.0)).sum()
        ),
        "maximum_absolute_cap20_supply_change": float(cap20_difference.abs().max()),
        "zero_supply_da_count": int(
            (pd.to_numeric(da_supply["unique_reachable_park_count"]) == 0).sum()
        ),
        "zero_population_das": int((pd.to_numeric(da_supply["da_pop"]) == 0).sum()),
        "zero_population_dbs_excluded_from_union": int((source_db["db_pop"] == 0).sum()),
        "crs": config["crs"]["projected"],
    }


def build_supply(config: dict[str, Any], final_paths: dict[str, Path]) -> dict[str, Any]:
    frames = build_supply_frames(config)
    crosswalk, pairs, union, db_supply, da_supply, entity_effects, da_map = frames
    slug = config["analysis_area"]["slug"]
    final_paths["da_supply_csv"].parent.mkdir(parents=True, exist_ok=True)
    temp_parent = final_paths["da_supply_csv"].parent.parent
    with inherited_temp_directory(temp_parent, f"{slug}_05_") as temp_dir:
        temp_paths = supply_paths(config, temp_dir)
        temp_paths["da_supply_csv"].parent.mkdir(parents=True, exist_ok=True)
        crosswalk.to_csv(temp_paths["park_entity_crosswalk_csv"], index=False)
        pairs.to_csv(temp_paths["pairs_with_area_csv"], index=False)
        union.to_csv(temp_paths["da_park_union_csv"], index=False)
        db_supply.to_csv(temp_paths["db_supply_csv"], index=False)
        da_supply.to_csv(temp_paths["da_supply_csv"], index=False)
        entity_effects.to_csv(temp_paths["entity_effects_csv"], index=False)
        da_map.to_file(temp_paths["da_supply_gpkg"], layer="da_supply", driver="GPKG")
        metrics = validate_supply(config, temp_paths)
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
    summary_path = table_dir / f"{slug}_05_supply_summary.csv"
    pd.DataFrame(
        [
            {
                "analysis_area": slug,
                "status": status,
                "primary_area_cap_ha": config["analysis"]["park_area_cap_ha"],
                "sensitivity_area_cap_ha": config["analysis"]["park_area_sensitivity_cap_ha"],
                "primary_coverage_threshold": config["analysis"]["high_coverage_threshold"],
                "sensitivity_coverage_threshold": config["analysis"]["coverage_sensitivity_threshold"],
                **metrics,
                **{f"{key}_path": str(value) for key, value in paths.items()},
            }
        ]
    ).to_csv(summary_path, index=False)
    return summary_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = supply_paths(config)
    slug = config["analysis_area"]["slug"]
    print(f"\nStage 05: accessible park supply ({slug})")
    print(
        "Primary: unique reachable physical park entities, 20 ha per-entity cap, "
        "per 1,000 DA residents."
    )
    print("Sensitivity: 10 ha cap, uncapped area, and 50% coverage threshold.")

    if force:
        print("Force enabled: building new artifacts atomically in this repository.")
        metrics = build_supply(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing artifacts.")
        try:
            metrics = validate_supply(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Supply validation failed: {error}")
            print("Build this area's Stage 05 artifacts explicitly with --force.")
            return 1
        status = "reused"

    metrics = validate_supply(config, paths) | metrics
    summary_path = write_summary(config, paths, metrics, status)
    print(f"Validated DAs: {metrics['da_count']:,}")
    print(f"Merged physical park entities: {metrics['merged_park_entity_count']:,}")
    print(f"DA-park-entity union rows: {metrics['da_park_entity_union_count']:,}")
    print(f"Primary median: {metrics['median_supply_cap20']:.3f} ha/1,000")
    print(f"High-supply DAs: {metrics['high_supply_da_count']:,}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this area's generated Stage 05 artifacts",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
