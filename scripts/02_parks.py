"""Build or validate the shared park polygons, eligibility, and entrances."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point
from shapely.ops import nearest_points

from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    override_path,
    project_root,
    workspace_data_root,
)


PARK_FIELDS = {
    "park_id",
    "park_key",
    "municipality",
    "park_name",
    "area_ha_calc",
    "park_eligibility",
    "eligibility_reason",
    "manual_override",
}

ENTRANCE_FIELDS = {
    "entrance_id",
    "entrance_method",
    "entrance_rule",
    "park_id",
    "park_key",
    "municipality",
    "park_name",
    "nearest_node",
    "candidate_x",
    "candidate_y",
    "node_x",
    "node_y",
    "dist_to_boundary_m",
    "snap_dist_m",
}


def park_artifact_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    """Return all shared Stage 02 artifact paths."""
    if data_root is None:
        resolved_parks = data_path(config, "parks")
        data_root = resolved_parks.parents[2]
    processed = data_root / "processed" / "parks"
    qa = data_root / "interim" / "qa"
    slug = config["network"]["slug"]

    return {
        "parks_gpkg": processed / f"{slug}_parks_cleaned.gpkg",
        "parks_csv": processed / f"{slug}_parks_cleaned.csv",
        "eligibility_gpkg": processed / f"{slug}_parks_with_eligibility.gpkg",
        "eligibility_csv": processed / f"{slug}_parks_with_eligibility.csv",
        "entrances_gpkg": processed / f"{slug}_park_entrances.gpkg",
        "entrances_csv": processed / f"{slug}_park_entrances.csv",
        "candidates_gpkg": qa / f"{slug}_park_entrance_candidates.gpkg",
        "isolated_gpkg": qa / f"{slug}_02_zero_entrance_isolated_or_island_parks.gpkg",
    }


def _make_slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _fix_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fixed = frame.copy()
    try:
        fixed["geometry"] = fixed.geometry.make_valid()
    except Exception:
        fixed["geometry"] = fixed.geometry.buffer(0)
    return fixed[fixed.geometry.notna() & ~fixed.geometry.is_empty].copy()


def _park_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["municipality"].astype("string").str.strip().str.casefold()
        + "\x1f"
        + frame["park_name"].astype("string").str.strip().str.casefold()
    )


def _classify_park(name: Any) -> tuple[str, str]:
    """Apply the reviewed name rules from the multi-city workflow."""
    text = str(name or "").lower().strip()
    if text in {"", "nan", "none"}:
        return "exclude", "missing park name"

    exclude_terms = [
        "substation",
        "works yard",
        "work yard",
        "city works",
        "fire hall",
        "firehall",
        "police",
        "city hall",
        "public library",
        "animal shelter",
        "college",
        "university",
        "justice institute",
        "elementary",
        "secondary",
        "middle",
        "school",
        "cemetery",
        "hospital",
        "church",
        "catholic",
        "masonic",
        "oddfellows",
        "museum",
        "intermediate care",
        "care centre",
        "care center",
        "reservoir",
        "pump station",
        "treatment plant",
        "utility",
        "hydro",
        "parking",
    ]
    include_terms = [
        "park",
        "playground",
        "off leash",
        "dog area",
        "greenway",
        "trail",
        "walkway",
        "walk",
        "square",
        "plaza",
        "triangle",
        "boulevard",
        "green",
        "garden",
        "gardens",
        "lawn",
        "ravine",
        "waterfront",
        "pier",
        "landing",
        "field",
        "sports court",
        "open space",
    ]
    for term in exclude_terms:
        if term in text:
            return "exclude", f"name contains '{term}'"
    for term in include_terms:
        if term in text:
            return "include", f"name contains '{term}'"
    return "review", "no clear park/open-space keyword"


def _read_small_exceptions(config: dict[str, Any]) -> pd.DataFrame:
    path = override_path(config, "park_small_exceptions")
    required = {"municipality", "park_name", "exception_basis", "evidence"}
    exceptions = pd.read_csv(path, dtype="string")
    _require_columns(exceptions, path, required)
    exceptions["_park_key"] = _park_keys(exceptions)
    _assert_unique(exceptions, ["_park_key"], "Small-park exceptions")
    if exceptions[["exception_basis", "evidence"]].isna().any().any():
        raise ValueError(f"{path.name} contains an undocumented exception")
    return exceptions


def _apply_eligibility(
    config: dict[str, Any], inventory: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    parks = inventory.copy()
    decisions = parks["park_name"].map(_classify_park)
    parks["park_eligibility"] = [value[0] for value in decisions]
    parks["eligibility_reason"] = [value[1] for value in decisions]
    parks["manual_override"] = False
    parks["_park_key"] = _park_keys(parks)

    overrides_path = override_path(config, "park_eligibility")
    overrides = pd.read_csv(overrides_path, dtype="string")
    required = {
        "municipality",
        "park_name",
        "park_eligibility",
        "eligibility_reason",
    }
    _require_columns(overrides, overrides_path, required)
    overrides["_park_key"] = _park_keys(overrides)
    _assert_unique(overrides, ["_park_key"], "Park eligibility overrides")
    override_lookup = overrides.set_index("_park_key")
    matched = parks["_park_key"].isin(override_lookup.index)
    parks.loc[matched, "park_eligibility"] = parks.loc[matched, "_park_key"].map(
        override_lookup["park_eligibility"]
    )
    parks.loc[matched, "eligibility_reason"] = parks.loc[matched, "_park_key"].map(
        override_lookup["eligibility_reason"]
    )
    parks.loc[matched, "manual_override"] = True

    small = _read_small_exceptions(config)
    small_lookup = small.set_index("_park_key")
    small_match = parks["_park_key"].isin(small_lookup.index)
    parks.loc[small_match, "park_eligibility"] = "include"
    parks.loc[small_match, "eligibility_reason"] = parks.loc[
        small_match, "_park_key"
    ].map(lambda key: f"small-park exception: {small_lookup.at[key, 'exception_basis']}")
    parks.loc[small_match, "manual_override"] = True

    return parks[[*sorted(PARK_FIELDS), "geometry"]].sort_values("park_id").reset_index(
        drop=True
    )


def _prepare_park_inventory(
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    expected_crs = config["crs"]["projected"]
    source = gpd.read_file(data_path(config, "parks_source"))
    if source.crs is None:
        raise ValueError("Raw park polygons have no CRS")
    raw_count = len(source)
    source = _fix_geometries(source.to_crs(expected_crs))

    name_field = config["parks_source"]["name_field"]
    city_field = config["parks_source"]["city_field"]
    required = {name_field, city_field}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Raw parks are missing fields: {', '.join(sorted(missing))}")

    municipalities = set(config["parks_source"]["municipalities"])
    source = source[source[city_field].isin(municipalities)].copy()
    selected_count = len(source)
    source[name_field] = source[name_field].astype(str).str.strip()
    source = source[
        source[name_field].notna()
        & (source[name_field] != "")
        & (source[name_field].str.lower() != "nan")
    ].copy()
    source["municipality"] = source[city_field].astype(str).str.strip()
    source["park_name"] = source[name_field].astype(str).str.strip()

    dissolved = source.dissolve(
        by=["municipality", "park_name"], as_index=False, aggfunc="first"
    )
    dissolved = _fix_geometries(dissolved.to_crs(expected_crs))
    dissolved["area_ha_calc"] = dissolved.geometry.area / 10_000
    dissolved["park_key"] = (
        dissolved["municipality"].map(_make_slug)
        + "__"
        + dissolved["park_name"].map(_make_slug)
    )
    if dissolved["park_key"].duplicated().any():
        raise ValueError("Dissolved park keys are not unique")

    minimum_area = float(config["analysis"]["minimum_park_area_ha"])
    base = dissolved[dissolved["area_ha_calc"] >= minimum_area].copy()
    base = base.sort_values(["municipality", "park_name"]).reset_index(drop=True)
    base["park_id"] = [f"park_{index + 1:05d}" for index in range(len(base))]

    exceptions = _read_small_exceptions(config)
    dissolved["_park_key"] = _park_keys(dissolved)
    small = dissolved[
        (dissolved["area_ha_calc"] < minimum_area)
        & dissolved["_park_key"].isin(exceptions["_park_key"])
    ].copy()
    small = small.sort_values(["municipality", "park_name"]).reset_index(drop=True)
    small["park_id"] = [
        f"park_{len(base) + index + 1:05d}" for index in range(len(small))
    ]

    inventory = gpd.GeoDataFrame(
        pd.concat([base, small], ignore_index=True),
        geometry="geometry",
        crs=expected_crs,
    )

    # Append the separately maintained Metro Vancouver Regional Parks layer
    # after assigning IDs to municipal records. This preserves all existing
    # municipal park IDs and therefore their reviewed Google matches.
    regional_path = data_path(config, "regional_parks_source")
    regional = gpd.read_file(regional_path)
    if regional.crs is None:
        raise ValueError("Raw regional park polygons have no CRS")
    regional_raw_count = len(regional)
    regional = _fix_geometries(regional.to_crs(expected_crs))
    regional_settings = config["regional_parks_source"]
    regional_name_field = regional_settings["name_field"]
    if regional_name_field not in regional.columns:
        raise ValueError(
            f"Raw regional parks are missing field: {regional_name_field}"
        )
    regional[regional_name_field] = (
        regional[regional_name_field].astype("string").str.strip()
    )
    regional = regional[
        regional[regional_name_field].notna()
        & regional[regional_name_field].ne("")
    ].copy()
    regional["municipality"] = regional_settings["municipality_label"]
    regional["park_name"] = regional[regional_name_field]
    regional = regional.dissolve(
        by=["municipality", "park_name"], as_index=False, aggfunc="first"
    )
    regional = _fix_geometries(regional)
    regional["area_ha_calc"] = regional.geometry.area / 10_000
    regional["park_key"] = (
        regional["municipality"].map(_make_slug)
        + "__"
        + regional["park_name"].map(_make_slug)
    )
    regional = regional[
        regional["area_ha_calc"] >= minimum_area
    ].sort_values("park_name").reset_index(drop=True)
    regional["park_id"] = [
        f"park_{len(base) + len(small) + index + 1:05d}"
        for index in range(len(regional))
    ]

    # Add only reviewed Vancouver features that are absent or generically
    # represented in the older Metro merged layer. Appending them after all
    # established IDs keeps existing Google-match overrides stable.
    supplement_path = data_path(config, "vancouver_parks_supplement")
    supplement = gpd.read_file(supplement_path)
    if supplement.crs is None:
        raise ValueError("Raw Vancouver supplemental park polygons have no CRS")
    supplement_raw_count = len(supplement)
    supplement = _fix_geometries(supplement.to_crs(expected_crs))
    supplement_settings = config["vancouver_parks_supplement"]
    supplement_name_field = supplement_settings["name_field"]
    if supplement_name_field not in supplement.columns:
        raise ValueError(
            "Raw Vancouver supplemental parks are missing field: "
            f"{supplement_name_field}"
        )
    include_names = set(supplement_settings["include_names"])
    supplement[supplement_name_field] = (
        supplement[supplement_name_field].astype("string").str.strip()
    )
    supplement = supplement[
        supplement[supplement_name_field].isin(include_names)
    ].copy()
    missing_supplement_names = include_names - set(supplement[supplement_name_field])
    if missing_supplement_names:
        raise ValueError(
            "Configured Vancouver supplemental parks were not found: "
            f"{sorted(missing_supplement_names)}"
        )
    supplement["municipality"] = supplement_settings["municipality_label"]
    supplement["park_name"] = supplement[supplement_name_field]
    supplement = supplement.dissolve(
        by=["municipality", "park_name"], as_index=False, aggfunc="first"
    )
    supplement = _fix_geometries(supplement)
    supplement["area_ha_calc"] = supplement.geometry.area / 10_000
    supplement["park_key"] = (
        supplement["municipality"].map(_make_slug)
        + "__"
        + supplement["park_name"].map(_make_slug)
    )
    supplement["_park_key"] = _park_keys(supplement)
    supplement = supplement[
        (supplement["area_ha_calc"] >= minimum_area)
        | supplement["_park_key"].isin(exceptions["_park_key"])
    ].sort_values("park_name").reset_index(drop=True)
    supplement["park_id"] = [
        f"park_{len(base) + len(small) + len(regional) + index + 1:05d}"
        for index in range(len(supplement))
    ]

    boundary_root = Path(config["runtime"]["data_root"])
    boundary_path = (
        boundary_root
        / "interim"
        / "network"
        / config["network"]["slug"]
        / f"{config['network']['slug']}_network_boundary.gpkg"
    )
    boundary = gpd.read_file(boundary_path, layer="network_boundary").to_crs(expected_crs)
    boundary_geometry = boundary.geometry.union_all()
    regional_outside = regional[~regional.geometry.intersects(boundary_geometry)].copy()
    regional = regional[regional.geometry.intersects(boundary_geometry)].copy()
    regional = regional[
        ["park_id", "park_key", "municipality", "park_name", "area_ha_calc", "geometry"]
    ]
    supplement_outside = supplement[
        ~supplement.geometry.intersects(boundary_geometry)
    ].copy()
    supplement = supplement[
        supplement.geometry.intersects(boundary_geometry)
    ].copy()
    supplement = supplement[
        ["park_id", "park_key", "municipality", "park_name", "area_ha_calc", "geometry"]
    ]
    inventory = gpd.GeoDataFrame(
        pd.concat([inventory, regional, supplement], ignore_index=True),
        geometry="geometry",
        crs=expected_crs,
    )
    outside = inventory[~inventory.geometry.intersects(boundary_geometry)].copy()
    inventory = inventory[inventory.geometry.intersects(boundary_geometry)].copy()
    inventory = inventory[
        ["park_id", "park_key", "municipality", "park_name", "area_ha_calc", "geometry"]
    ]
    if inventory["park_key"].duplicated().any():
        duplicates = inventory.loc[
            inventory["park_key"].duplicated(keep=False), "park_key"
        ].tolist()
        raise ValueError(f"Prepared park keys are not unique: {duplicates[:10]}")

    prepared_exception_keys = set(_park_keys(inventory))
    missing_exception_keys = set(exceptions["_park_key"]) - prepared_exception_keys
    if missing_exception_keys:
        names = exceptions[exceptions["_park_key"].isin(missing_exception_keys)][
            ["municipality", "park_name"]
        ].to_dict("records")
        raise ValueError(f"Small-park exceptions do not match prepared parks: {names}")

    eligibility = _apply_eligibility(config, inventory)
    eligible = eligibility[eligibility["park_eligibility"] == "include"].copy()
    metrics = {
        "raw_park_records": raw_count,
        "records_after_municipality_filter": selected_count,
        "parks_after_dissolve": len(dissolved),
        "raw_regional_park_records": regional_raw_count,
        "regional_parks_in_network_context": len(regional),
        "regional_parks_outside_network_context": len(regional_outside),
        "raw_vancouver_supplement_records": supplement_raw_count,
        "vancouver_supplement_parks_in_network_context": len(supplement),
        "vancouver_supplement_parks_outside_network_context": len(supplement_outside),
        "base_parks_at_or_above_minimum": len(base),
        "small_park_exceptions": len(small),
        "parks_outside_network_boundary": len(outside),
    }
    return eligibility, eligible, metrics


def _line_endpoints(geometry: Any) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coordinates = list(geometry.coords)
        return [Point(coordinates[0]), Point(coordinates[-1])] if len(coordinates) >= 2 else []
    if geometry.geom_type == "MultiLineString":
        endpoints: list[Point] = []
        for part in geometry.geoms:
            endpoints.extend(_line_endpoints(part))
        return endpoints
    return []


def _deduplicate_candidates(
    candidates: gpd.GeoDataFrame, distance_m: float
) -> gpd.GeoDataFrame:
    kept: list[pd.Series] = []
    for _, group in candidates.groupby("park_id"):
        group = group.sort_values("dist_to_boundary_m")
        geometries: list[Any] = []
        for _, row in group.iterrows():
            if not geometries or min(row.geometry.distance(item) for item in geometries) >= distance_m:
                kept.append(row)
                geometries.append(row.geometry)
    if not kept:
        return gpd.GeoDataFrame(columns=candidates.columns, crs=candidates.crs)
    return gpd.GeoDataFrame(kept, geometry="geometry", crs=candidates.crs).reset_index(
        drop=True
    )


def _network_layers(
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    slug = config["network"]["slug"]
    network_dir = Path(config["runtime"]["data_root"]) / "interim" / "network" / slug
    crs = config["crs"]["projected"]
    edges = gpd.read_file(network_dir / f"{slug}_osm_edges.gpkg", layer="osm_edges")
    nodes = gpd.read_file(network_dir / f"{slug}_osm_nodes.gpkg", layer="osm_nodes")
    edges = edges.to_crs(crs)
    nodes = nodes.to_crs(crs)
    edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty].copy()
    nodes = nodes[nodes.geometry.notna() & ~nodes.geometry.is_empty].copy()
    return edges, nodes


def _nodes_for_join(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "osmid" in nodes.columns:
        node_id = "osmid"
    elif "node" in nodes.columns:
        node_id = "node"
    else:
        nodes = nodes.copy()
        node_id = "nearest_node"
        nodes[node_id] = nodes.index.astype(str)
    result = nodes[[node_id, "geometry"]].rename(columns={node_id: "nearest_node"})
    result["node_x"] = result.geometry.x
    result["node_y"] = result.geometry.y
    return result


def _entrance_candidates(
    parks: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    edge_index = edges.sindex
    rows: list[dict[str, Any]] = []
    parks = parks.reset_index(drop=True)
    for index, park in parks.iterrows():
        if (index + 1) % 200 == 0:
            print(f"  entrance candidates: {index + 1:,}/{len(parks):,} parks")
        boundary = park.geometry.boundary
        if boundary is None or boundary.is_empty:
            continue
        possible = list(edge_index.query(boundary.buffer(buffer_m), predicate="intersects"))
        for _, edge in edges.iloc[possible].iterrows():
            geometry = edge.geometry
            if geometry is None or geometry.is_empty:
                continue
            try:
                point_on_edge, point_on_boundary = nearest_points(geometry, boundary)
                distance = point_on_edge.distance(point_on_boundary)
            except Exception:
                continue
            if distance > buffer_m:
                continue
            crosses = geometry.intersects(boundary)
            endpoint_near = any(
                endpoint.distance(boundary) <= buffer_m
                for endpoint in _line_endpoints(geometry)
            )
            if not crosses and not endpoint_near:
                continue
            rows.append(
                {
                    "park_id": park["park_id"],
                    "park_key": park["park_key"],
                    "municipality": park["municipality"],
                    "park_name": park["park_name"],
                    "dist_to_boundary_m": distance,
                    "entrance_rule": (
                        "edge_touches_or_crosses_boundary"
                        if crosses
                        else "edge_endpoint_within_boundary_buffer"
                    ),
                    "geometry": point_on_boundary,
                }
            )
    if not rows:
        raise ValueError("No park entrance candidates were created")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=parks.crs)


def _snap_candidates(
    candidates: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    cluster_m: float,
) -> gpd.GeoDataFrame:
    deduplicated = _deduplicate_candidates(candidates, cluster_m)
    deduplicated["entrance_temp_id"] = [
        f"entrance_tmp_{index + 1:07d}" for index in range(len(deduplicated))
    ]
    snapped = gpd.sjoin_nearest(
        deduplicated, _nodes_for_join(nodes), how="left", distance_col="snap_dist_m"
    )
    snapped = snapped.sort_values(["entrance_temp_id", "snap_dist_m"])
    snapped = snapped.drop_duplicates("entrance_temp_id").copy()
    snapped["candidate_x"] = snapped.geometry.x
    snapped["candidate_y"] = snapped.geometry.y
    snapped = snapped.sort_values(
        ["park_id", "nearest_node", "dist_to_boundary_m", "snap_dist_m"]
    ).drop_duplicates(["park_id", "nearest_node"])
    snapped["entrance_id"] = None
    snapped["entrance_method"] = "strict_boundary_or_endpoint"
    snapped["fallback_threshold_m"] = np.nan
    return gpd.GeoDataFrame(snapped, geometry="geometry", crs=candidates.crs)


def _fallback_entrances(
    parks: gpd.GeoDataFrame,
    strict: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    first_m: float,
    maximum_m: float,
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    edge_index = edges.sindex
    strict_ids = set(strict["park_id"].astype("string"))
    missing_parks = parks[~parks["park_id"].astype("string").isin(strict_ids)]
    fallback_rows: list[dict[str, Any]] = []
    isolated_rows: list[dict[str, Any]] = []
    for _, park in missing_parks.iterrows():
        boundary = park.geometry.boundary
        possible = list(
            edge_index.query(boundary.buffer(maximum_m), predicate="intersects")
        )
        best_distance: float | None = None
        best_boundary_point = None
        for _, edge in edges.iloc[possible].iterrows():
            try:
                point_on_edge, point_on_boundary = nearest_points(edge.geometry, boundary)
                distance = point_on_edge.distance(point_on_boundary)
            except Exception:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_boundary_point = point_on_boundary
        if best_distance is None or best_boundary_point is None or best_distance > maximum_m:
            isolated_rows.append(
                {
                    "park_id": park["park_id"],
                    "park_key": park["park_key"],
                    "municipality": park["municipality"],
                    "park_name": park["park_name"],
                    "reason": f"no_osm_edge_within_{int(maximum_m)}m",
                    "nearest_edge_dist_m": best_distance,
                    "geometry": park.geometry,
                }
            )
            continue
        fallback_rows.append(
            {
                "entrance_id": None,
                "entrance_method": "fallback_nearest_edge",
                "entrance_rule": "zero_entrance_fallback_nearest_edge",
                "fallback_threshold_m": first_m if best_distance <= first_m else maximum_m,
                "park_id": park["park_id"],
                "park_key": park["park_key"],
                "municipality": park["municipality"],
                "park_name": park["park_name"],
                "nearest_node": None,
                "candidate_x": best_boundary_point.x,
                "candidate_y": best_boundary_point.y,
                "node_x": None,
                "node_y": None,
                "dist_to_boundary_m": best_distance,
                "snap_dist_m": None,
                "geometry": best_boundary_point,
            }
        )

    columns = [
        "entrance_id",
        "entrance_method",
        "entrance_rule",
        "fallback_threshold_m",
        "park_id",
        "park_key",
        "municipality",
        "park_name",
        "nearest_node",
        "candidate_x",
        "candidate_y",
        "node_x",
        "node_y",
        "dist_to_boundary_m",
        "snap_dist_m",
        "geometry",
    ]
    if not fallback_rows:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=parks.crs), isolated_rows

    fallback = gpd.GeoDataFrame(fallback_rows, geometry="geometry", crs=parks.crs)
    fallback["entrance_temp_id"] = [
        f"fallback_tmp_{index + 1:07d}" for index in range(len(fallback))
    ]
    snapped = gpd.sjoin_nearest(
        fallback, _nodes_for_join(nodes), how="left", distance_col="snap_dist_m"
    )
    snapped = snapped.sort_values(["entrance_temp_id", "snap_dist_m"])
    snapped = snapped.drop_duplicates("entrance_temp_id").copy()
    for field in ["nearest_node", "node_x", "node_y"]:
        right = f"{field}_right"
        left = f"{field}_left"
        if right in snapped:
            snapped[field] = snapped[right]
        elif field not in snapped and left in snapped:
            snapped[field] = snapped[left]
    return gpd.GeoDataFrame(snapped[columns], geometry="geometry", crs=parks.crs), isolated_rows


def _apply_entrance_overrides(
    config: dict[str, Any],
    entrances: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Apply explicit add/remove/replace records; the current table is empty."""
    path = override_path(config, "park_entrances")
    overrides = pd.read_csv(path, dtype="string")
    if overrides.empty:
        return entrances

    result = entrances.copy()
    parks = parks.copy()
    parks["_park_key"] = _park_keys(parks)
    nodes_for_join = _nodes_for_join(nodes).copy()
    nodes_for_join["_node_key"] = _normalise_ids(nodes_for_join["nearest_node"])
    node_lookup = nodes_for_join.set_index("_node_key")

    for _, override in overrides.iterrows():
        key = (
            str(override["municipality"]).strip().casefold()
            + "\x1f"
            + str(override["park_name"]).strip().casefold()
        )
        park_match = parks[parks["_park_key"] == key]
        if len(park_match) != 1:
            raise ValueError(f"Entrance override has no unique park match: {override['park_name']}")
        park = park_match.iloc[0]
        action = str(override["action"]).strip().casefold()
        node_value = override.get("nearest_node")
        node_key = None if pd.isna(node_value) or not str(node_value).strip() else _normalise_ids(pd.Series([node_value])).iloc[0]

        if action in {"remove", "replace"}:
            park_mask = result["park_id"].astype("string") == str(park["park_id"])
            if action == "remove" and node_key is not None:
                park_mask &= _normalise_ids(result["nearest_node"]) == node_key
            result = result[~park_mask].copy()
        if action == "remove":
            continue

        longitude = pd.to_numeric(pd.Series([override.get("longitude")]), errors="coerce").iloc[0]
        latitude = pd.to_numeric(pd.Series([override.get("latitude")]), errors="coerce").iloc[0]
        point = None
        if pd.notna(longitude) and pd.notna(latitude):
            point = gpd.GeoSeries([Point(longitude, latitude)], crs=config["crs"]["geographic"]).to_crs(
                config["crs"]["projected"]
            ).iloc[0]
        if node_key is None:
            if point is None:
                raise ValueError(f"Entrance {action} requires coordinates or nearest_node")
            nearest = gpd.sjoin_nearest(
                gpd.GeoDataFrame({"geometry": [point]}, crs=config["crs"]["projected"]),
                nodes_for_join,
                how="left",
                distance_col="snap_dist_m",
            ).iloc[0]
            node_key = _normalise_ids(pd.Series([nearest["nearest_node"]])).iloc[0]
        if node_key not in node_lookup.index:
            raise ValueError(f"Entrance override references unknown node {node_key}")
        node = node_lookup.loc[node_key]
        if isinstance(node, pd.DataFrame):
            node = node.iloc[0]
        if point is None:
            point = node.geometry
        manual = gpd.GeoDataFrame(
            [
                {
                    "entrance_id": None,
                    "entrance_method": "manual_override",
                    "entrance_rule": f"manual_{action}",
                    "fallback_threshold_m": np.nan,
                    "park_id": park["park_id"],
                    "park_key": park["park_key"],
                    "municipality": park["municipality"],
                    "park_name": park["park_name"],
                    "nearest_node": node["nearest_node"],
                    "candidate_x": point.x,
                    "candidate_y": point.y,
                    "node_x": node["node_x"],
                    "node_y": node["node_y"],
                    "dist_to_boundary_m": point.distance(park.geometry.boundary),
                    "snap_dist_m": point.distance(node.geometry),
                    "geometry": point,
                }
            ],
            geometry="geometry",
            crs=config["crs"]["projected"],
        )
        result = gpd.GeoDataFrame(
            pd.concat([result, manual], ignore_index=True),
            geometry="geometry",
            crs=config["crs"]["projected"],
        )
        result["_node_key"] = _normalise_ids(result["nearest_node"])
        result = result.drop_duplicates(["park_id", "_node_key"], keep="last").drop(
            columns="_node_key"
        )
    return result


def _prepare_entrances(
    config: dict[str, Any], eligible: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    analysis = config["analysis"]
    edges, nodes = _network_layers(config)
    candidates = _entrance_candidates(
        eligible, edges, float(analysis["entrance_buffer_m"])
    )
    strict_full = _snap_candidates(
        candidates, nodes, float(analysis["entrance_cluster_m"])
    )
    output_columns = [
        "entrance_id",
        "entrance_method",
        "entrance_rule",
        "fallback_threshold_m",
        "park_id",
        "park_key",
        "municipality",
        "park_name",
        "nearest_node",
        "candidate_x",
        "candidate_y",
        "node_x",
        "node_y",
        "dist_to_boundary_m",
        "snap_dist_m",
        "geometry",
    ]
    strict = strict_full[output_columns].copy()
    fallback, provisional_isolated = _fallback_entrances(
        eligible,
        strict,
        edges,
        nodes,
        float(analysis["fallback_entrance_first_m"]),
        float(analysis["fallback_entrance_max_m"]),
    )
    entrances = gpd.GeoDataFrame(
        pd.concat([strict, fallback], ignore_index=True),
        geometry="geometry",
        crs=eligible.crs,
    )
    entrances = _apply_entrance_overrides(config, entrances, eligible, nodes)

    parks_with_entrances = set(entrances["park_id"].astype("string"))
    isolated_parks = eligible[
        ~eligible["park_id"].astype("string").isin(parks_with_entrances)
    ].copy()
    reason_lookup = {str(row["park_id"]): row for row in provisional_isolated}
    isolated_rows = []
    for _, park in isolated_parks.iterrows():
        prior = reason_lookup.get(str(park["park_id"]), {})
        isolated_rows.append(
            {
                "park_id": park["park_id"],
                "park_key": park["park_key"],
                "municipality": park["municipality"],
                "park_name": park["park_name"],
                "reason": prior.get("reason", "manual_override_removed_all_entrances"),
                "nearest_edge_dist_m": prior.get("nearest_edge_dist_m"),
                "geometry": park.geometry,
            }
        )
    isolated = gpd.GeoDataFrame(isolated_rows, geometry="geometry", crs=eligible.crs)
    final_parks = eligible[
        ~eligible["park_id"].astype("string").isin(isolated_parks["park_id"].astype("string"))
    ].copy()
    entrances = entrances.sort_values(
        ["park_id", "entrance_method", "nearest_node"]
    ).reset_index(drop=True)
    entrances["entrance_id"] = [
        f"entrance_{index + 1:07d}" for index in range(len(entrances))
    ]
    metrics = {
        "raw_entrance_candidates": len(candidates),
        "strict_entrances": int((entrances["entrance_method"] == "strict_boundary_or_endpoint").sum()),
        "fallback_entrances": int((entrances["entrance_method"] == "fallback_nearest_edge").sum()),
        "manual_entrances": int((entrances["entrance_method"] == "manual_override").sum()),
        "isolated_parks_removed": len(isolated),
    }
    return final_parks, entrances[output_columns], candidates, isolated, metrics


def _write_build_artifacts(
    paths: dict[str, Path],
    eligibility: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    entrances: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    isolated: gpd.GeoDataFrame,
) -> None:
    paths["parks_gpkg"].parent.mkdir(parents=True, exist_ok=True)
    paths["candidates_gpkg"].parent.mkdir(parents=True, exist_ok=True)
    parks.to_file(paths["parks_gpkg"], layer="parks_cleaned", driver="GPKG")
    parks.drop(columns="geometry").to_csv(paths["parks_csv"], index=False)
    eligibility.to_file(
        paths["eligibility_gpkg"], layer="parks_with_eligibility", driver="GPKG"
    )
    eligibility.drop(columns="geometry").to_csv(paths["eligibility_csv"], index=False)
    entrances.to_file(
        paths["entrances_gpkg"], layer="park_entrances", driver="GPKG"
    )
    entrances.drop(columns="geometry").to_csv(paths["entrances_csv"], index=False)
    candidates.to_file(
        paths["candidates_gpkg"], layer="entrance_candidates", driver="GPKG"
    )
    isolated.to_file(
        paths["isolated_gpkg"],
        layer="zero_entrance_isolated_or_island_parks",
        driver="GPKG",
    )


def _semantic_rebuild_comparison(
    reference: dict[str, Path], rebuilt: dict[str, Path]
) -> dict[str, Any]:
    """Compare a no-exception rebuild with the reviewed legacy reference."""
    reference_parks = gpd.read_file(reference["parks_gpkg"], layer="parks_cleaned")
    rebuilt_parks = gpd.read_file(rebuilt["parks_gpkg"], layer="parks_cleaned")
    reference_eligibility = gpd.read_file(
        reference["eligibility_gpkg"], layer="parks_with_eligibility"
    )
    rebuilt_eligibility = gpd.read_file(
        rebuilt["eligibility_gpkg"], layer="parks_with_eligibility"
    )
    reference_entrances = gpd.read_file(
        reference["entrances_gpkg"], layer="park_entrances"
    )
    rebuilt_entrances = gpd.read_file(rebuilt["entrances_gpkg"], layer="park_entrances")

    reference_park_ids = set(reference_parks["park_id"].astype("string"))
    rebuilt_park_ids = set(rebuilt_parks["park_id"].astype("string"))
    reference_inventory_ids = set(reference_eligibility["park_id"].astype("string"))
    rebuilt_inventory_ids = set(rebuilt_eligibility["park_id"].astype("string"))
    reference_pairs = set(
        zip(
            reference_entrances["park_id"].astype("string"),
            _normalise_ids(reference_entrances["nearest_node"]),
            strict=True,
        )
    )
    rebuilt_pairs = set(
        zip(
            rebuilt_entrances["park_id"].astype("string"),
            _normalise_ids(rebuilt_entrances["nearest_node"]),
            strict=True,
        )
    )
    common_inventory = reference_eligibility.set_index("park_id").join(
        rebuilt_eligibility.set_index("park_id"),
        lsuffix="_reference",
        rsuffix="_rebuilt",
        how="left",
    )
    if common_inventory["park_key_rebuilt"].isna().any():
        raise ValueError("Rebuild removed a legacy park inventory ID")
    if not (
        common_inventory["park_key_reference"].astype("string")
        == common_inventory["park_key_rebuilt"].astype("string")
    ).all():
        raise ValueError("Rebuild changed a legacy park ID-to-key assignment")
    if not np.allclose(
        pd.to_numeric(common_inventory["area_ha_calc_reference"]),
        pd.to_numeric(common_inventory["area_ha_calc_rebuilt"]),
    ):
        raise ValueError("Rebuild changed a legacy park area")
    if not reference_park_ids <= rebuilt_park_ids:
        raise ValueError("Rebuild removed a previously included park")
    if not reference_pairs <= rebuilt_pairs:
        raise ValueError("Rebuild removed a previously validated park-node entrance")

    return {
        "reference_final_parks": len(reference_parks),
        "reference_eligibility_rows": len(reference_eligibility),
        "reference_entrances": len(reference_entrances),
        "park_ids_added_vs_reference": len(rebuilt_park_ids - reference_park_ids),
        "park_ids_removed_vs_reference": len(reference_park_ids - rebuilt_park_ids),
        "inventory_ids_added_vs_reference": len(
            rebuilt_inventory_ids - reference_inventory_ids
        ),
        "inventory_ids_removed_vs_reference": len(
            reference_inventory_ids - rebuilt_inventory_ids
        ),
        "entrance_pairs_added_vs_reference": len(rebuilt_pairs - reference_pairs),
        "entrance_pairs_removed_vs_reference": len(reference_pairs - rebuilt_pairs),
    }


def build_parks(
    config: dict[str, Any], final_paths: dict[str, Path]
) -> dict[str, Any]:
    eligibility, eligible, park_metrics = _prepare_park_inventory(config)
    print(
        f"Prepared eligibility inventory: {len(eligibility):,} "
        f"({park_metrics['small_park_exceptions']:,} small exceptions)"
    )
    parks, entrances, candidates, isolated, entrance_metrics = _prepare_entrances(
        config, eligible
    )
    print(f"Prepared final parks: {len(parks):,}")
    print(f"Prepared entrances: {len(entrances):,}")

    workspace_root = workspace_data_root(config)
    workspace_root.mkdir(parents=True, exist_ok=True)
    reference_paths = park_artifact_paths(config, Path(config["runtime"]["data_root"]))
    with inherited_temp_directory(
        workspace_root, f"{config['network']['slug']}_02_"
    ) as temp_dir:
        temp_paths = park_artifact_paths(config, temp_dir)
        _write_build_artifacts(
            temp_paths, eligibility, parks, entrances, candidates, isolated
        )
        validated = validate_parks(config, temp_paths)
        comparison = _semantic_rebuild_comparison(reference_paths, temp_paths)
        for key, target in final_paths.items():
            source = temp_paths[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    return park_metrics | entrance_metrics | comparison | validated


def _spatial_info(
    path: Path,
    expected_crs: str,
    required_fields: set[str],
) -> int:
    info = pyogrio.read_info(path, force_feature_count=True)
    missing = required_fields - set(info["fields"])
    if missing:
        raise ValueError(f"{path.name} is missing fields: {', '.join(sorted(missing))}")

    features = int(info["features"])
    if features < 1:
        raise ValueError(f"{path.name} contains no features")

    crs = str(info["crs"])
    if crs.upper() != expected_crs.upper():
        raise ValueError(f"{path.name} CRS is {crs}; expected {expected_crs}")
    return features


def _require_columns(frame: pd.DataFrame, path: Path, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")


def _normalise_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def _normalise_ids(series: pd.Series) -> pd.Series:
    """Normalize IDs read differently by CSV and GeoPackage drivers."""
    values = series.astype("string").str.strip()
    return values.str.replace(r"\.0$", "", regex=True)


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicate = frame.duplicated(columns, keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, columns].head(3).to_dict("records")
        raise ValueError(f"{label} has duplicate keys {columns}: {sample}")


def _validate_override_decisions(
    config: dict[str, Any],
    eligibility: pd.DataFrame,
) -> dict[str, int]:
    path = override_path(config, "park_eligibility")
    required = {
        "municipality",
        "park_name",
        "park_eligibility",
        "eligibility_reason",
    }
    overrides = pd.read_csv(path, dtype="string")
    _require_columns(overrides, path, required)

    allowed = {"include", "exclude", "review"}
    decisions = set(_normalise_key(overrides["park_eligibility"]).dropna())
    if not decisions <= allowed:
        raise ValueError(f"{path.name} contains invalid eligibility values: {decisions - allowed}")

    for frame in (overrides, eligibility):
        frame["_municipality_key"] = _normalise_key(frame["municipality"])
        frame["_park_name_key"] = _normalise_key(frame["park_name"])

    keys = ["_municipality_key", "_park_name_key"]
    _assert_unique(overrides, keys, "Park eligibility overrides")
    _assert_unique(eligibility, keys, "Park eligibility inventory")

    matched = overrides.merge(
        eligibility[
            keys
            + [
                "park_eligibility",
                "eligibility_reason",
                "manual_override",
            ]
        ],
        on=keys,
        how="left",
        suffixes=("_override", "_artifact"),
        indicator=True,
    )
    missing = matched[matched["_merge"] != "both"]
    if not missing.empty:
        names = missing[["municipality", "park_name"]].head(5).to_dict("records")
        raise ValueError(f"Park eligibility overrides do not match an artifact row: {names}")

    same_decision = (
        _normalise_key(matched["park_eligibility_override"])
        == _normalise_key(matched["park_eligibility_artifact"])
    )
    same_reason = (
        matched["eligibility_reason_override"].astype("string").str.strip()
        == matched["eligibility_reason_artifact"].astype("string").str.strip()
    )
    manual = (
        matched["manual_override"].astype("string").str.strip().str.casefold()
        == "true"
    )
    if not (same_decision & same_reason & manual).all():
        bad = matched.loc[
            ~(same_decision & same_reason & manual), ["municipality", "park_name"]
        ].head(5)
        raise ValueError(
            "Reviewed park decisions are not fully represented in the artifacts: "
            f"{bad.to_dict('records')}"
        )

    return {
        "eligibility_override_rows": len(overrides),
        "manual_include_rows": int(
            (_normalise_key(overrides["park_eligibility"]) == "include").sum()
        ),
        "manual_exclude_rows": int(
            (_normalise_key(overrides["park_eligibility"]) == "exclude").sum()
        ),
    }


def _validate_small_park_exceptions(
    config: dict[str, Any], eligibility: pd.DataFrame
) -> int:
    exceptions = _read_small_exceptions(config)
    inventory = eligibility.copy()
    inventory["_park_key"] = _park_keys(inventory)
    joined = exceptions.merge(
        inventory[
            [
                "_park_key",
                "area_ha_calc",
                "park_eligibility",
                "manual_override",
            ]
        ],
        on="_park_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        missing = joined.loc[
            joined["_merge"] != "both", ["municipality", "park_name"]
        ].to_dict("records")
        raise ValueError(f"Small-park exceptions are absent from the inventory: {missing}")
    area = pd.to_numeric(joined["area_ha_calc"], errors="coerce")
    included = _normalise_key(joined["park_eligibility"]) == "include"
    manual = _normalise_key(joined["manual_override"]) == "true"
    if not (
        (area < float(config["analysis"]["minimum_park_area_ha"]))
        & included
        & manual
    ).all():
        raise ValueError("A small-park exception is not retained as a manual include")
    return len(exceptions)


def _validate_entrance_overrides(config: dict[str, Any]) -> int:
    path = override_path(config, "park_entrances")
    required = {
        "municipality",
        "park_name",
        "action",
        "longitude",
        "latitude",
        "nearest_node",
        "reason",
    }
    overrides = pd.read_csv(path, dtype="string")
    _require_columns(overrides, path, required)
    if not overrides.empty:
        allowed = {"add", "remove", "replace"}
        actions = set(_normalise_key(overrides["action"]).dropna())
        if not actions <= allowed:
            raise ValueError(f"{path.name} contains invalid actions: {actions - allowed}")
    return len(overrides)


def validate_parks(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Validate relationships between every reviewed Stage 02 artifact."""
    core = [
        "parks_gpkg",
        "parks_csv",
        "eligibility_gpkg",
        "eligibility_csv",
        "entrances_gpkg",
        "entrances_csv",
    ]
    missing = [paths[key] for key in core if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 02 artifacts: " + ", ".join(path.name for path in missing)
        )

    expected_crs = config["crs"]["projected"]
    park_gpkg_count = _spatial_info(paths["parks_gpkg"], expected_crs, PARK_FIELDS)
    eligibility_gpkg_count = _spatial_info(
        paths["eligibility_gpkg"], expected_crs, PARK_FIELDS
    )
    entrance_gpkg_count = _spatial_info(
        paths["entrances_gpkg"], expected_crs, ENTRANCE_FIELDS
    )

    parks = pd.read_csv(paths["parks_csv"], dtype="string")
    eligibility = pd.read_csv(paths["eligibility_csv"], dtype="string")
    entrances = pd.read_csv(paths["entrances_csv"], dtype="string")
    _require_columns(parks, paths["parks_csv"], PARK_FIELDS)
    _require_columns(eligibility, paths["eligibility_csv"], PARK_FIELDS)
    _require_columns(entrances, paths["entrances_csv"], ENTRANCE_FIELDS)

    expected_counts = {
        "parks": (len(parks), park_gpkg_count),
        "eligibility": (len(eligibility), eligibility_gpkg_count),
        "entrances": (len(entrances), entrance_gpkg_count),
    }
    mismatches = {
        name: counts for name, counts in expected_counts.items() if counts[0] != counts[1]
    }
    if mismatches:
        raise ValueError(f"CSV and GeoPackage feature counts differ: {mismatches}")

    _assert_unique(parks, ["park_id"], "Final parks")
    _assert_unique(parks, ["park_key"], "Final parks")
    _assert_unique(eligibility, ["park_id"], "Eligibility inventory")
    _assert_unique(eligibility, ["park_key"], "Eligibility inventory")
    _assert_unique(entrances, ["entrance_id"], "Park entrances")
    _assert_unique(entrances, ["park_id", "nearest_node"], "Park entrances")

    eligibility_values = set(_normalise_key(eligibility["park_eligibility"]).dropna())
    allowed = {"include", "exclude", "review"}
    if not eligibility_values <= allowed:
        raise ValueError(f"Eligibility artifacts contain invalid values: {eligibility_values - allowed}")

    inventory_by_id = eligibility.set_index("park_id")
    final_ids = set(parks["park_id"])
    inventory_ids = set(eligibility["park_id"])
    entrance_park_ids = set(entrances["park_id"])
    if not final_ids <= inventory_ids:
        raise ValueError("Final parks contain IDs absent from the eligibility inventory")
    if entrance_park_ids != final_ids:
        missing_entrances = final_ids - entrance_park_ids
        unknown_parks = entrance_park_ids - final_ids
        raise ValueError(
            "Entrance-to-park coverage mismatch: "
            f"{len(missing_entrances)} parks without entrances; "
            f"{len(unknown_parks)} unknown park IDs"
        )

    final_inventory = inventory_by_id.loc[list(final_ids)]
    if not (_normalise_key(final_inventory["park_eligibility"]) == "include").all():
        raise ValueError("Final park layer contains an excluded or unresolved park")

    # Check that CSV and GeoPackage identity fields describe the same records.
    park_spatial = pyogrio.read_dataframe(
        paths["parks_gpkg"], columns=["park_id", "park_key"], read_geometry=False
    )
    entrance_spatial = pyogrio.read_dataframe(
        paths["entrances_gpkg"],
        columns=["entrance_id", "park_id", "nearest_node"],
        read_geometry=False,
    )
    if set(park_spatial["park_id"].astype("string")) != final_ids:
        raise ValueError("Park CSV and GeoPackage contain different park IDs")
    if set(entrance_spatial["entrance_id"].astype("string")) != set(entrances["entrance_id"]):
        raise ValueError("Entrance CSV and GeoPackage contain different entrance IDs")

    csv_pairs = set(
        zip(entrances["park_id"], _normalise_ids(entrances["nearest_node"]), strict=True)
    )
    gpkg_pairs = set(
        zip(
            entrance_spatial["park_id"].astype("string"),
            _normalise_ids(entrance_spatial["nearest_node"]),
            strict=True,
        )
    )
    if csv_pairs != gpkg_pairs:
        raise ValueError("Entrance CSV and GeoPackage contain different park-node pairs")

    override_metrics = _validate_override_decisions(config, eligibility)
    small_exception_rows = _validate_small_park_exceptions(config, eligibility)
    manual_entrance_rows = _validate_entrance_overrides(config)

    isolated_count = 0
    if paths["isolated_gpkg"].exists():
        isolated_count = _spatial_info(
            paths["isolated_gpkg"],
            expected_crs,
            {"park_id", "park_name", "municipality", "reason"},
        )
        isolated = pyogrio.read_dataframe(
            paths["isolated_gpkg"], columns=["park_id"], read_geometry=False
        )
        if set(isolated["park_id"].astype("string")) & final_ids:
            raise ValueError("An isolated zero-entrance park remains in the final park layer")

    methods = entrances["entrance_method"].value_counts()
    return {
        "eligibility_inventory_rows": len(eligibility),
        "included_before_isolated_filter": int(
            (_normalise_key(eligibility["park_eligibility"]) == "include").sum()
        ),
        "excluded_rows": int(
            (_normalise_key(eligibility["park_eligibility"]) == "exclude").sum()
        ),
        "review_rows": int(
            (_normalise_key(eligibility["park_eligibility"]) == "review").sum()
        ),
        "final_parks": len(parks),
        "final_entrances": len(entrances),
        "strict_entrances": int(methods.get("strict_boundary_or_endpoint", 0)),
        "fallback_entrances": int(methods.get("fallback_nearest_edge", 0)),
        "isolated_parks_removed": isolated_count,
        "manual_entrance_override_rows": manual_entrance_rows,
        "small_park_exception_rows": small_exception_rows,
        "crs": expected_crs,
        **override_metrics,
    }


def write_summary(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    status: str,
) -> Path:
    output_dir = project_root() / "outputs" / config["network"]["slug"] / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{config['network']['slug']}_02_parks_summary.csv"
    row = {
        "network_slug": config["network"]["slug"],
        "status": status,
        **metrics,
        **{f"{key}_path": str(value) for key, value in paths.items()},
    }
    pd.DataFrame([row]).to_csv(summary_path, index=False)

    if status == "rebuilt":
        comparison_keys = [
            "reference_final_parks",
            "reference_eligibility_rows",
            "reference_entrances",
            "park_ids_added_vs_reference",
            "park_ids_removed_vs_reference",
            "inventory_ids_added_vs_reference",
            "inventory_ids_removed_vs_reference",
            "entrance_pairs_added_vs_reference",
            "entrance_pairs_removed_vs_reference",
        ]
        comparison_path = (
            output_dir / f"{config['network']['slug']}_02_rebuild_comparison.csv"
        )
        comparison = {key: metrics[key] for key in comparison_keys}
        pd.DataFrame([comparison]).to_csv(comparison_path, index=False)

    return summary_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = (
        park_artifact_paths(config, workspace_data_root(config))
        if force
        else park_artifact_paths(config)
    )

    print("\nStage 02: shared parks and entrances")
    print(f"Artifact folder: {paths['parks_gpkg'].parent}")

    if force:
        print("Force enabled: rebuilding atomically from raw parks and network artifacts.")
        try:
            metrics = build_parks(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Park/entrance rebuild failed: {error}")
            return 1
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing artifacts and tracked manual decisions.")
        try:
            metrics = validate_parks(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Park/entrance validation failed: {error}")
            return 1
        status = "reused"

    metrics = validate_parks(config, paths) | metrics
    summary_path = write_summary(config, paths, metrics, status)
    print(f"Validated final parks: {metrics['final_parks']:,}")
    print(f"Validated entrances: {metrics['final_entrances']:,}")
    print(f"Tracked eligibility decisions: {metrics['eligibility_override_rows']:,}")
    print(f"Isolated parks removed: {metrics['isolated_parks_removed']:,}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", default="metro", help="Configured area (parks are shared)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this repository's Stage 02 artifacts",
    )
    args = parser.parse_args()

    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
