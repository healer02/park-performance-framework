"""Prepare or validate 400 m DB-to-park pedestrian-network reachability."""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pyogrio

from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    output_root,
    workspace_data_root,
)


def clean_id_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def normalise_node_id(value: Any, integer_nodes: bool) -> int | str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if integer_nodes:
        try:
            return int(text)
        except ValueError:
            return None
    return text


def network_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["network"]["slug"]
    base = Path(config["runtime"]["data_root"]) / "interim" / "network" / slug
    return {
        "graphml": base / f"{slug}_walk.graphml",
        "graph_cache": base / f"{slug}_walk_graph.pkl",
        "nodes": base / f"{slug}_osm_nodes.gpkg",
        "undirected_cache": (
            workspace_data_root(config)
            / "interim"
            / "network"
            / slug
            / f"{slug}_walk_undirected.pkl"
        ),
    }


def reachability_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    root = data_root or workspace_data_root(config)
    base = root / "interim" / "reach" / slug
    return {
        "snapped_gpkg": base / f"{slug}_db_points_snapped.gpkg",
        "snapped_csv": base / f"{slug}_db_points_snapped.csv",
        "pairs_csv": base / f"{slug}_db_park_reachability.csv",
        "db_summary_csv": base / f"{slug}_db_reachability_summary.csv",
        "db_summary_gpkg": base / f"{slug}_db_reachability_summary.gpkg",
        "da_summary_csv": base / f"{slug}_da_reachability_summary.csv",
    }


def population_points_path(config: dict[str, Any]) -> Path:
    slug = config["analysis_area"]["slug"]
    return (
        workspace_data_root(config)
        / "interim"
        / "census"
        / slug
        / f"{slug}_db_points.gpkg"
    )


def _required_csv(path: Path, required: set[str], **kwargs: Any) -> pd.DataFrame:
    frame = pd.read_csv(path, **kwargs)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    return frame


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{label} contains duplicate keys: {columns}")


def validate_reachability(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 04 artifacts: " + ", ".join(path.name for path in missing)
        )

    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    threshold = float(config["analysis"]["network_distance_m"])
    expected_crs = config["crs"]["projected"]
    id_types = {db_id: "string", da_id: "string"}

    snapped_required = {
        db_id,
        da_id,
        "db_pop",
        "db_nearest_node",
        "db_snap_dist_m",
        "point_method",
    }
    pair_required = {
        db_id,
        da_id,
        "db_pop",
        "db_nearest_node",
        "db_snap_dist_m",
        "park_id",
        "park_name",
        "park_municipality",
        "nearest_entrance_id",
        "nearest_entrance_node",
        "network_dist_m",
        "total_access_dist_m",
    }
    db_required = {
        db_id,
        da_id,
        "db_pop",
        "db_nearest_node",
        "db_snap_dist_m",
        "reachable_park_count",
        "nearest_park_network_distance_m",
        "has_park_within_threshold",
        "population_with_access",
    }
    da_required = {
        da_id,
        "da_pop",
        "db_count",
        "dbs_with_access",
        "population_with_access",
        "mean_reachable_park_count",
        "park_access_pop_coverage",
    }

    snapped = _required_csv(paths["snapped_csv"], snapped_required, dtype=id_types)
    pairs = _required_csv(paths["pairs_csv"], pair_required, dtype=id_types | {"park_id": "string"})
    db_summary = _required_csv(paths["db_summary_csv"], db_required, dtype=id_types)
    da_summary = _required_csv(paths["da_summary_csv"], da_required, dtype={da_id: "string"})

    _assert_unique(snapped, [db_id], "Snapped DB points")
    _assert_unique(pairs, [db_id, "park_id"], "DB-park reachability")
    _assert_unique(db_summary, [db_id], "DB reachability summary")
    _assert_unique(da_summary, [da_id], "DA reachability summary")

    source_path = population_points_path(config)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Stage 03 DB points: {source_path}")
    source = pyogrio.read_dataframe(
        source_path, columns=[db_id, da_id, "db_pop"], read_geometry=False
    )
    source[db_id] = clean_id_series(source[db_id])
    source[da_id] = clean_id_series(source[da_id])
    source_ids = set(source[db_id])
    if set(snapped[db_id]) != source_ids or set(db_summary[db_id]) != source_ids:
        raise ValueError("Stage 04 does not contain exactly the Stage 03 DB IDs")
    if not set(pairs[db_id]) <= source_ids:
        raise ValueError("A reachability pair references an unknown DB")

    info = pyogrio.read_info(paths["snapped_gpkg"], force_feature_count=True)
    if int(info["features"]) != len(snapped) or str(info["crs"]).upper() != expected_crs.upper():
        raise ValueError("Snapped DB GeoPackage count or CRS differs from its CSV")
    info = pyogrio.read_info(paths["db_summary_gpkg"], force_feature_count=True)
    if int(info["features"]) != len(db_summary) or str(info["crs"]).upper() != expected_crs.upper():
        raise ValueError("DB summary GeoPackage count or CRS differs from its CSV")

    network_distance = pd.to_numeric(pairs["network_dist_m"], errors="coerce")
    snap_distance = pd.to_numeric(pairs["db_snap_dist_m"], errors="coerce")
    total_distance = pd.to_numeric(pairs["total_access_dist_m"], errors="coerce")
    if network_distance.isna().any() or (network_distance < 0).any():
        raise ValueError("Reachability pairs contain invalid network distances")
    if (network_distance > threshold + 1e-8).any():
        raise ValueError("A reachability pair exceeds the configured network threshold")
    if not np.allclose(total_distance, network_distance + snap_distance, atol=1e-6):
        raise ValueError("Total access distance is not network distance plus DB snap distance")

    entrances = pyogrio.read_dataframe(
        data_path(config, "entrances"),
        columns=["entrance_id", "park_id"],
        read_geometry=False,
    )
    valid_pairs = set(
        zip(
            entrances["entrance_id"].astype("string"),
            entrances["park_id"].astype("string"),
            strict=True,
        )
    )
    used_pairs = set(
        zip(
            pairs["nearest_entrance_id"].astype("string"),
            pairs["park_id"].astype("string"),
            strict=True,
        )
    )
    if not used_pairs <= valid_pairs:
        raise ValueError("A reachability row references an invalid entrance-park pair")

    actual_counts = pairs.groupby(db_id)["park_id"].nunique()
    reported_counts = (
        pd.to_numeric(db_summary.set_index(db_id)["reachable_park_count"], errors="raise")
        .reindex(db_summary[db_id])
    )
    expected_counts = db_summary[db_id].map(actual_counts).fillna(0).astype(int)
    if not np.array_equal(reported_counts.to_numpy(dtype=int), expected_counts.to_numpy(dtype=int)):
        raise ValueError("DB reachable-park counts do not match the pair table")

    has_access = db_summary["has_park_within_threshold"].astype("string").str.casefold() == "true"
    if not np.array_equal(has_access.to_numpy(), (expected_counts > 0).to_numpy()):
        raise ValueError("DB access flags do not match reachable-park counts")

    db_population = pd.to_numeric(db_summary["db_pop"], errors="raise")
    access_population = pd.to_numeric(db_summary["population_with_access"], errors="raise")
    expected_access_population = db_population.where(has_access, 0)
    if not np.allclose(access_population, expected_access_population):
        raise ValueError("DB access population does not match DB access flags")

    expected_da = (
        db_summary.assign(
            db_pop_numeric=db_population,
            access_pop_numeric=access_population,
            has_access_numeric=has_access.astype(int),
            park_count_numeric=expected_counts,
        )
        .groupby(da_id, as_index=False)
        .agg(
            da_pop=("db_pop_numeric", "sum"),
            db_count=(db_id, "size"),
            dbs_with_access=("has_access_numeric", "sum"),
            population_with_access=("access_pop_numeric", "sum"),
            mean_reachable_park_count=("park_count_numeric", "mean"),
        )
        .sort_values(da_id)
        .reset_index(drop=True)
    )
    observed_da = da_summary.sort_values(da_id).reset_index(drop=True)
    if list(expected_da[da_id]) != list(observed_da[da_id]):
        raise ValueError("DA summary IDs do not match DB aggregation")
    for column in (
        "da_pop",
        "db_count",
        "dbs_with_access",
        "population_with_access",
        "mean_reachable_park_count",
    ):
        if not np.allclose(
            pd.to_numeric(expected_da[column]),
            pd.to_numeric(observed_da[column]),
            equal_nan=True,
        ):
            raise ValueError(f"DA summary column does not match DB aggregation: {column}")

    expected_coverage = (
        pd.to_numeric(observed_da["population_with_access"])
        / pd.to_numeric(observed_da["da_pop"]).replace(0, np.nan)
    )
    if not np.allclose(
        expected_coverage,
        pd.to_numeric(observed_da["park_access_pop_coverage"]),
        equal_nan=True,
    ):
        raise ValueError("DA population coverage is inconsistent")

    municipality_lookup = config["parks_source"]["csd_municipality_names"]
    analysis_municipalities = {
        municipality_lookup[str(code)]
        for code in config["analysis_area"]["analysis_csd_codes"]
    }
    external_inventory = ~pairs["park_municipality"].isin(analysis_municipalities)

    return {
        "db_count": len(db_summary),
        "da_count": len(da_summary),
        "reachable_db_park_pairs": len(pairs),
        "dbs_with_access": int(has_access.sum()),
        "dbs_without_access": int((~has_access).sum()),
        "total_population": int(db_population.sum()),
        "population_with_access": int(access_population.sum()),
        "population_access_coverage": (
            float(access_population.sum() / db_population.sum())
            if db_population.sum() > 0
            else np.nan
        ),
        "db_snap_warning_count": int(
            (pd.to_numeric(snapped["db_snap_dist_m"]) > config["analysis"]["db_snap_warning_m"]).sum()
        ),
        "network_distance_m": threshold,
        "distance_rule": "network distance only; point snap distance retained as QA",
        "all_reachable_parks_retained": True,
        "reachable_pairs_to_non_analysis_inventory": int(external_inventory.sum()),
        "reachable_non_analysis_inventory_parks": int(
            pairs.loc[external_inventory, "park_id"].nunique()
        ),
        "dbs_reaching_non_analysis_inventory": int(
            pairs.loc[external_inventory, db_id].nunique()
        ),
        "das_reaching_non_analysis_inventory": int(
            pairs.loc[external_inventory, da_id].nunique()
        ),
        "analysis_inventory_municipalities": " | ".join(
            sorted(analysis_municipalities)
        ),
        "crs": expected_crs,
    }


def _load_walk_graph(config: dict[str, Any]) -> tuple[nx.Graph, bool]:
    paths = network_paths(config)
    source_path = paths["graph_cache"] if paths["graph_cache"].exists() else paths["graphml"]
    if not source_path.exists():
        raise FileNotFoundError("Shared pedestrian graph is missing")

    source_signature = {
        "source_path": str(source_path.resolve()),
        "source_size": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
    }
    if paths["undirected_cache"].exists():
        with paths["undirected_cache"].open("rb") as stream:
            cached_graph = pickle.load(stream)
        cached_signature = cached_graph.graph.get("source_signature")
        if cached_signature == source_signature:
            first_node = next(iter(cached_graph.nodes))
            return cached_graph, isinstance(first_node, (int, np.integer))

    if paths["graph_cache"].exists():
        with paths["graph_cache"].open("rb") as stream:
            graph = pickle.load(stream)
    elif paths["graphml"].exists():
        graph = ox.load_graphml(paths["graphml"])
    for _, _, _, edge_data in graph.edges(keys=True, data=True):
        edge_data["length"] = float(edge_data.get("length", 0) or 0)
    try:
        walk_graph = ox.convert.to_undirected(graph)
    except Exception:
        walk_graph = graph.to_undirected()
    walk_graph.graph["source_signature"] = source_signature

    paths["undirected_cache"].parent.mkdir(parents=True, exist_ok=True)
    with inherited_temp_directory(
        paths["undirected_cache"].parent, "undirected_graph_"
    ) as temp_dir:
        temp_cache = temp_dir / paths["undirected_cache"].name
        with temp_cache.open("wb") as stream:
            pickle.dump(walk_graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_cache, paths["undirected_cache"])

    first_node = next(iter(walk_graph.nodes))
    return walk_graph, isinstance(first_node, (int, np.integer))


def _snap_db_points(
    config: dict[str, Any], graph: nx.Graph, integer_nodes: bool
) -> gpd.GeoDataFrame:
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    projected_crs = config["crs"]["projected"]
    points = gpd.read_file(population_points_path(config), layer="db_points").to_crs(projected_crs)
    points[db_id] = clean_id_series(points[db_id])
    points[da_id] = clean_id_series(points[da_id])
    points["db_pop"] = pd.to_numeric(points["db_pop"], errors="raise").astype("int64")

    node_path = network_paths(config)["nodes"]
    nodes = gpd.read_file(node_path, layer="osm_nodes").to_crs(projected_crs)
    nodes = nodes[nodes.geometry.notna() & ~nodes.geometry.is_empty].copy()
    node_field = "osmid" if "osmid" in nodes.columns else "node"
    if node_field not in nodes.columns:
        raise ValueError("OSM node layer has neither osmid nor node field")
    nodes = nodes[[node_field, "geometry"]].rename(columns={node_field: "db_nearest_node"})
    nodes["db_nearest_node"] = nodes["db_nearest_node"].map(
        lambda value: normalise_node_id(value, integer_nodes)
    )

    snapped = gpd.sjoin_nearest(
        points,
        nodes,
        how="left",
        distance_col="db_snap_dist_m",
    )
    snapped = snapped.sort_values([db_id, "db_snap_dist_m", "db_nearest_node"])
    snapped = snapped.drop_duplicates(db_id).drop(columns="index_right", errors="ignore")
    if snapped["db_nearest_node"].isna().any():
        raise ValueError("At least one DB point did not snap to an OSM node")
    if not set(snapped["db_nearest_node"]) <= set(graph.nodes):
        raise ValueError("At least one snapped DB node is absent from the pedestrian graph")
    return snapped


def _entrance_lookup(
    config: dict[str, Any], graph: nx.Graph, integer_nodes: bool
) -> tuple[dict[Any, list[dict[str, Any]]], int, int]:
    entrances = gpd.read_file(data_path(config, "entrances"), layer="park_entrances")
    required = {"entrance_id", "park_id", "park_name", "municipality", "nearest_node"}
    missing = required - set(entrances.columns)
    if missing:
        raise ValueError(f"Park entrances are missing fields: {', '.join(sorted(missing))}")
    entrances["nearest_node"] = entrances["nearest_node"].map(
        lambda value: normalise_node_id(value, integer_nodes)
    )
    input_count = len(entrances)
    entrances = entrances[entrances["nearest_node"].isin(graph.nodes)].copy()
    lookup: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in entrances.itertuples(index=False):
        lookup[row.nearest_node].append(
            {
                "entrance_id": str(row.entrance_id),
                "park_id": str(row.park_id),
                "park_name": str(row.park_name),
                "park_municipality": str(row.municipality),
            }
        )
    return lookup, input_count, len(entrances)


def _calculate_pairs(
    config: dict[str, Any],
    graph: nx.Graph,
    snapped: gpd.GeoDataFrame,
    entrance_lookup: dict[Any, list[dict[str, Any]]],
) -> pd.DataFrame:
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    threshold = float(config["analysis"]["network_distance_m"])
    output_columns = [
        db_id,
        da_id,
        "db_pop",
        "db_nearest_node",
        "db_snap_dist_m",
        "park_id",
        "park_name",
        "park_municipality",
        "nearest_entrance_id",
        "nearest_entrance_node",
        "network_dist_m",
        "total_access_dist_m",
    ]
    rows: list[dict[str, Any]] = []
    records = snapped[[db_id, da_id, "db_pop", "db_nearest_node", "db_snap_dist_m"]]
    total = len(records)

    for number, values in enumerate(records.itertuples(index=False, name=None), start=1):
        current_db, current_da, db_pop, source, snap_distance = values
        lengths = nx.single_source_dijkstra_path_length(
            graph, source, cutoff=threshold, weight="length"
        )
        best_by_park: dict[str, dict[str, Any]] = {}
        for node, distance in lengths.items():
            for entrance in entrance_lookup.get(node, []):
                candidate = (
                    float(distance),
                    entrance["entrance_id"],
                    str(node),
                )
                existing = best_by_park.get(entrance["park_id"])
                if existing is None or candidate < existing["sort_key"]:
                    best_by_park[entrance["park_id"]] = {
                        **entrance,
                        "nearest_entrance_node": node,
                        "network_dist_m": float(distance),
                        "sort_key": candidate,
                    }

        for park_id, best in best_by_park.items():
            network_distance = best["network_dist_m"]
            rows.append(
                {
                    db_id: current_db,
                    da_id: current_da,
                    "db_pop": int(db_pop),
                    "db_nearest_node": source,
                    "db_snap_dist_m": float(snap_distance),
                    "park_id": park_id,
                    "park_name": best["park_name"],
                    "park_municipality": best["park_municipality"],
                    "nearest_entrance_id": best["entrance_id"],
                    "nearest_entrance_node": best["nearest_entrance_node"],
                    "network_dist_m": network_distance,
                    "total_access_dist_m": network_distance + float(snap_distance),
                }
            )
        if number % 500 == 0 or number == total:
            print(f"  Processed {number:,}/{total:,} DB points")

    pairs = pd.DataFrame(rows, columns=output_columns)
    if pairs.empty:
        raise ValueError("No DB-to-park pairs were reachable within the configured threshold")
    return pairs.sort_values([db_id, "park_id"]).reset_index(drop=True)


def _summaries(
    config: dict[str, Any],
    snapped: gpd.GeoDataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    fields = config["fields"]
    db_id = fields["db_id"]
    da_id = fields["da_id"]
    pair_summary = (
        pairs.groupby([db_id, da_id], as_index=False)
        .agg(
            reachable_park_count=("park_id", "nunique"),
            nearest_park_network_distance_m=("network_dist_m", "min"),
            nearest_park_total_access_distance_m=("total_access_dist_m", "min"),
            mean_reachable_park_total_access_distance_m=("total_access_dist_m", "mean"),
        )
    )
    base = snapped[[db_id, da_id, "db_pop", "db_nearest_node", "db_snap_dist_m"]].copy()
    db_summary = base.merge(pair_summary, on=[db_id, da_id], how="left")
    db_summary["reachable_park_count"] = db_summary["reachable_park_count"].fillna(0).astype(int)
    db_summary["has_park_within_threshold"] = db_summary["reachable_park_count"] > 0
    db_summary["population_with_access"] = db_summary["db_pop"].where(
        db_summary["has_park_within_threshold"], 0
    )
    db_summary = db_summary.sort_values(db_id).reset_index(drop=True)

    da_summary = (
        db_summary.groupby(da_id, as_index=False)
        .agg(
            da_pop=("db_pop", "sum"),
            db_count=(db_id, "nunique"),
            dbs_with_access=("has_park_within_threshold", "sum"),
            population_with_access=("population_with_access", "sum"),
            mean_reachable_park_count=("reachable_park_count", "mean"),
        )
        .sort_values(da_id)
        .reset_index(drop=True)
    )
    da_summary["park_access_pop_coverage"] = (
        da_summary["population_with_access"] / da_summary["da_pop"].replace(0, np.nan)
    )

    db_map = snapped.merge(
        db_summary[
            [
                db_id,
                "reachable_park_count",
                "nearest_park_network_distance_m",
                "nearest_park_total_access_distance_m",
                "mean_reachable_park_total_access_distance_m",
                "has_park_within_threshold",
                "population_with_access",
            ]
        ],
        on=db_id,
        how="left",
    )
    return db_summary, da_summary, db_map


def build_reachability(
    config: dict[str, Any], final_paths: dict[str, Path]
) -> dict[str, Any]:
    print("Loading the shared pedestrian graph...")
    graph, integer_nodes = _load_walk_graph(config)
    print(f"Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print("Snapping Stage 03 DB points to the graph...")
    snapped = _snap_db_points(config, graph, integer_nodes)
    print("Loading validated park entrances...")
    entrance_lookup, entrance_input_count, entrance_used_count = _entrance_lookup(
        config, graph, integer_nodes
    )
    print(
        f"Entrances represented in graph: {entrance_used_count:,}/{entrance_input_count:,} "
        f"across {len(entrance_lookup):,} nodes"
    )
    print("Calculating all unique DB-park pairs within the network threshold...")
    pairs = _calculate_pairs(config, graph, snapped, entrance_lookup)
    db_summary, da_summary, db_map = _summaries(config, snapped, pairs)

    final_paths["snapped_gpkg"].parent.mkdir(parents=True, exist_ok=True)
    temp_parent = final_paths["snapped_gpkg"].parent.parent
    slug = config["analysis_area"]["slug"]
    with inherited_temp_directory(temp_parent, f"{slug}_04_") as temp_dir:
        temp_paths = reachability_paths(config, temp_dir)
        temp_paths["snapped_gpkg"].parent.mkdir(parents=True, exist_ok=True)
        snapped.to_file(temp_paths["snapped_gpkg"], layer="db_points_snapped", driver="GPKG")
        snapped.drop(columns="geometry").to_csv(temp_paths["snapped_csv"], index=False)
        pairs.to_csv(temp_paths["pairs_csv"], index=False)
        db_summary.to_csv(temp_paths["db_summary_csv"], index=False)
        db_map.to_file(temp_paths["db_summary_gpkg"], layer="db_reachability_summary", driver="GPKG")
        da_summary.to_csv(temp_paths["da_summary_csv"], index=False)

        metrics = validate_reachability(config, temp_paths)
        for key, target in final_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_paths[key], target)

    metrics.update(
        {
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "park_entrances_input": entrance_input_count,
            "park_entrances_used": entrance_used_count,
            "unique_entrance_nodes": len(entrance_lookup),
        }
    )
    return metrics


def write_summary(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    status: str,
) -> Path:
    slug = config["analysis_area"]["slug"]
    table_dir = output_root(config) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / f"{slug}_04_reachability_summary.csv"
    row = {
        "analysis_area": slug,
        "status": status,
        **metrics,
        **{f"{key}_path": str(value) for key, value in paths.items()},
    }
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    return summary_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = reachability_paths(config)
    slug = config["analysis_area"]["slug"]
    print(f"\nStage 04: 400 m park reachability ({slug})")
    print("Rule: network distance from snapped DB node to park entrance <= 400 m")
    print("DB snap distance is retained for QA and is not added to the threshold.")

    if force:
        print("Force enabled: building new artifacts atomically in this repository.")
        metrics = build_reachability(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing artifacts.")
        try:
            metrics = validate_reachability(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Reachability validation failed: {error}")
            print("Build this area's Stage 04 artifacts explicitly with --force.")
            return 1
        status = "reused"

    metrics = validate_reachability(config, paths) | metrics
    summary_path = write_summary(config, paths, metrics, status)
    print(f"Validated DBs: {metrics['db_count']:,}")
    print(f"Reachable DB-park pairs: {metrics['reachable_db_park_pairs']:,}")
    print(f"Population access coverage: {metrics['population_access_coverage']:.1%}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this area's generated Stage 04 artifacts",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
