"""Prepare or validate the shared Metro Vancouver pedestrian network."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
import pyogrio

from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    project_root,
)


def read_csv_with_fallback(
    path: Path,
    usecols: list[str] | None = None,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtype,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Could not read CSV using supported encodings: {last_error}")


def network_paths(config: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    network_slug = config["network"]["slug"]
    base = (root or Path(config["runtime"]["data_root"])) / "interim" / "network" / network_slug

    return {
        "boundary": base / f"{network_slug}_network_boundary.gpkg",
        "graphml": base / f"{network_slug}_walk.graphml",
        "graph_cache": base / f"{network_slug}_walk_graph.pkl",
        "nodes": base / f"{network_slug}_osm_nodes.gpkg",
        "edges": base / f"{network_slug}_osm_edges.gpkg",
    }


def _check_spatial(
    path: Path,
    expected_crs: str,
    required_fields: set[str],
) -> tuple[int, str]:
    info = pyogrio.read_info(path, force_feature_count=True)
    fields = set(info["fields"])
    missing = required_fields - fields
    if missing:
        raise ValueError(f"{path.name} is missing fields: {', '.join(sorted(missing))}")

    features = int(info["features"])
    if features < 1:
        raise ValueError(f"{path.name} contains no features")

    crs = str(info["crs"])
    if crs.upper() != expected_crs.upper():
        raise ValueError(f"{path.name} CRS is {crs}; expected {expected_crs}")

    return features, crs


def validate_network(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing network artifacts: {names}")

    projected_crs = config["crs"]["projected"]
    boundary_count, boundary_crs = _check_spatial(
        paths["boundary"],
        projected_crs,
        {"boundary_csd_codes", "selected_da_count"},
    )
    node_count, node_crs = _check_spatial(
        paths["nodes"], projected_crs, {"osmid", "x", "y"}
    )
    edge_count, edge_crs = _check_spatial(
        paths["edges"], projected_crs, {"u", "v", "key", "length"}
    )

    with paths["graphml"].open("rb") as stream:
        header = stream.read(4096).lower()
    if b"graphml" not in header:
        raise ValueError(f"{paths['graphml'].name} does not have a GraphML header")

    if paths["graph_cache"].stat().st_size == 0:
        raise ValueError(f"{paths['graph_cache'].name} is empty")

    boundary = gpd.read_file(paths["boundary"], rows=1)
    stored_codes = {
        value.strip()
        for value in str(boundary.iloc[0]["boundary_csd_codes"]).split(",")
        if value.strip()
    }
    configured_codes = {str(value) for value in config["network"]["boundary_csd_codes"]}
    if stored_codes != configured_codes:
        raise ValueError(
            "Stored network boundary CSD codes do not match config/settings.yaml"
        )

    return {
        "boundary_features": boundary_count,
        "selected_da_count": int(boundary.iloc[0]["selected_da_count"]),
        "node_count": node_count,
        "edge_count": edge_count,
        "crs": node_crs,
        "graphml_mb": paths["graphml"].stat().st_size / (1024 * 1024),
        "graph_cache_mb": paths["graph_cache"].stat().st_size / (1024 * 1024),
        "boundary_crs": boundary_crs,
        "edge_crs": edge_crs,
    }


def build_network(config: dict[str, Any], paths: dict[str, Path]) -> None:
    network = config["network"]
    fields = config["fields"]
    projected_crs = config["crs"]["projected"]
    geographic_crs = config["crs"]["geographic"]
    boundary_codes = [str(value) for value in network["boundary_csd_codes"]]

    gaf_path = data_path(config, "census_gaf")
    da_path = data_path(config, "da_boundaries")
    gaf_csd = fields["gaf_csd_id"]
    gaf_da = fields["gaf_da_id"]
    da_id = fields["da_id"]

    print("Reading Census geography lookup...")
    gaf = read_csv_with_fallback(
        gaf_path,
        usecols=[gaf_csd, gaf_da],
        dtype={gaf_csd: str, gaf_da: str},
    )
    selected_da_ids = set(
        gaf.loc[gaf[gaf_csd].isin(boundary_codes), gaf_da].dropna().astype(str)
    )
    if not selected_da_ids:
        raise ValueError("No DAs matched network.boundary_csd_codes")

    print("Reading and filtering DA boundaries...")
    da = gpd.read_file(da_path)
    da[da_id] = da[da_id].astype(str)
    selected = da[da[da_id].isin(selected_da_ids)].to_crs(projected_crs)
    if selected.empty:
        raise ValueError("No DA polygons matched the selected network DAs")

    boundary = gpd.GeoDataFrame(
        {
            "slug": [network["slug"]],
            "name": ["Metro Vancouver network context"],
            "boundary_csd_codes": [",".join(boundary_codes)],
            "selected_da_count": [len(selected)],
            "projected_crs": [projected_crs],
        },
        geometry=[selected.geometry.union_all()],
        crs=projected_crs,
    )

    paths["boundary"].parent.mkdir(parents=True, exist_ok=True)
    with inherited_temp_directory(
        paths["boundary"].parent.parent, "network_build_"
    ) as temp_dir:
        temp_paths = network_paths(config, temp_dir)
        temp_paths["boundary"].parent.mkdir(parents=True, exist_ok=True)

        boundary.to_file(temp_paths["boundary"], layer="network_boundary", driver="GPKG")
        polygon = boundary.to_crs(geographic_crs).geometry.iloc[0]

        print("Downloading the OSM pedestrian network...")
        ox.settings.use_cache = True
        ox.settings.log_console = True
        graph = ox.graph_from_polygon(
            polygon,
            network_type=network.get("network_type", "walk"),
            simplify=True,
            retain_all=bool(network.get("retain_all", True)),
            truncate_by_edge=bool(network.get("truncate_by_edge", True)),
        )
        graph = ox.project_graph(graph, to_crs=projected_crs)

        ox.save_graphml(graph, temp_paths["graphml"])
        with temp_paths["graph_cache"].open("wb") as stream:
            pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)

        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        nodes.to_file(temp_paths["nodes"], layer="osm_nodes", driver="GPKG")
        edges.to_file(temp_paths["edges"], layer="osm_edges", driver="GPKG")

        validate_network(config, temp_paths)
        for key, target in paths.items():
            os.replace(temp_paths[key], target)


def write_summary(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    status: str,
) -> Path:
    output_dir = project_root() / "outputs" / config["network"]["slug"] / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{config['network']['slug']}_01_network_summary.csv"

    row = {
        "network_slug": config["network"]["slug"],
        "status": status,
        "boundary_csd_codes": ",".join(config["network"]["boundary_csd_codes"]),
        **metrics,
        **{f"{key}_path": str(value) for key, value in paths.items()},
    }
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    return summary_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = network_paths(config)

    print("\nStage 01: shared pedestrian network")
    print(f"Network slug: {config['network']['slug']}")
    print(f"Artifact folder: {paths['graphml'].parent}")

    if force:
        print("Force enabled: rebuilding network artifacts.")
        build_network(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing network artifacts.")
        try:
            metrics = validate_network(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Network validation failed: {error}")
            print("Rebuild explicitly with --force after reviewing the configured boundary.")
            return 1
        status = "reused"

    metrics = validate_network(config, paths)
    summary_path = write_summary(config, paths, metrics, status)

    print(f"Validated nodes: {metrics['node_count']:,}")
    print(f"Validated edges: {metrics['edge_count']:,}")
    print(f"CRS: {metrics['crs']}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", default="metro", help="Configured area (network is shared)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and atomically replace the shared network artifacts",
    )
    args = parser.parse_args()

    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
