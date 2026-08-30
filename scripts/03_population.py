"""Prepare or validate DA boundaries, DB boundaries, and DB population points."""

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
    output_root,
    workspace_data_root,
)


def clean_id_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def read_gaf(config: dict[str, Any]) -> pd.DataFrame:
    fields = config["fields"]
    columns = [
        fields["gaf_csd_id"],
        fields["gaf_da_id"],
        fields["gaf_db_id"],
        fields["gaf_db_population"],
    ]
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                data_path(config, "census_gaf"),
                usecols=columns,
                dtype="string",
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Could not read the Census GAF: {last_error}")


def census_artifact_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    root = data_root or workspace_data_root(config)
    base = root / "interim" / "census" / slug
    return {
        "da_boundaries": base / f"{slug}_da_boundaries.gpkg",
        "db_boundaries": base / f"{slug}_db_boundaries.gpkg",
        "db_points": base / f"{slug}_db_points.gpkg",
        "db_points_csv": base / f"{slug}_db_points.csv",
    }


def _spatial_count(
    path: Path,
    expected_crs: str,
    required_fields: set[str],
) -> int:
    info = pyogrio.read_info(path, force_feature_count=True)
    missing = required_fields - set(info["fields"])
    if missing:
        raise ValueError(f"{path.name} is missing fields: {', '.join(sorted(missing))}")
    if str(info["crs"]).upper() != expected_crs.upper():
        raise ValueError(f"{path.name} CRS is {info['crs']}; expected {expected_crs}")
    count = int(info["features"])
    if count < 1:
        raise ValueError(f"{path.name} contains no features")
    return count


def validate_population(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 03 artifacts: " + ", ".join(path.name for path in missing)
        )

    fields = config["fields"]
    da_id = fields["da_id"]
    db_id = fields["db_id"]
    representative_x = fields["db_representative_x"]
    representative_y = fields["db_representative_y"]
    expected_crs = config["crs"]["projected"]
    common = {"analysis_area", "analysis_csd_codes", "csd_id"}

    da_count = _spatial_count(paths["da_boundaries"], expected_crs, {da_id} | common)
    db_count = _spatial_count(
        paths["db_boundaries"],
        expected_crs,
        {
            db_id,
            da_id,
            representative_x,
            representative_y,
            "db_pop",
            "db_area_m2",
            "db_area_ha",
        }
        | common,
    )
    point_count = _spatial_count(
        paths["db_points"],
        expected_crs,
        {
            db_id,
            da_id,
            representative_x,
            representative_y,
            "db_pop",
            "point_x",
            "point_y",
            "point_method",
        }
        | common,
    )

    points_csv = pd.read_csv(paths["db_points_csv"], dtype="string")
    required_csv = {
        db_id,
        da_id,
        "db_pop",
        "point_x",
        "point_y",
        "point_method",
        "csd_id",
    }
    missing_columns = required_csv - set(points_csv.columns)
    if missing_columns:
        raise ValueError(
            f"{paths['db_points_csv'].name} is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if not (db_count == point_count == len(points_csv)):
        raise ValueError(
            "DB boundary, point GeoPackage, and point CSV counts differ: "
            f"{db_count}, {point_count}, {len(points_csv)}"
        )

    da = pyogrio.read_dataframe(
        paths["da_boundaries"], columns=[da_id, "csd_id"], read_geometry=False
    )
    db = pyogrio.read_dataframe(
        paths["db_boundaries"], columns=[db_id, da_id, "db_pop"], read_geometry=False
    )
    points = pyogrio.read_dataframe(
        paths["db_points"],
        columns=[db_id, da_id, "db_pop", "point_method", "csd_id"],
        read_geometry=False,
    )

    for frame, identifier, label in (
        (da, da_id, "DA boundaries"),
        (db, db_id, "DB boundaries"),
        (points, db_id, "DB points"),
        (points_csv, db_id, "DB point CSV"),
    ):
        if frame[identifier].astype("string").duplicated().any():
            raise ValueError(f"{label} contains duplicate {identifier} values")

    da_ids = set(da[da_id].astype("string"))
    db_ids = set(db[db_id].astype("string"))
    point_ids = set(points[db_id].astype("string"))
    csv_ids = set(points_csv[db_id].astype("string"))
    if db_ids != point_ids or db_ids != csv_ids:
        raise ValueError("DB boundary, point GeoPackage, and point CSV IDs differ")
    if not set(db[da_id].astype("string")) <= da_ids:
        raise ValueError("A DB references a DA absent from the prepared DA layer")

    db_geometry = pyogrio.read_dataframe(
        paths["db_boundaries"], columns=[db_id]
    ).set_index(db_id).geometry
    point_geometry_frame = pyogrio.read_dataframe(
        paths["db_points"],
        columns=[db_id, representative_x, representative_y, "point_x", "point_y"],
    )
    point_geometry = point_geometry_frame.set_index(db_id).geometry
    point_geometry = point_geometry.loc[db_geometry.index]
    if not db_geometry.covers(point_geometry, align=True).all():
        raise ValueError("A representative DB point falls outside its source DB polygon")

    stored_x = pd.to_numeric(point_geometry_frame["point_x"], errors="coerce")
    stored_y = pd.to_numeric(point_geometry_frame["point_y"], errors="coerce")
    if stored_x.isna().any() or stored_y.isna().any():
        raise ValueError("Prepared DB points contain missing projected coordinates")
    if not (
        np.allclose(stored_x, point_geometry_frame.geometry.x, atol=1e-6)
        and np.allclose(stored_y, point_geometry_frame.geometry.y, atol=1e-6)
    ):
        raise ValueError("Stored point_x/point_y values differ from DB point geometry")

    official_points = gpd.GeoSeries(
        gpd.points_from_xy(
            pd.to_numeric(point_geometry_frame[representative_x], errors="coerce"),
            pd.to_numeric(point_geometry_frame[representative_y], errors="coerce"),
        ),
        crs=config["crs"]["statcan_lambert"],
    ).to_crs(expected_crs)
    if not np.allclose(
        official_points.x,
        point_geometry_frame.geometry.x.reset_index(drop=True),
        atol=0.01,
    ) or not np.allclose(
        official_points.y,
        point_geometry_frame.geometry.y.reset_index(drop=True),
        atol=0.01,
    ):
        raise ValueError("DB point geometry does not match official DBRPLAMX/DBRPLAMY coordinates")

    configured_codes = {str(code) for code in config["analysis_area"]["analysis_csd_codes"]}
    stored_codes = set(points["csd_id"].astype("string"))
    if stored_codes != configured_codes:
        raise ValueError(
            f"Stored CSD codes {sorted(stored_codes)} do not match configuration "
            f"{sorted(configured_codes)}"
        )

    methods = set(points["point_method"].astype("string"))
    if methods != {"statcan_db_representative_point"}:
        raise ValueError(f"Unexpected DB point methods: {sorted(methods)}")

    db_population = pd.to_numeric(db["db_pop"], errors="coerce")
    point_population = pd.to_numeric(points["db_pop"], errors="coerce")
    csv_population = pd.to_numeric(points_csv["db_pop"], errors="coerce")
    if db_population.isna().any() or (db_population < 0).any():
        raise ValueError("Prepared DB population contains missing or negative values")
    if not (
        float(db_population.sum())
        == float(point_population.sum())
        == float(csv_population.sum())
    ):
        raise ValueError("DB population totals differ across Stage 03 artifacts")

    return {
        "da_count": da_count,
        "db_count": db_count,
        "db_point_count": point_count,
        "total_population": int(db_population.sum()),
        "zero_population_dbs": int((db_population == 0).sum()),
        "csd_count": len(stored_codes),
        "point_method": "statcan_db_representative_point",
        "points_inside_source_db": True,
        "crs": expected_crs,
    }


def _check_unique_gaf_rows(frame: pd.DataFrame, db_id: str) -> pd.DataFrame:
    duplicated = frame[frame.duplicated(db_id, keep=False)]
    if not duplicated.empty:
        comparison_columns = ["csd_id", "da_id", "db_pop"]
        conflicts = duplicated.groupby(db_id)[comparison_columns].nunique(dropna=False)
        if (conflicts > 1).any(axis=None):
            raise ValueError("The Census GAF contains conflicting duplicate DB records")
    return frame.drop_duplicates(db_id).copy()


def build_population(config: dict[str, Any], final_paths: dict[str, Path]) -> dict[str, Any]:
    fields = config["fields"]
    da_id = fields["da_id"]
    db_id = fields["db_id"]
    representative_x = fields["db_representative_x"]
    representative_y = fields["db_representative_y"]
    gaf_csd = fields["gaf_csd_id"]
    gaf_da = fields["gaf_da_id"]
    gaf_db = fields["gaf_db_id"]
    gaf_pop = fields["gaf_db_population"]
    slug = config["analysis_area"]["slug"]
    codes = [str(code) for code in config["analysis_area"]["analysis_csd_codes"]]
    codes_label = ";".join(codes)
    projected_crs = config["crs"]["projected"]

    print("Reading the required Census GAF columns...")
    gaf = read_gaf(config)
    for column in (gaf_csd, gaf_da, gaf_db):
        gaf[column] = clean_id_series(gaf[column])
    selected = gaf[gaf[gaf_csd].isin(codes)].copy()
    if selected.empty:
        raise ValueError("No GAF rows matched the configured CSD codes")

    population = selected[[gaf_csd, gaf_da, gaf_db, gaf_pop]].rename(
        columns={gaf_csd: "csd_id", gaf_da: "da_id", gaf_db: "db_id", gaf_pop: "db_pop"}
    )
    population["db_pop"] = (
        pd.to_numeric(population["db_pop"], errors="coerce").fillna(0).astype("int64")
    )
    population = _check_unique_gaf_rows(population, "db_id")
    da_ids = set(population["da_id"].dropna())
    db_ids = set(population["db_id"].dropna())

    print("Reading and filtering DA boundaries...")
    da = gpd.read_file(data_path(config, "da_boundaries"), columns=[da_id])
    if da.crs is None or da_id not in da.columns:
        raise ValueError(f"DA layer must have a CRS and the {da_id} field")
    da[da_id] = clean_id_series(da[da_id])
    da = da[da[da_id].isin(da_ids)].to_crs(projected_crs).copy()
    if da.empty:
        raise ValueError("No DA polygons matched the configured analysis area")
    da_lookup = population[["da_id", "csd_id"]].drop_duplicates("da_id")
    da = da.merge(da_lookup, left_on=da_id, right_on="da_id", how="left")
    if da_id != "da_id":
        da = da.drop(columns=["da_id"])
    da["analysis_area"] = slug
    da["analysis_csd_codes"] = codes_label

    print("Reading and filtering DB boundaries...")
    db = gpd.read_file(
        data_path(config, "db_boundaries"),
        columns=[db_id, representative_x, representative_y],
    )
    if db.crs is None or db_id not in db.columns:
        raise ValueError(f"DB layer must have a CRS and the {db_id} field")
    db[db_id] = clean_id_series(db[db_id])
    db = db[db[db_id].isin(db_ids)].copy()
    if db.empty:
        raise ValueError("No DB polygons matched the configured analysis area")

    for coordinate in (representative_x, representative_y):
        db[coordinate] = pd.to_numeric(db[coordinate], errors="coerce")
        if db[coordinate].isna().any():
            raise ValueError(f"DB layer contains missing official coordinates: {coordinate}")

    official_points = gpd.GeoDataFrame(
        {db_id: db[db_id].copy()},
        geometry=gpd.points_from_xy(db[representative_x], db[representative_y]),
        crs=db.crs,
    ).to_crs(projected_crs)
    official_by_id = official_points.set_index(db_id).geometry

    db = db.to_crs(projected_crs)
    db = db.merge(population, left_on=db_id, right_on="db_id", how="left")
    if db_id != "db_id":
        db = db.drop(columns=["db_id"])
    if da_id != "da_id":
        db = db.rename(columns={"da_id": da_id})
    db["analysis_area"] = slug
    db["analysis_csd_codes"] = codes_label
    db["db_area_m2"] = db.geometry.area
    db["db_area_ha"] = db["db_area_m2"] / 10_000

    points = db.copy()
    points["geometry"] = gpd.GeoSeries(
        points[db_id].map(official_by_id),
        index=points.index,
        crs=projected_crs,
    )
    points["point_x"] = points.geometry.x
    points["point_y"] = points.geometry.y
    points["point_method"] = "statcan_db_representative_point"
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=projected_crs)

    final_paths["da_boundaries"].parent.mkdir(parents=True, exist_ok=True)
    temp_parent = final_paths["da_boundaries"].parent.parent
    with inherited_temp_directory(temp_parent, f"{slug}_03_") as temp_dir:
        temp_paths = census_artifact_paths(config, temp_dir)
        temp_paths["da_boundaries"].parent.mkdir(parents=True, exist_ok=True)
        da.to_file(temp_paths["da_boundaries"], layer="da_boundaries", driver="GPKG")
        db.to_file(temp_paths["db_boundaries"], layer="db_boundaries", driver="GPKG")
        points.to_file(temp_paths["db_points"], layer="db_points", driver="GPKG")
        points.drop(columns="geometry").to_csv(temp_paths["db_points_csv"], index=False)

        metrics = validate_population(config, temp_paths)
        for key, target in final_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_paths[key], target)

    metrics.update(
        {
            "gaf_rows_selected": len(selected),
            "gaf_unique_dbs": len(population),
            "gaf_unique_das": len(da_ids),
            "db_boundaries_missing": len(db_ids) - len(db),
            "da_boundaries_missing": len(da_ids) - len(da),
        }
    )
    return metrics


def write_summaries(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    status: str,
) -> tuple[Path, Path]:
    fields = config["fields"]
    da_id = fields["da_id"]
    slug = config["analysis_area"]["slug"]
    table_dir = output_root(config) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    points = pyogrio.read_dataframe(
        paths["db_points"], columns=[da_id, "db_pop"], read_geometry=False
    )
    points["db_pop"] = pd.to_numeric(points["db_pop"], errors="raise")
    da_population = (
        points.groupby(da_id, as_index=False)
        .agg(da_population=("db_pop", "sum"), db_count=("db_pop", "size"))
        .sort_values(da_id)
    )
    population_path = table_dir / f"{slug}_03_da_population.csv"
    da_population.to_csv(population_path, index=False)

    summary_path = table_dir / f"{slug}_03_population_summary.csv"
    row = {
        "analysis_area": slug,
        "status": status,
        "analysis_csd_codes": ";".join(config["analysis_area"]["analysis_csd_codes"]),
        **metrics,
        **{f"{key}_path": str(value) for key, value in paths.items()},
    }
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    return summary_path, population_path


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = census_artifact_paths(config)
    slug = config["analysis_area"]["slug"]

    print(f"\nStage 03: Census population preparation ({slug})")
    print(f"Generated-data folder: {paths['db_points'].parent}")

    if force:
        print("Force enabled: building new artifacts atomically in this repository.")
        metrics = build_population(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing artifacts.")
        try:
            metrics = validate_population(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Population validation failed: {error}")
            print("Build this area's Stage 03 artifacts explicitly with --force.")
            return 1
        status = "reused"

    metrics = validate_population(config, paths) | metrics
    summary_path, population_path = write_summaries(config, paths, metrics, status)
    print(f"Validated DAs: {metrics['da_count']:,}")
    print(f"Validated DB points: {metrics['db_count']:,}")
    print(f"Population represented: {metrics['total_population']:,}")
    print(f"Summary: {summary_path}")
    print(f"DA population table: {population_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this area's generated Stage 03 artifacts",
    )
    args = parser.parse_args()

    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
