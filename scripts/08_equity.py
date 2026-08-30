"""Prepare or validate DA-level equity data and divergence profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from equity_methods import (
    CLASS_ORDER,
    chi_square_association,
    equity_strata,
    kruskal_wallis_association,
)

from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    output_root,
    workspace_data_root,
)


PROFILE_VARIABLES = {
    243: ("medhhinc", "Median household income", "C1_COUNT_TOTAL"),
    2008: ("pct_bachelor_plus", "Education (bachelor's+)", "C10_RATE_TOTAL"),
    9: ("pct_age_0_14", "Children aged 0-14", "C10_RATE_TOTAL"),
    24: ("pct_age_65plus", "Age 65+", "C10_RATE_TOTAL"),
    345: ("pct_LIM_AT", "Low income (LIM-AT)", "C10_RATE_TOTAL"),
    1529: ("pct_immigrant", "Immigrant share", "C10_RATE_TOTAL"),
    1684: ("pct_visible_minority", "Visible minority", "C10_RATE_TOTAL"),
}
VARIABLES = [
    ("pct_bachelor_plus", "Education (bachelor's+)"),
    ("ALE_index", "Active living environment"),
    ("pct_age_0_14", "Children aged 0-14"),
    ("pct_age_65plus", "Age 65+"),
    ("pct_LIM_AT", "Low income (LIM-AT)"),
    ("pct_immigrant", "Immigrant share"),
    ("pct_visible_minority", "Visible minority"),
]
EQUITY_VARIABLES = [
    ("medhhinc", "Median household income"),
    *VARIABLES,
]
PERCENT_VARIABLES = {
    field
    for field, _, source_column in PROFILE_VARIABLES.values()
    if source_column == "C10_RATE_TOTAL"
}
CLASS_LABELS = {
    "HH": "High supply / high experience",
    "LH": "Low supply / high experience",
    "HL": "High supply / low experience",
    "LL": "Low supply / low experience",
}
SHARED_VERSION = "statcan_profile_income_rates_canale_exact_da_v4"


def _significance_marker(p_value: float) -> str:
    """Return the manuscript marker for a two-sided p-value."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _profile_tests(
    frame: pd.DataFrame, class_column: str, field: str, seed_key: str
) -> dict[str, dict[str, float | int | bool | str]]:
    """Run continuous-profile and categorized-strata association tests."""
    classified = frame[class_column].isin(CLASS_ORDER)
    continuous = kruskal_wallis_association(
        frame.loc[classified, field], frame.loc[classified, class_column]
    )
    strata = equity_strata(frame[field], field)
    categorized = chi_square_association(
        strata.loc[classified],
        frame.loc[classified, class_column],
        seed_key=seed_key,
    )
    return {"continuous": continuous, "strata": categorized}


def experience_variants(config: dict[str, Any]) -> dict[str, str]:
    variants = {"rating": "divergence_class"}
    if config.get("experience", {}).get("sentiment", {}).get("enabled", False):
        variants["sentiment"] = "divergence_sentiment_sensitivity"
    return variants


def equity_profile_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    table_dir = output_root(config) / "tables"
    paths = {"rating": table_dir / f"{slug}_08_equity_profiles.csv"}
    if "sentiment" in experience_variants(config):
        paths["sentiment"] = table_dir / f"{slug}_08_equity_profiles_sentiment.csv"
    return paths


def equity_paths(
    config: dict[str, Any], data_root: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    root = data_root or workspace_data_root(config)
    base = root / "interim" / "equity" / slug
    return {
        "da_equity_csv": base / f"{slug}_da_equity.csv",
        "da_equity_gpkg": base / f"{slug}_da_equity.gpkg",
    }


def shared_paths(config: dict[str, Any]) -> dict[str, Path]:
    base = workspace_data_root(config) / "interim" / "equity" / "_shared"
    return {
        "variables_csv": base / "six_city_da_equity_variables.csv",
        "metadata_json": base / "six_city_da_equity_variables_metadata.json",
    }


def divergence_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = workspace_data_root(config) / "interim" / "divergence" / slug
    return {
        "csv": base / f"{slug}_da_divergence.csv",
        "gpkg": base / f"{slug}_da_divergence.gpkg",
    }


def study_da_path(config: dict[str, Any]) -> Path:
    return (
        workspace_data_root(config)
        / "interim"
        / "census"
        / "metro"
        / "metro_da_boundaries.gpkg"
    )


def _study_da_ids(config: dict[str, Any]) -> set[str]:
    path = study_da_path(config)
    if not path.exists():
        raise FileNotFoundError(
            "The combined Stage 03 DA boundaries are required before Stage 08: "
            f"{path}"
        )
    da_id = config["fields"]["da_id"]
    frame = gpd.read_file(path, layer="da_boundaries", columns=[da_id])
    frame[da_id] = _clean_da_id(frame[da_id])
    _assert_unique(frame, [da_id], "Combined Stage 03 DA boundaries")
    if frame.empty or frame[da_id].isna().any():
        raise ValueError("Combined Stage 03 DA boundaries have invalid DA IDs")
    return set(frame[da_id])


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        examples = frame.loc[frame.duplicated(columns, keep=False), columns].head()
        raise ValueError(
            f"{label} contains duplicate keys {columns}: {examples.to_dict('records')}"
        )


def _clean_da_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _expected_shared_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metro_config = load_config("metro")
    return {
        "version": SHARED_VERSION,
        "census_profile": _source_signature(data_path(config, "census_profile")),
        "canale": _source_signature(data_path(config, "canale")),
        "study_das": _source_signature(study_da_path(config)),
        "census_profile_geo_level": "Dissemination area",
        "census_profile_variables": {
            str(key): value[0] for key, value in sorted(PROFILE_VARIABLES.items())
        },
        "analysis_csd_codes": sorted(
            str(value)
            for value in metro_config["analysis_area"]["analysis_csd_codes"]
        ),
    }


def _validate_shared(config: dict[str, Any]) -> pd.DataFrame:
    paths = shared_paths(config)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing shared equity extract: " + ", ".join(path.name for path in missing)
        )
    with paths["metadata_json"].open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected_metadata = _expected_shared_metadata(config)
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Shared equity extract is stale: {key}")

    da_id = config["fields"]["da_id"]
    required = {da_id, *(field for field, _ in EQUITY_VARIABLES)}
    frame = pd.read_csv(paths["variables_csv"], dtype={da_id: "string"})
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Shared equity extract is missing: "
            + ", ".join(sorted(missing_columns))
        )
    _assert_unique(frame, [da_id], "Shared equity extract")
    if frame.empty or frame[da_id].isna().any():
        raise ValueError("Shared equity extract has no usable DA identifiers")
    target_da_ids = _study_da_ids(config)
    if set(frame[da_id]) != target_da_ids:
        raise ValueError("Shared equity extract DA IDs differ from combined Stage 03")
    if int(metadata.get("da_count", -1)) != len(frame):
        raise ValueError("Shared equity extract row count differs from its metadata")

    for field, _ in EQUITY_VARIABLES:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    for field in PERCENT_VARIABLES:
        invalid = frame[field].notna() & ~frame[field].between(0, 100)
        if invalid.any():
            raise ValueError(f"{field} contains a value outside 0-100")
    return frame.sort_values(da_id).reset_index(drop=True)


def _extract_profile(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    profile_path = data_path(config, "census_profile")
    target_da_ids = _study_da_ids(config)
    characteristic_ids = set(PROFILE_VARIABLES)
    usecols = [
        "ALT_GEO_CODE",
        "GEO_LEVEL",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "C1_COUNT_TOTAL",
        "C10_RATE_TOTAL",
    ]
    pieces: list[pd.DataFrame] = []
    processed_rows = 0
    print("Scanning the Census Profile once for median income and six DA rates...")
    reader = pd.read_csv(
        profile_path,
        usecols=usecols,
        chunksize=500_000,
        encoding="latin1",
        low_memory=False,
        dtype={"ALT_GEO_CODE": "string", "GEO_LEVEL": "string"},
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        processed_rows += len(chunk)
        candidate = chunk.loc[
            chunk["GEO_LEVEL"].eq("Dissemination area")
            & chunk["CHARACTERISTIC_ID"].isin(characteristic_ids)
        ].copy()
        if not candidate.empty:
            candidate["DAUID"] = _clean_da_id(candidate["ALT_GEO_CODE"])
            candidate = candidate.loc[candidate["DAUID"].isin(target_da_ids)]
            if not candidate.empty:
                pieces.append(candidate)
        if chunk_number % 10 == 0:
            print(f"  Profile rows scanned: {processed_rows:,}")
    if not pieces:
        raise ValueError("No Census Profile rows matched the six-city DA geography")

    long = pd.concat(pieces, ignore_index=True)
    long["CHARACTERISTIC_ID"] = pd.to_numeric(
        long["CHARACTERISTIC_ID"], errors="raise"
    ).astype("int64")
    _assert_unique(long, ["DAUID", "CHARACTERISTIC_ID"], "Census Profile extract")
    long["variable"] = long["CHARACTERISTIC_ID"].map(
        {key: value[0] for key, value in PROFILE_VARIABLES.items()}
    )
    long["value"] = np.nan
    for characteristic_id, (_, _, source_column) in PROFILE_VARIABLES.items():
        selected = long["CHARACTERISTIC_ID"].eq(characteristic_id)
        long.loc[selected, "value"] = pd.to_numeric(
            long.loc[selected, source_column], errors="coerce"
        )
    names = {
        str(int(key)): str(value)
        for key, value in long.groupby("CHARACTERISTIC_ID")[
            "CHARACTERISTIC_NAME"
        ].first().items()
    }
    found_ids = set(long["CHARACTERISTIC_ID"])
    if found_ids != characteristic_ids:
        raise ValueError(
            "Census Profile characteristic IDs differ from configuration: "
            f"expected {sorted(characteristic_ids)}, found {sorted(found_ids)}"
        )
    row_counts = long.groupby("DAUID")["CHARACTERISTIC_ID"].nunique()
    if not row_counts.eq(len(PROFILE_VARIABLES)).all():
        raise ValueError("At least one DA lacks a configured Census Profile row")

    wide = long.pivot(index="DAUID", columns="variable", values="value").reset_index()
    wide.columns.name = None
    for field in PERCENT_VARIABLES:
        if field not in wide:
            wide[field] = np.nan
    return wide[["DAUID", *[value[0] for value in PROFILE_VARIABLES.values()]]], names


def _load_canale(config: dict[str, Any]) -> pd.DataFrame:
    path = data_path(config, "canale")
    canale = pd.read_csv(path, usecols=["DAUID", "ALE_index"], dtype={"DAUID": "string"})
    canale["DAUID"] = _clean_da_id(canale["DAUID"])
    canale["ALE_index"] = pd.to_numeric(canale["ALE_index"], errors="coerce")
    canale = canale.loc[canale["DAUID"].str.len().eq(8)].copy()
    _assert_unique(canale, ["DAUID"], "Can-ALE source")
    return canale


def _build_shared(config: dict[str, Any]) -> pd.DataFrame:
    paths = shared_paths(config)
    paths["variables_csv"].parent.mkdir(parents=True, exist_ok=True)
    census, characteristic_names = _extract_profile(config)
    canale = _load_canale(config)
    frame = census.merge(canale, on="DAUID", how="left", validate="one_to_one")
    frame = frame[["DAUID", *[field for field, _ in EQUITY_VARIABLES]]]
    frame = frame.sort_values("DAUID").reset_index(drop=True)

    metadata = {
        **_expected_shared_metadata(config),
        "da_count": len(frame),
        "characteristic_names": characteristic_names,
        "nonmissing_counts": {
            field: int(frame[field].notna().sum()) for field, _ in EQUITY_VARIABLES
        },
    }
    parent = paths["variables_csv"].parent
    with inherited_temp_directory(parent, "08_shared_") as temp_dir:
        temp_csv = temp_dir / paths["variables_csv"].name
        temp_json = temp_dir / paths["metadata_json"].name
        frame.to_csv(temp_csv, index=False)
        with temp_json.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        os.replace(temp_csv, paths["variables_csv"])
        os.replace(temp_json, paths["metadata_json"])
    return _validate_shared(config)


def load_shared(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    try:
        return _validate_shared(config), "reused"
    except (FileNotFoundError, ValueError) as error:
        print(f"Shared equity extract requires a rebuild: {error}")
        return _build_shared(config), "rebuilt"


def _zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = float(numeric.mean())
    standard_deviation = float(numeric.std(ddof=1))
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    if np.isfinite(standard_deviation) and standard_deviation > 0:
        result.loc[numeric.notna()] = (
            numeric.loc[numeric.notna()] - mean
        ) / standard_deviation
    return result, mean, standard_deviation


def build_equity_frames(
    config: dict[str, Any],
    shared: pd.DataFrame,
    build_profiles: bool = True,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    paths = divergence_paths(config)
    da_id = config["fields"]["da_id"]
    if not paths["csv"].exists() or not paths["gpkg"].exists():
        raise FileNotFoundError("Stage 07 divergence artifacts are required before Stage 08")
    divergence = pd.read_csv(paths["csv"], dtype={da_id: "string"})
    variants = experience_variants(config)
    required = {da_id, "municipality", "da_pop", *variants.values()}
    missing = required - set(divergence.columns)
    if missing:
        raise ValueError(
            "Stage 07 divergence table is missing: " + ", ".join(sorted(missing))
        )
    _assert_unique(divergence, [da_id], "Stage 07 divergence table")
    shared = shared.rename(columns={"DAUID": da_id})
    merged = divergence.merge(
        shared, on=da_id, how="left", validate="one_to_one", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        examples = merged.loc[merged["_merge"].ne("both"), da_id].head().tolist()
        raise ValueError(f"No shared equity row exists for DAs: {examples}")
    merged = merged.drop(columns="_merge")
    for field, _ in EQUITY_VARIABLES:
        merged[field] = pd.to_numeric(merged[field], errors="coerce")
        if field in PERCENT_VARIABLES:
            invalid = merged[field].notna() & ~merged[field].between(0, 100)
            if invalid.any():
                raise ValueError(f"{field} contains a value outside 0-100")
    rows = []
    slug = config["analysis_area"]["slug"]
    for variant, class_column in variants.items():
        classified = merged[class_column].isin(CLASS_ORDER)
        z_suffix = "_z" if variant == "rating" else f"_z_{variant}"
        standardization: dict[str, tuple[float, float]] = {}
        for field, _ in VARIABLES:
            scores, mean, standard_deviation = _zscore(
                merged.loc[classified, field]
            )
            z_column = f"{field}{z_suffix}"
            merged[z_column] = np.nan
            merged.loc[classified, z_column] = scores
            standardization[field] = (mean, standard_deviation)

        if not build_profiles:
            continue
        for field, label in VARIABLES:
            mean, standard_deviation = standardization[field]
            z_column = f"{field}{z_suffix}"
            tests = _profile_tests(
                merged,
                class_column,
                field,
                seed_key=f"{slug}|{variant}|{field}",
            )
            profile_test = tests["continuous"]
            strata_test = tests["strata"]
            high_experience = pd.to_numeric(
                merged.loc[
                    classified & merged[class_column].isin(["HH", "LH"]), z_column
                ],
                errors="coerce",
            )
            low_experience = pd.to_numeric(
                merged.loc[
                    classified & merged[class_column].isin(["HL", "LL"]), z_column
                ],
                errors="coerce",
            )
            experience_mean_difference = float(
                high_experience.mean() - low_experience.mean()
            )
            for class_name in CLASS_ORDER:
                subset = merged.loc[merged[class_column].eq(class_name)]
                values = pd.to_numeric(subset[field], errors="coerce")
                z_values = pd.to_numeric(subset[z_column], errors="coerce")
                rows.append(
                    {
                        "analysis_area": slug,
                        "experience_variant": variant,
                        "class_column": class_column,
                        "variable": field,
                        "variable_label": label,
                        "divergence_class": class_name,
                        "divergence_label": CLASS_LABELS[class_name],
                        "class_da_count": len(subset),
                        "n": int(values.notna().sum()),
                        "mean_raw": values.mean(),
                        "sd_raw": values.std(ddof=1),
                        "mean_z": z_values.mean(),
                        "standardization_mean": mean,
                        "standardization_sd": standard_deviation,
                        "standardization_scope": "classified DAs in analysis area",
                        "profile_test": "Kruskal-Wallis comparison of continuous indicator across divergence quadrants",
                        "profile_test_statistic": profile_test["statistic"],
                        "profile_test_p_value": profile_test["p_value"],
                        "profile_test_eta_squared": profile_test["eta_squared"],
                        "profile_test_n": profile_test["n"],
                        "profile_test_group_count": profile_test["group_count"],
                        "profile_test_p_method": profile_test["p_value_method"],
                        "significance_marker": _significance_marker(
                            float(profile_test["p_value"])
                        ),
                        "strata_test": "Chi-square association of equity strata and divergence quadrant",
                        "strata_test_statistic": strata_test["statistic"],
                        "strata_test_p_value": strata_test["p_value"],
                        "strata_test_cramers_v": strata_test["cramers_v"],
                        "strata_test_n": strata_test["n"],
                        "strata_test_p_method": strata_test["p_value_method"],
                        "strata_test_minimum_expected_count": strata_test[
                            "minimum_expected_count"
                        ],
                        "strata_test_expected_cells_ge_5_share": strata_test[
                            "expected_cells_ge_5_share"
                        ],
                        "strata_test_expected_count_assumptions_met": strata_test[
                            "expected_count_assumptions_met"
                        ],
                        "strata_test_monte_carlo_iterations": strata_test[
                            "monte_carlo_iterations"
                        ],
                        "strata_significance_marker": _significance_marker(
                            float(strata_test["p_value"])
                        ),
                        "experience_mean_difference": experience_mean_difference,
                    }
                )
    profile = pd.DataFrame(rows)
    merged = merged.sort_values(da_id).reset_index(drop=True)

    geometry = gpd.read_file(paths["gpkg"], layer="da_divergence")[[da_id, "geometry"]]
    geometry[da_id] = geometry[da_id].astype("string")
    _assert_unique(geometry, [da_id], "Stage 07 divergence GeoPackage")
    if set(geometry[da_id]) != set(merged[da_id]):
        raise ValueError("Stage 07 divergence CSV and GeoPackage DA IDs differ")
    mapped = geometry.merge(merged, on=da_id, how="left", validate="one_to_one")
    return merged, mapped, profile


def _compare_numeric(observed: pd.Series, expected: pd.Series, label: str) -> None:
    if not np.allclose(
        pd.to_numeric(observed, errors="coerce"),
        pd.to_numeric(expected, errors="coerce"),
        equal_nan=True,
        atol=1e-9,
    ):
        raise ValueError(f"Stored Stage 08 values are inconsistent: {label}")


def validate_equity(
    config: dict[str, Any], paths: dict[str, Path], shared: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 08 artifacts: " + ", ".join(path.name for path in missing)
        )
    da_id = config["fields"]["da_id"]
    variants = experience_variants(config)
    required = {
        da_id,
        *variants.values(),
        *(field for field, _ in EQUITY_VARIABLES),
        *(f"{field}_z" for field, _ in VARIABLES),
    }
    if "sentiment" in variants:
        required.update(f"{field}_z_sentiment" for field, _ in VARIABLES)
    observed = pd.read_csv(paths["da_equity_csv"], dtype={da_id: "string"})
    missing_columns = required - set(observed.columns)
    if missing_columns:
        raise ValueError(
            "DA equity table is missing: " + ", ".join(sorted(missing_columns))
        )
    _assert_unique(observed, [da_id], "DA equity table")
    expected, _, profile = build_equity_frames(config, shared)
    check = expected[[da_id]].merge(
        observed[[da_id]], on=da_id, how="outer", indicator=True, validate="one_to_one"
    )
    if not check["_merge"].eq("both").all():
        raise ValueError("Stage 08 DA IDs differ from the current Stage 07 data")
    expected = expected.set_index(da_id).loc[observed[da_id]].reset_index()
    for variant, class_column in variants.items():
        if not observed[class_column].astype("string").equals(
            expected[class_column].astype("string")
        ):
            raise ValueError(
                f"Stored Stage 08 {variant} divergence classes are stale"
            )
    for field, _ in EQUITY_VARIABLES:
        _compare_numeric(observed[field], expected[field], field)
    for field, _ in VARIABLES:
        _compare_numeric(observed[f"{field}_z"], expected[f"{field}_z"], f"{field}_z")
        if "sentiment" in variants:
            _compare_numeric(
                observed[f"{field}_z_sentiment"],
                expected[f"{field}_z_sentiment"],
                f"{field}_z_sentiment",
            )

    mapped = gpd.read_file(paths["da_equity_gpkg"], layer="da_equity")
    mapped[da_id] = mapped[da_id].astype("string")
    _assert_unique(mapped, [da_id], "DA equity GeoPackage")
    if set(mapped[da_id]) != set(observed[da_id]):
        raise ValueError("DA equity GeoPackage IDs differ from the CSV")
    if str(mapped.crs) != config["crs"]["projected"]:
        raise ValueError(
            f"DA equity CRS is {mapped.crs}; expected {config['crs']['projected']}"
        )

    variable_fields = [field for field, _ in EQUITY_VARIABLES]
    rating_classified = observed["divergence_class"].isin(CLASS_ORDER)
    metrics: dict[str, Any] = {
        "da_count": len(observed),
        "classified_da_count": int(rating_classified.sum()),
        "matched_any_da_count": int(observed[variable_fields].notna().any(axis=1).sum()),
        "matched_all_da_count": int(observed[variable_fields].notna().all(axis=1).sum()),
        "crs": config["crs"]["projected"],
    }
    for variant, class_column in variants.items():
        classified = observed[class_column].isin(CLASS_ORDER)
        metrics[f"{variant}_classified_da_count"] = int(classified.sum())
        for field, _ in EQUITY_VARIABLES:
            metrics[f"{variant}_{field}_classified_nonmissing_da_count"] = int(
                observed.loc[classified, field].notna().sum()
            )
    for field, _ in EQUITY_VARIABLES:
        metrics[f"{field}_nonmissing_da_count"] = int(observed[field].notna().sum())
        metrics[f"{field}_classified_nonmissing_da_count"] = int(
            observed.loc[rating_classified, field].notna().sum()
        )
    return metrics, profile


def build_equity(
    config: dict[str, Any], final_paths: dict[str, Path], shared: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Profile tests are calculated once by the validation pass below. Skipping
    # them here avoids repeating Monte Carlo work during a forced rebuild.
    frame, mapped, _ = build_equity_frames(config, shared, build_profiles=False)
    slug = config["analysis_area"]["slug"]
    final_paths["da_equity_csv"].parent.mkdir(parents=True, exist_ok=True)
    temp_parent = final_paths["da_equity_csv"].parent.parent
    with inherited_temp_directory(temp_parent, f"{slug}_08_") as temp_dir:
        temp_paths = equity_paths(config, temp_dir)
        temp_paths["da_equity_csv"].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(temp_paths["da_equity_csv"], index=False)
        mapped.to_file(temp_paths["da_equity_gpkg"], layer="da_equity", driver="GPKG")
        metrics, profile = validate_equity(config, temp_paths, shared)
        for key, target in final_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_paths[key], target)
    return metrics, profile


def write_summaries(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics: dict[str, Any],
    profile: pd.DataFrame,
    status: str,
    shared_status: str,
) -> tuple[Path, dict[str, Path]]:
    slug = config["analysis_area"]["slug"]
    table_dir = output_root(config) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / f"{slug}_08_equity_summary.csv"
    profile_paths = equity_profile_paths(config)
    rows = []
    for variant in experience_variants(config):
        for field, label in EQUITY_VARIABLES:
            rows.append(
                {
                    "analysis_area": slug,
                    "experience_variant": variant,
                    "status": status,
                    "shared_extract_status": shared_status,
                    "variable": field,
                    "variable_label": label,
                    "source": "Can-ALE 2021" if field == "ALE_index" else "2021 Census Profile",
                    "da_count": metrics["da_count"],
                    "classified_da_count": metrics[f"{variant}_classified_da_count"],
                    "nonmissing_da_count": metrics[f"{field}_nonmissing_da_count"],
                    "classified_nonmissing_da_count": metrics[
                        f"{variant}_{field}_classified_nonmissing_da_count"
                    ],
                    "nonmissing_share": (
                        metrics[f"{field}_nonmissing_da_count"] / metrics["da_count"]
                        if metrics["da_count"]
                        else np.nan
                    ),
                    "da_equity_csv_path": str(paths["da_equity_csv"]),
                    "da_equity_gpkg_path": str(paths["da_equity_gpkg"]),
                    "profile_path": str(profile_paths[variant]),
                }
            )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    # Keep the analytical profile byte-stable between rebuild and reuse runs.
    # Execution status belongs in the QA summary, not in a Stage 10 source table.
    for variant, profile_path in profile_paths.items():
        profile.loc[profile["experience_variant"].eq(variant)].to_csv(
            profile_path, index=False
        )
    return summary_path, profile_paths


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = equity_paths(config)
    slug = config["analysis_area"]["slug"]
    print(f"\nStage 08: equity data ({slug})")
    print(
        "Indicators: median household income, six published Census Profile rates, "
        "and the 2021 Can-ALE index."
    )
    shared, shared_status = load_shared(config)
    print(f"Shared six-city equity rows: {len(shared):,} ({shared_status})")

    if force:
        print("Force enabled: building new area artifacts atomically.")
        metrics, profile = build_equity(config, paths, shared)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing area artifacts.")
        try:
            metrics, profile = validate_equity(config, paths, shared)
        except (FileNotFoundError, ValueError) as error:
            print(f"Equity validation failed: {error}")
            print("Build this area's Stage 08 artifacts explicitly with --force.")
            return 1
        status = "reused"

    summary_path, profile_paths = write_summaries(
        config, paths, metrics, profile, status, shared_status
    )
    print(f"Validated DAs: {metrics['da_count']:,}")
    print(f"Classified DAs: {metrics['classified_da_count']:,}")
    print(
        f"Complete on all {len(EQUITY_VARIABLES)} indicators: "
        f"{metrics['matched_all_da_count']:,}/{metrics['da_count']:,} DAs"
    )
    print(f"Coverage summary: {summary_path}")
    for variant, profile_path in profile_paths.items():
        print(f"{variant.title()} divergence profiles: {profile_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build and atomically replace this area's Stage 08 artifacts",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
