"""Run reproducible statistical analyses for each enabled experience measure."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from equity_methods import (
    CLASS_ORDER,
    chi_square_association,
    equity_strata,
    kruskal_wallis_association,
)
from project_config import ConfigError, inherited_temp_directory, load_config, output_root, workspace_data_root


MODEL_CLASS_ORDER = ["LL", "HH", "LH", "HL"]
MODEL_PREDICTORS = [
    ("pct_age_0_14", "Children aged 0-14 (%)"),
    ("pct_age_65plus", "Age 65+ (%)"),
    ("pct_visible_minority", "Visible minority (%)"),
    ("pct_LIM_AT", "LIM-AT (%)"),
    ("pct_bachelor_plus", "Education (bachelor's+)"),
    ("ALE_index", "Active living environment"),
]
MIN_MODEL_N = 100
MIN_CLASS_N = 20
MORAN_PERMUTATIONS = 999

DESCRIPTIVE_COMMON_VARIABLES = [
    ("park_access_pop_coverage", "Population with park access", "%"),
    ("supply_ha_per_1000_cap20", "Accessible park area", "ha per 1,000"),
    ("digital_salience_reviews_per_1000", "Digital salience", "reviews per 1,000"),
    ("medhhinc", "Median household income", "2020 dollars"),
    ("pct_bachelor_plus", "Education (bachelor's+)", "%"),
    ("ALE_index", "Active living environment", "index"),
    ("pct_age_0_14", "Children aged 0-14", "%"),
    ("pct_age_65plus", "Age 65+", "%"),
    ("pct_LIM_AT", "Low income (LIM-AT)", "%"),
    ("pct_immigrant", "Immigrant share", "%"),
    ("pct_visible_minority", "Visible minority", "%"),
]


def _experience_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    specs = [{
        "name": "rating",
        "score_column": "experience_rating_mean",
        "score_label": "Google star rating",
        "score_unit": "stars",
        "class_column": "divergence_class",
        "suffix": "",
    }]
    if config.get("experience", {}).get("sentiment", {}).get("enabled", False):
        specs.append({
            "name": "sentiment",
            "score_column": "experience_sentiment_mean",
            "score_label": "Mean sentiment score",
            "score_unit": "score (-1 to 1)",
            "class_column": "divergence_sentiment_sensitivity",
            "suffix": "_sentiment",
        })
    return specs


def _significance_marker(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _paths(config: dict[str, Any], variant: str = "rating") -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    specs = {spec["name"]: spec for spec in _experience_specs(config)}
    if variant not in specs:
        raise ValueError(f"Experience variant is not enabled: {variant}")
    suffix = specs[variant]["suffix"]
    base = output_root(config) / "tables"
    equity = workspace_data_root(config) / "interim" / "equity" / slug
    return {
        "equity": equity / f"{slug}_da_equity.csv",
        "equity_gpkg": equity / f"{slug}_da_equity.gpkg",
        "descriptive": base / f"{slug}_09_descriptive_statistics{suffix}.csv",
        "spearman": base / f"{slug}_09_spearman{suffix}.csv",
        "moran": base / f"{slug}_09_global_moran{suffix}.csv",
        "kruskal": base / f"{slug}_09_equity_kruskal_wallis{suffix}.csv",
        "chi_square": base / f"{slug}_09_equity_chi_square{suffix}.csv",
        "vif": base / f"{slug}_09_vif{suffix}.csv",
        "multinomial": base / f"{slug}_09_multinomial_logit{suffix}.csv",
        "multinomial_formatted": base / f"{slug}_09_multinomial_logit_formatted{suffix}.csv",
        "model_fit": base / f"{slug}_09_model_fit{suffix}.csv",
        "summary": base / f"{slug}_09_statistics_summary{suffix}.csv",
    }


def _load_equity(config: dict[str, Any]) -> pd.DataFrame:
    path = _paths(config)["equity"]
    if not path.exists():
        raise FileNotFoundError(f"Stage 08 equity data is required: {path}")
    frame = pd.read_csv(path, dtype={config["fields"]["da_id"]: "string"})
    required = {
        "da_pop",
        "park_access_pop_coverage",
        "supply_ha_per_1000_cap20",
        "experience_rating_mean",
        "digital_salience_reviews_per_1000",
        *(field for field, _ in MODEL_PREDICTORS),
        "pct_immigrant",
        "medhhinc",
    }
    for spec in _experience_specs(config):
        required.update({spec["score_column"], spec["class_column"]})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Stage 08 equity data is missing: " + ", ".join(sorted(missing)))
    return frame


def _load_equity_geometry(config: dict[str, Any]) -> gpd.GeoDataFrame:
    path = _paths(config)["equity_gpkg"]
    if not path.exists():
        raise FileNotFoundError(f"Stage 08 equity geometry is required: {path}")
    da_id = config["fields"]["da_id"]
    geometry = gpd.read_file(path, layer="da_equity", columns=[da_id, "geometry"])
    geometry[da_id] = geometry[da_id].astype("string")
    if geometry.empty or geometry.geometry.isna().any():
        raise ValueError("Stage 08 equity geometry has missing DA polygons")
    if geometry[da_id].duplicated().any():
        raise ValueError("Stage 08 equity geometry contains duplicate DA IDs")
    return geometry


def _descriptive_statistics(
    frame: pd.DataFrame, area: str, spec: dict[str, str]
) -> pd.DataFrame:
    rows = []
    populated = pd.to_numeric(frame["da_pop"], errors="coerce").gt(0)
    variables = [
        *DESCRIPTIVE_COMMON_VARIABLES[:2],
        (spec["score_column"], spec["score_label"], spec["score_unit"]),
        *DESCRIPTIVE_COMMON_VARIABLES[2:],
    ]
    for field, label, unit in variables:
        values = pd.to_numeric(frame.loc[populated, field], errors="coerce")
        if field == "park_access_pop_coverage":
            values = values * 100
        observed = values.dropna()
        quantiles = observed.quantile([0.25, 0.5, 0.75]) if not observed.empty else {}
        rows.append({
            "analysis_area": area,
            "experience_variant": spec["name"],
            "variable": field,
            "variable_label": label,
            "unit": unit,
            "eligible_da_count": int(populated.sum()),
            "n": int(observed.size),
            "missing_n": int(populated.sum() - observed.size),
            "mean": observed.mean(),
            "standard_deviation": observed.std(ddof=1),
            "minimum": observed.min(),
            "p25": quantiles.get(0.25, np.nan),
            "median": quantiles.get(0.5, np.nan),
            "p75": quantiles.get(0.75, np.nan),
            "maximum": observed.max(),
        })
    return pd.DataFrame(rows)


def _spearman(
    frame: pd.DataFrame, area: str, spec: dict[str, str]
) -> pd.DataFrame:
    score_column = spec["score_column"]
    score_label = spec["score_label"]
    relationships = [
        ("Park coverage", "park_access_pop_coverage", score_label, score_column),
        ("Accessible park area (ha/1,000)", "supply_ha_per_1000_cap20", score_label, score_column),
        ("Park coverage", "park_access_pop_coverage", "Digital salience (reviews/1,000)", "digital_salience_reviews_per_1000"),
        ("Accessible park area (ha/1,000)", "supply_ha_per_1000_cap20", "Digital salience (reviews/1,000)", "digital_salience_reviews_per_1000"),
        (score_label, score_column, "Digital salience (reviews/1,000)", "digital_salience_reviews_per_1000"),
    ]
    rows = []
    for x_label, x_column, y_label, y_column in relationships:
        pairs = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pairs) < 3 or pairs[x_column].nunique() < 2 or pairs[y_column].nunique() < 2:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(pairs[x_column], pairs[y_column])
        rows.append({
            "analysis_area": area,
            "experience_variant": spec["name"],
            "x_variable": x_column,
            "x_label": x_label,
            "y_variable": y_column,
            "y_label": y_label,
            "n": len(pairs),
            "spearman_rho": rho,
            "p_value": p_value,
            "significance": _significance_marker(p_value),
        })
    return pd.DataFrame(rows)


def _stable_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _queen_edges(geometry: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return unique directed Queen-contiguity edges for DA polygons."""
    base = geometry.reset_index(drop=True)[["geometry"]]
    joined = gpd.sjoin(base, base, how="inner", predicate="touches")
    left = joined.index.to_numpy(dtype="int64")
    right = joined["index_right"].to_numpy(dtype="int64")
    different = left != right
    left, right = left[different], right[different]
    size = len(base)
    encoded = np.concatenate([left * size + right, right * size + left])
    encoded = np.unique(encoded)
    return encoded // size, encoded % size


def _moran_statistic(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, int, int]:
    centered = values - values.mean()
    denominator = float(np.square(centered).sum())
    row_counts = np.bincount(left, minlength=len(values))
    non_islands = row_counts > 0
    island_count = int((~non_islands).sum())
    if denominator <= 0 or not left.size or not non_islands.any():
        return np.nan, island_count, 0
    weights = 1.0 / row_counts[left]
    s0 = float(weights.sum())
    statistic = (len(values) / s0) * float(
        np.sum(weights * centered[left] * centered[right])
    ) / denominator
    return statistic, island_count, int(left.size)


def _global_moran(
    frame: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    area: str,
    area_slug: str,
    spec: dict[str, str],
) -> pd.DataFrame:
    da_id = next(
        column for column in geometry.columns if column != geometry.geometry.name
    )
    attributes = frame.copy()
    attributes[da_id] = attributes[da_id].astype("string")
    spatial = geometry[[da_id, "geometry"]].copy()
    spatial[da_id] = spatial[da_id].astype("string")
    spatial = spatial.merge(attributes, on=da_id, how="left", validate="one_to_one")
    all_left, all_right = _queen_edges(spatial)
    variables: list[tuple[str, str, pd.Series]] = [
        (
            "park_access_pop_coverage",
            "Population park-access coverage",
            pd.to_numeric(spatial["park_access_pop_coverage"], errors="coerce"),
        ),
        (
            "supply_ha_per_1000_cap20",
            "Accessible park area (ha per 1,000)",
            pd.to_numeric(spatial["supply_ha_per_1000_cap20"], errors="coerce"),
        ),
        (
            spec["score_column"],
            spec["score_label"],
            pd.to_numeric(spatial[spec["score_column"]], errors="coerce"),
        ),
    ]
    class_column = spec["class_column"]
    classified = spatial[class_column].isin(CLASS_ORDER)
    for class_name in CLASS_ORDER:
        indicator = pd.Series(np.nan, index=spatial.index, dtype="float64")
        indicator.loc[classified] = spatial.loc[
            classified, class_column
        ].eq(class_name).astype(float)
        variables.append(
            (
                f"divergence_{class_name}",
                f"{class_name} divergence membership",
                indicator,
            )
        )

    rows = []
    for field, label, series in variables:
        valid = series.notna().to_numpy()
        original_indices = np.flatnonzero(valid)
        lookup = np.full(len(spatial), -1, dtype="int64")
        lookup[original_indices] = np.arange(len(original_indices))
        edge_valid = valid[all_left] & valid[all_right]
        left = lookup[all_left[edge_valid]]
        right = lookup[all_right[edge_valid]]
        values = series.iloc[original_indices].to_numpy(dtype="float64")
        observed, islands, directed_edges = _moran_statistic(values, left, right)
        expected = -1 / (len(values) - 1) if len(values) > 1 else np.nan
        p_value = np.nan
        if np.isfinite(observed) and len(values) >= 3:
            rng = np.random.default_rng(
                _stable_seed(f"{area_slug}|{spec['name']}|moran|{field}")
            )
            extreme = 0
            observed_distance = abs(observed - expected)
            for _ in range(MORAN_PERMUTATIONS):
                simulated, _, _ = _moran_statistic(
                    rng.permutation(values), left, right
                )
                if abs(simulated - expected) >= observed_distance - 1e-12:
                    extreme += 1
            p_value = (extreme + 1) / (MORAN_PERMUTATIONS + 1)
        rows.append({
            "analysis_area": area,
            "experience_variant": spec["name"],
            "variable": field,
            "variable_label": label,
            "n": len(values),
            "queen_directed_edge_count": directed_edges,
            "island_da_count": islands,
            "weights": "Queen contiguity, row standardized",
            "moran_i": observed,
            "expected_i_randomization": expected,
            "permutation_p_value_two_sided": p_value,
            "permutations": MORAN_PERMUTATIONS,
            "significance": _significance_marker(p_value),
        })
    return pd.DataFrame(rows)


def _chi_square(
    frame: pd.DataFrame,
    area: str,
    area_slug: str,
    spec: dict[str, str],
) -> pd.DataFrame:
    variables = [
        ("medhhinc", "Median household income"),
        ("pct_bachelor_plus", "Education (bachelor's+)"),
        ("ALE_index", "Active living environment"),
        ("pct_age_0_14", "Children aged 0-14"),
        ("pct_age_65plus", "Age 65+"),
        ("pct_LIM_AT", "Low income (LIM-AT)"),
        ("pct_immigrant", "Immigrant share"),
        ("pct_visible_minority", "Visible minority"),
    ]
    class_column = spec["class_column"]
    classified = frame[class_column].isin(CLASS_ORDER)
    rows = []
    for field, label in variables:
        # Define within-area strata (ALE tertiles and income bands) before
        # restricting the test to DAs with a valid divergence classification.
        strata = equity_strata(frame[field], field)
        test = chi_square_association(
            strata.loc[classified],
            frame.loc[classified, class_column],
            seed_key=f"{area_slug}|{spec['name']}|{field}",
        )
        rows.append({
            "analysis_area": area,
            "experience_variant": spec["name"],
            "variable": field,
            "variable_label": label,
            "test": "Chi-square association of equity strata and divergence quadrant",
            "n": test["n"],
            "stratum_count": test["stratum_count"],
            "quadrant_count": test["quadrant_count"],
            "chi_square": test["statistic"],
            "degrees_of_freedom": test["degrees_of_freedom"],
            "p_value": test["p_value"],
            "p_value_method": test["p_value_method"],
            "minimum_expected_count": test["minimum_expected_count"],
            "expected_cells_ge_5_share": test["expected_cells_ge_5_share"],
            "expected_count_assumptions_met": test[
                "expected_count_assumptions_met"
            ],
            "monte_carlo_iterations": test["monte_carlo_iterations"],
            "significance": _significance_marker(float(test["p_value"])),
            "cramers_v": test["cramers_v"],
        })
    return pd.DataFrame(rows)


def _kruskal_wallis(
    frame: pd.DataFrame, area: str, spec: dict[str, str]
) -> pd.DataFrame:
    variables = [
        ("pct_bachelor_plus", "Education (bachelor's+)"),
        ("ALE_index", "Active living environment"),
        ("pct_age_0_14", "Children aged 0-14"),
        ("pct_age_65plus", "Age 65+"),
        ("pct_LIM_AT", "Low income (LIM-AT)"),
        ("pct_immigrant", "Immigrant share"),
        ("pct_visible_minority", "Visible minority"),
    ]
    class_column = spec["class_column"]
    rows = []
    for field, label in variables:
        test = kruskal_wallis_association(frame[field], frame[class_column])
        rows.append({
            "analysis_area": area,
            "experience_variant": spec["name"],
            "variable": field,
            "variable_label": label,
            "test": "Kruskal-Wallis comparison across divergence quadrants",
            "n": test["n"],
            "group_count": test["group_count"],
            "h_statistic": test["statistic"],
            "degrees_of_freedom": test["degrees_of_freedom"],
            "p_value": test["p_value"],
            "p_value_method": test["p_value_method"],
            "eta_squared": test["eta_squared"],
            "significance": _significance_marker(float(test["p_value"])),
        })
    return pd.DataFrame(rows)


def _fit_models(
    frame: pd.DataFrame, area: str, spec: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fields = [field for field, _ in MODEL_PREDICTORS]
    labels = dict(MODEL_PREDICTORS)
    class_column = spec["class_column"]
    data = frame.loc[
        frame[class_column].isin(CLASS_ORDER), [class_column, *fields]
    ].copy()
    data[fields] = data[fields].apply(pd.to_numeric, errors="coerce")
    data = data.dropna().copy()
    class_counts = data[class_column].value_counts().reindex(CLASS_ORDER, fill_value=0)
    eligible = len(data) >= MIN_MODEL_N and int(class_counts.min()) >= MIN_CLASS_N
    fit_row = {
        "analysis_area": area,
        "experience_variant": spec["name"],
        "model": "Multinomial logit; reference = LL",
        "n": len(data),
        **{f"{name}_count": int(class_counts[name]) for name in CLASS_ORDER},
        "minimum_model_n": MIN_MODEL_N,
        "minimum_class_n": MIN_CLASS_N,
        "model_eligible": eligible,
    }
    empty_vif = pd.DataFrame(columns=["analysis_area", "experience_variant", "predictor", "vif"])
    empty_results = pd.DataFrame(columns=["analysis_area", "experience_variant", "outcome_vs_LL", "predictor", "odds_ratio", "ci_lower", "ci_upper", "p_value", "significance"])
    if not eligible:
        fit_row.update({"status": "not_run_insufficient_sample_or_class_count"})
        return empty_vif, empty_results, pd.DataFrame([fit_row])

    standardized = (data[fields] - data[fields].mean()) / data[fields].std(ddof=1)
    design = sm.add_constant(standardized, has_constant="add")
    vif = pd.DataFrame({
        "analysis_area": area,
        "experience_variant": spec["name"],
        "predictor": fields,
        "predictor_label": [labels[field] for field in fields],
        "vif": [variance_inflation_factor(design.to_numpy(), index + 1) for index in range(len(fields))],
    })
    outcome = pd.Categorical(
        data[class_column], categories=MODEL_CLASS_ORDER
    ).codes
    result = sm.MNLogit(outcome, design).fit(method="newton", maxiter=200, disp=False)
    rows = []
    for column, outcome_name in enumerate(MODEL_CLASS_ORDER[1:]):
        for field in fields:
            coefficient = float(result.params.iloc[design.columns.get_loc(field), column])
            standard_error = float(result.bse.iloc[design.columns.get_loc(field), column])
            p_value = float(result.pvalues.iloc[design.columns.get_loc(field), column])
            rows.append({
                "analysis_area": area,
                "experience_variant": spec["name"],
                "outcome_vs_LL": outcome_name,
                "predictor": field,
                "predictor_label": labels[field],
                "odds_ratio": np.exp(coefficient),
                "ci_lower": np.exp(coefficient - 1.96 * standard_error),
                "ci_upper": np.exp(coefficient + 1.96 * standard_error),
                "p_value": p_value,
                "significance": _significance_marker(p_value),
            })
    log_likelihood_null = float(result.llnull)
    log_likelihood_model = float(result.llf)
    cox_snell = 1 - np.exp((2 / len(data)) * (log_likelihood_null - log_likelihood_model))
    fit_row.update({
        "status": "fitted",
        "log_likelihood": log_likelihood_model,
        "null_log_likelihood": log_likelihood_null,
        "likelihood_ratio_p_value": float(result.llr_pvalue),
        "mcfadden_r2": 1 - (log_likelihood_model / log_likelihood_null),
        "cox_snell_r2": cox_snell,
        "nagelkerke_r2": cox_snell / (1 - np.exp((2 / len(data)) * log_likelihood_null)),
    })
    return vif, pd.DataFrame(rows), pd.DataFrame([fit_row])


def _format_multinomial(
    results: pd.DataFrame, model_fit: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "analysis_area",
        "experience_variant",
        "outcome_vs_LL",
        "predictor_label",
        "odds_ratio",
        "confidence_interval_95",
        "p_value",
        "significance",
        "reference_category",
        "model_n",
        "mcfadden_r2",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    fit = model_fit.iloc[0]
    formatted = results.copy()
    formatted["confidence_interval_95"] = formatted.apply(
        lambda row: f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]", axis=1
    )
    formatted["reference_category"] = "LL — Low supply / low experience"
    formatted["model_n"] = int(fit["n"])
    formatted["mcfadden_r2"] = fit.get("mcfadden_r2", np.nan)
    return formatted[columns]


def _write_tables(paths: dict[str, Path], tables: dict[str, pd.DataFrame]) -> None:
    parent = paths["summary"].parent
    parent.mkdir(parents=True, exist_ok=True)
    with inherited_temp_directory(parent, "09_statistics_") as temp_dir:
        for name, table in tables.items():
            target = paths[name]
            temporary = temp_dir / target.name
            table.to_csv(temporary, index=False)
            os.replace(temporary, target)


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    specs = _experience_specs(config)
    paths_by_variant = {
        spec["name"]: _paths(config, spec["name"]) for spec in specs
    }
    outputs = [
        path
        for paths in paths_by_variant.values()
        for name, path in paths.items()
        if name not in {"equity", "equity_gpkg"}
    ]
    if not force and all(path.exists() for path in outputs):
        print("Stage 09 outputs already exist. Use --force to recalculate them.")
        return 0
    frame = _load_equity(config)
    geometry = _load_equity_geometry(config)
    area_name = config["analysis_area"]["name"]
    area_slug = config["analysis_area"]["slug"]
    for spec in specs:
        variant = spec["name"]
        paths = paths_by_variant[variant]
        descriptive = _descriptive_statistics(frame, area_name, spec)
        spearman = _spearman(frame, area_name, spec)
        moran = _global_moran(frame, geometry, area_name, area_slug, spec)
        kruskal_table = _kruskal_wallis(frame, area_name, spec)
        chi_square = _chi_square(frame, area_name, area_slug, spec)
        vif, multinomial, model_fit = _fit_models(frame, area_name, spec)
        multinomial_formatted = _format_multinomial(multinomial, model_fit)
        summary = pd.DataFrame([{
            "analysis_area": area_name,
            "experience_variant": variant,
            "descriptive_variables": len(descriptive),
            "spearman_tests": len(spearman),
            "global_moran_tests": len(moran),
            "kruskal_wallis_tests": len(kruskal_table),
            "chi_square_tests": len(chi_square),
            "model_status": model_fit.loc[0, "status"],
            "model_n": model_fit.loc[0, "n"],
        }])
        _write_tables(paths, {
            "descriptive": descriptive,
            "spearman": spearman,
            "moran": moran,
            "kruskal": kruskal_table,
            "chi_square": chi_square,
            "vif": vif,
            "multinomial": multinomial,
            "multinomial_formatted": multinomial_formatted,
            "model_fit": model_fit,
            "summary": summary,
        })
        print(f"Stage 09 {variant} statistics complete: {area_name}")
        print(f"Descriptive table: {paths['descriptive']}")
        print(f"Spearman table: {paths['spearman']}")
        print(f"Global Moran table: {paths['moran']}")
        print(f"Equity Kruskal-Wallis table: {paths['kruskal']}")
        print(f"Equity chi-square table: {paths['chi_square']}")
        print(f"Model fit: {paths['model_fit']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force", action="store_true", help="Recalculate Stage 09 tables"
    )
    arguments = parser.parse_args()
    try:
        raise SystemExit(main(arguments.area, force=arguments.force))
    except (ConfigError, FileNotFoundError, ValueError) as error:
        print(f"Stage 09 failed: {error}")
        raise SystemExit(2)
