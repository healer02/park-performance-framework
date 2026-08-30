"""Shared equity strata and assumption-aware association testing."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


CLASS_ORDER = ("HH", "LH", "HL", "LL")
MONTE_CARLO_ITERATIONS = 4_999


def equity_strata(values: pd.Series, field: str) -> pd.Series:
    """Apply the documented within-area or fixed equity strata."""
    numeric = pd.to_numeric(values, errors="coerce")
    if field == "medhhinc":
        area_median = float(numeric.median())
        if not np.isfinite(area_median) or area_median <= 0:
            return pd.Series(pd.NA, index=values.index, dtype="object")
        return pd.cut(
            numeric,
            bins=[0, area_median * 0.6, area_median * 1.4, np.inf],
            include_lowest=True,
        )
    if field == "ALE_index":
        # Active living environment is grouped using within-area tertiles.
        return pd.qcut(numeric, q=3, duplicates="drop")
    bins = {
        "pct_visible_minority": [0, 20, 50, 100],
        "pct_age_0_14": [0, 10, 20, 100],
        "pct_age_65plus": [0, 10, 20, 100],
        "pct_LIM_AT": [0, 20, 35, 100],
        "pct_immigrant": [0, 30, 50, 100],
        "pct_bachelor_plus": [0, 30, 50, 100],
    }
    if field not in bins:
        raise ValueError(f"No equity stratum is configured for {field}")
    return pd.cut(numeric, bins=bins[field], include_lowest=True)


def kruskal_wallis_association(
    values: pd.Series, classes: pd.Series
) -> dict[str, float | int | str]:
    """Compare a continuous indicator across divergence classes.

    The eta-squared estimate follows ``(H - k + 1) / (n - k)`` and is bounded
    at zero. This test supplies the asterisks in the continuous equity-profile
    dot plot; chi-square remains available for categorized equity strata.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & classes.isin(CLASS_ORDER)
    numeric = numeric.loc[valid]
    classes_valid = classes.loc[valid].astype("string")
    groups = [
        numeric.loc[classes_valid.eq(name)].to_numpy(dtype="float64")
        for name in CLASS_ORDER
        if classes_valid.eq(name).any()
    ]
    sample_size = int(len(numeric))
    group_count = int(len(groups))
    empty = {
        "statistic": np.nan,
        "p_value": np.nan,
        "degrees_of_freedom": max(group_count - 1, 0),
        "eta_squared": np.nan,
        "n": sample_size,
        "group_count": group_count,
        "p_value_method": "not_tested",
    }
    if group_count < 2 or sample_size <= group_count or numeric.nunique() < 2:
        return empty
    statistic, p_value = kruskal(*groups)
    eta_squared = max(
        0.0, float((statistic - group_count + 1) / (sample_size - group_count))
    )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": group_count - 1,
        "eta_squared": eta_squared,
        "n": sample_size,
        "group_count": group_count,
        "p_value_method": "scipy_asymptotic",
    }


def _stable_seed(seed_key: str) -> int:
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def chi_square_association(
    strata: pd.Series,
    classes: pd.Series,
    seed_key: str,
    iterations: int = MONTE_CARLO_ITERATIONS,
) -> dict[str, float | int | bool | str]:
    """Test independence, using a Monte Carlo p-value when counts are sparse.

    The Pearson statistic and Cramer's V are always reported. The asymptotic
    p-value is used only when every expected count is at least one and at least
    80% of expected counts are five or greater. Otherwise, class labels are
    permuted while both observed margins remain fixed.
    """
    valid = strata.notna() & classes.isin(CLASS_ORDER)
    strata_valid = strata.loc[valid]
    classes_valid = classes.loc[valid].astype("string")
    table = pd.crosstab(strata_valid, classes_valid)
    observed_classes = [name for name in CLASS_ORDER if name in table.columns]
    table = table.reindex(columns=observed_classes)
    table = table.loc[table.sum(axis=1).gt(0), table.sum(axis=0).gt(0)]
    sample_size = int(table.to_numpy().sum())
    empty = {
        "statistic": np.nan,
        "p_value": np.nan,
        "degrees_of_freedom": np.nan,
        "cramers_v": np.nan,
        "n": sample_size,
        "stratum_count": int(table.shape[0]),
        "quadrant_count": int(table.shape[1]),
        "p_value_method": "not_tested",
        "minimum_expected_count": np.nan,
        "expected_cells_ge_5_share": np.nan,
        "expected_count_assumptions_met": False,
        "monte_carlo_iterations": 0,
    }
    if table.shape[0] < 2 or table.shape[1] < 2 or sample_size == 0:
        return empty

    statistic, asymptotic_p, dof, expected = chi2_contingency(
        table.to_numpy(), correction=False
    )
    minimum_expected = float(expected.min())
    expected_ge_5_share = float((expected >= 5).mean())
    assumptions_met = minimum_expected >= 1 and expected_ge_5_share >= 0.8
    p_value = float(asymptotic_p)
    method = "pearson_asymptotic"
    iterations_used = 0

    if not assumptions_met:
        if iterations < 1:
            raise ValueError("Monte Carlo iterations must be positive")
        # ``table.index`` can be a CategoricalIndex that still carries unused
        # source categories. Convert it to ordinary values so the codes match
        # the table's actual row count exactly.
        row_categories = table.index.astype("object").tolist()
        row_codes = pd.Categorical(
            strata_valid.astype("object"),
            categories=row_categories,
            ordered=True,
        ).codes
        class_lookup = {name: index for index, name in enumerate(table.columns)}
        class_codes = classes_valid.map(class_lookup).to_numpy(dtype="int64")
        row_count, column_count = table.shape
        rng = np.random.default_rng(_stable_seed(seed_key))
        extreme_count = 0
        for _ in range(iterations):
            shuffled = rng.permutation(class_codes)
            permuted = np.bincount(
                row_codes * column_count + shuffled,
                minlength=row_count * column_count,
            ).reshape(row_count, column_count)
            permuted_statistic = float(
                np.sum(np.square(permuted - expected) / expected)
            )
            if permuted_statistic >= statistic - 1e-12:
                extreme_count += 1
        p_value = (extreme_count + 1) / (iterations + 1)
        method = "monte_carlo_label_permutation"
        iterations_used = iterations

    minimum_dimension = min(table.shape) - 1
    cramers_v = np.sqrt(statistic / (sample_size * minimum_dimension))
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "n": sample_size,
        "stratum_count": int(table.shape[0]),
        "quadrant_count": int(table.shape[1]),
        "p_value_method": method,
        "minimum_expected_count": minimum_expected,
        "expected_cells_ge_5_share": expected_ge_5_share,
        "expected_count_assumptions_met": assumptions_met,
        "monte_carlo_iterations": iterations_used,
    }
