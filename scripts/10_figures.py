"""Create or validate publication-ready divergence and equity figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from equity_methods import equity_strata
from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    output_root,
    workspace_data_root,
)


STYLE_VERSION = "divergence_2x2_titleless_v4"
EQUITY_STYLE_VERSION = "equity_profiles_proposal_profile_v6"
RELATIONSHIP_STYLE_VERSION = "proposal_figure2_rating_v3"
STACKED_STYLE_VERSION = "equity_strata_stacked_dynamic_v2"
TRANSFERABILITY_STYLE_VERSION = "six_city_equal_2x3_v1"
EQUITY_X_LIMIT = 0.8
TRANSFERABILITY_CITIES = [
    ("vancouver", "Vancouver", "Vancouver"),
    ("burnaby", "Burnaby", "Burnaby"),
    ("richmond", "Richmond", "Richmond"),
    ("surrey", "Surrey", "Surrey"),
    ("new_westminster", "New Westminster", "New Westminster"),
    ("coquitlam", "Coquitlam", "Coquitlam"),
]
CLASS_ORDER = [
    "HH",
    "HL",
    "LH",
    "LL",
    "insufficient_experience",
    "insufficient_population",
]
CLASS_LABELS = {
    "HH": "HH — High supply, high experience",
    "HL": "HL — High supply, low experience",
    "LH": "LH — Low supply, high experience",
    "LL": "LL — Low supply, low experience",
    "insufficient_experience": "Insufficient experience data",
    "insufficient_population": "Insufficient population",
}
EQUITY_VARIABLES = [
    ("pct_bachelor_plus", "Education (bachelor's+)"),
    ("ALE_index", "Active living environment"),
    ("pct_age_0_14", "Children aged 0-14"),
    ("pct_age_65plus", "Age 65+"),
    ("pct_LIM_AT", "Low income (LIM-AT)"),
    ("pct_immigrant", "Immigrant share"),
    ("pct_visible_minority", "Visible minority"),
]
EQUITY_CLASS_ORDER = ["HH", "LH", "HL", "LL"]
SHORT_CLASS_LABELS = {
    "HH": "High supply, high experience",
    "LH": "Low supply, high experience",
    "HL": "High supply, low experience",
    "LL": "Low supply, low experience",
}
STRATUM_LABELS = {
    "pct_bachelor_plus": ["Low (<30%)", "Middle (30–50%)", "High (>50%)"],
    "ALE_index": ["Low", "Middle", "High"],
    "pct_age_0_14": ["Low (<10%)", "Middle (10–20%)", "High (>20%)"],
    "pct_age_65plus": ["Younger (<10%)", "Middle (10–20%)", "Older (>20%)"],
    "pct_LIM_AT": ["Low (<20%)", "Middle (20–35%)", "High (>35%)"],
    "pct_immigrant": ["Low (<30%)", "Middle (30–50%)", "High (>50%)"],
    "pct_visible_minority": ["Low (<20%)", "Middle (20–50%)", "High (>50%)"],
}


def experience_variants(config: dict[str, Any]) -> list[str]:
    variants = ["rating"]
    if config.get("experience", {}).get("sentiment", {}).get("enabled", False):
        variants.append("sentiment")
    return variants


def _variant_spec(variant: str) -> dict[str, str]:
    if variant == "rating":
        return {
            "class_column": "divergence_class",
            "threshold_column": "experience_rating_threshold",
            "score_column": "experience_rating_mean",
            "score_label": "Google star rating",
            "missing_label": "rating",
            "file_suffix": "",
        }
    if variant == "sentiment":
        return {
            "class_column": "divergence_sentiment_sensitivity",
            "threshold_column": "experience_sentiment_threshold",
            "score_column": "experience_sentiment_mean",
            "score_label": "review sentiment score",
            "missing_label": "sentiment",
            "file_suffix": "_sentiment",
        }
    raise ValueError(f"Unknown experience variant: {variant}")


def figure_paths(
    config: dict[str, Any], output_base: Path | None = None
) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = output_base or output_root(config)
    paths = {"summary_csv": base / "tables" / f"{slug}_10_figures_summary.csv"}
    for variant in experience_variants(config):
        suffix = _variant_spec(variant)["file_suffix"]
        paths[f"{variant}_map_png"] = (
            base / "maps" / f"{slug}_da_divergence{suffix}.png"
        )
        paths[f"{variant}_equity_png"] = (
            base / "figures" / f"{slug}_equity_profiles{suffix}.png"
        )
        paths[f"{variant}_relationships_png"] = (
            base / "figures" / f"{slug}_supply_experience_relationships{suffix}.png"
        )
        paths[f"{variant}_stacked_png"] = (
            base / "figures" / f"{slug}_equity_strata_stacked{suffix}.png"
        )
    if slug == "metro":
        paths["transferability_map_png"] = (
            base / "maps" / "six_city_divergence_comparison.png"
        )
        paths["transferability_table_csv"] = (
            base / "tables" / "six_city_transferability_summary.csv"
        )
    return paths


def divergence_path(config: dict[str, Any]) -> Path:
    slug = config["analysis_area"]["slug"]
    return (
        workspace_data_root(config)
        / "interim"
        / "divergence"
        / slug
        / f"{slug}_da_divergence.gpkg"
    )


def equity_profile_path(config: dict[str, Any], variant: str = "rating") -> Path:
    slug = config["analysis_area"]["slug"]
    suffix = _variant_spec(variant)["file_suffix"]
    return output_root(config) / "tables" / f"{slug}_08_equity_profiles{suffix}.csv"


def equity_data_path(config: dict[str, Any]) -> Path:
    slug = config["analysis_area"]["slug"]
    return (
        workspace_data_root(config)
        / "interim"
        / "equity"
        / slug
        / f"{slug}_da_equity.csv"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def _single_numeric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{column} must contain one finite value; found {values}")
    return float(values[0])


def _palette(config: dict[str, Any]) -> dict[str, str]:
    configured = config["plot"]["palette"]
    missing = set(CLASS_ORDER) - set(configured)
    if missing:
        raise ValueError(f"Plot palette is missing: {', '.join(sorted(missing))}")
    return {name: str(configured[name]).upper() for name in CLASS_ORDER}


def _figure_caption(
    config: dict[str, Any],
    counts: dict[str, int],
    supply_threshold: float,
    experience_threshold: float,
    variant: str,
) -> str:
    spec = _variant_spec(variant)
    area_name = config["analysis_area"]["name"]
    coverage_percent = float(config["analysis"]["high_coverage_threshold"]) * 100
    missing = counts["insufficient_experience"]
    zero_population = counts["insufficient_population"]
    caption = (
        f"Supply–experience divergence across {area_name} dissemination areas. "
        f"High supply requires at least {coverage_percent:.0f}% population coverage "
        f"and accessible park area, using a 20 ha per-park cap, at or above the "
        f"analysis-area median ({supply_threshold:.3f} ha per 1,000 residents). "
        f"High experience indicates a mean {spec['score_label']} at or above the "
        f"analysis-area median ({experience_threshold:.3f}). Green outlines show "
        f"park polygons. Grey areas lack sufficient {spec['missing_label']} data "
        f"(n={missing:,})."
    )
    if zero_population:
        caption += f" Light-grey areas have no resident population (n={zero_population:,})."
    return caption


def _equity_caption(
    config: dict[str, Any], classified_da_count: int, variant: str
) -> str:
    area_name = config["analysis_area"]["name"]
    score_label = _variant_spec(variant)["score_label"]
    return (
        f"Neighbourhood profiles by divergence quadrant in {area_name} (n="
        f"{classified_da_count:,} classified dissemination areas), using "
        f"{score_label} as experience. Variables are ordered by the difference "
        f"between higher- and lower-experience neighbourhoods. Asterisks indicate "
        f"Kruskal-Wallis comparisons of the continuous indicator across divergence "
        f"quadrants (* p<.05, ** p<.01, *** p<.001). Each indicator "
        f"is standardized across classified "
        f"DAs within the analysis area; positive values are above the area mean. "
        f"Census indicators use published 2021 Census Profile rates, and active "
        f"living environment uses the 2021 Can-ALE index. Missing indicator values "
        f"are omitted separately for each mean."
    )


def _source_data(
    config: dict[str, Any], variant: str
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, int], float, float]:
    spec = _variant_spec(variant)
    source = divergence_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Missing Stage 07 divergence GeoPackage: {source}")
    divergence = gpd.read_file(source, layer="da_divergence")
    required = {
        config["fields"]["da_id"],
        spec["class_column"],
        "supply_area_threshold",
        spec["threshold_column"],
        "geometry",
    }
    missing = required - set(divergence.columns)
    if missing:
        raise ValueError(
            f"Stage 07 divergence data are missing: {', '.join(sorted(missing))}"
        )
    divergence["divergence_class"] = divergence[spec["class_column"]].astype(
        "string"
    )
    invalid = set(divergence["divergence_class"].dropna()) - set(CLASS_ORDER)
    if invalid:
        raise ValueError(f"Unexpected divergence classes: {sorted(invalid)}")
    if divergence.empty or divergence.geometry.isna().any():
        raise ValueError("Divergence map source has no usable DA geometry")
    if str(divergence.crs) != config["crs"]["projected"]:
        raise ValueError(
            f"Divergence CRS is {divergence.crs}; expected {config['crs']['projected']}"
        )

    parks_path = data_path(config, "parks")
    parks = gpd.read_file(parks_path, layer="parks_cleaned")
    if parks.crs != divergence.crs:
        parks = parks.to_crs(divergence.crs)
    study_geometry = divergence.geometry.union_all()
    parks = parks.loc[parks.geometry.intersects(study_geometry)].copy()
    if parks.empty:
        raise ValueError("No cleaned parks intersect the selected analysis area")

    counts = {
        name: int(divergence["divergence_class"].eq(name).sum())
        for name in CLASS_ORDER
    }
    supply_threshold = _single_numeric(divergence, "supply_area_threshold")
    experience_threshold = _single_numeric(divergence, spec["threshold_column"])
    return divergence, parks, counts, supply_threshold, experience_threshold


def render_map(
    config: dict[str, Any], destination: Path, variant: str
) -> dict[str, Any]:
    divergence, parks, counts, supply_threshold, experience_threshold = _source_data(
        config, variant
    )
    spec = _variant_spec(variant)
    palette = _palette(config)
    dpi = int(config["plot"]["dpi"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    # Draw high-high, low-high, high-low, low-low, then missing data. The
    # explicit z-order keeps park outlines visible.
    draw_order = [
        "HH",
        "LH",
        "HL",
        "LL",
        "insufficient_experience",
        "insufficient_population",
    ]
    for class_name in draw_order:
        subset = divergence.loc[divergence["divergence_class"].eq(class_name)]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=palette[class_name],
                edgecolor="white",
                linewidth=0.2,
                zorder=1,
            )
    parks.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#2D6A2D",
        linewidth=0.8,
        zorder=2,
    )

    xmin, ymin, xmax, ymax = divergence.total_bounds
    x_padding = max((xmax - xmin) * 0.015, 1)
    y_padding = max((ymax - ymin) * 0.015, 1)
    ax.set_xlim(xmin - x_padding, xmax + x_padding)
    ax.set_ylim(ymin - y_padding, ymax + y_padding)

    patches = [
        mpatches.Patch(
            color=palette[name], label=f"{CLASS_LABELS[name]} (n={counts[name]:,})"
        )
        for name in CLASS_ORDER
        if counts[name] > 0 or name != "insufficient_population"
    ]
    ax.legend(
        handles=patches,
        title="Divergence class",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
    )

    ax_inset = fig.add_axes([0.02, 0.02, 0.18, 0.18])
    matrix_data = np.array(
        [
            [counts["LH"], counts["HH"]],
            [counts["LL"], counts["HL"]],
        ]
    )
    matrix_colours = np.array(
        [
            [palette["LH"], palette["HH"]],
            [palette["LL"], palette["HL"]],
        ]
    )
    for row in range(2):
        for column in range(2):
            ax_inset.add_patch(
                plt.Rectangle(
                    (column, 1 - row),
                    1,
                    1,
                    color=matrix_colours[row, column],
                    ec="white",
                    lw=1.5,
                )
            )
            ax_inset.text(
                column + 0.5,
                1.5 - row,
                f"{matrix_data[row, column]:,}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
    ax_inset.set_xlim(0, 2)
    ax_inset.set_ylim(0, 2)
    ax_inset.set_xticks([0.5, 1.5])
    ax_inset.set_xticklabels(["Low supply", "High supply"], fontsize=7)
    ax_inset.set_yticks([0.5, 1.5])
    ax_inset.set_yticklabels(["Low exp", "High exp"], fontsize=7)
    ax_inset.tick_params(length=0)
    ax_inset.set_title("n by quadrant", fontsize=7, pad=3)
    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    missing_note = (
        f"Grey = insufficient {spec['missing_label']} data "
        f"(n={counts['insufficient_experience']:,})"
    )
    if counts["insufficient_population"]:
        missing_note += (
            f"; light grey = no residents (n={counts['insufficient_population']:,})"
        )
    ax.annotate(
        missing_note,
        xy=(0.02, 0.22),
        xycoords="axes fraction",
        fontsize=9,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "none",
        },
    )

    ax.set_axis_off()
    # The quadrant inset is an independently positioned axes, so tight_layout
    # is inappropriate and emits a warning. These margins retain the inset
    # composition without clipping either axes. Publication titles and
    # thresholds are kept in the generated caption rather than on the map.
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    width, height = _png_dimensions(destination)
    return {
        "style_version": STYLE_VERSION,
        "analysis_area": config["analysis_area"]["slug"],
        "experience_variant": variant,
        "experience_score_label": spec["score_label"],
        "status": "rebuilt",
        "dpi": dpi,
        "width_px": width,
        "height_px": height,
        "file_size_bytes": destination.stat().st_size,
        "source_divergence_sha256": _sha256(divergence_path(config)),
        "source_parks_sha256": _sha256(data_path(config, "parks")),
        "supply_area_threshold": supply_threshold,
        "experience_threshold": experience_threshold,
        "experience_threshold_column": spec["threshold_column"],
        "figure_caption": _figure_caption(
            config, counts, supply_threshold, experience_threshold, variant
        ),
        "palette_json": json.dumps(palette, sort_keys=True),
        **{f"{name}_da_count": counts[name] for name in CLASS_ORDER},
    }


def build_transferability_summary() -> pd.DataFrame:
    """Build the paper's six-city Table 3 from city-specific classifications."""
    metro_config = load_config("metro")
    parks = gpd.read_file(data_path(metro_config, "parks"), layer="parks_cleaned")
    rows: list[dict[str, Any]] = []
    for slug, city_name, municipality in TRANSFERABILITY_CITIES:
        config = load_config(slug)
        source = divergence_path(config)
        if not source.exists():
            raise FileNotFoundError(f"Missing Stage 07 city classification: {source}")
        frame = gpd.read_file(source, layer="da_divergence")
        required = {
            "da_pop",
            "divergence_class",
            "experience_rating_mean",
            "supply_area_threshold",
            "experience_rating_threshold",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{city_name} Stage 07 data are missing: {', '.join(sorted(missing))}"
            )
        classes = frame["divergence_class"].astype("string")
        population = pd.to_numeric(frame["da_pop"], errors="raise")
        populated = population.gt(0)
        classified = populated & classes.isin(["HH", "HL", "LH", "LL"])
        classified_n = int(classified.sum())
        populated_n = int(populated.sum())
        if classified_n == 0 or populated_n == 0:
            raise ValueError(f"{city_name} has no classified populated DAs")
        counts = {
            name: int((classified & classes.eq(name)).sum())
            for name in ["HH", "HL", "LH", "LL"]
        }
        rating = pd.to_numeric(frame["experience_rating_mean"], errors="coerce")
        supply_threshold = _single_numeric(
            frame.loc[populated], "supply_area_threshold"
        )
        rating_threshold = _single_numeric(
            frame.loc[classified], "experience_rating_threshold"
        )
        rows.append(
            {
                "City": city_name,
                "DAs classified (n; %)": (
                    f"{classified_n} ({classified_n / populated_n * 100:.1f}%)"
                ),
                "High supply (%)": round(
                    (counts["HH"] + counts["HL"]) / classified_n * 100, 1
                ),
                "Mean Google star rating": round(float(rating.loc[classified].mean()), 2),
                "HH (%)": round(counts["HH"] / classified_n * 100, 1),
                "HL (%)": round(counts["HL"] / classified_n * 100, 1),
                "LH (%)": round(counts["LH"] / classified_n * 100, 1),
                "LL (%)": round(counts["LL"] / classified_n * 100, 1),
                "Total populated DAs (n)": populated_n,
                "Classified DAs (n)": classified_n,
                "Excluded: no eligible reachable rating (n)": int(
                    (populated & classes.eq("insufficient_experience")).sum()
                ),
                "Cleaned park inventory (n)": int(
                    parks.loc[parks["municipality"].eq(municipality), "park_id"].nunique()
                ),
                "Minimum Google reviews per rated place": int(
                    config["analysis"]["minimum_google_reviews"]
                ),
                "Accessible park area median (ha/1,000)": supply_threshold,
                "Google star rating median": rating_threshold,
                "Percentage denominator": "classified DAs",
                "Classification rule": (
                    "city-specific accessible-area and DA mean-rating medians; "
                    "high supply also requires >=80% park coverage"
                ),
                "source_divergence_sha256": _sha256(source),
            }
        )
    return pd.DataFrame(rows)


def render_transferability_map(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Render Figure 8 as six equal panels using each city's own thresholds."""
    if config["analysis_area"]["slug"] != "metro":
        raise ValueError("The six-city comparison must use the metro output folder")
    palette = _palette(config)
    dpi = int(config["plot"]["dpi"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    draw_order = [
        "HH",
        "LH",
        "HL",
        "LL",
        "insufficient_experience",
        "insufficient_population",
    ]
    for axis, (slug, city_name, _) in zip(axes.flat, TRANSFERABILITY_CITIES):
        city_config = load_config(slug)
        divergence, parks, _, _, _ = _source_data(city_config, "rating")
        for class_name in draw_order:
            subset = divergence.loc[
                divergence["divergence_class"].eq(class_name)
            ]
            if not subset.empty:
                subset.plot(
                    ax=axis,
                    color=palette[class_name],
                    edgecolor="white",
                    linewidth=0.18,
                    zorder=1,
                )
        parks.plot(
            ax=axis,
            facecolor="none",
            edgecolor="#2D6A2D",
            linewidth=0.45,
            zorder=2,
        )
        xmin, ymin, xmax, ymax = divergence.total_bounds
        x_pad = max((xmax - xmin) * 0.02, 1)
        y_pad = max((ymax - ymin) * 0.02, 1)
        axis.set_xlim(xmin - x_pad, xmax + x_pad)
        axis.set_ylim(ymin - y_pad, ymax + y_pad)
        axis.set_title(city_name, fontsize=12, fontweight="bold", pad=3)
        axis.set_axis_off()

    legend_order = ["HH", "HL", "LH", "LL", "insufficient_experience"]
    legend_labels = {
        "HH": "High supply / high experience",
        "HL": "High supply / low experience",
        "LH": "Low supply / high experience",
        "LL": "Low supply / low experience",
        "insufficient_experience": "Insufficient experience data",
    }
    handles = [
        mpatches.Patch(color=palette[name], label=legend_labels[name])
        for name in legend_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(
        left=0.015, right=0.985, top=0.97, bottom=0.12, wspace=0.05, hspace=0.10
    )
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    width, height = _png_dimensions(destination)
    return {
        "style_version": TRANSFERABILITY_STYLE_VERSION,
        "width_px": width,
        "height_px": height,
        "file_size_bytes": destination.stat().st_size,
    }


def validate_transferability(paths: dict[str, Path]) -> dict[str, Any]:
    table_path = paths["transferability_table_csv"]
    map_path = paths["transferability_map_png"]
    table = pd.read_csv(table_path)
    expected_cities = [city for _, city, _ in TRANSFERABILITY_CITIES]
    if table["City"].tolist() != expected_cities:
        raise ValueError("The six-city transferability table has the wrong city order")
    shares = table[["HH (%)", "HL (%)", "LH (%)", "LL (%)"]].sum(axis=1)
    if not np.allclose(shares, 100, atol=0.2):
        raise ValueError("Table 3 quadrant percentages do not sum to 100%")
    for slug, city_name, _ in TRANSFERABILITY_CITIES:
        observed = table.loc[
            table["City"].eq(city_name), "source_divergence_sha256"
        ].iloc[0]
        if observed != _sha256(divergence_path(load_config(slug))):
            raise ValueError(f"Table 3 is stale relative to {city_name} Stage 07")
    width, height = _png_dimensions(map_path)
    if width < 2500 or height < 1600 or map_path.stat().st_size < 50_000:
        raise ValueError(
            f"Six-city comparison map is unexpectedly small: {width}x{height}"
        )
    return {
        "style_version": TRANSFERABILITY_STYLE_VERSION,
        "width_px": width,
        "height_px": height,
        "file_size_bytes": map_path.stat().st_size,
        "table_rows": len(table),
    }


def _equity_source(
    config: dict[str, Any], variant: str
) -> tuple[pd.DataFrame, int]:
    source = equity_profile_path(config, variant)
    if not source.exists():
        raise FileNotFoundError(f"Missing Stage 08 equity profile table: {source}")
    profile = pd.read_csv(source)
    required = {
        "analysis_area",
        "experience_variant",
        "variable",
        "variable_label",
        "divergence_class",
        "class_da_count",
        "n",
        "mean_z",
        "standardization_scope",
        "profile_test",
        "profile_test_statistic",
        "profile_test_p_value",
        "profile_test_eta_squared",
        "profile_test_p_method",
        "significance_marker",
        "strata_test",
        "strata_test_statistic",
        "strata_test_p_value",
        "strata_test_cramers_v",
        "strata_significance_marker",
        "experience_mean_difference",
    }
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(
            "Stage 08 equity profile is missing: " + ", ".join(sorted(missing))
        )
    slug = config["analysis_area"]["slug"]
    if len(profile) != len(EQUITY_VARIABLES) * len(EQUITY_CLASS_ORDER):
        raise ValueError(
            "Stage 08 equity profile does not contain every variable-class combination"
        )
    if not profile["analysis_area"].eq(slug).all():
        raise ValueError("Stage 08 equity profile has the wrong analysis area")
    if not profile["experience_variant"].eq(variant).all():
        raise ValueError("Stage 08 equity profile has the wrong experience variant")
    expected_variables = {field for field, _ in EQUITY_VARIABLES}
    if set(profile["variable"]) != expected_variables:
        raise ValueError("Stage 08 equity profile variables differ from the figure design")
    if set(profile["divergence_class"]) != set(EQUITY_CLASS_ORDER):
        raise ValueError("Stage 08 equity profile divergence classes are incomplete")
    if profile.duplicated(["variable", "divergence_class"]).any():
        raise ValueError("Stage 08 equity profile contains duplicate combinations")
    if not profile["standardization_scope"].eq(
        "classified DAs in analysis area"
    ).all():
        raise ValueError("Stage 08 equity standardization scope is unexpected")
    if not profile["profile_test"].eq(
        "Kruskal-Wallis comparison of continuous indicator across divergence quadrants"
    ).all():
        raise ValueError("Stage 08 equity profile uses an outdated star test")
    if not profile["strata_test"].eq(
        "Chi-square association of equity strata and divergence quadrant"
    ).all():
        raise ValueError("Stage 08 equity profile uses an outdated strata test")
    profile["class_da_count"] = pd.to_numeric(
        profile["class_da_count"], errors="raise"
    ).astype("int64")
    profile["n"] = pd.to_numeric(profile["n"], errors="raise").astype("int64")
    profile["mean_z"] = pd.to_numeric(profile["mean_z"], errors="coerce")
    profile["profile_test_p_value"] = pd.to_numeric(
        profile["profile_test_p_value"], errors="coerce"
    )
    profile["profile_test_statistic"] = pd.to_numeric(
        profile["profile_test_statistic"], errors="coerce"
    )
    profile["profile_test_eta_squared"] = pd.to_numeric(
        profile["profile_test_eta_squared"], errors="coerce"
    )
    profile["strata_test_statistic"] = pd.to_numeric(
        profile["strata_test_statistic"], errors="coerce"
    )
    profile["strata_test_p_value"] = pd.to_numeric(
        profile["strata_test_p_value"], errors="coerce"
    )
    profile["strata_test_cramers_v"] = pd.to_numeric(
        profile["strata_test_cramers_v"], errors="coerce"
    )
    profile["experience_mean_difference"] = pd.to_numeric(
        profile["experience_mean_difference"], errors="coerce"
    )
    if (profile["n"] < 0).any() or (profile["n"] > profile["class_da_count"]).any():
        raise ValueError("Stage 08 equity profile has invalid sample counts")
    if profile.loc[profile["n"].gt(0), "mean_z"].isna().any():
        raise ValueError("Stage 08 equity profile is missing a class mean z-score")
    class_counts = profile.groupby("divergence_class")["class_da_count"].nunique()
    if not class_counts.eq(1).all():
        raise ValueError("Stage 08 equity class counts vary between indicators")
    classified_da_count = int(
        profile.drop_duplicates("divergence_class")["class_da_count"].sum()
    )
    return profile, classified_da_count


def render_equity(
    config: dict[str, Any], destination: Path, variant: str
) -> dict[str, Any]:
    profile, classified_da_count = _equity_source(config, variant)
    palette = _palette(config)
    dpi = int(config["plot"]["dpi"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    labels = profile.drop_duplicates("variable").set_index("variable")
    profile_order = labels["experience_mean_difference"].sort_values(
        ascending=False
    ).index.tolist()
    y_positions = np.arange(len(profile_order))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for y_position, field in zip(y_positions, profile_order, strict=True):
        values = pd.to_numeric(
            profile.loc[profile["variable"].eq(field), "mean_z"], errors="coerce"
        )
        ax.hlines(
            y_position,
            values.min(),
            values.max(),
            color="#BDBDBD",
            linewidth=0.9,
            zorder=1,
        )
    for class_name in EQUITY_CLASS_ORDER:
        subset = profile.loc[
            profile["divergence_class"].eq(class_name)
        ].set_index("variable")
        x_values = [subset.loc[field, "mean_z"] for field in profile_order]
        ax.scatter(
            x_values,
            y_positions,
            s=68,
            color=palette[class_name],
            label=SHORT_CLASS_LABELS[class_name],
            alpha=0.95,
            edgecolor="none",
            zorder=3,
        )

    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"{labels.loc[field, 'variable_label']} "
            f"{str(labels.loc[field, 'significance_marker']) if pd.notna(labels.loc[field, 'significance_marker']) else ''}".rstrip()
            for field in profile_order
        ],
        fontsize=9,
    )
    ax.set_xlabel(
        "Mean standardized value (z-score) by divergence quadrant", fontsize=9
    )
    ax.set_xlim(-EQUITY_X_LIMIT, EQUITY_X_LIMIT)
    ax.set_xticks(np.arange(-EQUITY_X_LIMIT, EQUITY_X_LIMIT + 0.01, 0.2))
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.0)
    ax.grid(axis="y", alpha=0.26, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(
        title="Divergence class",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        markerscale=0.9,
    )
    fig.subplots_adjust(left=0.27, right=0.99, bottom=0.22, top=0.98)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    width, height = _png_dimensions(destination)
    return {
        "equity_style_version": EQUITY_STYLE_VERSION,
        "equity_width_px": width,
        "equity_height_px": height,
        "equity_file_size_bytes": destination.stat().st_size,
        "source_equity_profile_sha256": _sha256(
            equity_profile_path(config, variant)
        ),
        "equity_classified_da_count": classified_da_count,
        "equity_x_limit": EQUITY_X_LIMIT,
        "equity_figure_caption": _equity_caption(
            config, classified_da_count, variant
        ),
    }


def _equity_da_source(config: dict[str, Any], variant: str) -> pd.DataFrame:
    source = equity_data_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Missing Stage 08 DA equity table: {source}")
    frame = pd.read_csv(source)
    spec = _variant_spec(variant)
    required = {
        "da_pop",
        "park_access_pop_coverage",
        "supply_ha_per_1000_cap20",
        spec["score_column"],
        spec["class_column"],
        *(field for field, _ in EQUITY_VARIABLES),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Stage 08 DA equity table is missing: " + ", ".join(sorted(missing))
        )
    return frame


def _relationship_caption(
    config: dict[str, Any], variant: str, panel_counts: list[int]
) -> str:
    return (
        f"Relationships between neighbourhood park supply and expressed "
        f"satisfaction across {config['analysis_area']['name']} dissemination "
        f"areas. Experience is measured using "
        f"{_variant_spec(variant)['score_label']}. Panel A uses hexagonal bins "
        f"to show the coverage ceiling while retaining full-coverage DAs as "
        f"points. Panel B uses a logarithmic area axis; dashed lines mark the "
        f"sample medians. Annotations report Spearman rank correlations. Panel "
        f"sample sizes are n={panel_counts[0]:,} and n={panel_counts[1]:,}."
    )


def render_relationships(
    config: dict[str, Any], destination: Path, variant: str
) -> dict[str, Any]:
    frame = _equity_da_source(config, variant)
    spec = _variant_spec(variant)
    score = spec["score_column"]
    dpi = int(config["plot"]["dpi"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    counts: list[int] = []
    populated = pd.to_numeric(frame["da_pop"], errors="coerce").gt(0)

    coverage = frame.loc[
        populated, ["park_access_pop_coverage", score]
    ].apply(pd.to_numeric, errors="coerce").dropna()
    coverage_x = coverage["park_access_pop_coverage"].to_numpy(dtype="float64")
    coverage_y = coverage[score].to_numpy(dtype="float64")
    counts.append(len(coverage))
    full_coverage = np.isclose(coverage_x, 1.0, atol=1e-12)
    incomplete = ~full_coverage
    if incomplete.any():
        axes[0].hexbin(
            coverage_x[incomplete],
            coverage_y[incomplete],
            gridsize=24,
            mincnt=1,
            bins="log",
            cmap="YlGn",
            linewidths=0.15,
            edgecolors="white",
            alpha=0.92,
        )
    if full_coverage.any():
        axes[0].scatter(
            coverage_x[full_coverage],
            coverage_y[full_coverage],
            s=8,
            color="#1F5C4D",
            alpha=0.28,
            edgecolor="none",
            rasterized=True,
        )
        axes[0].annotate(
            f"n={int(full_coverage.sum()):,} DAs\nat full coverage",
            xy=(1.0, float(np.nanpercentile(coverage_y[full_coverage], 12))),
            xytext=(0.78, 0.12),
            textcoords="axes fraction",
            fontsize=7,
            color="#666666",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7},
        )
    if len(coverage) >= 3 and np.unique(coverage_x).size > 1:
        coverage_r, coverage_p = spearmanr(coverage_x, coverage_y)
    else:
        coverage_r, coverage_p = np.nan, np.nan

    area = frame.loc[
        populated, ["supply_ha_per_1000_cap20", score]
    ].apply(pd.to_numeric, errors="coerce").dropna()
    area = area.loc[area["supply_ha_per_1000_cap20"].gt(0)]
    area_x = area["supply_ha_per_1000_cap20"].to_numpy(dtype="float64")
    area_y = area[score].to_numpy(dtype="float64")
    counts.append(len(area))
    axes[1].scatter(
        area_x,
        area_y,
        s=10,
        color="#1F5C4D",
        alpha=0.38,
        edgecolor="none",
        rasterized=True,
    )
    axes[1].axvline(
        float(np.median(area_x)), color="#888888", lw=0.8, ls=(0, (4, 4))
    )
    axes[1].axhline(
        float(np.median(area_y)), color="#888888", lw=0.8, ls=(0, (4, 4))
    )
    axes[1].set_xscale("log")
    if len(area) >= 3 and np.unique(area_x).size > 1:
        area_r, area_p = spearmanr(area_x, area_y)
    else:
        area_r, area_p = np.nan, np.nan

    def _p_text(value: float) -> str:
        if not np.isfinite(value):
            return "p unavailable"
        return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"

    axes[0].set_title(
        f"A. Coverage × Experience\nSpearman r = {coverage_r:.3f}, "
        f"{_p_text(coverage_p)}, n = {len(coverage):,}",
        fontsize=8.5,
        pad=7,
    )
    axes[1].set_title(
        f"B. Park Area × Experience\nSpearman r = {area_r:.3f}, "
        f"{_p_text(area_p)}, n = {len(area):,}",
        fontsize=8.5,
        pad=7,
    )
    axes[0].set_xlabel("Park coverage (proportion within 400 m)", fontsize=8.5)
    axes[1].set_xlabel(
        "Accessible park area (ha per 1,000 residents; log scale)", fontsize=8.5
    )
    axes[0].set_ylabel(spec["score_label"].capitalize(), fontsize=8.5)
    axes[0].set_xlim(-0.03, 1.04)
    for ax in axes:
        ax.grid(color="#E0E0E0", linewidth=0.55, alpha=0.55, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.16, top=0.88, wspace=0.14)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    width, height = _png_dimensions(destination)
    return {
        "relationship_style_version": RELATIONSHIP_STYLE_VERSION,
        "relationship_width_px": width,
        "relationship_height_px": height,
        "relationship_file_size_bytes": destination.stat().st_size,
        "source_equity_da_sha256": _sha256(equity_data_path(config)),
        "relationship_figure_caption": _relationship_caption(
            config, variant, counts
        ),
    }


def _stacked_caption(config: dict[str, Any], variant: str) -> str:
    return (
        f"Distribution of supply–experience divergence quadrants across equity "
        f"strata in {config['analysis_area']['name']}, using "
        f"{_variant_spec(variant)['score_label']} as experience. Bars show the "
        f"percentage of classified dissemination areas within each stratum. "
        f"Asterisks report assumption-aware chi-square tests of equity strata "
        f"and divergence class (* p<.05, ** p<.01, *** p<.001)."
    )


def render_equity_stacked(
    config: dict[str, Any], destination: Path, variant: str
) -> dict[str, Any]:
    frame = _equity_da_source(config, variant)
    profile, _ = _equity_source(config, variant)
    spec = _variant_spec(variant)
    palette = _palette(config)
    dpi = int(config["plot"]["dpi"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    classes = frame[spec["class_column"]].astype("string")
    classified = classes.isin(EQUITY_CLASS_ORDER)
    markers = (
        profile.drop_duplicates("variable")
        .set_index("variable")["strata_significance_marker"]
        .fillna("")
        .astype(str)
    )
    test_rows = profile.drop_duplicates("variable").set_index("variable")
    column_count = 3
    row_count = int(np.ceil(len(EQUITY_VARIABLES) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.8, 3.2 * row_count),
        sharex=False,
        squeeze=False,
    )
    for ax, (field, label) in zip(axes.flat, EQUITY_VARIABLES):
        strata = equity_strata(frame[field], field)
        valid = classified & strata.notna()
        table = pd.crosstab(strata.loc[valid], classes.loc[valid])
        table = table.reindex(columns=EQUITY_CLASS_ORDER, fill_value=0)
        table = table.loc[table.sum(axis=1).gt(0)]
        percentages = table.div(table.sum(axis=1), axis=0) * 100
        labels = STRATUM_LABELS[field][: len(percentages)]
        bottom = np.zeros(len(percentages), dtype="float64")
        x = np.arange(len(percentages))
        for class_name in EQUITY_CLASS_ORDER:
            values = percentages[class_name].to_numpy(dtype="float64")
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=palette[class_name],
                width=0.72,
                label=SHORT_CLASS_LABELS[class_name],
            )
            for position, (value, base) in enumerate(zip(values, bottom, strict=True)):
                if value > 6:
                    ax.text(
                        position,
                        base + value / 2,
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if class_name in {"HH", "LL"} else "black",
                    )
            bottom += values
        marker = markers.get(field, "")
        test = test_rows.loc[field]
        ax.set_title(
            f"{label}\nχ²={test['strata_test_statistic']:.1f}, "
            f"p={test['strata_test_p_value']:.3g} {marker}, "
            f"V={test['strata_test_cramers_v']:.2f}".rstrip(),
            fontsize=8.7,
            pad=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    for row_index in range(row_count):
        axes[row_index, 0].set_ylabel(
            "Classified DAs within stratum (%)", fontsize=8.5
        )
    for unused in axes.flat[len(EQUITY_VARIABLES) :]:
        unused.set_visible(False)
    handles = [
        mpatches.Patch(color=palette[name], label=SHORT_CLASS_LABELS[name])
        for name in EQUITY_CLASS_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.12, top=0.98, hspace=0.48, wspace=0.22
    )
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    width, height = _png_dimensions(destination)
    return {
        "stacked_style_version": STACKED_STYLE_VERSION,
        "stacked_width_px": width,
        "stacked_height_px": height,
        "stacked_file_size_bytes": destination.stat().st_size,
        "stacked_figure_caption": _stacked_caption(config, variant),
    }


def write_summary(
    config: dict[str, Any],
    paths: dict[str, Path],
    metrics_by_variant: dict[str, dict[str, Any]],
    status: str,
) -> Path:
    path = paths["summary_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in experience_variants(config):
        rows.append(
            {
                **metrics_by_variant[variant],
                "status": status,
                "map_png_path": str(paths[f"{variant}_map_png"]),
                "equity_png_path": str(paths[f"{variant}_equity_png"]),
                "relationships_png_path": str(
                    paths[f"{variant}_relationships_png"]
                ),
                "equity_stacked_png_path": str(paths[f"{variant}_stacked_png"]),
                "source_divergence_path": str(divergence_path(config)),
                "source_parks_path": str(data_path(config, "parks")),
                "source_equity_da_path": str(equity_data_path(config)),
                "source_equity_profile_path": str(
                    equity_profile_path(config, variant)
                ),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def validate_figure(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 10 artifacts: " + ", ".join(path.name for path in missing)
        )
    summary = pd.read_csv(paths["summary_csv"])
    variants = experience_variants(config)
    if (
        "experience_variant" not in summary
        or set(summary["experience_variant"]) != set(variants)
        or len(summary) != len(variants)
    ):
        raise ValueError("Stage 10 figure summary has the wrong experience variants")

    validated: dict[str, dict[str, Any]] = {}
    for variant in variants:
        row = summary.loc[summary["experience_variant"].eq(variant)].iloc[0]
        map_path = paths[f"{variant}_map_png"]
        equity_path = paths[f"{variant}_equity_png"]
        relationships_path = paths[f"{variant}_relationships_png"]
        stacked_path = paths[f"{variant}_stacked_png"]
        width, height = _png_dimensions(map_path)
        if width < 2000 or height < 1500 or map_path.stat().st_size < 50_000:
            raise ValueError(
                f"{variant.title()} divergence map is unexpectedly small: "
                f"{width}x{height}, {map_path.stat().st_size:,} bytes"
            )
        equity_width, equity_height = _png_dimensions(equity_path)
        if (
            equity_width < 1800
            or equity_height < 1100
            or equity_path.stat().st_size < 25_000
        ):
            raise ValueError(
                f"{variant.title()} equity figure is unexpectedly small: "
                f"{equity_width}x{equity_height}, "
                f"{equity_path.stat().st_size:,} bytes"
            )
        relationship_width, relationship_height = _png_dimensions(
            relationships_path
        )
        if (
            relationship_width < 1800
            or relationship_height < 800
            or relationships_path.stat().st_size < 20_000
        ):
            raise ValueError(
                f"{variant.title()} relationship figure is unexpectedly small: "
                f"{relationship_width}x{relationship_height}, "
                f"{relationships_path.stat().st_size:,} bytes"
            )
        stacked_width, stacked_height = _png_dimensions(stacked_path)
        if (
            stacked_width < 2400
            or stacked_height < 1500
            or stacked_path.stat().st_size < 25_000
        ):
            raise ValueError(
                f"{variant.title()} stacked equity figure is unexpectedly small: "
                f"{stacked_width}x{stacked_height}, "
                f"{stacked_path.stat().st_size:,} bytes"
            )
        if row.get("style_version") != STYLE_VERSION:
            raise ValueError("Divergence map uses an old style version; rebuild with --force")
        if row.get("equity_style_version") != EQUITY_STYLE_VERSION:
            raise ValueError("Equity plot uses an old style version; rebuild with --force")
        if row.get("relationship_style_version") != RELATIONSHIP_STYLE_VERSION:
            raise ValueError("Relationship plot uses an old style version; rebuild with --force")
        if row.get("stacked_style_version") != STACKED_STYLE_VERSION:
            raise ValueError("Stacked equity plot uses an old style version; rebuild with --force")
        if row.get("source_divergence_sha256") != _sha256(divergence_path(config)):
            raise ValueError("Divergence map is stale relative to Stage 07")
        if row.get("source_parks_sha256") != _sha256(data_path(config, "parks")):
            raise ValueError("Divergence map is stale relative to the cleaned park inventory")
        if row.get("source_equity_profile_sha256") != _sha256(
            equity_profile_path(config, variant)
        ):
            raise ValueError("Equity plot is stale relative to Stage 08")
        if row.get("source_equity_da_sha256") != _sha256(equity_data_path(config)):
            raise ValueError("Stage 10 figures are stale relative to Stage 08 DA data")
        if int(row["dpi"]) != int(config["plot"]["dpi"]):
            raise ValueError("Divergence map DPI differs from configuration")
        if row.get("palette_json") != json.dumps(_palette(config), sort_keys=True):
            raise ValueError("Divergence map palette differs from configuration")

        _, _, counts, supply_threshold, experience_threshold = _source_data(
            config, variant
        )
        _, classified_da_count = _equity_source(config, variant)
        for name in CLASS_ORDER:
            if int(row[f"{name}_da_count"]) != counts[name]:
                raise ValueError(f"{variant.title()} map count is stale for {name}")
        if not np.isclose(float(row["supply_area_threshold"]), supply_threshold):
            raise ValueError("Divergence map supply threshold is stale")
        if not np.isclose(float(row["experience_threshold"]), experience_threshold):
            raise ValueError("Divergence map experience threshold is stale")
        spec = _variant_spec(variant)
        validated[variant] = {
            "style_version": STYLE_VERSION,
            "analysis_area": config["analysis_area"]["slug"],
            "experience_variant": variant,
            "experience_score_label": spec["score_label"],
            "dpi": int(row["dpi"]),
            "width_px": width,
            "height_px": height,
            "file_size_bytes": map_path.stat().st_size,
            "source_divergence_sha256": row["source_divergence_sha256"],
            "source_parks_sha256": row["source_parks_sha256"],
            "equity_style_version": EQUITY_STYLE_VERSION,
            "equity_width_px": equity_width,
            "equity_height_px": equity_height,
            "equity_file_size_bytes": equity_path.stat().st_size,
            "source_equity_profile_sha256": row[
                "source_equity_profile_sha256"
            ],
            "equity_classified_da_count": classified_da_count,
            "equity_x_limit": EQUITY_X_LIMIT,
            "equity_figure_caption": _equity_caption(
                config, classified_da_count, variant
            ),
            "relationship_style_version": RELATIONSHIP_STYLE_VERSION,
            "relationship_width_px": relationship_width,
            "relationship_height_px": relationship_height,
            "relationship_file_size_bytes": relationships_path.stat().st_size,
            "source_equity_da_sha256": row["source_equity_da_sha256"],
            "relationship_figure_caption": row["relationship_figure_caption"],
            "stacked_style_version": STACKED_STYLE_VERSION,
            "stacked_width_px": stacked_width,
            "stacked_height_px": stacked_height,
            "stacked_file_size_bytes": stacked_path.stat().st_size,
            "stacked_figure_caption": _stacked_caption(config, variant),
            "supply_area_threshold": supply_threshold,
            "experience_threshold": experience_threshold,
            "experience_threshold_column": spec["threshold_column"],
            "figure_caption": _figure_caption(
                config, counts, supply_threshold, experience_threshold, variant
            ),
            "palette_json": row["palette_json"],
            **{f"{name}_da_count": counts[name] for name in CLASS_ORDER},
        }
    if config["analysis_area"]["slug"] == "metro":
        validate_transferability(paths)
    return validated


def build_figure(
    config: dict[str, Any], final_paths: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    slug = config["analysis_area"]["slug"]
    parent = output_root(config).parent
    parent.mkdir(parents=True, exist_ok=True)
    with inherited_temp_directory(parent, f"{slug}_10_") as temp_dir:
        temp_paths = figure_paths(config, temp_dir)
        metrics = {}
        for variant in experience_variants(config):
            metrics[variant] = {
                **render_map(config, temp_paths[f"{variant}_map_png"], variant),
                **render_equity(
                    config, temp_paths[f"{variant}_equity_png"], variant
                ),
                **render_relationships(
                    config, temp_paths[f"{variant}_relationships_png"], variant
                ),
                **render_equity_stacked(
                    config, temp_paths[f"{variant}_stacked_png"], variant
                ),
            }
        if slug == "metro":
            temp_paths["transferability_table_csv"].parent.mkdir(
                parents=True, exist_ok=True
            )
            build_transferability_summary().to_csv(
                temp_paths["transferability_table_csv"], index=False
            )
            render_transferability_map(
                config, temp_paths["transferability_map_png"]
            )
        write_summary(config, temp_paths, metrics, "rebuilt")
        metrics = validate_figure(config, temp_paths)
        for key, target in final_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and _sha256(temp_paths[key]) == _sha256(target):
                continue
            os.replace(temp_paths[key], target)
    return metrics


def main(area: str, force: bool = False) -> int:
    config = load_config(area)
    paths = figure_paths(config)
    slug = config["analysis_area"]["slug"]
    print(f"\nStage 10: publication figures ({slug})")
    print("Style: titleless Python figures with the shared teal–brown palette.")

    if force:
        print("Force enabled: rendering and atomically replacing active figures.")
        metrics = build_figure(config, paths)
        status = "rebuilt"
    else:
        print("Reuse mode: validating existing figures and their source fingerprints.")
        try:
            metrics = validate_figure(config, paths)
        except (FileNotFoundError, ValueError) as error:
            print(f"Figure validation failed: {error}")
            print("Build this area's figures explicitly with --force.")
            return 1
        status = "reused"

    write_summary(config, paths, metrics, status)
    for variant in experience_variants(config):
        variant_metrics = metrics[variant]
        print(
            f"{variant.title()} map size: {variant_metrics['width_px']:,} × "
            f"{variant_metrics['height_px']:,} pixels"
        )
        print(f"Saved: {paths[f'{variant}_map_png']}")
        print(
            f"{variant.title()} equity figure size: "
            f"{variant_metrics['equity_width_px']:,} × "
            f"{variant_metrics['equity_height_px']:,} pixels"
        )
        print(f"Saved: {paths[f'{variant}_equity_png']}")
        print(f"Saved: {paths[f'{variant}_relationships_png']}")
        print(f"Saved: {paths[f'{variant}_stacked_png']}")
    if slug == "metro":
        transferability = validate_transferability(paths)
        print(
            "Six-city comparison size: "
            f"{transferability['width_px']:,} Ã— {transferability['height_px']:,} pixels"
        )
        print(f"Saved: {paths['transferability_map_png']}")
        print(f"Saved: {paths['transferability_table_csv']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="City slug or metro")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Render and atomically replace this area's divergence map",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.area, force=args.force))
    except ConfigError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(2)
