"""Complete Vancouver sentiment validation and perceived-usability analyses."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_config import (
    ConfigError,
    data_path,
    inherited_temp_directory,
    load_config,
    output_root,
    workspace_data_root,
)


CLASS_ORDER = ["HH", "LH", "HL", "LL"]
MODEL_CLASS_ORDER = ["LL", "HH", "LH", "HL"]
AMENITIES = {
    "playground": ("Playground", ["playground", "playgrounds", "play structure"]),
    "sports_fields": (
        "Sports fields",
        ["soccer field", "football field", "baseball diamond", "baseball field", "sports field"],
    ),
    "courts": (
        "Courts",
        ["basketball court", "tennis court", "pickleball court", "sports court"],
    ),
    "trails": ("Trails", ["walking trail", "hiking trail", "walking path", "bike path", "forest trail"]),
    "dog_offleash": ("Dog off-leash", ["off leash area", "off-leash area"]),
    "water_play": ("Water play", ["spray park", "splash pad", "wading pool"]),
    "beach_waterfront": ("Beach/waterfront", ["beach", "shoreline", "waterfront"]),
    "picnic": ("Picnic area", ["picnic area", "picnic table"]),
    "washroom": ("Washroom", ["washroom", "washrooms", "restroom", "restrooms"]),
    "community_garden": ("Community garden", ["community garden", "allotment", "garden plot"]),
    "seating_shelter": (
        "Seating & shelter",
        ["park bench", "benches", "picnic shelter", "covered shelter"],
    ),
}
MODEL_SOCIO = [
    ("pct_age_0_14", "Children aged 0-14 (%)"),
    ("pct_age_65plus", "Age 65+ (%)"),
    ("pct_visible_minority", "Visible minority (%)"),
    ("pct_LIM_AT", "LIM-AT (%)"),
    ("pct_bachelor_plus", "Education (bachelor's+)"),
    ("ALE_index", "Active living environment"),
]
MODEL_AMENITIES = [
    "sports_fields",
    "courts",
    "trails",
    "dog_offleash",
    "water_play",
    "beach_waterfront",
    "picnic",
    "community_garden",
]
FACILITY_MAP = {
    "Playgrounds": "playground",
    "Soccer Fields": "sports_fields",
    "Football Fields": "sports_fields",
    "Baseball Diamonds": "sports_fields",
    "Softball": "sports_fields",
    "Ultimate Fields": "sports_fields",
    "Rugby Fields": "sports_fields",
    "Field Hockey": "sports_fields",
    "Basketball Courts": "courts",
    "Tennis Courts": "courts",
    "Pickleball": "courts",
    "Ball Hockey": "courts",
    "Outdoor Roller Hockey Rinks": "courts",
    "Dogs Off-Leash Areas": "dog_offleash",
    "Water/Spray Parks": "water_play",
    "Wading Pool": "water_play",
    "Swimming Pools": "water_play",
    "Beaches": "beach_waterfront",
    "Picnic Sites": "picnic",
    "Jogging Trails": "trails",
}
VALIDATION_AMENITIES = [
    "playground",
    "sports_fields",
    "courts",
    "dog_offleash",
    "water_play",
    "picnic",
    "washroom",
]
NEGATIONS = {"no", "not", "without", "lack", "lacks", "lacking", "never"}


def _significance(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    pairs = pd.DataFrame({"left": left, "right": right}).dropna()
    if pairs.empty:
        return np.nan
    observed = float(pairs["left"].eq(pairs["right"]).mean())
    left_share = float(pairs["left"].mean())
    right_share = float(pairs["right"].mean())
    expected = left_share * right_share + (1 - left_share) * (1 - right_share)
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def _configured_path(config: dict[str, Any], key: str) -> Path:
    value = config["experience"]["sentiment"].get(key)
    if not value:
        raise ValueError(f"Vancouver sentiment configuration is missing {key}")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    workspace = workspace_data_root(config) / path
    if workspace.exists():
        return workspace.resolve()
    return (Path(config["runtime"]["data_root"]) / path).resolve()


def _paths(config: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    slug = config["analysis_area"]["slug"]
    base = root or output_root(config)
    tables = base / "tables"
    figures = base / "figures"
    interim = workspace_data_root(config) / "interim"
    return {
        "ratings": data_path(config, "google_ratings_validated"),
        "sentiment": _configured_path(config, "path"),
        "reviews": _configured_path(config, "review_text_path"),
        "park_metrics": _configured_path(config, "park_metrics_path"),
        "facilities": _configured_path(config, "official_facilities_path"),
        "washrooms": _configured_path(config, "official_washrooms_path"),
        "experience": interim / "experience" / slug / f"{slug}_da_experience.csv",
        "da_entities": interim / "experience" / slug / f"{slug}_da_google_entities.csv",
        "reach_pairs": interim / "reach" / slug / f"{slug}_db_park_reachability.csv",
        "park_crosswalk": interim / "supply" / slug / f"{slug}_park_entity_crosswalk.csv",
        "equity": interim / "equity" / slug / f"{slug}_da_equity.csv",
        "baseline_model_fit": output_root(config) / "tables" / f"{slug}_09_model_fit_sentiment.csv",
        "validation": tables / f"{slug}_11_sentiment_rating_validation.csv",
        "agreement": tables / f"{slug}_11_sentiment_classification_agreement.csv",
        "linkage_audit": tables / f"{slug}_11_sentiment_linkage_audit.csv",
        "sensitivity": tables / f"{slug}_11_sentiment_weighting_sensitivity.csv",
        "park_amenities": tables / f"{slug}_11_park_amenities.csv",
        "da_usability": tables / f"{slug}_11_da_usability.csv",
        "amenity_prevalence": tables / f"{slug}_11_amenity_prevalence.csv",
        "amenity_chi_square": tables / f"{slug}_11_amenity_chi_square.csv",
        "amenity_correlations": tables / f"{slug}_11_amenity_correlations.csv",
        "amenity_validation": tables / f"{slug}_11_amenity_validation.csv",
        "amenity_vif": tables / f"{slug}_11_amenity_model_vif.csv",
        "amenity_model": tables / f"{slug}_11_amenity_multinomial_logit.csv",
        "amenity_model_formatted": tables / f"{slug}_11_amenity_multinomial_logit_formatted.csv",
        "amenity_model_fit": tables / f"{slug}_11_amenity_model_fit.csv",
        "summary": tables / f"{slug}_11_vancouver_extension_summary.csv",
        "validation_figure": figures / f"{slug}_sentiment_rating_validation.png",
        "pairwise_figure": figures / f"{slug}_pairwise_relationships_sentiment.png",
        "amenity_dotplot": figures / f"{slug}_amenity_profiles_sentiment.png",
        "amenity_heatmap": figures / f"{slug}_amenity_heatmap_sentiment.png",
    }


def _require_files(paths: dict[str, Path]) -> None:
    for key in ["ratings", "sentiment", "reviews", "park_metrics", "facilities", "washrooms", "experience", "da_entities", "reach_pairs", "park_crosswalk", "equity", "baseline_model_fit"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Required Vancouver extension input is missing: {paths[key]}")


def _correlation_row(level: str, x: pd.Series, y: pd.Series) -> dict[str, Any]:
    pairs = pd.DataFrame({"sentiment": pd.to_numeric(x, errors="coerce"), "rating": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pairs) < 3 or pairs.nunique().min() < 2:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    else:
        pearson_r, pearson_p = pearsonr(pairs["sentiment"], pairs["rating"])
        spearman_r, spearman_p = spearmanr(pairs["sentiment"], pairs["rating"])
    return {
        "level": level,
        "n": len(pairs),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "pearson_significance": _significance(pearson_p),
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "spearman_significance": _significance(spearman_p),
    }


def _sentiment_validation(
    reviews: pd.DataFrame,
    park_metrics: pd.DataFrame,
    experience: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_rating = pd.to_numeric(reviews.get("Rating"), errors="coerce")
    if review_rating.notna().sum() == 0 and "rating" in reviews:
        review_rating = pd.to_numeric(reviews["rating"], errors="coerce")
    validation = pd.DataFrame([
        _correlation_row("review", reviews["Sentiment"], review_rating),
        _correlation_row(
            "park",
            park_metrics.loc[park_metrics["has_valid_sentiment"].astype(str).str.lower().eq("true"), "MeanSentiment"],
            park_metrics.loc[park_metrics["has_valid_sentiment"].astype(str).str.lower().eq("true"), "AvgRating"],
        ),
        _correlation_row(
            "dissemination area",
            experience["experience_sentiment_mean"],
            experience["experience_rating_mean"],
        ),
    ])

    joint = experience[["DAUID", "experience_sentiment_mean", "experience_rating_mean"]].dropna().copy()
    sentiment_median = float(joint["experience_sentiment_mean"].median())
    rating_median = float(joint["experience_rating_mean"].median())
    joint["sentiment_high"] = joint["experience_sentiment_mean"].ge(sentiment_median).astype(int)
    joint["rating_high"] = joint["experience_rating_mean"].ge(rating_median).astype(int)
    agreement = pd.DataFrame([{
        "n": len(joint),
        "sentiment_median": sentiment_median,
        "rating_median": rating_median,
        "agreement_n": int(joint["sentiment_high"].eq(joint["rating_high"]).sum()),
        "agreement_percent": float(joint["sentiment_high"].eq(joint["rating_high"]).mean() * 100),
        "cohen_kappa": _cohen_kappa(joint["sentiment_high"], joint["rating_high"]),
        "sentiment_high_rating_low_n": int(((joint["sentiment_high"] == 1) & (joint["rating_high"] == 0)).sum()),
        "sentiment_low_rating_high_n": int(((joint["sentiment_high"] == 0) & (joint["rating_high"] == 1)).sum()),
    }])
    return validation, agreement, joint


def _make_slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _map_park_metrics(
    config: dict[str, Any], park_metrics: pd.DataFrame, park_crosswalk: pd.DataFrame
) -> pd.DataFrame:
    settings = config["experience"]["sentiment"]
    legacy_id = settings.get("park_metrics_id_field", "park_id")
    park_name = settings.get("park_metrics_name_field", "park_name")
    required = {legacy_id, park_name, "MeanSentiment", "TotalReviews", "has_valid_sentiment"}
    missing = required - set(park_metrics.columns)
    if missing:
        raise ValueError("Park sentiment metrics are missing: " + ", ".join(sorted(missing)))

    mapped = park_metrics.loc[
        park_metrics["has_valid_sentiment"].astype(str).str.lower().eq("true")
    ].copy()
    mapped = mapped.rename(columns={legacy_id: "sentiment_park_id", park_name: "sentiment_park_name"})
    mapped["legacy_source"] = mapped["sentiment_park_id"].astype(str).str.rsplit("_", n=1).str[0]
    mapped["municipality"] = mapped["legacy_source"].map(
        settings.get("legacy_source_municipalities", {})
    )
    if mapped["municipality"].isna().any():
        sources = sorted(mapped.loc[mapped["municipality"].isna(), "legacy_source"].unique())
        raise ValueError(f"Park-level sentiment contains unmapped legacy sources: {sources}")
    mapped["park_key"] = (
        mapped["municipality"].map(_make_slug)
        + "__"
        + mapped["sentiment_park_name"].map(_make_slug)
    )
    overrides = settings.get("park_key_overrides", {})
    if overrides:
        mapped["park_key"] = mapped["sentiment_park_id"].map(overrides).fillna(mapped["park_key"])
    current = park_crosswalk[["park_id", "park_key"]].drop_duplicates().copy()
    if current["park_key"].duplicated().any():
        raise ValueError("Stage 05 park-key crosswalk contains duplicate park keys")
    mapped = mapped.merge(current, on="park_key", how="left", validate="one_to_one")
    mapped = mapped.dropna(subset=["park_id"]).copy()
    if mapped["park_id"].duplicated().any():
        raise ValueError("Multiple valid sentiment parks map to one current park")
    return mapped


def _weighted_sentiment_sensitivity(
    experience: pd.DataFrame,
    reach_pairs: pd.DataFrame,
    mapped_park_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = reach_pairs.loc[
        pd.to_numeric(reach_pairs["db_pop"], errors="coerce").gt(0),
        ["DAUID", "park_id"],
    ].drop_duplicates().merge(
        mapped_park_metrics[["park_id", "sentiment_park_id", "MeanSentiment", "TotalReviews"]],
        on="park_id",
        how="inner",
        validate="many_to_one",
    )
    # Match the proposal workflow: damp destination-park dominance with the
    # natural log of total Google reviews rather than raw review counts.
    scored["review_weight"] = np.log1p(pd.to_numeric(scored["TotalReviews"], errors="coerce"))
    scored["weighted_score"] = scored["MeanSentiment"] * scored["review_weight"]
    weighted = scored.groupby("DAUID", as_index=False).agg(
        review_weighted_numerator=("weighted_score", "sum"),
        review_weighted_denominator=("review_weight", "sum"),
        weighted_sentiment_place_count=("sentiment_park_id", "nunique"),
    )
    weighted["experience_sentiment_review_weighted"] = (
        weighted["review_weighted_numerator"] / weighted["review_weighted_denominator"]
    )
    compared = experience[["DAUID", "experience_sentiment_mean", "high_supply"]].merge(
        weighted[["DAUID", "experience_sentiment_review_weighted", "weighted_sentiment_place_count"]],
        on="DAUID",
        how="left",
    )
    joint = compared.dropna(subset=["experience_sentiment_mean", "experience_sentiment_review_weighted"]).copy()
    primary_median = float(joint["experience_sentiment_mean"].median())
    weighted_median = float(joint["experience_sentiment_review_weighted"].median())
    joint["primary_high"] = joint["experience_sentiment_mean"].ge(primary_median).astype(int)
    joint["weighted_high"] = joint["experience_sentiment_review_weighted"].ge(weighted_median).astype(int)
    joint["primary_divergence"] = np.where(
        joint["high_supply"], np.where(joint["primary_high"].eq(1), "HH", "HL"), np.where(joint["primary_high"].eq(1), "LH", "LL")
    )
    joint["weighted_divergence"] = np.where(
        joint["high_supply"], np.where(joint["weighted_high"].eq(1), "HH", "HL"), np.where(joint["weighted_high"].eq(1), "LH", "LL")
    )
    summary = pd.DataFrame([{
        "weight_definition": "log1p total Google reviews",
        "n": len(joint),
        "primary_median": primary_median,
        "review_weighted_median": weighted_median,
        "high_low_agreement_percent": float(joint["primary_high"].eq(joint["weighted_high"]).mean() * 100),
        "high_low_cohen_kappa": _cohen_kappa(joint["primary_high"], joint["weighted_high"]),
        "divergence_changed_n": int(joint["primary_divergence"].ne(joint["weighted_divergence"]).sum()),
        "divergence_changed_percent": float(joint["primary_divergence"].ne(joint["weighted_divergence"]).mean() * 100),
    }])
    return summary, weighted


def _has_positive_mention(text: str, pattern: re.Pattern[str]) -> int:
    for match in pattern.finditer(text):
        prior_words = re.findall(r"[a-z]+(?:[-'][a-z]+)?", text[: match.start()].lower())[-4:]
        if not NEGATIONS.intersection(prior_words):
            return 1
    return 0


def _extract_amenities(reviews: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    text = reviews["Review"].fillna("").astype(str).str.lower()
    mention_columns: dict[str, pd.Series] = {}
    for field, (_, keywords) in AMENITIES.items():
        expression = r"(?<!\w)(?:" + "|".join(re.escape(keyword) for keyword in keywords) + r")(?!\w)"
        pattern = re.compile(expression, flags=re.IGNORECASE)
        mention_columns[field] = text.map(lambda value: _has_positive_mention(value, pattern))
    mentions = pd.DataFrame(mention_columns, index=reviews.index)
    mentions["PlaceID"] = reviews["PlaceID"].astype("string")
    counts = mentions.groupby("PlaceID", as_index=False).agg({field: "sum" for field in AMENITIES})
    for field in AMENITIES:
        counts[field] = counts[field].ge(2).astype(int)
    counts["amenity_type_count"] = counts[list(AMENITIES)].sum(axis=1)
    metadata = ratings.loc[ratings["match_validated"] & ratings["PlaceID"].notna(), ["PlaceID", "park_id", "park_name", "municipality"]].copy()
    park_labels = metadata.groupby("PlaceID", as_index=False).agg(
        matched_park_count=("park_id", "nunique"),
        park_ids=("park_id", lambda values: "|".join(sorted(set(values.astype(str))))),
        park_names=("park_name", lambda values: "|".join(sorted(set(values.astype(str))))),
        municipalities=("municipality", lambda values: "|".join(sorted(set(values.astype(str))))),
    )
    return counts.merge(park_labels, on="PlaceID", how="inner", validate="one_to_one")


def _da_usability(equity: pd.DataFrame, entities: pd.DataFrame, park_amenities: pd.DataFrame) -> pd.DataFrame:
    joined = entities[["DAUID", "PlaceID"]].drop_duplicates().merge(
        park_amenities[["PlaceID", *AMENITIES]], on="PlaceID", how="inner"
    )
    aggregated = joined.groupby("DAUID", as_index=False).agg(
        **{field: (field, "max") for field in AMENITIES},
        amenity_google_entity_count=("PlaceID", "nunique"),
    )
    result = equity[["DAUID"]].merge(aggregated, on="DAUID", how="left", validate="one_to_one")
    for field in AMENITIES:
        result[field] = result[field].fillna(0).astype(int)
    result["amenity_google_entity_count"] = result["amenity_google_entity_count"].fillna(0).astype(int)
    result["amenity_type_count"] = result[list(AMENITIES)].sum(axis=1)
    return result


def _amenity_associations(
    equity: pd.DataFrame,
    usability: pd.DataFrame,
    park_amenities: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = equity.merge(usability, on="DAUID", how="left", validate="one_to_one")
    frame = frame[frame["divergence_sentiment_sensitivity"].isin(CLASS_ORDER)].copy()
    rows = []
    prevalence_rows = []
    for field, (label, _) in AMENITIES.items():
        contingency = pd.crosstab(frame["divergence_sentiment_sensitivity"], frame[field]).reindex(CLASS_ORDER, fill_value=0)
        if contingency.shape[1] == 2:
            statistic, p_value, degrees, expected = chi2_contingency(contingency)
            minimum_expected = float(expected.min())
        else:
            statistic = p_value = minimum_expected = np.nan
            degrees = 0
        rows.append({
            "amenity": field,
            "amenity_label": label,
            "n": len(frame),
            "chi_square": statistic,
            "degrees_of_freedom": degrees,
            "p_value": p_value,
            "significance": _significance(p_value),
            "minimum_expected_count": minimum_expected,
            "cramers_v": np.sqrt(statistic / len(frame)) if len(frame) and np.isfinite(statistic) else np.nan,
        })
        for class_name in CLASS_ORDER:
            values = frame.loc[frame["divergence_sentiment_sensitivity"].eq(class_name), field]
            prevalence_rows.append({
                "amenity": field,
                "amenity_label": label,
                "divergence_class": class_name,
                "da_count": len(values),
                "prevalence_percent": float(values.mean() * 100) if len(values) else np.nan,
            })
    chi_square = pd.DataFrame(rows).sort_values("chi_square", ascending=False)
    prevalence = pd.DataFrame(prevalence_rows)

    park = park_amenities.merge(
        sentiment[["PlaceID", "MeanSentiment", "AvgRating", "n_text_reviews", "has_valid_sentiment"]],
        on="PlaceID",
        how="inner",
        validate="one_to_one",
    )
    park = park[park["has_valid_sentiment"].astype(str).str.lower().eq("true")].copy()
    correlation_rows = []
    for field, label in [("amenity_type_count", "Amenity type count"), *[(key, value[0]) for key, value in AMENITIES.items()]]:
        for outcome, outcome_label in [("MeanSentiment", "Mean sentiment"), ("AvgRating", "Google star rating")]:
            pairs = park[[field, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            rho, p_value = (spearmanr(pairs[field], pairs[outcome]) if len(pairs) >= 3 and pairs.nunique().min() > 1 else (np.nan, np.nan))
            correlation_rows.append({
                "amenity": field,
                "amenity_label": label,
                "outcome": outcome,
                "outcome_label": outcome_label,
                "n": len(pairs),
                "spearman_rho": rho,
                "p_value": p_value,
                "significance": _significance(p_value),
            })
    return prevalence, chi_square, pd.DataFrame(correlation_rows)


def _normalize_name(value: Any) -> str:
    text = str(value).casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def _official_validation(
    facilities_path: Path,
    washrooms_path: Path,
    ratings: pd.DataFrame,
    park_amenities: pd.DataFrame,
) -> pd.DataFrame:
    facilities = pd.read_csv(facilities_path, sep=";", encoding="utf-8-sig")
    facilities.columns = facilities.columns.str.strip()
    facilities["amenity"] = facilities["FacilityType"].map(FACILITY_MAP)
    facilities["park_key"] = facilities["Name"].map(_normalize_name)
    official = facilities.dropna(subset=["amenity"]).assign(present=1).pivot_table(
        index="park_key", columns="amenity", values="present", aggfunc="max", fill_value=0
    ).reset_index()
    washrooms = pd.read_csv(washrooms_path, sep=";", encoding="utf-8-sig")
    washrooms.columns = washrooms.columns.str.strip()
    washroom_keys = set(washrooms["Park Name"].dropna().map(_normalize_name))
    all_keys = sorted(set(official["park_key"]) | washroom_keys)
    official = pd.DataFrame({"park_key": all_keys}).merge(official, on="park_key", how="left").fillna(0)
    official["washroom"] = official["park_key"].isin(washroom_keys).astype(int)

    mapping = ratings.loc[
        ratings["match_validated"] & ratings["PlaceID"].notna() & ratings["municipality"].eq("Vancouver"),
        ["PlaceID", "park_name"],
    ].merge(park_amenities[["PlaceID", *AMENITIES]], on="PlaceID", how="inner")
    mapping["park_key"] = mapping["park_name"].map(_normalize_name)
    observed = mapping.groupby("park_key", as_index=False).agg({field: "max" for field in AMENITIES})
    compared = official.merge(observed, on="park_key", how="inner", suffixes=("_official", "_review"))
    rows = []
    for field in VALIDATION_AMENITIES:
        official_field = f"{field}_official" if f"{field}_official" in compared else field
        review_field = f"{field}_review" if f"{field}_review" in compared else field
        # Pivot columns only receive suffixes when both sources contain the field.
        if official_field == review_field:
            continue
        left = pd.to_numeric(compared[official_field], errors="coerce").fillna(0).astype(int)
        right = pd.to_numeric(compared[review_field], errors="coerce").fillna(0).astype(int)
        rows.append({
            "amenity": field,
            "amenity_label": AMENITIES[field][0],
            "parks_compared": len(compared),
            "official_present_n": int(left.sum()),
            "review_present_n": int(right.sum()),
            "agreement_percent": float(left.eq(right).mean() * 100),
            "cohen_kappa": _cohen_kappa(left, right),
        })
    return pd.DataFrame(rows)


def _amenity_model(
    equity: pd.DataFrame, usability: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_column = "divergence_sentiment_sensitivity"
    socio_fields = [field for field, _ in MODEL_SOCIO]
    data = equity[["DAUID", class_column, *socio_fields]].merge(
        usability[["DAUID", *MODEL_AMENITIES]], on="DAUID", how="left", validate="one_to_one"
    )
    data = data[data[class_column].isin(CLASS_ORDER)].copy()
    fields = [*socio_fields, *MODEL_AMENITIES]
    data[fields] = data[fields].apply(pd.to_numeric, errors="coerce")
    data = data.dropna().copy()
    counts = data[class_column].value_counts().reindex(MODEL_CLASS_ORDER, fill_value=0)
    fit_row: dict[str, Any] = {
        "model": "Amenity-adjusted multinomial logit; reference = LL",
        "n": len(data),
        **{f"{name}_count": int(counts[name]) for name in MODEL_CLASS_ORDER},
    }
    socio_z = (data[socio_fields] - data[socio_fields].mean()) / data[socio_fields].std(ddof=1)
    design = pd.concat([socio_z, data[MODEL_AMENITIES]], axis=1)
    constant_fields = design.columns[design.nunique(dropna=True).le(1)].tolist()
    design = design.drop(columns=constant_fields)
    design = sm.add_constant(design, has_constant="add")
    labels = {**dict(MODEL_SOCIO), **{field: AMENITIES[field][0] for field in MODEL_AMENITIES}}
    vif = pd.DataFrame({
        "predictor": design.columns[1:],
        "predictor_label": [labels[field] for field in design.columns[1:]],
        "vif": [variance_inflation_factor(design.to_numpy(), index) for index in range(1, design.shape[1])],
    })
    outcome = pd.Categorical(data[class_column], categories=MODEL_CLASS_ORDER).codes
    try:
        result = sm.MNLogit(outcome, design).fit(method="newton", maxiter=300, disp=False)
    except (np.linalg.LinAlgError, ValueError) as error:
        fit_row.update({"status": f"not_fitted: {error}", "constant_predictors_removed": "|".join(constant_fields)})
        empty = pd.DataFrame(columns=["outcome_vs_LL", "predictor", "odds_ratio", "ci_lower", "ci_upper", "p_value", "significance"])
        return vif, empty, empty.copy(), pd.DataFrame([fit_row])
    rows = []
    for column, outcome_name in enumerate(MODEL_CLASS_ORDER[1:]):
        for field in design.columns[1:]:
            position = design.columns.get_loc(field)
            coefficient = float(result.params.iloc[position, column])
            standard_error = float(result.bse.iloc[position, column])
            p_value = float(result.pvalues.iloc[position, column])
            rows.append({
                "outcome_vs_LL": outcome_name,
                "predictor": field,
                "predictor_label": labels[field],
                "odds_ratio": np.exp(coefficient),
                "ci_lower": np.exp(coefficient - 1.96 * standard_error),
                "ci_upper": np.exp(coefficient + 1.96 * standard_error),
                "p_value": p_value,
                "significance": _significance(p_value),
            })
    results = pd.DataFrame(rows)
    fit_row.update({
        "status": "fitted" if result.mle_retvals.get("converged", False) else "fitted_not_converged",
        "constant_predictors_removed": "|".join(constant_fields),
        "log_likelihood": float(result.llf),
        "null_log_likelihood": float(result.llnull),
        "likelihood_ratio_p_value": float(result.llr_pvalue),
        "mcfadden_r2": 1 - float(result.llf / result.llnull),
    })
    formatted = results.copy()
    formatted["confidence_interval_95"] = formatted.apply(lambda row: f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]", axis=1)
    formatted["reference_category"] = "LL - Low supply / low experience"
    formatted["model_n"] = len(data)
    formatted["mcfadden_r2"] = fit_row["mcfadden_r2"]
    return vif, results, formatted, pd.DataFrame([fit_row])


def _validation_figure(
    reviews: pd.DataFrame,
    park_metrics: pd.DataFrame,
    experience: pd.DataFrame,
    validation: pd.DataFrame,
    output: Path,
) -> None:
    review_rating = pd.to_numeric(reviews.get("Rating"), errors="coerce")
    if review_rating.notna().sum() == 0 and "rating" in reviews:
        review_rating = pd.to_numeric(reviews["rating"], errors="coerce")
    palette = {"point": "#01665E", "line": "#8C510A"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))

    review_pairs = pd.DataFrame({
        "sentiment": pd.to_numeric(reviews["Sentiment"], errors="coerce"),
        "rating": review_rating,
    }).dropna()
    review_row = validation.loc[validation["level"].eq("review")].iloc[0]
    review_groups = [
        review_pairs.loc[review_pairs["rating"].eq(rating), "sentiment"].to_numpy()
        for rating in range(1, 6)
    ]
    violin = axes[0].violinplot(
        review_groups,
        positions=np.arange(1, 6),
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(palette["point"])
        body.set_edgecolor("none")
        body.set_alpha(0.5)
    for rating, values in enumerate(review_groups, start=1):
        if len(values):
            median = float(np.median(values))
            axes[0].vlines(median, rating - 0.18, rating + 0.18, color=palette["line"], linewidth=1.2)
    axes[0].set_title(
        f"A. Review level\nPearson r={review_row['pearson_r']:.3f}; "
        f"Spearman ρ={review_row['spearman_rho']:.3f}; n={len(review_pairs):,}",
        fontsize=9,
    )
    axes[0].set_xlabel("Review sentiment")
    axes[0].set_ylabel("Google star rating")
    axes[0].set_yticks(np.arange(1, 6))
    axes[0].set_xlim(-1.05, 1.05)

    scatter_panels = [
        (
            park_metrics.loc[park_metrics["has_valid_sentiment"].astype(str).str.lower().eq("true"), "MeanSentiment"],
            park_metrics.loc[park_metrics["has_valid_sentiment"].astype(str).str.lower().eq("true"), "AvgRating"],
            "B. Park level",
            "Mean sentiment",
            "Mean star rating",
            validation.loc[validation["level"].eq("park")].iloc[0],
        ),
        (
            experience["experience_sentiment_mean"],
            experience["experience_rating_mean"],
            "C. Neighbourhood level",
            "Mean sentiment",
            "Mean star rating",
            validation.loc[validation["level"].eq("dissemination area")].iloc[0],
        ),
    ]
    for axis, (x, y, title, x_label, y_label, row) in zip(axes[1:], scatter_panels):
        pairs = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
        axis.scatter(pairs["x"], pairs["y"], s=10, alpha=0.22, color=palette["point"], edgecolors="none")
        if len(pairs) >= 3 and pairs.nunique().min() > 1:
            slope, intercept = np.polyfit(pairs["x"], pairs["y"], 1)
            x_line = np.linspace(pairs["x"].min(), pairs["x"].max(), 100)
            axis.plot(x_line, slope * x_line + intercept, color=palette["line"], linewidth=1.3)
        axis.set_title(f"{title}\nPearson r={row['pearson_r']:.3f}; Spearman ρ={row['spearman_rho']:.3f}; n={len(pairs):,}", fontsize=9)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#E5E5E5", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pairwise_figure(equity: pd.DataFrame, output: Path) -> None:
    """Render the proposal's six pairwise supply-experience relationships."""
    frame = equity.dropna(subset=["experience_sentiment_mean"]).copy()
    relationships = [
        ("park_access_pop_coverage", "supply_ha_per_1000_cap20", "Park coverage (proportion within 400 m)", "Accessible park area (ha/1,000 residents; log)", False, True),
        ("park_access_pop_coverage", "digital_salience_reviews_per_1000", "Park coverage (proportion within 400 m)", "Digital salience (reviews/1,000 residents; log)", False, True),
        ("park_access_pop_coverage", "experience_sentiment_mean", "Park coverage (proportion within 400 m)", "Mean sentiment score", False, False),
        ("supply_ha_per_1000_cap20", "digital_salience_reviews_per_1000", "Accessible park area (ha/1,000 residents; log)", "Digital salience (reviews/1,000 residents; log)", True, True),
        ("supply_ha_per_1000_cap20", "experience_sentiment_mean", "Accessible park area (ha/1,000 residents; log)", "Mean sentiment score", True, False),
        ("digital_salience_reviews_per_1000", "experience_sentiment_mean", "Digital salience (reviews/1,000 residents; log)", "Mean sentiment score", True, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4))
    for axis, (x_field, y_field, x_label, y_label, x_log, y_log) in zip(axes.flat, relationships):
        pairs = frame[[x_field, y_field]].apply(pd.to_numeric, errors="coerce").dropna()
        rho, p_value = spearmanr(pairs[x_field], pairs[y_field])
        plotted = pairs.copy()
        if x_log:
            plotted = plotted[plotted[x_field] > 0]
            axis.set_xscale("log")
        if y_log:
            plotted = plotted[plotted[y_field] > 0]
            axis.set_yscale("log")
        axis.scatter(plotted[x_field], plotted[y_field], s=11, alpha=0.35, color="#4F8F83", edgecolors="none")
        axis.axvline(float(pairs[x_field].median()), color="#777777", linestyle="--", linewidth=0.8)
        axis.axhline(float(pairs[y_field].median()), color="#777777", linestyle="--", linewidth=0.8)
        p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        axis.set_title(f"ρ={rho:.3f}, {p_text}, n={len(pairs):,}", fontsize=9)
        axis.set_xlabel(x_label, fontsize=8)
        axis.set_ylabel(y_label, fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _amenity_figures(
    prevalence: pd.DataFrame,
    chi_square: pd.DataFrame,
    palette: dict[str, str],
    dotplot_path: Path,
    heatmap_path: Path,
) -> None:
    matrix = prevalence.pivot(index="amenity_label", columns="divergence_class", values="prevalence_percent").reindex(columns=CLASS_ORDER)
    sig = chi_square.set_index("amenity_label")["significance"].replace("ns", "").to_dict()
    matrix["contrast"] = matrix[["HH", "LH"]].mean(axis=1) - matrix[["HL", "LL"]].mean(axis=1)
    matrix = matrix.sort_values("contrast").drop(columns="contrast")
    labels = [f"{label} {sig.get(label, '')}".strip() for label in matrix.index]
    y = np.arange(len(matrix))
    fig, axis = plt.subplots(figsize=(8, 6.4))
    for position in y:
        axis.axhline(position, color="#E6E6E6", linewidth=0.7, zorder=0)
    for class_name in CLASS_ORDER:
        axis.scatter(matrix[class_name], y, s=55, color=palette[class_name], label={"HH": "High supply, high experience", "LH": "Low supply, high experience", "HL": "High supply, low experience", "LL": "Low supply, low experience"}[class_name], zorder=2)
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Prevalence (% of DAs with amenity present)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        title="Divergence class",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(dotplot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    proposal_order = [value[0] for value in AMENITIES.values()]
    heat = prevalence.pivot(index="amenity_label", columns="divergence_class", values="prevalence_percent").reindex(index=proposal_order, columns=CLASS_ORDER)
    class_counts = prevalence.groupby("divergence_class")["da_count"].first().reindex(CLASS_ORDER)
    heat["All"] = heat[CLASS_ORDER].mul(class_counts, axis=1).sum(axis=1) / class_counts.sum()
    heat_columns = ["HH", "LH", "HL", "LL", "All"]
    heat = heat[heat_columns]
    ylabels = [f"{label} {sig.get(label, '')}".strip() for label in heat.index]
    fig, axis = plt.subplots(figsize=(8.4, 6.5))
    image = axis.imshow(heat, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            value = heat.iloc[row, column]
            axis.text(column, row, f"{value:.0f}%", ha="center", va="center", fontsize=8, color="white" if value > 60 else "black")
    header_labels = [
        f"High - High\n(n={int(class_counts['HH'])})",
        f"Low Supply\nHigh Exp.\n(n={int(class_counts['LH'])})",
        f"High Supply\nLow Exp.\n(n={int(class_counts['HL'])})",
        f"Low - Low\n(n={int(class_counts['LL'])})",
        f"All DAs\n(n={int(class_counts.sum())})",
    ]
    axis.set_xticks(np.arange(len(heat.columns)), header_labels)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", pad=8, labelsize=8)
    header_colors = [palette["HH"], palette["LH"], palette["HL"], palette["LL"], "#666666"]
    for label, color in zip(axis.get_xticklabels(), header_colors):
        label.set_color("white")
        label.set_fontweight("bold")
        label.set_bbox({"facecolor": color, "edgecolor": "none", "boxstyle": "round,pad=0.45"})
    axis.set_yticks(np.arange(len(heat.index)), ylabels)
    axis.set_title(
        "Socially Perceived Recreational Affordances by Divergence Quadrant — Vancouver\n"
        "% of DAs with ≥1 reachable park mentioning each amenity type (≥2 review mentions)",
        fontsize=9,
        pad=18,
    )
    fig.colorbar(image, ax=axis, label="% DAs", shrink=0.8)
    fig.subplots_adjust(top=0.78, left=0.25, right=0.9, bottom=0.06)
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_outputs(paths: dict[str, Path], tables: dict[str, pd.DataFrame]) -> None:
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)


def run(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    _require_files(paths)
    ratings = pd.read_csv(paths["ratings"], dtype={"PlaceID": "string"})
    ratings["match_validated"] = ratings["match_validated"].astype(str).str.lower().eq("true")
    validated_place_ids = set(ratings.loc[ratings["match_validated"] & ratings["PlaceID"].notna(), "PlaceID"].astype(str))
    sentiment = pd.read_csv(paths["sentiment"], dtype={"PlaceID": "string"})
    required_sentiment = {"PlaceID", "MeanSentiment", "AvgRating", "TotalReviews", "n_text_reviews", "has_valid_sentiment"}
    missing = required_sentiment - set(sentiment.columns)
    if missing:
        raise ValueError("Sentiment table is missing: " + ", ".join(sorted(missing)))
    sentiment = sentiment.drop_duplicates("PlaceID")
    park_metrics = pd.read_csv(paths["park_metrics"])
    required_park_metrics = {"MeanSentiment", "AvgRating", "has_valid_sentiment"}
    missing = required_park_metrics - set(park_metrics.columns)
    if missing:
        raise ValueError("Park sentiment metrics are missing: " + ", ".join(sorted(missing)))
    reviews = pd.read_csv(
        paths["reviews"],
        usecols=lambda column: column in {"PlaceID", "ReviewID", "Review", "Sentiment", "Rating", "rating"},
        dtype={"PlaceID": "string", "ReviewID": "string"},
        low_memory=False,
    )
    reviews = reviews.drop_duplicates("ReviewID") if "ReviewID" in reviews and reviews["ReviewID"].notna().any() else reviews.drop_duplicates(["PlaceID", "Review"])
    amenity_reviews = reviews[reviews["PlaceID"].isin(validated_place_ids)].copy()
    experience = pd.read_csv(paths["experience"], dtype={"DAUID": "string"})
    entities = pd.read_csv(paths["da_entities"], dtype={"DAUID": "string", "PlaceID": "string"})
    reach_pairs = pd.read_csv(paths["reach_pairs"], dtype={"DAUID": "string", "park_id": "string"})
    park_crosswalk = pd.read_csv(paths["park_crosswalk"], dtype={"park_id": "string"})
    equity = pd.read_csv(paths["equity"], dtype={"DAUID": "string"})

    rating_labels = ratings.loc[
        ratings["match_validated"] & ratings["PlaceID"].notna(),
        ["PlaceID", "park_id", "park_name", "municipality"],
    ].groupby("PlaceID", as_index=False).agg(
        matched_park_ids=("park_id", lambda values: "|".join(sorted(set(values.astype(str))))),
        matched_park_names=("park_name", lambda values: "|".join(sorted(set(values.astype(str))))),
        matched_municipalities=("municipality", lambda values: "|".join(sorted(set(values.astype(str))))),
    )
    reachable_ids = set(entities["PlaceID"].dropna().astype(str))
    linkage_audit = sentiment.merge(rating_labels, on="PlaceID", how="left", validate="one_to_one")
    linkage_audit["in_validated_shared_inventory"] = linkage_audit["matched_park_ids"].notna()
    linkage_audit["reachable_from_populated_vancouver_da"] = linkage_audit["PlaceID"].isin(reachable_ids)
    linkage_audit["linkage_status"] = np.select(
        [
            ~linkage_audit["in_validated_shared_inventory"],
            ~linkage_audit["reachable_from_populated_vancouver_da"],
            ~linkage_audit["has_valid_sentiment"].astype(str).str.lower().eq("true"),
        ],
        ["not in current validated match table", "validated but not reachable in Vancouver", "reachable but below sentiment review threshold"],
        default="reachable and sentiment eligible",
    )

    validation, agreement, _ = _sentiment_validation(reviews, park_metrics, experience)
    mapped_park_metrics = _map_park_metrics(config, park_metrics, park_crosswalk)
    sensitivity, weighted = _weighted_sentiment_sensitivity(
        experience, reach_pairs, mapped_park_metrics
    )
    park_amenities = _extract_amenities(amenity_reviews, ratings)
    usability = _da_usability(equity, entities, park_amenities)
    prevalence, amenity_chi, correlations = _amenity_associations(equity, usability, park_amenities, sentiment)
    amenity_validation = _official_validation(paths["facilities"], paths["washrooms"], ratings, park_amenities)
    vif, model, model_formatted, model_fit = _amenity_model(equity, usability)
    baseline_fit = pd.read_csv(paths["baseline_model_fit"])
    baseline_r2 = pd.to_numeric(baseline_fit.get("mcfadden_r2"), errors="coerce").dropna()
    baseline_r2_value = float(baseline_r2.iloc[0]) if not baseline_r2.empty else np.nan
    model_fit["baseline_sociodemographic_mcfadden_r2"] = baseline_r2_value
    model_fit["mcfadden_r2_improvement"] = model_fit.get("mcfadden_r2", np.nan) - baseline_r2_value

    summary = pd.DataFrame([{
        "analysis_area": "Vancouver",
        "review_rows_used_for_validation": len(reviews),
        "review_rows_used_for_current_amenities": len(amenity_reviews),
        "validated_place_ids_shared_inventory": len(validated_place_ids),
        "valid_sentiment_place_ids_in_shared_matches": int((sentiment["PlaceID"].isin(validated_place_ids) & sentiment["has_valid_sentiment"].astype(str).str.lower().eq("true")).sum()),
        "reachable_validated_place_ids_vancouver": int(entities["PlaceID"].nunique()),
        "reachable_valid_sentiment_place_ids_vancouver": int((sentiment["PlaceID"].isin(set(entities["PlaceID"].dropna().astype(str))) & sentiment["has_valid_sentiment"].astype(str).str.lower().eq("true")).sum()),
        "valid_sentiment_parks_mapped_to_current_inventory": len(mapped_park_metrics),
        "das_with_primary_park_level_sentiment": int(experience["experience_sentiment_mean"].notna().sum()),
        "das_with_joint_sentiment_and_rating": int(agreement.loc[0, "n"]),
        "sentiment_rating_agreement_percent": float(agreement.loc[0, "agreement_percent"]),
        "sentiment_rating_cohen_kappa": float(agreement.loc[0, "cohen_kappa"]),
        "review_weighted_divergence_changes": int(sensitivity.loc[0, "divergence_changed_n"]),
        "review_weighted_divergence_change_percent": float(sensitivity.loc[0, "divergence_changed_percent"]),
        "park_google_entities_with_amenities": len(park_amenities),
        "das_with_amenity_entities": int(usability["amenity_google_entity_count"].gt(0).sum()),
        "official_validation_park_count": int(amenity_validation["parks_compared"].max()) if not amenity_validation.empty else 0,
        "mean_amenity_kappa": float(amenity_validation["cohen_kappa"].mean()) if not amenity_validation.empty else np.nan,
        "amenity_model_status": model_fit.loc[0, "status"],
        "amenity_model_n": int(model_fit.loc[0, "n"]),
        "amenity_model_mcfadden_r2": model_fit.loc[0].get("mcfadden_r2", np.nan),
        "baseline_sociodemographic_mcfadden_r2": baseline_r2_value,
        "mcfadden_r2_improvement": model_fit.loc[0].get("mcfadden_r2_improvement", np.nan),
        "sentiment_validation_figure_caption": "Agreement between review-derived sentiment scores and Google star ratings at review, park, and neighbourhood levels in Vancouver.",
        "pairwise_figure_caption": "Pairwise relationships among park coverage, accessible park area, digital salience, and expressed satisfaction across Vancouver dissemination areas.",
        "amenity_profiles_figure_caption": "Perceived recreational affordances by sentiment-based divergence class in Vancouver. * p<.05, ** p<.01, *** p<.001.",
        "amenity_heatmap_figure_caption": "Prevalence of review-derived amenity categories by sentiment-based divergence class in Vancouver.",
    }])
    return {
        "validation": validation,
        "agreement": agreement,
        "linkage_audit": linkage_audit,
        "sensitivity": sensitivity,
        "park_amenities": park_amenities,
        "da_usability": usability,
        "amenity_prevalence": prevalence,
        "amenity_chi_square": amenity_chi,
        "amenity_correlations": correlations,
        "amenity_validation": amenity_validation,
        "amenity_vif": vif,
        "amenity_model": model,
        "amenity_model_formatted": model_formatted,
        "amenity_model_fit": model_fit,
        "summary": summary,
        "_reviews": reviews,
        "_sentiment": sentiment,
        "_park_metrics": park_metrics,
        "_experience": experience,
        "_equity": equity,
        "_weighted": weighted,
    }


def main(area: str, force: bool = False) -> int:
    if area != "vancouver":
        print("Stage 11 is Vancouver-specific; no output is required for this area.")
        return 0
    config = load_config(area)
    if not config.get("experience", {}).get("sentiment", {}).get("enabled", False):
        raise ValueError("Enable Vancouver sentiment before running Stage 11")
    final_paths = _paths(config)
    output_keys = [key for key in final_paths if key not in {"ratings", "sentiment", "reviews", "park_metrics", "facilities", "washrooms", "experience", "da_entities", "reach_pairs", "park_crosswalk", "equity", "baseline_model_fit"}]
    if not force and all(final_paths[key].exists() for key in output_keys):
        print("Stage 11 outputs already exist. Use --force to rebuild them.")
        return 0

    parent = output_root(config).parent
    with inherited_temp_directory(parent, "vancouver_11_") as temp_dir:
        temp_paths = _paths(config, temp_dir / "vancouver")
        temp_paths["validation_figure"].parent.mkdir(parents=True, exist_ok=True)
        tables = run(config, final_paths)
        _write_outputs(temp_paths, {key: value for key, value in tables.items() if not key.startswith("_")})
        _validation_figure(tables["_reviews"], tables["_park_metrics"], tables["_experience"], tables["validation"], temp_paths["validation_figure"])
        _pairwise_figure(tables["_equity"], temp_paths["pairwise_figure"])
        palette = {key: value for key, value in config["plot"]["palette"].items() if key in CLASS_ORDER}
        _amenity_figures(tables["amenity_prevalence"], tables["amenity_chi_square"], palette, temp_paths["amenity_dotplot"], temp_paths["amenity_heatmap"])
        for key in output_keys:
            final_paths[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_paths[key], final_paths[key])

    summary = pd.read_csv(final_paths["summary"])
    print("Stage 11 Vancouver sentiment validation and usability analysis complete.")
    print(f"Reviews analysed: {int(summary.loc[0, 'review_rows_used_for_validation']):,}")
    print(f"Amenity model: {summary.loc[0, 'amenity_model_status']}")
    print(f"Summary: {final_paths['summary']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, help="Use vancouver")
    parser.add_argument("--force", action="store_true", help="Rebuild Stage 11 outputs")
    arguments = parser.parse_args()
    try:
        raise SystemExit(main(arguments.area, force=arguments.force))
    except (ConfigError, FileNotFoundError, ValueError, KeyError) as error:
        print(f"Stage 11 failed: {error}")
        raise SystemExit(2)
