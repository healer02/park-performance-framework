"""Retrieve or rescore Google candidates without selecting on rating outcomes.

This preparation step keeps entity matching separate from the experience
outcome. Google ratings and review counts are copied only after a candidate is
selected; they never contribute to the match score. A fresh-only mode lets the
final rebuild stand independently of the legacy candidate search while retaining
the legacy ratings solely as a comparison benchmark.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from project_config import (
    data_path,
    load_config,
    output_root,
    override_path,
    workspace_data_root,
)


GENERIC_NAME_TOKENS = {
    "area",
    "city",
    "community",
    "conservation",
    "garden",
    "greenbelt",
    "greenway",
    "linear",
    "management",
    "municipal",
    "natural",
    "nature",
    "neighborhood",
    "neighbourhood",
    "park",
    "parkway",
    "recreation",
    "recreational",
    "regional",
    "reserve",
    "site",
    "trail",
    "waterfront",
}

STRONG_PLACE_TYPES = {
    "beach",
    "botanical_garden",
    "city_park",
    "cycling_park",
    "dog_park",
    "garden",
    "hiking_area",
    "national_park",
    "natural_feature",
    "nature_preserve",
    "park",
    "playground",
    "scenic_spot",
    "wildlife_refuge",
}

CONTEXT_PLACE_TYPES = {
    "athletic_field",
    "community_center",
    "cultural_landmark",
    "fishing_pier",
    "historical_landmark",
    "historical_place",
    "marina",
    "museum",
    "plaza",
    "skateboard_park",
    "sports_activity_location",
    "sports_complex",
    "stadium",
    "tourist_attraction",
}

INCOMPATIBLE_PLACE_TYPES = {
    "administrative_area",
    "apartment_building",
    "bus_station",
    "church",
    "clothing_store",
    "condominium_complex",
    "laundry",
    "light_rail_station",
    "locality",
    "neighborhood",
    "parking_lot",
    "pharmacy",
    "place_of_worship",
    "political",
    "premise",
    "route",
    "school",
    "shopping_mall",
    "store",
    "street_address",
    "subway_station",
    "train_station",
    "transit_station",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "park_id",
    "PlaceID",
    "google_name",
    "google_latitude",
    "google_longitude",
    "google_types",
    "google_rating",
    "google_review_count",
}


def _normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"\bneighborhood\b", "neighbourhood", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _name_scores(park_name: object, google_name: object) -> tuple[float, float, float]:
    park = _normalise_text(park_name)
    google = _normalise_text(google_name)
    sequence = SequenceMatcher(None, park, google).ratio() if park and google else 0.0

    park_tokens = set(park.split())
    google_tokens = set(google.split())
    denominator = len(park_tokens) + len(google_tokens)
    dice = (
        2 * len(park_tokens & google_tokens) / denominator if denominator else 0.0
    )

    park_info = park_tokens - GENERIC_NAME_TOKENS
    google_info = google_tokens - GENERIC_NAME_TOKENS
    informative_overlap = park_info & google_info
    if len(informative_overlap) >= 2:
        containment = len(park_info & google_info) / min(len(park_info), len(google_info))
    else:
        containment = 0.0

    combined = max(sequence, dice, containment * 0.9)
    return sequence, dice, combined


def _type_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {token.strip().casefold() for token in str(value).split("|") if token.strip()}


def _type_class(value: object, primary_type: object = None) -> str:
    tokens = _type_tokens(value) | _type_tokens(primary_type)
    if tokens & INCOMPATIBLE_PLACE_TYPES and not tokens & STRONG_PLACE_TYPES:
        return "incompatible"
    if tokens & STRONG_PLACE_TYPES:
        return "strong_recreation"
    if tokens & CONTEXT_PLACE_TYPES:
        return "context_only"
    return "unsupported"


def _distance_score(distance: float) -> float:
    if not np.isfinite(distance):
        return -100.0
    if distance <= 1:
        return 35.0
    if distance <= 50:
        return 30.0
    if distance <= 100:
        return 25.0
    if distance <= 250:
        return 15.0
    if distance <= 500:
        return 0.0
    return -min(60.0, (distance - 500.0) / 25.0)


def _type_score(place_type_class: str) -> float:
    return {
        "strong_recreation": 35.0,
        "context_only": 5.0,
        "unsupported": -20.0,
        "incompatible": -80.0,
    }[place_type_class]


def _key(municipality: object, park_name: object) -> str:
    return f"{_normalise_text(municipality)}|{_normalise_text(park_name)}"


def google_paths(config: dict[str, Any]) -> dict[str, Path]:
    slug = config["network"]["slug"]
    workspace = workspace_data_root(config)
    return {
        "parks": data_path(config, "parks"),
        "source_candidates": data_path(config, "google_candidates"),
        "source_ratings": data_path(config, "google_ratings"),
        "refresh_candidates": workspace
        / "interim"
        / "google"
        / slug
        / f"{slug}_google_place_candidates_refresh.csv",
        "api_log": workspace
        / "interim"
        / "google"
        / slug
        / f"{slug}_google_api_request_log.csv",
        "place_details_log": workspace
        / "interim"
        / "google"
        / slug
        / f"{slug}_google_place_details_request_log.csv",
        "candidate_scores": workspace
        / "interim"
        / "google"
        / slug
        / f"{slug}_google_candidate_scores.csv",
        "validated_ratings": workspace
        / "processed"
        / "google"
        / f"{slug}_google_ratings_validated.csv",
        "review_queue": output_root(config)
        / "tables"
        / f"{slug}_google_match_review_queue.csv",
        "comparison": output_root(config)
        / "tables"
        / f"{slug}_google_match_comparison.csv",
        "summary": output_root(config)
        / "tables"
        / f"{slug}_google_matching_summary.csv",
    }


def _read_csv(path: Path, required: set[str], **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, **kwargs)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    return frame


def _score_candidates(
    parks: gpd.GeoDataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    parks = parks[["park_id", "park_name", "municipality", "geometry"]].copy()
    parks["park_id"] = parks["park_id"].astype("string")
    candidates = candidates.copy()
    candidates["park_id"] = candidates["park_id"].astype("string")
    candidates = candidates.merge(
        parks,
        on="park_id",
        how="inner",
        validate="many_to_one",
        suffixes=("_source", ""),
    )
    if candidates.empty:
        empty_columns = {
            "distance_to_park_polygon_m": "float64",
            "name_sequence": "float64",
            "name_token_dice": "float64",
            "name_similarity_revised": "float64",
            "place_type_class": "string",
            "match_score_revised": "float64",
            "revised_candidate_rank": "Int64",
            "next_best_score": "float64",
            "score_margin": "float64",
        }
        for column, dtype in empty_columns.items():
            candidates[column] = pd.Series(dtype=dtype)
        return candidates.drop(columns="geometry")

    google_points = gpd.GeoSeries(
        gpd.points_from_xy(
            pd.to_numeric(candidates["google_longitude"], errors="coerce"),
            pd.to_numeric(candidates["google_latitude"], errors="coerce"),
        ),
        crs="EPSG:4326",
    ).to_crs(parks.crs)
    candidates["distance_to_park_polygon_m"] = [
        point.distance(polygon)
        if point is not None and not point.is_empty and polygon is not None
        else np.nan
        for point, polygon in zip(google_points, candidates["geometry"], strict=True)
    ]

    scores = candidates.apply(
        lambda row: _name_scores(row["park_name"], row["google_name"]), axis=1
    )
    candidates[["name_sequence", "name_token_dice", "name_similarity_revised"]] = (
        pd.DataFrame(scores.tolist(), index=candidates.index)
    )
    if "google_primary_type" not in candidates.columns:
        candidates["google_primary_type"] = pd.NA
    candidates["place_type_class"] = [
        _type_class(types, primary_type)
        for types, primary_type in zip(
            candidates["google_types"],
            candidates["google_primary_type"],
            strict=True,
        )
    ]
    candidates["match_score_revised"] = (
        candidates["name_similarity_revised"] * 100.0
        + candidates["distance_to_park_polygon_m"].map(_distance_score)
        + candidates["place_type_class"].map(_type_score)
    )
    candidates = candidates.sort_values(
        ["park_id", "match_score_revised", "PlaceID"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    candidates["revised_candidate_rank"] = (
        candidates.groupby("park_id").cumcount() + 1
    )
    candidates["next_best_score"] = candidates.groupby("park_id")[
        "match_score_revised"
    ].shift(-1)
    candidates["score_margin"] = (
        candidates["match_score_revised"] - candidates["next_best_score"]
    )
    return candidates.drop(columns="geometry")


def _load_overrides(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = _read_csv(
        override_path(config, "google_matches"),
        {
            "municipality",
            "park_name",
            "google_match_decision",
            "google_match_reason",
        },
        dtype="string",
    )
    place_ids = _read_csv(
        override_path(config, "google_place_ids"),
        {"park_id", "override_place_id", "override_reason"},
        dtype="string",
    )
    shared_entities = _read_csv(
        override_path(config, "google_shared_entities"),
        {"PlaceID", "shared_entity_decision", "shared_entity_reason"},
        dtype="string",
    )
    matches["park_key"] = [
        _key(city, name)
        for city, name in zip(matches["municipality"], matches["park_name"], strict=True)
    ]
    if matches["park_key"].duplicated().any():
        raise ValueError("Google match overrides contain duplicate park keys")
    if place_ids["park_id"].duplicated().any():
        raise ValueError("Google Place ID overrides contain duplicate park IDs")
    if shared_entities["PlaceID"].duplicated().any():
        raise ValueError("Google shared-entity overrides contain duplicate Place IDs")
    invalid_shared = set(
        shared_entities["shared_entity_decision"].str.casefold().dropna()
    ) - {"accept", "reject"}
    if invalid_shared:
        raise ValueError(
            "Google shared-entity decisions must be accept or reject: "
            + ", ".join(sorted(invalid_shared))
        )
    return matches, place_ids, shared_entities


def _candidate_for_place_id(
    candidates: pd.DataFrame, park_id: str, place_id: str
) -> pd.Series | None:
    match = candidates[
        candidates["park_id"].eq(park_id) & candidates["PlaceID"].eq(place_id)
    ]
    return None if match.empty else match.iloc[0]


def _select_matches(
    config: dict[str, Any],
    parks: gpd.GeoDataFrame,
    candidates: pd.DataFrame,
    old_ratings: pd.DataFrame,
    allow_legacy_override_fallback: bool,
) -> pd.DataFrame:
    matches, place_ids, shared_entities = _load_overrides(config)
    match_lookup = matches.set_index("park_key")
    place_lookup = place_ids.set_index("park_id")
    old_lookup = old_ratings.set_index("park_id")
    threshold_distance = float(
        config["google_matching"]["auto_accept_polygon_distance_m"]
    )
    threshold_name = float(config["google_matching"]["auto_accept_name_similarity"])
    threshold_margin = float(config["google_matching"]["ambiguous_score_margin"])

    rows: list[dict[str, Any]] = []
    for _, park in parks.sort_values("park_id").iterrows():
        park_id = str(park["park_id"])
        park_key = _key(park["municipality"], park["park_name"])
        park_candidates = candidates[candidates["park_id"].eq(park_id)]
        old = old_lookup.loc[park_id] if park_id in old_lookup.index else None
        override = match_lookup.loc[park_key] if park_key in match_lookup.index else None
        forced = place_lookup.loc[park_id] if park_id in place_lookup.index else None
        selected: pd.Series | None = None
        decision = "no_candidate"
        reason = "no cached Google candidate"

        if forced is not None:
            selected = _candidate_for_place_id(
                candidates, park_id, str(forced["override_place_id"])
            )
            if (
                allow_legacy_override_fallback
                and selected is None
                and old is not None
                and str(old.get("PlaceID")) == str(forced["override_place_id"])
            ):
                # A legacy forced candidate may have been found through a
                # different park's search. The reviewed PlaceID remains the
                # authority even when that row is absent from this park's
                # cached candidate subset.
                selected = old
            if selected is None:
                decision = "review"
                reason = "forced PlaceID is not present in the cached candidates"
            else:
                decision = "manual_accept"
                reason = str(forced["override_reason"])
        elif override is not None and str(override["google_match_decision"]).casefold() == "reject":
            decision = "manual_reject"
            reason = str(override["google_match_reason"])
        elif override is not None and str(override["google_match_decision"]).casefold() == "accept":
            old_place_id = None if old is None or pd.isna(old.get("PlaceID")) else str(old["PlaceID"])
            selected = (
                None
                if old_place_id is None
                else _candidate_for_place_id(candidates, park_id, old_place_id)
            )
            if selected is None:
                decision = "review"
                reason = "accepted legacy PlaceID is not present in the cached candidates"
            else:
                decision = "manual_accept"
                reason = str(override["google_match_reason"])
        elif not park_candidates.empty:
            selected = park_candidates.iloc[0]
            clear_margin = pd.isna(selected["score_margin"]) or (
                float(selected["score_margin"]) >= threshold_margin
            )
            acceptable = (
                selected["place_type_class"] == "strong_recreation"
                and float(selected["distance_to_park_polygon_m"]) <= threshold_distance
                and float(selected["name_similarity_revised"]) >= threshold_name
                and clear_margin
            )
            if acceptable:
                decision = "auto_accept"
                reason = "compatible type, polygon proximity, name similarity, and score margin"
            elif selected["place_type_class"] == "incompatible":
                decision = "review"
                reason = "top candidate has an incompatible place type"
            elif float(selected["distance_to_park_polygon_m"]) > threshold_distance:
                decision = "review"
                reason = "top candidate is too far from the park polygon"
            elif float(selected["name_similarity_revised"]) < threshold_name:
                decision = "review"
                reason = "top candidate has low normalized name similarity"
            elif not clear_margin:
                decision = "review"
                reason = "top two candidates have similar scores"
            else:
                decision = "review"
                reason = "top candidate lacks a strong recreational place type"

        row: dict[str, Any] = {
            "park_id": park_id,
            "park_name": park["park_name"],
            "municipality": park["municipality"],
            "match_decision": decision,
            "match_reason": reason,
            "match_validated": decision in {"auto_accept", "manual_accept"},
        }
        copy_columns = [
            "PlaceID",
            "google_name",
            "google_address",
            "google_rating",
            "google_review_count",
            "google_latitude",
            "google_longitude",
            "google_types",
            "google_primary_type",
            "google_maps_url",
            "distance_to_park_polygon_m",
            "name_similarity_revised",
            "place_type_class",
            "match_score_revised",
            "score_margin",
            "retrieved_at_utc",
            "candidate_source",
        ]
        for column in copy_columns:
            row[column] = np.nan if selected is None else selected.get(column, np.nan)
        row["legacy_PlaceID"] = np.nan if old is None else old.get("PlaceID", np.nan)
        row["legacy_match_status"] = (
            "missing_new_park" if old is None else old.get("match_status", np.nan)
        )
        rows.append(row)

    selected = pd.DataFrame(rows)
    selected["google_rating"] = pd.to_numeric(selected["google_rating"], errors="coerce")
    selected["google_review_count"] = pd.to_numeric(
        selected["google_review_count"], errors="coerce"
    )

    # A Place ID is one Google entity even when several municipal/regional
    # park components select it. Normalize dynamic rating fields to the most
    # recently retrieved observation so review-count growth between API runs
    # cannot create contradictory entity records downstream.
    selected["_retrieved_at"] = pd.to_datetime(
        selected["retrieved_at_utc"], errors="coerce", utc=True
    )
    canonical = (
        selected.loc[selected["PlaceID"].notna()]
        .sort_values(
            ["PlaceID", "_retrieved_at", "park_id"],
            na_position="first",
        )
        .drop_duplicates("PlaceID", keep="last")
        .set_index("PlaceID")
    )
    for column in ["google_rating", "google_review_count"]:
        selected.loc[selected["PlaceID"].notna(), column] = selected.loc[
            selected["PlaceID"].notna(), "PlaceID"
        ].map(canonical[column])
    selected = selected.drop(columns="_retrieved_at")
    selected["duplicate_selected_placeid_count"] = (
        selected["PlaceID"].map(selected.loc[selected["PlaceID"].notna(), "PlaceID"].value_counts())
    ).astype("Int64")
    selected["duplicate_validated_placeid_count"] = (
        selected["PlaceID"].map(
            selected.loc[
                selected["match_validated"] & selected["PlaceID"].notna(), "PlaceID"
            ].value_counts()
        )
    ).astype("Int64")
    shared_lookup = shared_entities.set_index("PlaceID")
    selected["shared_entity_decision"] = selected["PlaceID"].map(
        shared_lookup["shared_entity_decision"]
    )
    selected["shared_entity_reason"] = selected["PlaceID"].map(
        shared_lookup["shared_entity_reason"]
    )
    unreviewed_duplicates = (
        selected["match_validated"]
        & selected["duplicate_validated_placeid_count"].gt(1)
        & ~selected["shared_entity_decision"].str.casefold().eq("accept").fillna(False)
    )
    selected.loc[unreviewed_duplicates, "match_decision"] = "review"
    selected.loc[unreviewed_duplicates, "match_reason"] = (
        "selected PlaceID is shared by multiple parks without a reviewed shared-entity decision"
    )
    selected.loc[unreviewed_duplicates, "match_validated"] = False
    minimum_reviews = int(config["analysis"]["minimum_google_reviews"])
    selected["rating_eligible"] = (
        selected["match_validated"]
        & selected["google_rating"].between(1, 5)
        & selected["google_review_count"].ge(minimum_reviews)
    )
    selected["placeid_changed_vs_legacy"] = (
        selected["PlaceID"].astype("string") != selected["legacy_PlaceID"].astype("string")
    ).fillna(False)
    return selected


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}_", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_optional_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _search_query(park_name: object, municipality: object) -> str:
    name = str(park_name).strip()
    normalised = set(_normalise_text(name).split())
    recreational_words = {
        "beach",
        "garden",
        "greenbelt",
        "greenway",
        "park",
        "playground",
        "ravine",
        "reserve",
        "trail",
        "waterfront",
    }
    descriptor = name if normalised & recreational_words else f"{name} park"
    return f"{descriptor}, {str(municipality).strip()}, British Columbia, Canada"


def _api_search(
    api_key: str, query: str, latitude: float, longitude: float
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.location,places.rating,places.userRatingCount,"
            "places.googleMapsUri,places.types,places.primaryType"
        ),
    }
    payload = {
        "textQuery": query,
        "pageSize": 20,
        "regionCode": "CA",
        "languageCode": "en",
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": 1500.0,
            }
        },
    }
    response = requests.post(
        "https://places.googleapis.com/v1/places:searchText",
        headers=headers,
        json=payload,
        timeout=30,
    )
    try:
        content = response.json()
    except ValueError:
        content = {"raw_response": response.text[:500]}
    return response.status_code, content


def _api_place_details(
    api_key: str, place_id: str
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "id,displayName,formattedAddress,location,rating,userRatingCount,"
            "googleMapsUri,types,primaryType,movedPlaceId"
        ),
    }
    response = requests.get(
        f"https://places.googleapis.com/v1/places/{place_id}",
        headers=headers,
        params={"languageCode": "en", "regionCode": "CA"},
        timeout=30,
    )
    try:
        content = response.json()
    except ValueError:
        content = {"raw_response": response.text[:500]}
    return response.status_code, content


def _missing_override_plan(
    config: dict[str, Any], paths: dict[str, Path], parks: gpd.GeoDataFrame
) -> pd.DataFrame:
    _, place_ids, _ = _load_overrides(config)
    refresh = _read_optional_csv(
        paths["refresh_candidates"], dtype={"park_id": "string", "PlaceID": "string"}
    )
    existing_pairs = (
        set(zip(refresh["park_id"], refresh["PlaceID"], strict=True))
        if {"park_id", "PlaceID"} <= set(refresh.columns)
        else set()
    )
    planned = place_ids[
        [
            (str(park_id), str(place_id)) not in existing_pairs
            for park_id, place_id in zip(
                place_ids["park_id"], place_ids["override_place_id"], strict=True
            )
        ]
    ].copy()
    planned = planned.merge(
        parks[["park_id"]], on="park_id", how="inner", validate="one_to_one"
    )
    return planned.sort_values(["municipality", "park_name"]).reset_index(drop=True)


def _fetch_override_details(
    config: dict[str, Any],
    paths: dict[str, Path],
    parks: gpd.GeoDataFrame,
    max_requests: int,
    sleep_seconds: float,
) -> int:
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY is not available in this terminal. "
            "Set it locally; never paste it into a tracked file."
        )
    planned = _missing_override_plan(config, paths, parks).head(max_requests).copy()
    if planned.empty:
        print("No missing reviewed Place IDs require a details refresh.")
        return 0
    refresh = _read_optional_csv(
        paths["refresh_candidates"], dtype={"park_id": "string", "PlaceID": "string"}
    )
    details_log = _read_optional_csv(
        paths["place_details_log"],
        dtype={"park_id": "string", "requested_place_id": "string"},
    )
    completed = 0
    for row in planned.itertuples(index=False):
        place_id = str(row.override_place_id)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        print(
            f"Place Details {completed + 1:,}/{len(planned):,}: "
            f"{row.municipality} — {row.park_name}"
        )
        try:
            status, content = _api_place_details(api_key, place_id)
            error = "" if status == 200 else str(content)[:500]
        except requests.RequestException as request_error:
            status, content, error = 0, {}, str(request_error)

        if status == 200:
            location = content.get("location", {})
            candidate = pd.DataFrame(
                [
                    {
                        "park_id": str(row.park_id),
                        "park_name": row.park_name,
                        "municipality": row.municipality,
                        "candidate_rank": 0,
                        "PlaceID": content.get("id", place_id),
                        "google_name": content.get("displayName", {}).get("text"),
                        "google_address": content.get("formattedAddress"),
                        "google_rating": content.get("rating"),
                        "google_review_count": content.get("userRatingCount"),
                        "google_latitude": location.get("latitude"),
                        "google_longitude": location.get("longitude"),
                        "google_types": "|".join(content.get("types", [])),
                        "google_primary_type": content.get("primaryType"),
                        "google_maps_url": content.get("googleMapsUri"),
                        "google_search_query": f"place_details:{place_id}",
                        "api_status_code": status,
                        "retrieved_at_utc": retrieved_at,
                        "candidate_source": "place_details_override_refresh",
                    }
                ]
            )
            if not refresh.empty and {"park_id", "PlaceID"} <= set(refresh.columns):
                refresh = refresh[
                    ~(
                        refresh["park_id"].astype("string").eq(str(row.park_id))
                        & refresh["PlaceID"].astype("string").eq(place_id)
                    )
                ].copy()
            refresh = pd.concat([refresh, candidate], ignore_index=True, sort=False)
            refresh["park_id"] = refresh["park_id"].astype("string")
            refresh["PlaceID"] = refresh["PlaceID"].astype("string")
            refresh = refresh.drop_duplicates(["park_id", "PlaceID"], keep="last")

        log_row = pd.DataFrame(
            [
                {
                    "park_id": str(row.park_id),
                    "park_name": row.park_name,
                    "municipality": row.municipality,
                    "requested_place_id": place_id,
                    "returned_place_id": content.get("id"),
                    "moved_place_id": content.get("movedPlaceId"),
                    "api_status_code": status,
                    "api_complete": status == 200,
                    "error": error,
                    "retrieved_at_utc": retrieved_at,
                }
            ]
        )
        details_log = pd.concat([details_log, log_row], ignore_index=True, sort=False)
        details_log = details_log.drop_duplicates(
            ["park_id", "requested_place_id"], keep="last"
        )
        _atomic_csv(refresh, paths["refresh_candidates"])
        _atomic_csv(details_log, paths["place_details_log"])
        completed += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return completed


def _api_plan(
    selected: pd.DataFrame,
    parks: gpd.GeoDataFrame,
    api_log: pd.DataFrame,
    scope: str,
    force_api: bool,
) -> gpd.GeoDataFrame:
    if scope == "all":
        ids = set(selected["park_id"].astype("string"))
    elif scope == "new":
        ids = set(
            selected.loc[
                selected["legacy_match_status"].eq("missing_new_park"), "park_id"
            ].astype("string")
        )
    else:
        ids = set(
            selected.loc[
                selected["match_decision"].isin(["review", "no_candidate"]),
                "park_id",
            ].astype("string")
        )

    if not force_api and not api_log.empty and {"park_id", "api_complete"} <= set(api_log.columns):
        completed = set(
            api_log.loc[
                api_log["api_complete"].astype("string").str.casefold().eq("true"),
                "park_id",
            ].astype("string")
        )
        ids -= completed

    planned = parks[parks["park_id"].astype("string").isin(ids)].copy()
    planned = planned.sort_values(["municipality", "park_name"]).reset_index(drop=True)
    return planned


def _fetch_candidates(
    config: dict[str, Any],
    paths: dict[str, Path],
    selected: pd.DataFrame,
    parks: gpd.GeoDataFrame,
    scope: str,
    max_requests: int,
    force_api: bool,
    sleep_seconds: float,
) -> int:
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY is not available in this terminal. "
            "Set it locally; never paste it into a tracked file."
        )
    refresh = _read_optional_csv(
        paths["refresh_candidates"], dtype={"park_id": "string", "PlaceID": "string"}
    )
    api_log = _read_optional_csv(paths["api_log"], dtype={"park_id": "string"})
    planned = _api_plan(selected, parks, api_log, scope, force_api)
    planned = planned.head(max_requests).copy()
    if planned.empty:
        print("No API requests are needed for the selected scope.")
        return 0

    planned_wgs = planned.to_crs("EPSG:4326")
    representatives = planned_wgs.geometry.representative_point()
    successful = 0
    for request_number, (index, park) in enumerate(planned_wgs.iterrows(), start=1):
        latitude = float(representatives.loc[index].y)
        longitude = float(representatives.loc[index].x)
        query = _search_query(park["park_name"], park["municipality"])
        retrieved_at = datetime.now(timezone.utc).isoformat()
        print(
            f"API {request_number:,}/{len(planned):,}: "
            f"{park['municipality']} — {park['park_name']}"
        )
        try:
            status, content = _api_search(
                api_key, query, latitude=latitude, longitude=longitude
            )
            places = content.get("places", []) if isinstance(content, dict) else []
            error = "" if status == 200 else str(content)[:500]
        except requests.RequestException as request_error:
            status, places, error = 0, [], str(request_error)

        new_candidates = []
        for rank, place in enumerate(places, start=1):
            location = place.get("location", {})
            new_candidates.append(
                {
                    "park_id": str(park["park_id"]),
                    "park_name": park["park_name"],
                    "municipality": park["municipality"],
                    "candidate_rank": rank,
                    "PlaceID": place.get("id"),
                    "google_name": place.get("displayName", {}).get("text"),
                    "google_address": place.get("formattedAddress"),
                    "google_rating": place.get("rating"),
                    "google_review_count": place.get("userRatingCount"),
                    "google_latitude": location.get("latitude"),
                    "google_longitude": location.get("longitude"),
                    "google_types": "|".join(place.get("types", [])),
                    "google_primary_type": place.get("primaryType"),
                    "google_maps_url": place.get("googleMapsUri"),
                    "google_search_query": query,
                    "api_status_code": status,
                    "retrieved_at_utc": retrieved_at,
                    "candidate_source": "places_api_refresh",
                }
            )
        if status == 200 and not refresh.empty and "park_id" in refresh.columns:
            refresh = refresh[
                ~refresh["park_id"].astype("string").eq(str(park["park_id"]))
            ].copy()
        if new_candidates:
            refresh = pd.concat(
                [refresh, pd.DataFrame(new_candidates)], ignore_index=True
            )
            refresh["park_id"] = refresh["park_id"].astype("string")
            refresh["PlaceID"] = refresh["PlaceID"].astype("string")
            refresh = refresh.drop_duplicates(["park_id", "PlaceID"], keep="last")

        log_row = pd.DataFrame(
            [
                {
                    "park_id": str(park["park_id"]),
                    "park_name": park["park_name"],
                    "municipality": park["municipality"],
                    "query": query,
                    "api_status_code": status,
                    "candidate_count": len(places),
                    "api_complete": status == 200,
                    "error": error,
                    "retrieved_at_utc": retrieved_at,
                }
            ]
        )
        api_log = pd.concat([api_log, log_row], ignore_index=True)
        api_log["park_id"] = api_log["park_id"].astype("string")
        api_log = api_log.drop_duplicates("park_id", keep="last")
        _atomic_csv(refresh, paths["refresh_candidates"])
        _atomic_csv(api_log, paths["api_log"])
        if status == 200:
            successful += 1
        elif "API_KEY_INVALID" in error or "API key not valid" in error:
            raise ValueError(
                "Google rejected GOOGLE_MAPS_API_KEY as invalid. "
                "Copy the active key again and retry; the key is never stored by the workflow."
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return successful


def run(area: str, candidate_source: str = "combined") -> dict[str, Any]:
    config = load_config(area)
    if config["network"]["slug"] != "metro":
        raise ValueError("Google rating preparation must use the shared metro configuration")
    paths = google_paths(config)
    parks = gpd.read_file(paths["parks"], layer="parks_cleaned")
    source_candidates = _read_csv(
        paths["source_candidates"], REQUIRED_CANDIDATE_COLUMNS, dtype={"park_id": "string", "PlaceID": "string"}
    )
    refresh_candidates = _read_optional_csv(
        paths["refresh_candidates"], dtype={"park_id": "string", "PlaceID": "string"}
    )
    if candidate_source == "fresh":
        candidates = refresh_candidates.copy()
    elif candidate_source == "combined":
        candidates = pd.concat(
            [source_candidates, refresh_candidates], ignore_index=True, sort=False
        )
    else:
        raise ValueError(f"Unsupported candidate source: {candidate_source}")
    if candidates.empty:
        candidates = pd.DataFrame(columns=sorted(REQUIRED_CANDIDATE_COLUMNS))
    candidates = candidates.drop_duplicates(["park_id", "PlaceID"], keep="last")
    old_ratings = _read_csv(
        paths["source_ratings"],
        {"park_id", "PlaceID", "match_status"},
        dtype={"park_id": "string", "PlaceID": "string"},
    )
    if parks["park_id"].astype("string").duplicated().any():
        raise ValueError("Final park IDs are not unique")
    if old_ratings["park_id"].duplicated().any():
        raise ValueError("Legacy Google ratings contain duplicate park IDs")

    scored = _score_candidates(parks, candidates)
    selected = _select_matches(
        config,
        parks,
        scored,
        old_ratings,
        allow_legacy_override_fallback=candidate_source == "combined",
    )
    api_log = _read_optional_csv(paths["api_log"], dtype={"park_id": "string"})
    if not api_log.empty and {"park_id", "api_complete"} <= set(api_log.columns):
        fresh_complete_ids = set(
            api_log.loc[
                api_log["api_complete"].astype("string").str.casefold().eq("true"),
                "park_id",
            ].astype("string")
        )
    else:
        fresh_complete_ids = set()
    review = selected[selected["match_decision"].isin(["review", "no_candidate"])].copy()
    validated_placeid_counts = selected.loc[
        selected["match_validated"] & selected["PlaceID"].notna(), "PlaceID"
    ].value_counts()
    eligible_placeid_counts = selected.loc[
        selected["rating_eligible"] & selected["PlaceID"].notna(), "PlaceID"
    ].value_counts()
    comparison = selected[
        [
            "park_id",
            "park_name",
            "municipality",
            "legacy_PlaceID",
            "PlaceID",
            "legacy_match_status",
            "match_decision",
            "match_reason",
            "placeid_changed_vs_legacy",
            "rating_eligible",
        ]
    ].copy()
    summary = {
        "candidate_source_mode": candidate_source,
        "park_rows": len(selected),
        "cached_candidate_rows_used": len(scored),
        "cached_parks_with_candidates": scored["park_id"].nunique(),
        "auto_accepted": int(selected["match_decision"].eq("auto_accept").sum()),
        "manually_accepted": int(selected["match_decision"].eq("manual_accept").sum()),
        "manually_rejected": int(selected["match_decision"].eq("manual_reject").sum()),
        "needs_review": int(selected["match_decision"].eq("review").sum()),
        "no_cached_candidate": int(selected["match_decision"].eq("no_candidate").sum()),
        "rating_eligible": int(selected["rating_eligible"].sum()),
        "placeids_changed_vs_legacy": int(selected["placeid_changed_vs_legacy"].sum()),
        "selected_duplicate_placeid_groups": int(
            selected.loc[selected["duplicate_selected_placeid_count"].gt(1), "PlaceID"].nunique()
        ),
        "validated_duplicate_placeid_groups": int(validated_placeid_counts.gt(1).sum()),
        "rating_eligible_duplicate_placeid_groups": int(
            eligible_placeid_counts.gt(1).sum()
        ),
        "unreviewed_validated_duplicate_placeid_groups": int(
            selected.loc[
                selected["match_validated"]
                & selected["duplicate_validated_placeid_count"].gt(1)
                & ~selected["shared_entity_decision"]
                .str.casefold()
                .eq("accept")
                .fillna(False),
                "PlaceID",
            ].nunique()
        ),
        "fresh_api_complete_parks": len(
            set(parks["park_id"].astype("string")) & fresh_complete_ids
        ),
        "fresh_api_incomplete_parks": len(
            set(parks["park_id"].astype("string")) - fresh_complete_ids
        ),
        **{f"{key}_path": str(value) for key, value in paths.items()},
    }
    _atomic_csv(scored, paths["candidate_scores"])
    _atomic_csv(selected, paths["validated_ratings"])
    _atomic_csv(review, paths["review_queue"])
    _atomic_csv(comparison, paths["comparison"])
    _atomic_csv(pd.DataFrame([summary]), paths["summary"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", default="metro")
    parser.add_argument(
        "--candidate-source",
        choices=["combined", "fresh"],
        default="combined",
        help=(
            "Use legacy plus fresh candidates, or only candidates retrieved "
            "by this reproducible Places API workflow"
        ),
    )
    parser.add_argument(
        "--api-scope",
        choices=["none", "new", "unresolved", "all"],
        default="none",
        help="Plan or execute a targeted Places API refresh after offline rescoring",
    )
    parser.add_argument(
        "--execute-api",
        action="store_true",
        help="Actually make billable API requests; otherwise only print the plan",
    )
    parser.add_argument(
        "--refresh-missing-overrides",
        action="store_true",
        help="Refresh reviewed Place IDs omitted from this park's Text Search candidates",
    )
    parser.add_argument("--max-api-requests", type=int)
    parser.add_argument("--force-api", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    try:
        summary = run(args.area, args.candidate_source)
    except (FileNotFoundError, ValueError) as error:
        print(f"Google rating preparation failed: {error}")
        return 1
    print("\nInitial Google candidate rescoring completed.")
    for key in (
        "park_rows",
        "cached_candidate_rows_used",
        "cached_parks_with_candidates",
        "auto_accepted",
        "manually_accepted",
        "manually_rejected",
        "needs_review",
        "no_cached_candidate",
        "rating_eligible",
        "placeids_changed_vs_legacy",
        "fresh_api_complete_parks",
        "fresh_api_incomplete_parks",
    ):
        print(f"{key}: {summary[key]:,}")
    print(f"Summary: {summary['summary_path']}")

    if args.api_scope != "none":
        config = load_config(args.area)
        paths = google_paths(config)
        selected = pd.read_csv(
            paths["validated_ratings"], dtype={"park_id": "string", "PlaceID": "string"}
        )
        parks = gpd.read_file(paths["parks"], layer="parks_cleaned")
        api_log = _read_optional_csv(paths["api_log"], dtype={"park_id": "string"})
        planned = _api_plan(selected, parks, api_log, args.api_scope, args.force_api)
        print(f"Planned API requests for scope '{args.api_scope}': {len(planned):,}")
        if not args.execute_api:
            print("Planning only: no API requests were made.")
            return 0
        if args.max_api_requests is None or args.max_api_requests <= 0:
            print("API execution requires a positive --max-api-requests safety cap.")
            return 2
        try:
            completed = _fetch_candidates(
                config,
                paths,
                selected,
                parks,
                args.api_scope,
                args.max_api_requests,
                args.force_api,
                args.sleep,
            )
            print(f"Successful API requests: {completed:,}")
            if completed:
                refreshed = run(args.area, args.candidate_source)
                print(f"Rating-eligible parks after refresh: {refreshed['rating_eligible']:,}")
        except (FileNotFoundError, ValueError) as error:
            print(f"Google API refresh failed: {error}")
            return 1
    if args.refresh_missing_overrides:
        config = load_config(args.area)
        paths = google_paths(config)
        parks = gpd.read_file(paths["parks"], layer="parks_cleaned")
        planned = _missing_override_plan(config, paths, parks)
        print(f"Planned Place Details requests for missing overrides: {len(planned):,}")
        if not args.execute_api:
            print("Planning only: no Place Details requests were made.")
            return 0
        if args.max_api_requests is None or args.max_api_requests <= 0:
            print("API execution requires a positive --max-api-requests safety cap.")
            return 2
        try:
            completed = _fetch_override_details(
                config, paths, parks, args.max_api_requests, args.sleep
            )
            print(f"Completed Place Details requests: {completed:,}")
            if completed:
                refreshed = run(args.area, args.candidate_source)
                print(
                    "Rating-eligible parks after Place Details refresh: "
                    f"{refreshed['rating_eligible']:,}"
                )
        except (FileNotFoundError, ValueError) as error:
            print(f"Google Place Details refresh failed: {error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
