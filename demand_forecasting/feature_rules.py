from __future__ import annotations

import pandas as pd


def build_geo_grid_series(lat: pd.Series, lon: pd.Series, precision: int = 2) -> pd.Series:
    lat_num = pd.to_numeric(lat, errors="coerce").round(precision)
    lon_num = pd.to_numeric(lon, errors="coerce").round(precision)

    lat_str = lat_num.astype("string").fillna("NA")
    lon_str = lon_num.astype("string").fillna("NA")
    grid = ("Grid_" + lat_str + "_" + lon_str).astype("string")

    missing_mask = lat_num.isna() | lon_num.isna()
    grid = grid.where(~missing_mask, "Missing_Grid")
    return grid.astype(str)


def assign_geo_grid(
    df: pd.DataFrame,
    precision: int = 2,
    lat_col: str = "location_latitude",
    lon_col: str = "location_longitude",
    out_col: str = "geo_grid_auto",
) -> pd.DataFrame:
    frame = df.copy()
    if lat_col in frame.columns and lon_col in frame.columns:
        frame[out_col] = build_geo_grid_series(frame[lat_col], frame[lon_col], precision=precision)
    else:
        frame[out_col] = "Missing_Grid"
    frame[out_col] = frame[out_col].fillna("Missing_Grid")
    return frame


def ap_bucket_label(ap: object) -> str:
    if pd.isna(ap):
        return "Exclude"
    try:
        ap_val = int(float(ap))
    except Exception:
        return "Exclude"
    if ap_val < 0:
        return "Exclude"
    if ap_val <= 10:
        return f"AP_{ap_val:02d}"
    for high in range(15, 51, 5):
        if ap_val <= high:
            return f"AP_{high-4:02d}-{high:02d}"
    return "AP_>50"


def duration_bucket_label(duration: object) -> str:
    if pd.isna(duration):
        return "Exclude"
    try:
        d = int(float(duration))
    except Exception:
        return "Exclude"
    if d <= 0:
        return "Exclude"
    if d <= 7:
        return f"Stay_{d}"
    return "Stay_>7"


def rating_bucket_label(rating: object) -> str:
    if pd.isna(rating):
        return "Rate_NA"
    try:
        r = float(rating)
    except Exception:
        return "Rate_NA"
    if r <= 1.5:
        return "Rate_1-1.5"
    if 2.0 <= r <= 2.5:
        return "Rate_2-2.5"
    if 3.0 <= r <= 3.5:
        return "Rate_3-3.5"
    if 4.0 <= r <= 4.5:
        return "Rate_4-4.5"
    if r == 5.0:
        return "Rate_5"
    return "Exclude"
