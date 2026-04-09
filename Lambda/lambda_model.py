from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


HORIZON_BREAKS = [1, 3, 7, 30]
HORIZON_LABELS = ["same_day", "short", "mid", "long", "very_long"]
DEFAULT_TOP_N_HOTELS = 300


@dataclass
class KMSummary:
    n_intervals: int
    n_events: int
    total_exposure_hours: float
    median_hours: float
    rmst_hours: float
    lambda_km_median: float
    lambda_km_rmst: float
    lambda_empirical: float


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _assign_bucket_label(value: int) -> str:
    for idx, boundary in enumerate(HORIZON_BREAKS):
        if value <= boundary:
            return HORIZON_LABELS[idx]
    return HORIZON_LABELS[-1]


def _bucket_midpoint(label: str) -> float:
    if label == "same_day":
        return 1.0
    if label == "short":
        return 3.0
    if label == "mid":
        return 7.0
    if label == "long":
        return 30.0
    return 60.0


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not np.any(mask):
        return float("nan")
    if float(w[mask].sum()) <= 0:
        return float(np.nanmean(v[mask]))
    return float(np.average(v[mask], weights=w[mask]))


def _resolve_hotel_series(df_enriched: pd.DataFrame) -> pd.Series:
    if "hotel_code" in df_enriched.columns:
        return df_enriched["hotel_code"].astype(str)
    if "chain_code" in df_enriched.columns:
        return df_enriched["chain_code"].astype(str)
    return pd.Series(["unknown"] * len(df_enriched), index=df_enriched.index, dtype="object")


def _ensure_hotel_grouped(df_enriched: pd.DataFrame, top_n_hotels: int = DEFAULT_TOP_N_HOTELS) -> pd.DataFrame:
    out = df_enriched.copy()

    if "hotel_grouped" in out.columns:
        out["hotel_grouped"] = out["hotel_grouped"].astype(str).replace({"": "Other", "nan": "Other"}).fillna("Other")
        return out

    hotel = _resolve_hotel_series(out)
    out["hotel_code"] = hotel

    vc = hotel.value_counts(dropna=False)
    keep = set(vc.head(max(1, int(top_n_hotels))).index.astype(str).tolist())
    out["hotel_grouped"] = hotel.where(hotel.isin(keep), other="Other").astype(str)
    return out


def _ensure_horizon_bucket(df_enriched: pd.DataFrame) -> pd.DataFrame:
    out = df_enriched.copy()

    if "lead_time" not in out.columns:
        if {"rq_stay_start_date", "rq_timestamp"}.issubset(out.columns):
            stay = pd.to_datetime(out["rq_stay_start_date"], errors="coerce")
            ts = pd.to_datetime(out["rq_timestamp"], errors="coerce", utc=True)
            ts_local = ts.dt.tz_convert(None)
            out["lead_time"] = (stay - ts_local.dt.normalize()).dt.days
        else:
            out["lead_time"] = 0

    out["lead_time"] = pd.to_numeric(out["lead_time"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    if "horizon_bucket" not in out.columns:
        out["horizon_bucket"] = out["lead_time"].apply(_assign_bucket_label)
    else:
        out["horizon_bucket"] = out["horizon_bucket"].astype(str)

    return out


def _prepare_base(df_enriched: pd.DataFrame, top_n_hotels: int = DEFAULT_TOP_N_HOTELS) -> pd.DataFrame:
    out = df_enriched.copy()
    out = _ensure_hotel_grouped(out, top_n_hotels=top_n_hotels)
    out = _ensure_horizon_bucket(out)

    if "rate_source" not in out.columns:
        out["rate_source"] = "unknown"
    out["rate_source"] = out["rate_source"].astype(str).replace({"": "unknown", "nan": "unknown"}).fillna("unknown")

    if "price_change" not in out.columns:
        raise ValueError("df_enriched must include 'price_change' column.")

    if "rq_timestamp" not in out.columns:
        raise ValueError("df_enriched must include 'rq_timestamp' column.")

    out["rq_timestamp"] = pd.to_datetime(out["rq_timestamp"], errors="coerce", utc=True)
    out["price_change"] = pd.to_numeric(out["price_change"], errors="coerce").fillna(0).astype(int).clip(lower=0)

    out = out.dropna(subset=["rq_timestamp", "hotel_grouped", "rate_source", "horizon_bucket"]).copy()
    return out


def _group_level_stats(df_enriched: pd.DataFrame, top_n_hotels: int = DEFAULT_TOP_N_HOTELS) -> pd.DataFrame:
    """Build per-group interval/event/exposure statistics used by all methods."""
    frame = _prepare_base(df_enriched, top_n_hotels=top_n_hotels)
    rows = []
    group_cols = ["hotel_grouped", "rate_source", "horizon_bucket"]

    for keys, g in frame.groupby(group_cols, dropna=False):
        summary = estimate_lambda_km(g)
        rows.append(
            {
                "hotel_grouped": str(keys[0]),
                "rate_source": str(keys[1]),
                "horizon_bucket": str(keys[2]),
                "lead_time": _safe_int(pd.to_numeric(g.get("lead_time"), errors="coerce").median(), default=0),
                "n_intervals": summary.n_intervals,
                "n_events": summary.n_events,
                "exposure_hours": summary.total_exposure_hours,
                "median_hours": summary.median_hours,
                "rmst_hours": summary.rmst_hours,
                "lambda_km_median": summary.lambda_km_median,
                "lambda_km_rmst": summary.lambda_km_rmst,
                "lambda_empirical": summary.lambda_empirical,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["horizon_mid"] = out["horizon_bucket"].apply(_bucket_midpoint)
    return out


def _km_survival(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    mask = np.isfinite(durations) & (durations > 0)
    durations = durations[mask]
    events = events[mask]

    if len(durations) == 0:
        return np.array([]), np.array([])

    times = np.sort(np.unique(durations))
    survival = []
    s = 1.0

    for t in times:
        n_at_risk = np.sum(durations >= t)
        d_t = np.sum((durations == t) & (events == 1))
        if n_at_risk > 0:
            s *= (1.0 - d_t / n_at_risk)
        survival.append(s)

    return times, np.array(survival)


def _median_from_survival(times: np.ndarray, survival: np.ndarray) -> float:
    if len(times) == 0:
        return math.inf
    hit = np.where(survival <= 0.5)[0]
    if len(hit) == 0:
        return math.inf
    return float(times[hit[0]])


def _rmst_from_survival(times: np.ndarray, survival: np.ndarray) -> float:
    if len(times) == 0:
        return math.nan
    prev_t = 0.0
    prev_s = 1.0
    area = 0.0
    for t, s in zip(times, survival):
        area += prev_s * (t - prev_t)
        prev_t = float(t)
        prev_s = float(s)
    return area


def estimate_lambda_km(group_df: pd.DataFrame) -> KMSummary:
    g = group_df.sort_values("rq_timestamp").copy()
    g["delta_hours"] = g["rq_timestamp"].diff().dt.total_seconds() / 3600.0
    g["event_change"] = pd.to_numeric(g["price_change"], errors="coerce").fillna(0).astype(int)

    km_df = g.dropna(subset=["delta_hours"])
    km_df = km_df[km_df["delta_hours"] > 0]

    durations = km_df["delta_hours"].to_numpy(dtype=float)
    events = km_df["event_change"].to_numpy(dtype=int)

    times, survival = _km_survival(durations, events)
    median_hours = _median_from_survival(times, survival)
    rmst_hours = _rmst_from_survival(times, survival)

    total_exposure_hours = float(durations.sum()) if len(durations) else 0.0
    n_events = int(events.sum())
    n_intervals = int(len(durations))

    lambda_empirical = (n_events / total_exposure_hours) if total_exposure_hours > 0 else 0.0
    lambda_km_median = (math.log(2.0) / median_hours) if math.isfinite(median_hours) and median_hours > 0 else math.nan
    lambda_km_rmst = (1.0 / rmst_hours) if np.isfinite(rmst_hours) and rmst_hours > 0 else math.nan

    return KMSummary(
        n_intervals=n_intervals,
        n_events=n_events,
        total_exposure_hours=total_exposure_hours,
        median_hours=median_hours,
        rmst_hours=rmst_hours,
        lambda_km_median=lambda_km_median,
        lambda_km_rmst=lambda_km_rmst,
        lambda_empirical=lambda_empirical,
    )


def _hierarchical_fill(
    out: pd.DataFrame,
    raw_lambda_col: str,
    min_intervals: int,
    method_name: str,
) -> pd.DataFrame:
    if out.empty:
        return out

    frame = out.copy()
    frame[raw_lambda_col] = pd.to_numeric(frame[raw_lambda_col], errors="coerce")
    frame["sparse_group"] = frame["n_intervals"].fillna(0).astype(int) < int(max(1, min_intervals))

    valid = frame[(~frame["sparse_group"]) & frame[raw_lambda_col].notna() & (frame[raw_lambda_col] > 0)].copy()
    global_lambda = _weighted_mean(valid[raw_lambda_col], valid["exposure_hours"])
    if not np.isfinite(global_lambda) or global_lambda <= 0:
        fallback_base = frame[raw_lambda_col].replace([np.inf, -np.inf], np.nan).dropna()
        global_lambda = float(fallback_base.median()) if not fallback_base.empty else 0.1

    frame["lambda_final"] = frame[raw_lambda_col]
    frame["fallback_level"] = np.where(frame["sparse_group"], "pending", "exact")

    lvl_hotel_horizon = (
        valid.groupby(["hotel_grouped", "horizon_bucket"], dropna=False)[[raw_lambda_col, "exposure_hours"]]
        .apply(lambda g: _weighted_mean(g[raw_lambda_col], g["exposure_hours"]))
        .to_dict()
    )
    lvl_source_horizon = (
        valid.groupby(["rate_source", "horizon_bucket"], dropna=False)[[raw_lambda_col, "exposure_hours"]]
        .apply(lambda g: _weighted_mean(g[raw_lambda_col], g["exposure_hours"]))
        .to_dict()
    )
    lvl_horizon = (
        valid.groupby(["horizon_bucket"], dropna=False)[[raw_lambda_col, "exposure_hours"]]
        .apply(lambda g: _weighted_mean(g[raw_lambda_col], g["exposure_hours"]))
        .to_dict()
    )

    for idx, row in frame.iterrows():
        lam = row["lambda_final"]
        if (not bool(row["sparse_group"])) and pd.notna(lam) and float(lam) > 0:
            continue

        h = str(row["hotel_grouped"])
        s = str(row["rate_source"])
        b = str(row["horizon_bucket"])

        candidates = [
            (lvl_hotel_horizon.get((h, b)), "hotel_horizon"),
            (lvl_source_horizon.get((s, b)), "source_horizon"),
            (lvl_horizon.get(b), "horizon"),
            (global_lambda, "global"),
        ]

        resolved = None
        level = "global"
        for value, lvl in candidates:
            if value is None:
                continue
            if np.isfinite(value) and float(value) > 0:
                resolved = float(value)
                level = lvl
                break

        if resolved is None:
            resolved = float(global_lambda)
            level = "global"

        frame.at[idx, "lambda_final"] = resolved
        frame.at[idx, "fallback_level"] = level

    frame["lambda_final"] = pd.to_numeric(frame["lambda_final"], errors="coerce").fillna(global_lambda).clip(lower=1e-9)
    frame["lambda_method"] = method_name
    return frame


def build_lambda_table_km(
    df_enriched: pd.DataFrame,
    min_intervals: int = 5,
    top_n_hotels: int = DEFAULT_TOP_N_HOTELS,
) -> pd.DataFrame:
    """KM-based lambda with hierarchical sparse-group fallback."""
    out = _group_level_stats(df_enriched, top_n_hotels=top_n_hotels)
    if out.empty:
        return out

    out["lambda_km"] = pd.to_numeric(out["lambda_km_rmst"], errors="coerce").fillna(out["lambda_empirical"])
    out = _hierarchical_fill(out, raw_lambda_col="lambda_km", min_intervals=min_intervals, method_name="km_hier")
    return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table_poisson(
    df_enriched: pd.DataFrame,
    min_intervals: int = 5,
    top_n_hotels: int = DEFAULT_TOP_N_HOTELS,
) -> pd.DataFrame:
    """Poisson process lambda with hierarchical sparse-group fallback."""
    out = _group_level_stats(df_enriched, top_n_hotels=top_n_hotels)
    if out.empty:
        return out

    out["lambda_poisson"] = pd.to_numeric(out["lambda_empirical"], errors="coerce")
    out = _hierarchical_fill(out, raw_lambda_col="lambda_poisson", min_intervals=min_intervals, method_name="poisson_hier")
    return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table_glm(
    df_enriched: pd.DataFrame,
    min_intervals: int = 5,
    use_negative_binomial: bool = True,
    top_n_hotels: int = DEFAULT_TOP_N_HOTELS,
) -> pd.DataFrame:
    """Notebook-style GLM lambda model on grouped counts with hierarchical fallback.

    Model:
        n_events ~ horizon_mid + C(rate_source) + C(hotel_grouped)
        offset = log(exposure_hours)
    """
    out = _group_level_stats(df_enriched, top_n_hotels=top_n_hotels)
    if out.empty:
        return out

    out["log_exposure"] = np.log(pd.to_numeric(out["exposure_hours"], errors="coerce").clip(lower=1e-9))
    out["n_events"] = pd.to_numeric(out["n_events"], errors="coerce").fillna(0).astype(int)

    train = out[(out["n_intervals"] >= int(max(1, min_intervals))) & (out["exposure_hours"] > 0)].copy()
    if train.empty:
        fb = out.copy()
        fb["lambda_glm"] = fb["lambda_empirical"]
        fb = _hierarchical_fill(fb, raw_lambda_col="lambda_glm", min_intervals=max(1, min_intervals), method_name="glm_fallback_hier")
        fb["lambda_debug_reason"] = "No train groups passed min_intervals/exposure filters"
        return fb.sort_values("n_intervals", ascending=False).reset_index(drop=True)

    total_exposure = float(train["exposure_hours"].sum())
    total_events = float(train["n_events"].sum())
    global_lambda = (total_events / total_exposure) if total_exposure > 0 else 0.1

    try:
        import importlib

        smf = importlib.import_module("statsmodels.formula.api")
        sm = importlib.import_module("statsmodels.api")

        family = sm.families.NegativeBinomial() if bool(use_negative_binomial) else sm.families.Poisson()

        model = smf.glm(
            formula="n_events ~ horizon_mid + C(rate_source) + C(hotel_grouped)",
            data=train,
            family=family,
            offset=train["log_exposure"],
        ).fit()

        pred = out.copy()
        pred_events = model.predict(pred, offset=pred["log_exposure"])
        pred["lambda_glm"] = pd.to_numeric(pred_events, errors="coerce") / pred["exposure_hours"].clip(lower=1e-9)
        pred["lambda_glm"] = pred["lambda_glm"].replace([np.inf, -np.inf], np.nan).fillna(global_lambda).clip(lower=1e-9)

        pred = _hierarchical_fill(
            pred,
            raw_lambda_col="lambda_glm",
            min_intervals=max(1, min_intervals),
            method_name="glm_nb_hier" if use_negative_binomial else "glm_poisson_hier",
        )
        pred["lambda_debug_reason"] = ""
        return pred.sort_values("n_intervals", ascending=False).reset_index(drop=True)

    except Exception as exc:
        fallback = out.copy()
        fallback["lambda_glm"] = fallback["lambda_empirical"].replace([np.inf, -np.inf], np.nan).fillna(global_lambda)
        fallback = _hierarchical_fill(
            fallback,
            raw_lambda_col="lambda_glm",
            min_intervals=max(1, min_intervals),
            method_name="glm_fallback_hier",
        )
        fallback["lambda_debug_reason"] = f"{type(exc).__name__}: {exc}"
        return fallback.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table_fallback(
    df_enriched: pd.DataFrame,
    min_intervals: int = 1,
    top_n_hotels: int = DEFAULT_TOP_N_HOTELS,
) -> pd.DataFrame:
    """Robust empirical lambda with hierarchical fallback."""
    out = _group_level_stats(df_enriched, top_n_hotels=top_n_hotels)
    if out.empty:
        return out

    out["lambda_fallback"] = out["lambda_empirical"]
    out = _hierarchical_fill(out, raw_lambda_col="lambda_fallback", min_intervals=max(1, min_intervals), method_name="fallback_hier")
    out["lambda_debug_reason"] = ""
    return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table(
    df_enriched: pd.DataFrame,
    min_intervals: int = 5,
    method: str = "km",
    top_n_hotels: int = DEFAULT_TOP_N_HOTELS,
) -> pd.DataFrame:
    """Unified lambda table builder for multiple estimation methods.

    Args:
        df_enriched: Enriched dataframe with rq_timestamp and price_change.
        min_intervals: Minimum intervals required before a group can use its own estimate.
        method: One of ["km", "poisson", "glm", "fallback"].
        top_n_hotels: Number of most frequent hotels to keep before mapping others to "Other".
    """
    method_norm = str(method).strip().lower()
    if method_norm == "km":
        return build_lambda_table_km(df_enriched=df_enriched, min_intervals=min_intervals, top_n_hotels=top_n_hotels)
    if method_norm == "poisson":
        return build_lambda_table_poisson(df_enriched=df_enriched, min_intervals=min_intervals, top_n_hotels=top_n_hotels)
    if method_norm == "glm":
        return build_lambda_table_glm(df_enriched=df_enriched, min_intervals=min_intervals, top_n_hotels=top_n_hotels)
    if method_norm == "fallback":
        return build_lambda_table_fallback(df_enriched=df_enriched, min_intervals=max(1, min_intervals), top_n_hotels=top_n_hotels)
    raise ValueError(f"Unsupported method='{method}'. Use one of: km, poisson, glm, fallback.")
