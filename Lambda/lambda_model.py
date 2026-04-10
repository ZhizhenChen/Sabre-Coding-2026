from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

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


def _resolve_group_key(df_enriched: pd.DataFrame) -> str:
	"""Use hotel-level grouping for lambda estimation."""
	if "hotel_code" in df_enriched.columns:
		return "hotel_code"
	raise ValueError("df_enriched must include 'hotel_code'.")


def _group_level_stats(df_enriched: pd.DataFrame) -> pd.DataFrame:
	"""Build per-group interval/event/exposure statistics used by all methods."""
	rows = []
	group_key = _resolve_group_key(df_enriched)
	group_cols = [group_key, "lead_time", "rate_source"]

	for keys, g in df_enriched.groupby(group_cols, dropna=False):
		summary = estimate_lambda_km(g)
		rows.append(
			{
				"hotel_code": str(keys[0]),
				"lead_time": _safe_int(keys[1]),
				"rate_source": str(keys[2]),
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

	return pd.DataFrame(rows)


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
	#delta_hours = time between consecutive requests in hours
	g["delta_hours"] = g["rq_timestamp"].diff().dt.total_seconds() / 3600.0
	g["event_change"] = g["price_change"].astype(int)

	# First row has no preceding interval.
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


def build_lambda_table_km(df_enriched: pd.DataFrame, min_intervals: int = 5) -> pd.DataFrame:
	"""Kaplan-Meier based lambda estimation (existing/default behavior)."""
	out = _group_level_stats(df_enriched)
	if out.empty:
		return out

	out = out[out["n_intervals"] >= min_intervals].copy()
	if out.empty:
		return out

	out["lambda_final"] = out["lambda_km_rmst"].fillna(out["lambda_empirical"])
	out["lambda_method"] = "km"
	return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table_poisson(df_enriched: pd.DataFrame, min_intervals: int = 5) -> pd.DataFrame:
	"""Poisson process rate estimation: lambda = total_changes / total_exposure_hours."""
	out = _group_level_stats(df_enriched) 
	if out.empty:
		return out

	out = out[out["n_intervals"] >= min_intervals].copy()
	if out.empty:
		return out

	out["lambda_poisson"] = out["lambda_empirical"]
	out["lambda_final"] = out["lambda_poisson"].fillna(0.0)
	out["lambda_method"] = "poisson"
	return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table_glm(
	df_enriched: pd.DataFrame,
	min_intervals: int = 5,
	use_negative_binomial: bool = True,
) -> pd.DataFrame:
	"""GLM-based lambda estimation using event counts with exposure offset.

	Model:
		n_events ~ lead_time + C(rate_source) + offset(log(exposure_hours))
	"""
	out = _group_level_stats(df_enriched)
	if out.empty:
		return out

	train = out[(out["n_intervals"] >= min_intervals) & (out["exposure_hours"] > 0)].copy()
	if train.empty:
		return pd.DataFrame()

	# Global fallback lambda for unseen/sparse groups.
	total_exposure = float(train["exposure_hours"].sum())
	total_events = float(train["n_events"].sum())
	global_lambda = (total_events / total_exposure) if total_exposure > 0 else 0.1
	fallback_reason = ""

	try:
		import importlib
		smf = importlib.import_module("statsmodels.formula.api")
		sm = importlib.import_module("statsmodels.api")

		train["log_exposure"] = np.log(train["exposure_hours"].clip(lower=1e-9))
		formula = "n_events ~ lead_time + C(rate_source)"

		if use_negative_binomial:
			family = sm.families.NegativeBinomial()
		else:
			family = sm.families.Poisson()

		model = smf.glm(
			formula=formula,
			data=train,
			family=family,
			offset=train["log_exposure"],
		).fit()

		pred = out.copy()
		pred_exposure = pred["exposure_hours"].clip(lower=1e-9)
		predicted_events = model.predict(pred, offset=np.log(pred_exposure))
		pred["lambda_glm"] = predicted_events / pred_exposure
		pred["lambda_glm"] = pred["lambda_glm"].replace([np.inf, -np.inf], np.nan).fillna(global_lambda)
		pred["lambda_glm"] = pred["lambda_glm"].clip(lower=0.0)

		pred["lambda_final"] = pred["lambda_glm"]
		pred["lambda_method"] = "glm_nb" if use_negative_binomial else "glm_poisson"
		pred["lambda_debug_reason"] = ""
		return pred.sort_values("n_intervals", ascending=False).reset_index(drop=True)

	except Exception as exc:
		# Safe fallback if statsmodels is unavailable or model fitting fails.
		fallback = out.copy()
		fallback["lambda_final"] = fallback["lambda_empirical"].replace(0.0, np.nan).fillna(global_lambda)
		fallback["lambda_method"] = "glm_fallback_empirical"
		fallback_reason = f"{type(exc).__name__}: {exc}"
		fallback["lambda_debug_reason"] = fallback_reason
		return fallback.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table_fallback(df_enriched: pd.DataFrame, min_intervals: int = 1) -> pd.DataFrame:
	"""Robust fallback: empirical per-group lambda with global fill, minimal filtering."""
	out = _group_level_stats(df_enriched)
	if out.empty:
		return out

	out = out[out["n_intervals"] >= min_intervals].copy()
	if out.empty:
		return out

	total_exposure = float(out["exposure_hours"].sum())
	total_events = float(out["n_events"].sum())
	global_lambda = (total_events / total_exposure) if total_exposure > 0 else 0.1

	out["lambda_fallback"] = out["lambda_empirical"].replace(0.0, np.nan).fillna(global_lambda)
	out["lambda_final"] = out["lambda_fallback"]
	out["lambda_method"] = "fallback"
	out["lambda_debug_reason"] = ""
	return out.sort_values("n_intervals", ascending=False).reset_index(drop=True)


def build_lambda_table(
	df_enriched: pd.DataFrame,
	min_intervals: int = 5,
	method: str = "km",
) -> pd.DataFrame:
	"""Unified lambda table builder for comparing multiple estimation methods.

	Args:
		df_enriched: Enriched dataframe with rq_timestamp, hotel_code, and price_change.
		min_intervals: Minimum intervals required per group.
		method: One of ["km", "poisson", "glm", "fallback"].
	"""
	method_norm = str(method).strip().lower()
	if method_norm == "km":
		return build_lambda_table_km(df_enriched=df_enriched, min_intervals=min_intervals)
	if method_norm == "poisson":
		return build_lambda_table_poisson(df_enriched=df_enriched, min_intervals=min_intervals)
	if method_norm == "glm":
		return build_lambda_table_glm(df_enriched=df_enriched, min_intervals=min_intervals)
	if method_norm == "fallback":
		return build_lambda_table_fallback(df_enriched=df_enriched, min_intervals=max(1, min_intervals))
	raise ValueError(f"Unsupported method='{method}'. Use one of: km, poisson, glm, fallback.")
