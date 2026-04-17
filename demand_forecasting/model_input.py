from __future__ import annotations

from pathlib import Path
import importlib
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd

from demand_forecasting.feature_rules import (
    ap_bucket_label,
    build_geo_grid_series,
    duration_bucket_label,
    rating_bucket_label,
)


class DemandScoreGenerator:
    """
    Demand score generator backed by four GluonTS checkpoints.

    The final request score is:
        p_reuse = score_ap * score_location * score_duration * score_rating_loc

    Scores are derived from each checkpoint's one-step demand forecast and
    squashed to (0, 1) by x / (1 + x).
    """

    def __init__(self, model_path: Optional[str] = None, checkpoint_root: Optional[str] = None) -> None:
        # Keep model_path for backward compatibility; this implementation ignores XGBoost.
        _ = model_path
        root = Path(checkpoint_root) if checkpoint_root else Path(__file__).resolve().parent
        self.checkpoint_dirs: Dict[str, Path] = {
            "AP_bucket": root / "checkpoint_model_AP",
            "geo_grid_auto": root / "checkpoint_model_Location",
            "stay_bucket": root / "checkpoint_model_duration",
            "rating_loc": root / "checkpoint_model_rating_loc",
        }

    def generate_demand_scores(
        self,
        processed_df: pd.DataFrame,
        source_df: pd.DataFrame,
        num_samples: Optional[int] = None,
        output_parquet: Optional[str] = None,
    ) -> pd.DataFrame:
        _ = num_samples

        required_processed = {"timestamp_hour", "demand", "AP_bucket", "geo_grid_auto", "stay_bucket", "rating_loc"}
        missing_processed = required_processed - set(processed_df.columns)
        if missing_processed:
            raise ValueError(f"processed_df missing required columns: {sorted(missing_processed)}")

        required_source = {"cache_key", "lead_time", "duration", "location_latitude", "location_longitude", "sabre_rating"}
        missing_source = required_source - set(source_df.columns)
        if missing_source:
            raise ValueError(f"source_df missing required columns: {sorted(missing_source)}")

        predictors = self._load_predictors()

        ap_scores = self._score_by_category(processed_df, "AP_bucket", predictors["AP_bucket"])
        loc_scores = self._score_by_category(processed_df, "geo_grid_auto", predictors["geo_grid_auto"])
        dur_scores = self._score_by_category(processed_df, "stay_bucket", predictors["stay_bucket"])
        rating_loc_scores = self._score_by_category(processed_df, "rating_loc", predictors["rating_loc"])

        request_frame = self._build_request_feature_frame(source_df)

        ap_default = self._default_score(ap_scores)
        loc_default = self._default_score(loc_scores)
        dur_default = self._default_score(dur_scores)
        rating_loc_default = self._default_score(rating_loc_scores)

        request_frame["score_ap"] = request_frame["AP_bucket"].map(ap_scores).fillna(ap_default)
        request_frame["score_location"] = request_frame["geo_grid_auto"].map(loc_scores).fillna(loc_default)
        request_frame["score_duration"] = request_frame["stay_bucket"].map(dur_scores).fillna(dur_default)
        request_frame["score_rating_loc"] = request_frame["rating_loc"].map(rating_loc_scores).fillna(rating_loc_default)

        request_frame["p_reuse"] = (
            request_frame["score_ap"]
            * request_frame["score_location"]
            * request_frame["score_duration"]
            * request_frame["score_rating_loc"]
        ).clip(lower=0.0, upper=1.0)

        result_cols = ["cache_key", "p_reuse", "score_ap", "score_location", "score_duration", "score_rating_loc"]
        p_reuse_df = request_frame[result_cols].copy()

        if output_parquet:
            Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
            p_reuse_df.to_parquet(output_parquet, index=False)

        return p_reuse_df

    def _load_predictors(self) -> Dict[str, object]:
        try:
            predictor_mod = importlib.import_module("gluonts.model.predictor")
            Predictor = getattr(predictor_mod, "Predictor")
        except Exception as exc:
            raise ImportError("gluonts is required to load checkpoint models") from exc

        predictors: Dict[str, object] = {}
        for feature_name, checkpoint_dir in self.checkpoint_dirs.items():
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_dir}")
            predictors[feature_name] = Predictor.deserialize(checkpoint_dir)
        return predictors

    def _score_by_category(self, processed_df: pd.DataFrame, category_col: str, predictor: object) -> Dict[str, float]:
        category_scores: Dict[str, float] = {}

        grouped = (
            processed_df[["timestamp_hour", category_col, "demand"]]
            .dropna(subset=["timestamp_hour", category_col])
            .groupby([category_col, "timestamp_hour"], as_index=False)["demand"]
            .sum()
        )

        for category, group in grouped.groupby(category_col):
            score = self._forecast_score(group, predictor)
            category_scores[str(category)] = score

        return category_scores

    def _forecast_score(self, group: pd.DataFrame, predictor: object) -> float:
        try:
            dataset_mod = importlib.import_module("gluonts.dataset.common")
            ListDataset = getattr(dataset_mod, "ListDataset")
        except Exception as exc:
            raise ImportError("gluonts is required to build prediction datasets") from exc

        ts = group.sort_values("timestamp_hour").set_index("timestamp_hour")["demand"].astype(float)
        full_index = pd.date_range(ts.index.min(), ts.index.max(), freq="h", tz="UTC")
        ts = ts.reindex(full_index).fillna(0.0)

        if len(ts) < 2:
            return float(ts.mean() / (1.0 + ts.mean()))

        start_ts = ts.index[0]
        # Convert to tz-naive before Period conversion to avoid timezone-drop warning.
        if getattr(start_ts, "tzinfo", None) is not None:
            start_ts = start_ts.tz_localize(None)

        dataset = ListDataset([
            {
                "start": start_ts.to_period("h"),
                "target": ts.to_numpy(dtype=np.float32),
            }
        ], freq="h")

        try:
            # Suppress a known upstream warning in gluonts/torch indexing behavior.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Using a non-tuple sequence for multidimensional indexing is deprecated.*",
                    category=UserWarning,
                )
                forecast = next(predictor.predict(dataset))
            point = float(np.mean(forecast.mean))
        except Exception:
            point = float(ts.mean())

        point = max(point, 0.0)
        return float(point / (1.0 + point))

    def _build_request_feature_frame(self, source_df: pd.DataFrame) -> pd.DataFrame:
        frame = source_df.copy()

        frame["AP_bucket"] = frame["lead_time"].apply(ap_bucket_label)
        frame["stay_bucket"] = frame["duration"].apply(duration_bucket_label)
        frame["rating_bucket"] = frame["sabre_rating"].apply(rating_bucket_label)
        frame["geo_grid_auto"] = build_geo_grid_series(frame["location_latitude"], frame["location_longitude"])
        frame["rating_loc"] = frame["rating_bucket"].astype(str) + "_|_" + frame["geo_grid_auto"].astype(str)

        frame = frame[frame["AP_bucket"] != "Exclude"]
        frame = frame[frame["stay_bucket"] != "Exclude"]
        frame = frame[frame["rating_bucket"] != "Exclude"]

        return frame

    @staticmethod
    def _default_score(score_map: Dict[str, float]) -> float:
        if not score_map:
            return 0.5
        return float(np.mean(list(score_map.values())))
