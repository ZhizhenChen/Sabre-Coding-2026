from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

XGBOOST_AVAILABLE = False
xgb = None
_XGBOOST_IMPORT_ATTEMPTED = False


def _lazy_import_xgboost() -> bool:
    """Lazily import xgboost on first use (on-demand, not at module load)."""
    global XGBOOST_AVAILABLE, xgb, _XGBOOST_IMPORT_ATTEMPTED
    
    if _XGBOOST_IMPORT_ATTEMPTED:
        return XGBOOST_AVAILABLE
    
    _XGBOOST_IMPORT_ATTEMPTED = True
    try:
        import xgboost as xgb_temp
        xgb = xgb_temp
        XGBOOST_AVAILABLE = True
        print("✓ XGBoost successfully imported")
        return True
    except Exception as e:
        print(f"⚠ XGBoost import failed: {e}")
        print("Falling back to heuristic scoring...\n")
        return False


class DemandScoreGenerator:
    """Load XGBoost model, predict next-hour count, then map count to demand probability."""

    MODEL_FEATURES = [
        "lag_1h",
        "lag_2h",
        "past_3h",
        "past_6h",
        "request_time_gap",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "lead_time",
        "leadtime_x_recent",
        "recent_and_close",
        "lead_time_bucket",
    ]

    LEAD_TIME_BINS = [-1, 1, 3, 7, 30, 365]
    LEAD_TIME_LABELS = ["same_day", "short", "mid", "long", "very_long"]
    REQUIRED_BASE_COLUMNS = ["cache_key", "timestamp_hour", "request_count"]

    def __init__(self, model_path: str = "xgb_model.json") -> None:
        """Initialize the demand forecaster model.

        Args:
            model_path: Path to the XGBoost model JSON file.
        """
        self.model_path = model_path
        self.model = None
        self.model_features = None
        
        # Try lazy-loaded xgboost
        if _lazy_import_xgboost():
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                print(f"✓ XGBoost model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load XGBoost model: {e}")
                print("Using fallback scoring method.")
                self._load_model_metadata(model_path)
        else:
            self._load_model_metadata(model_path)
    
    def _load_model_metadata(self, model_path: str) -> None:
        """Load model feature names and metadata from JSON for fallback inference."""
        try:
            with open(model_path, 'r') as f:
                model_json = json.load(f)
                if 'learner' in model_json:
                    features = model_json['learner'].get('feature_names', [])
                    self.model_features = features
                    print(f"✓ Loaded model metadata with {len(features)} features")
        except Exception as e:
            print(f"Warning: Could not load model metadata: {e}")

    def _build_hourly_feature_frame(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a pre-built hourly feature frame from data pipeline."""
        frame = processed_df.copy()
        required = self.REQUIRED_BASE_COLUMNS + self.MODEL_FEATURES
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                "Input dataframe is missing required prepared columns. "
                f"Move preprocessing to data_pipeline.ipynb. Missing: {missing}"
            )

        if not pd.api.types.is_datetime64_any_dtype(frame["timestamp_hour"]):
            raise ValueError(
                "Column 'timestamp_hour' must be datetime dtype prepared in data_pipeline.ipynb."
            )

        numeric_features = [c for c in self.MODEL_FEATURES if c != "lead_time_bucket"]
        non_numeric = [c for c in numeric_features if not pd.api.types.is_numeric_dtype(frame[c])]
        if non_numeric:
            raise ValueError(
                "Numeric model features must be precomputed as numeric dtypes in data_pipeline.ipynb. "
                f"Non-numeric columns: {non_numeric}"
            )

        if not pd.api.types.is_categorical_dtype(frame["lead_time_bucket"]):
            raise ValueError(
                "Column 'lead_time_bucket' must be categorical dtype prepared in data_pipeline.ipynb."
            )

        return frame

    def extract_features(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Select model features from a prepared hourly dataframe."""
        feature_frame = self._build_hourly_feature_frame(processed_df)
        return feature_frame[self.MODEL_FEATURES].copy()

    @staticmethod
    def _compute_lead_time_weight(lead_time: int) -> float:
        """Compute exponential decay weight for lead_time (from demand_forecasting.ipynb)."""
        if lead_time <= 50:
            return float(np.exp(-lead_time / 20))
        else:
            return 0.05

    def _fallback_demand_score_predictor(self, feature_frame: pd.DataFrame, processed_df: pd.DataFrame) -> pd.Series:
        """Fallback predictor using demand_score method: hotel × duration × lead_time × rating.
        
        Based on demand_forecasting.ipynb weighted scoring approach.
        """
        # # If source_df not available, fall back to simple heuristic
        # if not hasattr(self, '_source_df_for_fallback') or self._source_df_for_fallback is None:
        #     pred = (
        #         0.45 * feature_frame["lag_1h"]
        #         + 0.25 * feature_frame["lag_2h"]
        #         + 0.2 * (feature_frame["past_3h"] / 3.0)
        #         + 0.1 * (feature_frame["past_6h"] / 6.0)
        #     )
        #     pred = np.maximum(pred, 0.6 * feature_frame["request_count"])
        #     return pd.Series(pred, index=feature_frame.index)
        
        # Use source_df to compute demand score weights
        id_col = 'hotel_code' if 'hotel_code' in self._source_df_for_fallback.columns else 'chain_code'
        source = self._source_df_for_fallback[['cache_key', id_col, 'duration', 'lead_time', 'sabre_rating']].copy()
        
        # 1. Hotel weight: log1p normalization of hotel frequencies
        hotel_counts = source[id_col].value_counts()
        hotel_weight_map = np.log1p(hotel_counts) / np.log1p(hotel_counts).max()
        source['hotel_weight'] = source[id_col].map(hotel_weight_map).fillna(0.0)
        
        # 2. Duration weight: log1p normalization of duration frequencies
        duration_counts = source['duration'].value_counts()
        duration_weight_map = np.log1p(duration_counts) / np.log1p(duration_counts).max()
        source['duration_weight'] = source['duration'].map(duration_weight_map).fillna(0.0)
        
        # 3. Lead time weight: exponential decay
        source['lead_time_weight'] = source['lead_time'].apply(self._compute_lead_time_weight)
        
        # 4. Rating weight: log1p normalization of rating frequencies
        rating_counts = source['sabre_rating'].value_counts()
        rating_weight_map = np.log1p(rating_counts) / np.log1p(rating_counts).max()
        source['rating_weight'] = source['sabre_rating'].map(rating_weight_map).fillna(0.0)
        
        # 5. Compute demand score as product of all weights
        source['demand_score'] = (
            source['hotel_weight'] * 
            source['duration_weight'] * 
            source['lead_time_weight'] * 
            source['rating_weight']
        )
        
        # Aggregate by cache_key: mean of demand_score
        source_agg = source.groupby('cache_key')[['demand_score']].mean().reset_index()
        
        # Merge with feature_frame on cache_key
        feature_frame = feature_frame.merge(source_agg, on='cache_key', how='left')
        feature_frame['demand_score'] = feature_frame['demand_score'].fillna(0.5)  # Default to 0.5
        
        # Compute prediction: demand_score * request_count + 0.6 * request_count
        pred = (
            feature_frame['demand_score'].values * feature_frame['request_count'].values +
            0.6 * feature_frame['request_count'].values
        )
        
        return pd.Series(pred, index=feature_frame.index)

    def predict_next_hour_request_count(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Predict next_hour_request_count on a prepared hourly feature frame."""
        feature_frame = self._build_hourly_feature_frame(processed_df)
        X = self.extract_features(feature_frame)

        if self.model is not None:
            try:
                d_matrix = xgb.DMatrix(X, enable_categorical=True)
                pred = self.model.predict(d_matrix)
                feature_frame["pred_next_hour_request_count"] = np.maximum(pred, 0.0)
                return feature_frame
            except Exception as e:
                print(f"Warning: XGBoost prediction failed: {e}")
                print("Falling back to heuristic next-hour count.")

        # Fallback: demand_score based predictor (from demand_forecasting.ipynb)
        pred = self._fallback_demand_score_predictor(feature_frame, processed_df)
        feature_frame["pred_next_hour_request_count"] = np.maximum(pred.values, 0.0)
        return feature_frame

    @staticmethod
    def count_to_probability(counts: np.ndarray, method: str = "log_minmax") -> np.ndarray:
        """Convert predicted count to demand score (probability in [0, 1]).

        methods:
        - minmax: plain min-max scaling
        - log_minmax: log1p then min-max scaling (recommended)
        """
        counts = np.nan_to_num(np.asarray(counts, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        counts = np.maximum(counts, 0.0)

        if method == "minmax":
            transformed = counts
        else:
            transformed = np.log1p(counts)

        mn = float(np.min(transformed)) if transformed.size else 0.0
        mx = float(np.max(transformed)) if transformed.size else 0.0
        if mx <= mn:
            return np.zeros_like(transformed)
        return (transformed - mn) / (mx - mn)

    @staticmethod
    def validate_scores(scores: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """Validate and clamp demand scores to valid range.

        Args:
            scores: Array of predicted scores.
            min_val: Minimum valid value (default 0.0).
            max_val: Maximum valid value (default 1.0).

        Returns:
            Validated scores clamped to [min_val, max_val].
        """
        # Replace NaN and inf with 0
        scores = np.nan_to_num(scores, nan=0.0, posinf=max_val, neginf=min_val)

        # Clamp to valid range
        scores = np.clip(scores, min_val, max_val)

        return scores

    def generate_demand_scores(
        self,
        processed_df: pd.DataFrame,
        source_df: Optional[pd.DataFrame] = None,
        num_samples: Optional[int] = None,
        prob_method: str = "log_minmax",
        output_parquet: Optional[str] = None,
    ) -> pd.DataFrame:
        """Predict next-hour count and p_reuse on prepared hourly dataframe.

        Args:
            processed_df: Prepared hourly dataframe from data_pipeline.ipynb.
            source_df: Optional row-level source data for fallback predictor weight calculation.
            num_samples: Optional limit on samples.
            prob_method: 'log_minmax' or 'minmax'.
            output_parquet: Optional path to save results as Parquet.

        Returns:
            DataFrame with input columns + pred_next_hour_request_count + p_reuse.
        """
        data_df = processed_df.copy()
        if num_samples is not None:
            data_df = data_df.head(num_samples).copy()

        # Store source_df for fallback predictor if provided
        self._source_df_for_fallback = source_df

        print("\nPredicting next_hour_request_count...")
        result_df = self.predict_next_hour_request_count(data_df)

        print("\nMapping count to probability demand score...")
        probs = self.count_to_probability(result_df["pred_next_hour_request_count"].values, method=prob_method)
        result_df["p_reuse"] = self.validate_scores(probs)

        # Print statistics
        print("\n=== Predicted Count Statistics ===")
        print(f"Mean pred count: {result_df['pred_next_hour_request_count'].mean():.4f}")
        print(f"Median pred count: {result_df['pred_next_hour_request_count'].median():.4f}")
        print(f"Min pred count: {result_df['pred_next_hour_request_count'].min():.4f}")
        print(f"Max pred count: {result_df['pred_next_hour_request_count'].max():.4f}")

        print("\n=== Demand Score Statistics ===")
        print(f"Score method: {prob_method}")
        print(f"Mean p_reuse: {result_df['p_reuse'].mean():.4f}")
        print(f"Median p_reuse: {result_df['p_reuse'].median():.4f}")
        print(f"Min p_reuse: {result_df['p_reuse'].min():.4f}")
        print(f"Max p_reuse: {result_df['p_reuse'].max():.4f}")
        print(f"Std p_reuse: {result_df['p_reuse'].std():.4f}")

        # Check for any scores outside valid range
        invalid_count = ((result_df["p_reuse"] < 0.0) | (result_df["p_reuse"] > 1.0)).sum()
        if invalid_count > 0:
            print(f"Warning: {invalid_count} scores outside [0, 1] range (after validation)")
        else:
            print("All scores successfully validated to [0, 1] range")

        if output_parquet:
            result_df.to_parquet(output_parquet, compression="snappy")
            print(f"\nResults saved to {output_parquet}")

        return result_df


def main() -> None:
    """Entrypoint disabled: use the class methods with a provided DataFrame."""
    raise SystemExit("Use DemandScoreGenerator.generate_demand_scores(processed_df=...) with a prepared dataframe.")



if __name__ == "__main__":
    main()
