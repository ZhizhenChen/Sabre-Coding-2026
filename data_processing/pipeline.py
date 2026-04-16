"""
Data pipeline processor: loads raw parquet data and produces prepared features.
Converts logic from data_pipeline.ipynb into a reusable class interface.
"""

import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

from demand_forecasting.feature_rules import (
    ap_bucket_label,
    assign_geo_grid,
    duration_bucket_label,
    rating_bucket_label,
)


class DataPipelineProcessor:
    """
    Loads cleaned_partitioned data by date and produces:
    - source_df: enriched row-level data with base features
    - df (prepared): hourly-aggregated features ready for models
    """

    def __init__(self, data_root: str = "../data/cleaned_partitioned"):
        """
        Args:
            data_root: Path to cleaned_partitioned directory
        """
        self.data_root = Path(data_root)

    def process(self, start_date: str, end_date: str, max_requests: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: load data for a date range and produce prepared features.

        Args:
            start_date: Start date string (e.g., '2026-02-07')
            end_date: End date string (e.g., '2026-02-07')
            max_requests: Optional row limit for testing/sampling

        Returns:
            (source_df, prepared_df) where:
            - source_df: enriched row-level data with timestamps, lead_time, cache_key, etc.
            - prepared_df: hourly-aggregated features with all engineered columns
        """
        # Step 1: Load raw data
        df = self._load_raw_data(start_date, end_date)

        # Step 1.5: sampling for testing
        df = self._sample_raw_requests(df, max_requests=max_requests)

        # Step 2: Data preprocessing (timestamps, lead_time, cache_key)
        df = self._preprocess_raw_data(df)

        # Step 3: Data cleanup (remove invalid geolocation)
        df = self._cleanup_location_data(df)

        # Step 4: Rate and price processing
        # df = self._process_rates_and_prices(df)

        # Step 5: Market and price change features
        # df = self._compute_market_and_price_change(df)

        # Step 6: Create source_df (enriched row-level) and produce prepared hourly features
        source_df = df.copy()
        prepared_df = self._create_hourly_features(df)

        return source_df, prepared_df

    def _load_raw_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load parquet data for specific date range."""
        date_list = pd.date_range(start=start_date, end=end_date, freq="D")
        if len(date_list) == 0:
            raise ValueError(f"No dates found between start_date={start_date} and end_date={end_date}.")
        dfs = []
        for dt in date_list:
            day = dt.strftime("%Y-%m-%d")
            path = self.data_root / f"date={day}"
            if not path.exists():
                raise FileNotFoundError(f"Partition path not found: {path}")

            # Find all parquet files
            parquet_files = list(path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {path}")
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                dfs.append(df)
                
        total_df = pd.concat(dfs, ignore_index=True) 

        return total_df


    def _sample_raw_requests(self, raw_df: pd.DataFrame, max_requests: int | None = None) -> pd.DataFrame:
        """Sample request rows before feature engineering."""
        if raw_df.empty:
            return raw_df.copy()

        sampled = raw_df.copy()
        sampled["rq_timestamp"] = pd.to_datetime(sampled["rq_timestamp"], errors="coerce", utc=True)
        sampled = sampled.dropna(subset=["rq_timestamp"]).sort_values("rq_timestamp", kind="stable")

        if max_requests is not None and max_requests > 0:
            sampled = sampled.head(max_requests)

        return sampled.reset_index(drop=True)
    
    def _preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamps, compute lead_time, cache_key, hour."""
        df = df.copy()

        # Ensure rq_timestamp is UTC
        df['rq_timestamp'] = pd.to_datetime(df['rq_timestamp'], utc=True)

        # Parse stay dates
        df['rq_stay_start_date'] = pd.to_datetime(df['rq_stay_start_date'], format='mixed')
        df['rq_stay_end_date'] = pd.to_datetime(df['rq_stay_end_date'], format='mixed')

        # Stay duration (days)
        df['duration'] = (df['rq_stay_end_date'] - df['rq_stay_start_date']).dt.days
        df['duration'] = df['duration'].clip(lower=1)

        # Convert stay dates to UTC using location timezone
        # NYC
        mask_nyc = df['location_city_code'].isin(['JFK', 'EWR', 'LGA', 'NYC'])
        df.loc[mask_nyc, 'rq_stay_start_date_utc'] = (
            df.loc[mask_nyc, 'rq_stay_start_date']
            .dt.tz_localize('America/New_York')
            .dt.tz_convert('UTC')
        )
        df.loc[mask_nyc, 'rq_stay_end_date_utc'] = (
            df.loc[mask_nyc, 'rq_stay_end_date']
            .dt.tz_localize('America/New_York')
            .dt.tz_convert('UTC')
        )

        # DFW (Dallas)
        mask_dfw = df['location_city_code'].isin(['DFW', 'DAL'])
        df.loc[mask_dfw, 'rq_stay_start_date_utc'] = (
            df.loc[mask_dfw, 'rq_stay_start_date']
            .dt.tz_localize('America/Chicago')
            .dt.tz_convert('UTC')
        )
        df.loc[mask_dfw, 'rq_stay_end_date_utc'] = (
            df.loc[mask_dfw, 'rq_stay_end_date']
            .dt.tz_localize('America/Chicago')
            .dt.tz_convert('UTC')
        )

        # Lead time (days)
        df['lead_time'] = (df['rq_stay_start_date_utc'] - df['rq_timestamp']).dt.days
        df.loc[df['lead_time'] < 0, 'lead_time'] = 0

        # Cache key
        df['cache_key'] = (
            df['hotel_code'].astype(str) + '-' +
            df['location_city_code'] + '-' +
            df['rq_stay_start_date'].dt.strftime('%Y-%m-%d') + '-' +
            df['rq_stay_end_date'].dt.strftime('%Y-%m-%d')
        )

        # Hour
        df['timestamp_hour'] = df['rq_timestamp'].dt.floor('h')

        return df

    def _cleanup_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert geolocation to numeric and drop invalid rows."""
        df = df.copy()
        df['location_latitude'] = pd.to_numeric(df['location_latitude'], errors='coerce')
        df['location_longitude'] = pd.to_numeric(df['location_longitude'], errors='coerce')
        df = df.dropna(subset=['location_latitude', 'location_longitude'])
        return df

    def _process_rates_and_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract rate source and explode convertedrate_infos, compute price_per_day."""
        df = df.copy()

        # Rate group (hash of rate plan candidates)
        def make_rate_group(val):
            if pd.isna(val) or str(val).strip() in ('[]', '', 'nan'):
                return 'NO_RATE'
            tokens = re.findall(r'[a-f0-9]{10,}', str(val))
            if not tokens:
                return 'NO_RATE'
            key = '_'.join(sorted(tokens))
            return hashlib.md5(key.encode()).hexdigest()[:12]

        if 'rq_rate_plan_candidates' in df.columns:
            rate_source_series = df['rq_rate_plan_candidates']
        elif 'hotelrateinfo' in df.columns:
            rate_source_series = df['hotelrateinfo']
        else:
            rate_source_series = pd.Series(['NO_RATE'] * len(df), index=df.index)

        df['rate_group'] = rate_source_series.apply(make_rate_group)

        # Explode convertedrate_infos
        df = df.explode('convertedrate_infos')
        df['price'] = df['convertedrate_infos'].apply(
            lambda x: x['amount_after_tax'] if isinstance(x, dict) else None
        )
        df['currencies'] = df['convertedrate_infos'].apply(
            lambda x: x['currency_code'] if isinstance(x, dict) else None
        )
        df['rate_source'] = df['convertedrate_infos'].apply(
            lambda x: x['rate_source'] if isinstance(x, dict) else None
        )

        # FX conversion
        fx_to_usd = {
            'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'CAD': 0.74, 'AUD': 0.66,
            'JPY': 0.0067, 'CNY': 0.14, 'HKD': 0.13, 'TWD': 0.031, 'KRW': 0.00075,
            'INR': 0.012, 'BRL': 0.20, 'MXN': 0.058, 'COP': 0.00025, 'ZAR': 0.053,
            'TRY': 0.031, 'LKR': 0.0033
        }
        df['price_in_USD'] = df['price'] * df['currencies'].map(fx_to_usd)
        df['price_per_day'] = df['price_in_USD'] / df['duration']

        return df

    def _compute_market_and_price_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market_key and price_change flag."""
        df = df.copy()
        # df['market_key'] = (
        #     df['location_latitude'].astype(str) + '_' +
        #     df['location_longitude'].astype(str)
        # )
        df = df.sort_values(['hotel_code', 'lead_time'])
        df['price_change'] = (
            df.groupby(['hotel_code', 'lead_time', 'rate_source'])['price_per_day']
            .transform(lambda x: (np.abs(x - x.shift()) > 1))
            .fillna(False)
            .astype(int)
        )
        return df


    # ====================================================
    # Hourly Feature Generation (DeepAR Optimized)
    # ====================================================
    def _create_hourly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data preparation formatted specifically for DeepAR Walk-Forward inference.
        Applies strict bucketing and drops missing values. 
        Note: Manual lag features are omitted as DeepAR's LSTM handles temporal context internally.
        """
        source_df_copy = df.copy()

        # 1. Map variable names to match the forecasting pipeline
        if 'AP' not in source_df_copy.columns and 'lead_time' in source_df_copy.columns:
            source_df_copy['AP'] = source_df_copy['lead_time']
        if 'stay_duration' not in source_df_copy.columns and 'duration' in source_df_copy.columns:
            source_df_copy['stay_duration'] = source_df_copy['duration']

        # 2. Generate spatial grid (geo_grid_auto)
        source_df_copy = assign_geo_grid(source_df_copy)

        # 3. Purge rows with missing essential attributes (Crucial for clean DeepAR inference)
        source_df_copy = source_df_copy.dropna(subset=['sabre_rating', 'chain_code', 'stay_duration', 'AP'])

        # 4. Apply categorical bucketing
        source_df_copy['AP_bucket'] = source_df_copy['AP'].apply(ap_bucket_label)
        source_df_copy['stay_bucket'] = source_df_copy['stay_duration'].apply(duration_bucket_label)
        source_df_copy['rating_bucket'] = source_df_copy['sabre_rating'].apply(rating_bucket_label)
        
        # Chain bucketing (Top 50 logic)
        top_50_chains = set(source_df_copy['chain_code'].value_counts().nlargest(50).index.tolist())
        source_df_copy['chain_bucket'] = source_df_copy['chain_code'].apply(
            lambda x: f"Chain_{x}" if x in top_50_chains else 'Chain_others'
        )

        # Remove "Exclude" buckets
        for col in ['AP_bucket', 'stay_bucket', 'rating_bucket', 'chain_bucket']:
            source_df_copy = source_df_copy[source_df_copy[col] != "Exclude"]

        source_df_copy = source_df_copy[source_df_copy['geo_grid_auto'] != 'Unknown']

        # 5. Create conditional keys specific to DeepAR logic
        source_df_copy['rating_loc'] = source_df_copy['rating_bucket'].astype(str) + "_|_" + source_df_copy['geo_grid_auto'].astype(str)
        source_df_copy['rating_chain'] = source_df_copy['rating_bucket'].astype(str) + "_|_" + source_df_copy['chain_bucket'].astype(str)

        source_df_copy['timestamp_hour'] = pd.to_datetime(source_df_copy['rq_timestamp'], utc=True).dt.floor('h')

        # 6. Aggregate demand by all DeepAR target attributes
        deepar_features = [
            'geo_grid_auto', 'AP_bucket', 'stay_bucket', 'rating_bucket', 
            'chain_bucket', 'rating_loc', 'rating_chain'
        ]
        
        prepared_df = source_df_copy.groupby(['timestamp_hour'] + deepar_features).size().reset_index(name='demand')

        return prepared_df.sort_values('timestamp_hour').reset_index(drop=True)