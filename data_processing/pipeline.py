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

    def process(self, partition_date: str, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: load data for a date and produce prepared features.

        Args:
            partition_date: Date string (e.g., '2026-02-07')
            max_rows: Optional row limit for testing/sampling

        Returns:
            (source_df, prepared_df) where:
            - source_df: enriched row-level data with timestamps, lead_time, cache_key, etc.
            - prepared_df: hourly-aggregated features with all engineered columns
        """
        # Step 1: Load raw data
        df = self._load_raw_data(partition_date, max_rows)

        # Step 2: Data preprocessing (timestamps, lead_time, cache_key)
        df = self._preprocess_raw_data(df)

        # Step 3: Data cleanup (remove invalid geolocation)
        df = self._cleanup_location_data(df)

        # Step 4: Rate and price processing
        df = self._process_rates_and_prices(df)

        # Step 5: Market and price change features
        df = self._compute_market_and_price_change(df)

        # Step 6: Create source_df (enriched row-level) and produce prepared hourly features
        source_df = df.copy()
        prepared_df = self._create_hourly_features(df)

        return source_df, prepared_df

    def _load_raw_data(self, partition_date: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load parquet data for specific partition date."""
        path = self.data_root / f"date={partition_date}"
        
        if not path.exists():
            raise FileNotFoundError(f"Partition path not found: {path}")

        # Find all parquet files
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")

        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            if max_rows:
                df = df.head(max_rows)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        if max_rows:
            df = df.head(max_rows)

        return df

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

    def _create_hourly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate row-level data to hourly features."""
        source_df_copy = df.copy()
        source_df_copy['rq_timestamp'] = pd.to_datetime(source_df_copy['rq_timestamp'], utc=True)
        source_df_copy['timestamp_hour'] = source_df_copy['rq_timestamp'].dt.floor('h')

        # Request-level deduplication
        if 'rq_correlation_id' in source_df_copy.columns:
            request_level = source_df_copy.drop_duplicates(subset=['rq_correlation_id']).copy()
        else:
            request_level = source_df_copy.copy()

        request_level['request_count'] = 1

        # Aggregate to cache_key + hour
        hourly = (
            request_level.groupby(['cache_key', 'timestamp_hour'], dropna=False)
            .agg(
                request_count=('request_count', 'sum'),
                lead_time=('lead_time', 'median'),
            )
            .reset_index()
            .sort_values(['cache_key', 'timestamp_hour'])
            .reset_index(drop=True)
        )

        # Fill continuous hour series
        frames = []
        for cache_key, group in hourly.groupby('cache_key'):
            group = group.sort_values('timestamp_hour').set_index('timestamp_hour')
            full_index = pd.date_range(group.index.min(), group.index.max(), freq='h', tz='UTC')
            expanded = group.reindex(full_index)
            expanded['cache_key'] = cache_key
            expanded['request_count'] = expanded['request_count'].fillna(0)
            expanded['lead_time'] = expanded['lead_time'].ffill().bfill().fillna(0)
            expanded = expanded.reset_index().rename(columns={'index': 'timestamp_hour'})
            frames.append(expanded)

        df_hourly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=['cache_key', 'timestamp_hour', 'request_count', 'lead_time']
        )

        # Temporal features
        df_hourly['hour_of_day'] = df_hourly['timestamp_hour'].dt.hour
        df_hourly['day_of_week'] = df_hourly['timestamp_hour'].dt.dayofweek
        df_hourly['is_weekend'] = (df_hourly['day_of_week'] >= 5).astype(int)
        df_hourly['is_holiday'] = 0

        # Lags and rolling features
        df_hourly = df_hourly.sort_values(['cache_key', 'timestamp_hour'])
        df_hourly['lag_1h'] = df_hourly.groupby('cache_key')['request_count'].shift(1).fillna(0)
        df_hourly['lag_2h'] = df_hourly.groupby('cache_key')['request_count'].shift(2).fillna(0)
        df_hourly['past_3h'] = (
            df_hourly.groupby('cache_key')['request_count']
            .rolling(3).sum().reset_index(level=0, drop=True).fillna(0)
        )
        df_hourly['past_6h'] = (
            df_hourly.groupby('cache_key')['request_count']
            .rolling(6).sum().reset_index(level=0, drop=True).fillna(0)
        )

        # Request time gap
        df_hourly['last_request_time'] = (
            df_hourly.groupby('cache_key')['timestamp_hour']
            .transform(lambda x: x.where(df_hourly.loc[x.index, 'request_count'] > 0).ffill())
        )
        df_hourly['request_time_gap'] = (
            (df_hourly['timestamp_hour'] - df_hourly['last_request_time']).dt.total_seconds() / 3600.0
        ).fillna(999.0)

        # Interaction features
        df_hourly['leadtime_x_recent'] = df_hourly['lead_time'] * df_hourly['lag_1h']
        df_hourly['recent_and_close'] = (
            (df_hourly['lag_1h'] > 0) & (df_hourly['lead_time'] <= 1)
        ).astype(int)

        # Lead time bucket
        df_hourly['lead_time_bucket'] = pd.cut(
            df_hourly['lead_time'],
            bins=[-1, 1, 3, 7, 30, 365],
            labels=['same_day', 'short', 'mid', 'long', 'very_long'],
        )
        df_hourly['lead_time_bucket'] = (
            df_hourly['lead_time_bucket'].astype('object').fillna('very_long').astype('category')
        )

        # Target-aligned bookkeeping
        df_hourly['next_hour_request_count'] = (
            df_hourly.groupby('cache_key')['request_count'].shift(-1).fillna(0)
        )

        return df_hourly
