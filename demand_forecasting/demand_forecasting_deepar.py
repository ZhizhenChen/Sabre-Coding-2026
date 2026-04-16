import pandas as pd
import numpy as np
import os
import random  
from pathlib import Path
import torch

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput

# Import customized utilities for preprocessing and grid generation
from utils_df import (
    load_and_preprocess_data, 
    create_smart_mesh_grid
)

# ==========================================
# Fix Random Seed for Reproducibility
# ==========================================
def set_seed(seed=42):
    """
    Ensures that identical models are generated across different runs 
    by locking Python and PyTorch internal random states.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Prevent PyTorch unpickling error for historical checkpoints (Workaround for PyTorch 2.6+)
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

# ==========================================
# Bucketing Logic for Categorical Data
# ==========================================
def get_ap_bucket_label(ap):
    """Categorizes continuous Advance Purchase (lead time) into specific day ranges."""
    if pd.isna(ap) or ap < 0: return "Exclude"
    if ap <= 10: return f"AP_{int(ap):02d}"
    for high in range(15, 51, 5):
        if ap <= high: return f"AP_{high-4:02d}-{high:02d}"
    return "AP_>50"

def get_duration_bucket_label(stay):
    """Categorizes stay duration length."""
    if pd.isna(stay) or stay <= 0: return "Exclude"
    if stay <= 7: return f"Stay_{int(stay)}"
    return "Stay_>7"

def get_rating_bucket_label(rating):
    """Normalizes sabre ratings into standard buckets to mitigate data sparsity."""
    if pd.isna(rating): return "Rate_NA"
    try: r = float(rating)
    except: return "Rate_NA"
    if r <= 1.5: return "Rate_1-1.5"
    elif 2.0 <= r <= 2.5: return "Rate_2-2.5"
    elif 3.0 <= r <= 3.5: return "Rate_3-3.5"
    elif 4.0 <= r <= 4.5: return "Rate_4-4.5"
    elif r == 5.0: return "Rate_5"
    else: return "Exclude"

# ==========================================
# Global Parameters & Configuration
# ==========================================
print("[0% Complete] Initializing Demand Forecasting Engine...")

USE_DATA_CACHE = True
# Using the clean cache filename to avoid loading dirty historical data
CACHE_FILE_NAME = "cache_valid_grids_clean.parquet"

# Forecasting Mode
# WALK_FORWARD: Trains the model to predict exactly 1 hour ahead (rolling window approach).
# ONE_SHOT: Trains the model to predict the entire TEST_DAYS period at once.
FORECAST_MODE = "WALK_FORWARD"  

# Model Parameters (Optimized for specific attribute sparsity)
DEFAULT_MAX_EPOCHS = 30
DEFAULT_USE_NB = True       # Negative Binomial (Handles over-dispersed count data well)

SPECIAL_MAX_EPOCHS = 15
SPECIAL_USE_NB = False      # Student's T distribution (Better for extremely sparse conditional matrices)

LAG_SIZE = 24               # Lookback window (Hours)
TEST_DAYS = 7
TEST_STEPS = 24 * TEST_DAYS # Total validation/inference horizon in hours
TRAIN_WEEKS = 12            # Using 3 months of historical data
RECENT_WEEKS = TRAIN_WEEKS + (TEST_DAYS // 7)  

NUM_LAYERS = 2
HIDDEN_SIZE = 100                 

# The target output length for the internal Neural Network.
pred_len = 1 if FORECAST_MODE == "WALK_FORWARD" else TEST_STEPS

# ==========================================
# Phase 0: Data Loading, Purging & Preprocessing
# ==========================================
if USE_DATA_CACHE and os.path.exists(CACHE_FILE_NAME):
    print(f"\n[10% Complete] Loading pre-processed clean cache from '{CACHE_FILE_NAME}'...")
    valid_grids_df = pd.read_parquet(CACHE_FILE_NAME)
    
    # Generate continuous hourly timeline to handle missing hours (zero-fill later)
    ts_data_full = valid_grids_df.set_index('rq_timestamp').resample('h').size().fillna(0)
    recent_idx = ts_data_full.iloc[-24 * 7 * RECENT_WEEKS:].index
    full_idx = pd.date_range(start=recent_idx[0], end=recent_idx[-1], freq='h')
else:
    print("\n[10% Complete] Ingesting raw Parquet dataset...")
    file_pattern = 'data/raw/0225/*/get_hotel_avail_00000000*.parquet'
    df = load_and_preprocess_data(file_pattern)

    # Filter to only retain the recent historical period required for training
    cutoff_time_ts = df['rq_timestamp'].max() - pd.Timedelta(weeks=RECENT_WEEKS)
    df = df[df['rq_timestamp'] >= cutoff_time_ts].copy()

    # Clean extreme outlier coordinates before spatial partitioning
    df['location_latitude'] = pd.to_numeric(df['location_latitude'], errors='coerce')
    df['location_longitude'] = pd.to_numeric(df['location_longitude'], errors='coerce')
    df = df[~df['location_longitude'].between(-2, 12)]
    df = df[~df['location_longitude'].between(-125, -115)]

    # Purge rows with missing essential attributes to prevent sparse distributions & inaccurate coverages
    print("  -> Purging rows with missing rating, chain, stay duration, or AP...")
    df = df.dropna(subset=['sabre_rating', 'chain_code', 'stay_duration', 'AP'])

    # Generate spatial clustering
    MAX_HOTELS = 100
    MIN_HOTELS = 30
    df = create_smart_mesh_grid(df, max_hotels=MAX_HOTELS, min_hotels=MIN_HOTELS)

    valid_grids_df = df[df['geo_grid_auto'] != 'Unknown']

    ts_data_full = valid_grids_df.set_index('rq_timestamp').resample('h').size().fillna(0)
    recent_idx = ts_data_full.iloc[-24 * 7 * RECENT_WEEKS:].index
    full_idx = pd.date_range(start=recent_idx[0], end=recent_idx[-1], freq='h')
    
    if USE_DATA_CACHE:
        valid_grids_df.to_parquet(CACHE_FILE_NAME, index=False)

# Dynamic Chain Bucketing Setup
top_50_chains = set(valid_grids_df['chain_code'].value_counts().nlargest(50).index.tolist())
def get_chain_bucket_label(chain):
    if pd.isna(chain): return 'Chain_others'
    return f"Chain_{chain}" if chain in top_50_chains else 'Chain_others'

# ==========================================
# Phase 1: Train Base Location Demand Model
# ==========================================
print("\n==================================================")
print(f" [Phase 1] Location Forecast Modeling (Mode: {FORECAST_MODE}) ")
print("==================================================")

all_grids = valid_grids_df['geo_grid_auto'].unique().tolist()
all_train_list = []

recent_df = valid_grids_df[valid_grids_df['rq_timestamp'] >= recent_idx[0]]
grouped_df = recent_df.groupby(['geo_grid_auto', pd.Grouper(key='rq_timestamp', freq='h')]).size().unstack(level=0, fill_value=0)
grouped_df = grouped_df.reindex(full_idx, fill_value=0)

for grid in all_grids:
    grid_ts = grouped_df[grid] if grid in grouped_df.columns else pd.Series(0, index=full_idx)
    # The last TEST_STEPS are reserved for validation/inference simulation
    train_grid = grid_ts.iloc[:-TEST_STEPS]
    all_train_list.append({"target": train_grid.values, "start": pd.Period(train_grid.index[0], freq="H"), "item_id": grid})

print(f"Training DeepAR on {len(all_grids)} distinct spatial grids...")
print(f"  -> Applying DEFAULT config: MAX_EPOCHS={DEFAULT_MAX_EPOCHS}, NegativeBinomial={DEFAULT_USE_NB}, Pred_Len={pred_len}")

estimator_kwargs = {
    "freq": "H", "prediction_length": pred_len, "context_length": LAG_SIZE,
    "num_layers": NUM_LAYERS, "hidden_size": HIDDEN_SIZE,
    "trainer_kwargs": {'max_epochs': DEFAULT_MAX_EPOCHS, 'enable_progress_bar': False}
}
if DEFAULT_USE_NB:
    estimator_kwargs["distr_output"] = NegativeBinomialOutput()
    
estimator = DeepAREstimator(**estimator_kwargs)
predictor = estimator.train(ListDataset(all_train_list, freq="H"))

# Serialize and export the trained model for downstream engineering teams
model_dir = Path(f"checkpoint_model_Location_{FORECAST_MODE}")
model_dir.mkdir(exist_ok=True)
predictor.serialize(model_dir)
print(f"  -> Model successfully exported to '{model_dir}'")

# Execute inference loop to demonstrate the model functionality
if FORECAST_MODE == "WALK_FORWARD":
    print(f"  -> Demonstrating Walk-Forward Inference (1-step ahead shifting window)...")
    for h in range(TEST_STEPS):
        wf_list = []
        for grid in all_grids:
            grid_ts = grouped_df[grid] if grid in grouped_df.columns else pd.Series(0, index=full_idx)
            # Shift the observation window forward by 'h' steps to predict 'h+1'
            target_h = grid_ts.iloc[:-TEST_STEPS + h].values if h > 0 else grid_ts.iloc[:-TEST_STEPS].values
            wf_list.append({"target": target_h, "start": pd.Period(grid_ts.index[0], freq="H"), "item_id": grid})
        
        step_forecasts = list(predictor.predict(ListDataset(wf_list, freq="H")))
        if (h + 1) % 24 == 0:
            print(f"     ... {h+1}/{TEST_STEPS} hours successfully forecasted.")

# ==========================================
# Phase 2: Train Attribute Distribution Models
# ==========================================
print("\n==================================================")
print(f" [Phase 2] Attribute Models Construction")
print("==================================================")

# Apply categorization rules
df_attr_base = valid_grids_df.copy()
df_attr_base['AP_bucket'] = df_attr_base['AP'].apply(get_ap_bucket_label)
df_attr_base['stay_bucket'] = df_attr_base['stay_duration'].apply(get_duration_bucket_label)
df_attr_base['rating_bucket'] = df_attr_base['sabre_rating'].apply(get_rating_bucket_label)
df_attr_base['chain_bucket'] = df_attr_base['chain_code'].apply(get_chain_bucket_label)

# Generate conditional probability keys
mask_loc = df_attr_base['rating_bucket'].notna() & df_attr_base['geo_grid_auto'].notna()
df_attr_base.loc[mask_loc, 'rating_loc'] = df_attr_base.loc[mask_loc, 'rating_bucket'].astype(str) + "_|_" + df_attr_base.loc[mask_loc, 'geo_grid_auto'].astype(str)

mask_chain = df_attr_base['rating_bucket'].notna() & df_attr_base['chain_bucket'].notna()
df_attr_base.loc[mask_chain, 'rating_chain'] = df_attr_base.loc[mask_chain, 'rating_bucket'].astype(str) + "_|_" + df_attr_base.loc[mask_chain, 'chain_bucket'].astype(str)

# Remove invalid entries
for col in ['AP_bucket', 'stay_bucket', 'rating_bucket', 'chain_bucket']:
    df_attr_base = df_attr_base[df_attr_base[col] != "Exclude"]

# Train all models needed for up to M10 configurations
attributes_to_train = ['AP_bucket', 'stay_bucket', 'rating_bucket', 'chain_bucket', 'rating_loc', 'rating_chain']

for attr in attributes_to_train:
    print(f"\n--- Initiating Model Pipeline for Attribute: [{attr}] ---")
    
    # Dynamically select parameters based on attribute sparsity
    if attr in ['rating_loc', 'rating_chain']:
        current_epochs = SPECIAL_MAX_EPOCHS
        current_nb = SPECIAL_USE_NB
        print(f"  -> Applying SPECIAL config: MAX_EPOCHS={current_epochs}, NegativeBinomial={current_nb}, Pred_Len={pred_len}")
    else:
        current_epochs = DEFAULT_MAX_EPOCHS
        current_nb = DEFAULT_USE_NB
        print(f"  -> Applying DEFAULT config: MAX_EPOCHS={current_epochs}, NegativeBinomial={current_nb}, Pred_Len={pred_len}")

    df_attr = df_attr_base.dropna(subset=[attr]).copy()
    target_keys = df_attr[attr].unique().tolist()

    attr_grouped = df_attr.groupby([attr, pd.Grouper(key='rq_timestamp', freq='h')]).size().unstack(level=0, fill_value=0)
    attr_grouped = attr_grouped.reindex(full_idx, fill_value=0)

    attr_train_list = []
    for val in target_keys:
        ts = attr_grouped[val] if val in attr_grouped.columns else pd.Series(0, index=full_idx)
        attr_train_list.append({"target": ts.iloc[:-TEST_STEPS].values, "start": pd.Period(ts.index[0], freq="H"), "item_id": str(val)})

    estimator_kwargs = {
        "freq": "H", "prediction_length": pred_len, "context_length": LAG_SIZE,
        "num_layers": NUM_LAYERS, "hidden_size": HIDDEN_SIZE,
        "trainer_kwargs": {'max_epochs': current_epochs, 'enable_progress_bar': False}
    }
    
    if current_nb:
        estimator_kwargs["distr_output"] = NegativeBinomialOutput()
        
    estimator = DeepAREstimator(**estimator_kwargs)
    predictor = estimator.train(ListDataset(attr_train_list, freq="H"))

    # Serialize and export the attribute model
    model_dir = Path(f"checkpoint_model_{attr}_{FORECAST_MODE}")
    model_dir.mkdir(exist_ok=True)
    predictor.serialize(model_dir)
    print(f"  -> Model successfully exported to '{model_dir}'")

    # Optional: Walk-Forward demonstration for attribute
    if FORECAST_MODE == "WALK_FORWARD":
        print(f"  -> Demonstrating Walk-Forward Inference...")
        for h in range(TEST_STEPS):
            wf_list = []
            for val in target_keys:
                ts = attr_grouped[val] if val in attr_grouped.columns else pd.Series(0, index=full_idx)
                target_h = ts.iloc[:-TEST_STEPS + h].values if h > 0 else ts.iloc[:-TEST_STEPS].values
                wf_list.append({"target": target_h, "start": pd.Period(ts.index[0], freq="H"), "item_id": str(val)})
            
            step_forecasts = list(predictor.predict(ListDataset(wf_list, freq="H")))
            if (h + 1) % 24 == 0:
                print(f"     ... {h+1}/{TEST_STEPS} hours successfully forecasted.")

print("\n[Complete] All core models have been trained and exported as checkpoints.")