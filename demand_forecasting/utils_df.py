import pandas as pd
import glob
import numpy as np

def load_and_preprocess_data(file_pattern):
    """
    Loads Parquet files matching the given pattern and performs initial preprocessing.
    To optimize memory usage, only the necessary columns are extracted.
    """
    file_list = sorted(glob.glob(file_pattern))
    print(f"  -> Found {len(file_list)} files. Starting data ingestion...")
    
    # Explicitly specify target columns to discard large JSON/text columns, 
    # significantly reducing memory consumption.
    target_cols = [
        'rq_timestamp', 'rq_stay_start_date', 'rq_stay_end_date', 
        'hotel_code', 'chain_code', 'sabre_rating', 
        'location_latitude', 'location_longitude'
    ]
    
    # Check the first file to ensure columns exist before loading
    sample_df = pd.read_parquet(file_list[0])
    use_cols = [c for c in target_cols if c in sample_df.columns]
    
    # Load and concatenate all files
    dfs = [pd.read_parquet(f, columns=use_cols) for f in file_list]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  -> Total rows loaded successfully: {len(df)}")
    
    # Convert timestamp strings to datetime objects. 
    df['rq_timestamp'] = pd.to_datetime(df['rq_timestamp'], format='ISO8601').dt.tz_localize(None)
    df['rq_stay_start_date'] = pd.to_datetime(df['rq_stay_start_date'], errors='coerce').dt.tz_localize(None)
    df['rq_stay_end_date'] = pd.to_datetime(df['rq_stay_end_date'], errors='coerce').dt.tz_localize(None)
    
    # Calculate essential derived features: duration and Advance Purchase (AP) lead time
    df['stay_duration'] = (df['rq_stay_end_date'] - df['rq_stay_start_date']).dt.days
    df['lead_time'] = (df['rq_stay_start_date'].dt.normalize() - df['rq_timestamp'].dt.normalize()).dt.days

    df['AP'] = df['lead_time']
    
    # Fill missing identifiers with 'Unknown' to prevent errors during categorization
    if 'sabre_rating' not in df.columns:
        df['sabre_rating'] = 'Unknown'
    if 'chain_code' not in df.columns:
        df['chain_code'] = 'Unknown'
        
    return df

def create_smart_mesh_grid(df, max_hotels=100, min_hotels=30, lat_col='location_latitude', lon_col='location_longitude'):
    """
    Dynamically partitions geographical space into rectangular grids based on hotel density.
    This ensures that each defined area (e.g., NYC_001) contains a statistically stable number
    of hotels, avoiding sparse or overly dense clusters.
    """
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    
    unique_hotels = df[valid_mask].drop_duplicates(subset=['hotel_code', lat_col, lon_col]).copy()
    
    min_lat_init, max_lat_init = unique_hotels[lat_col].min(), unique_hotels[lat_col].max()
    min_lon_init, max_lon_init = unique_hotels[lon_col].min(), unique_hotels[lon_col].max()
    
    lat_margin, lon_margin = 0.05, 0.05
    queue = [(unique_hotels, min_lat_init - lat_margin, max_lat_init + lat_margin, min_lon_init - lon_margin, max_lon_init + lon_margin)]
    final_grids = []
    
    # Recursive spatial splitting based on hotel count thresholds
    while queue:
        pts, lat0, lat1, lon0, lon1 = queue.pop(0)
        count = len(pts)
        
        if count <= max_hotels or count < min_hotels * 2:
            final_grids.append({'pts': pts, 'lat0': lat0, 'lat1': lat1, 'lon0': lon0, 'lon1': lon1})
            continue
            
        lat_dist = lat1 - lat0
        lon_dist = lon1 - lon0
        
        if lat_dist == 0 and lon_dist == 0:
            final_grids.append({'pts': pts, 'lat0': lat0, 'lat1': lat1, 'lon0': lon0, 'lon1': lon1})
            continue
            
        if lon_dist >= lat_dist:
            axis_col = lon_col
            spatial_mid = (lon0 + lon1) / 2.0
        else:
            axis_col = lat_col
            spatial_mid = (lat0 + lat1) / 2.0
            
        pts_sorted = pts.sort_values(by=axis_col)
        valid_indices = np.arange(min_hotels, count - min_hotels + 1)
        valid_coords = pts_sorted[axis_col].values[valid_indices]
        
        best_relative_idx = np.argmin(np.abs(valid_coords - spatial_mid))
        split_idx = valid_indices[best_relative_idx]
        split_val = (pts_sorted.iloc[split_idx - 1][axis_col] + pts_sorted.iloc[split_idx][axis_col]) / 2.0
        if split_val == pts_sorted.iloc[split_idx - 1][axis_col]:
            split_val += 0.000001
            
        pts_left = pts_sorted[pts_sorted[axis_col] < split_val]
        pts_right = pts_sorted[pts_sorted[axis_col] >= split_val]
        
        if len(pts_left) < min_hotels or len(pts_right) < min_hotels:
            final_grids.append({'pts': pts, 'lat0': lat0, 'lat1': lat1, 'lon0': lon0, 'lon1': lon1})
            continue
            
        if lon_dist >= lat_dist:
            queue.append((pts_left, lat0, lat1, lon0, split_val))
            queue.append((pts_right, lat0, lat1, split_val, lon1))
        else:
            queue.append((pts_left, lat0, split_val, lon0, lon1))
            queue.append((pts_right, split_val, lat1, lon0, lon1))
            
    # Assign semantic names based on longitude for readability (e.g., NYC vs DFW)
    nyc_count, dfw_count, other_count = 1, 1, 1
    mapping_records = []
    
    for grid in final_grids:
        mid_lon = (grid['lon0'] + grid['lon1']) / 2.0
        if mid_lon > -85:
            grid_id = f"Grid_NYC_{nyc_count:03d}"
            nyc_count += 1
        elif mid_lon <= -85:
            grid_id = f"Grid_DFW_{dfw_count:03d}"
            dfw_count += 1
        else:
            grid_id = f"Grid_Other_{other_count:03d}"
            other_count += 1
            
        # Create a mapping record for every hotel in this newly defined grid
        for hc in grid['pts']['hotel_code'].values:
            mapping_records.append({
                'hotel_code': hc,
                'geo_grid_auto': grid_id
            })
            
    mapping_df = pd.DataFrame(mapping_records)
    
    # Merge mapping efficiently using Pandas left join
    cols_to_add = ['geo_grid_auto']
    df = df.drop(columns=[c for c in cols_to_add if c in df.columns], errors='ignore') 
    
    df = df.merge(mapping_df, on='hotel_code', how='left')
    
    # Handle edge cases where hotel coordinates were initially invalid
    df['geo_grid_auto'] = df['geo_grid_auto'].fillna('Unknown')
        
    return df