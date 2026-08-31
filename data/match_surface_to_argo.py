"""
Match preprocessed Argo profiles with real surface observations (SST, SSH, SSS) to produce the training dataset.
Enforces strict spatial (<=25 km) and temporal (<=24 h) matching constraints.
"""
from pathlib import Path
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
PROCESSED_DIR = BASE / 'processed'
DATASET_DIR = BASE / 'dataset'
DATASET_DIR.mkdir(parents=True, exist_ok=True)

try:
    from .surface_fetch import fetch_nearest_surface
except ImportError:
    from data.surface_fetch import fetch_nearest_surface

def match_surface_features(processed_parquet, spatial_radius_km=25.0, time_window_hours=24.0):
    df = pd.read_parquet(processed_parquet)
    logger.info(f"Loaded {len(df)} preprocessed Argo profiles for surface matching.")
    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        lat, lon = float(r['lat']), float(r['lon'])
        profile_time = r.get('profile_time')
        if pd.isna(profile_time) or profile_time is None: continue
        p_time = pd.to_datetime(profile_time)
        day_of_year = int(p_time.timetuple().tm_yday)
        surf = fetch_nearest_surface(lat, lon, dt=p_time, time_window_hours=time_window_hours, max_distance_km=spatial_radius_km)
        if surf is None: continue
        dist_km = surf.get('distance_km')
        if dist_km is not None and dist_km > spatial_radius_km: continue
        obs_time = surf.get('obs_time')
        if obs_time is None: continue
        time_diff_h = abs((pd.to_datetime(obs_time).tz_localize(None) - p_time.tz_localize(None)).total_seconds()) / 3600.0
        if time_diff_h > time_window_hours: continue

        sst_val = surf.get('sst')
        ssh_val = surf.get('ssh')
        sss_val = surf.get('sss')
        sss_source = surf.get('sss_source')
        if sss_val is None or pd.isna(sss_val):
            if 'salinity_0m' in r and not pd.isna(r['salinity_0m']):
                sss_val = float(r['salinity_0m']); sss_source = "Argo_CTD_in_situ"
        if sst_val is None or pd.isna(sst_val):
            if 'temperature_0m' in r and not pd.isna(r['temperature_0m']):
                sst_val = float(r['temperature_0m']); surf['sst_source'] = "Argo_CTD_in_situ"
            else: continue

        rec = {
            'argo_wmo': str(r.get('argo_wmo')), 'argo_cycle': int(r.get('argo_cycle')),
            'profile_id': str(r.get('source_file')), 'source_file': str(r.get('source_file')),
            'profile_time': p_time, 'surface_time': pd.to_datetime(obs_time),
            'surface_distance_km': float(dist_km) if dist_km is not None else 0.0,
            'surface_time_diff_hours': float(time_diff_h),
            'surface_obs_lat': float(surf.get('obs_lat')) if surf.get('obs_lat') is not None else lat,
            'surface_obs_lon': float(surf.get('obs_lon')) if surf.get('obs_lon') is not None else lon,
            'sst_source': str(surf.get('sst_source', 'NOAA_ERDDAP')),
            'ssh_source': str(surf.get('ssh_source', 'NOAA_NESDIS')),
            'sss_source': str(sss_source) if sss_source else 'ESA_SMOS_or_Argo',
            'lat': float(lat), 'lon': float(lon), 'day_of_year': int(day_of_year),
            'sst': float(sst_val),
            'ssh': float(ssh_val) if ssh_val is not None and not pd.isna(ssh_val) else 0.0,
            'sss': float(sss_val) if sss_val is not None and not pd.isna(sss_val) else 35.0,
            'temperature_0m': float(r['temperature_0m']) if not pd.isna(r.get('temperature_0m')) else float(sst_val),
            'temperature_50m': float(r['temperature_50m']),
            'temperature_100m': float(r['temperature_100m']),
            'temperature_200m': float(r['temperature_200m']),
            'temperature_500m': float(r['temperature_500m']),
            'temp_50m': float(r['temperature_50m']),
            'temp_100m': float(r['temperature_100m']),
            'temp_200m': float(r['temperature_200m']),
            'temp_500m': float(r['temperature_500m']),
        }
        rows.append(rec)

    if not rows: logger.warning('No matched surface-Argo rows produced'); return None
    out_df = pd.DataFrame(rows)
    out_parquet = DATASET_DIR / 'train_dataset.parquet'
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(DATASET_DIR / 'train_dataset.csv', index=False)
    logger.info(f"MATCHED {len(out_df)} rows -> {out_parquet}")
    return out_parquet

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pp = PROCESSED_DIR / 'argo_profiles.parquet'
    if pp.exists(): match_surface_features(pp)
    else: print('No processed profiles found; run data/argo_preprocess.py first')