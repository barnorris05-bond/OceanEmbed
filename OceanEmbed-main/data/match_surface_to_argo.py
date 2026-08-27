"""
Match preprocessed Argo profiles with surface observations and produce a final training dataset.
Aggregates surface samples in the window (mean, std, count) and stores alongside Argo target temps.
"""
from pathlib import Path
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
PROCESSED_DIR = BASE / 'processed'
DATASET_DIR = BASE / 'dataset'
DATASET_DIR.mkdir(parents=True, exist_ok=True)

from .surface_fetch import fetch_nearest_surface


def match_surface_features(processed_parquet, spatial_radius_km=25, time_window_hours=24):
    df = pd.read_parquet(processed_parquet)
    rows = []
    import pandas as pd
    from datetime import timedelta

    for _, r in df.iterrows():
        lat = float(r['lat'])
        lon = float(r['lon'])
        profile_time = r.get('profile_time', None)
        # If no profile_time, skip (we need time for temporal matching)
        if pd.isna(profile_time) or profile_time is None:
            logger.debug(f"Profile {r.get('source_file')} has no profile_time; skipping")
            continue

        # Fetch surface observation near profile time within window
        surf = fetch_nearest_surface(lat, lon, dt=profile_time, time_window_hours=time_window_hours)
        if surf is None:
            continue

        # Enforce spatial/time acceptance criteria
        if surf.get('distance_km') is not None and surf['distance_km'] > spatial_radius_km:
            logger.debug(f"Surface obs too far ({surf['distance_km']:.1f} km) for profile {r.get('source_file')}")
            continue
        if surf.get('obs_time') is None:
            logger.debug(f"Surface obs has no time for profile {r.get('source_file')}")
            continue
        # ensure temporal difference within window
        time_diff = abs((pd.to_datetime(surf['obs_time']) - pd.to_datetime(profile_time)).total_seconds())/3600.0
        if time_diff > time_window_hours:
            logger.debug(f"Surface obs time delta {time_diff:.1f} h > window for profile {r.get('source_file')}")
            continue

        rec = {
            'argo_wmo': r.get('argo_wmo'),
            'argo_cycle': r.get('argo_cycle'),
            'profile_id': r.get('source_file', f"p_{_}"),
            'lat': lat,
            'lon': lon,
            'profile_time': profile_time,
            'sst': surf.get('sst') if surf.get('sst') is not None else np.nan,
            'ssh': surf.get('ssh') if surf.get('ssh') is not None else np.nan,
            'sss': surf.get('sss') if surf.get('sss') is not None else np.nan,
            'wind_u': surf.get('wind_u') if surf.get('wind_u') is not None else np.nan,
            'wind_v': surf.get('wind_v') if surf.get('wind_v') is not None else np.nan,
            'wind_speed': surf.get('wind_speed') if surf.get('wind_speed') is not None else np.nan,
            'surface_time': surf.get('obs_time'),
            'surface_obs_lat': surf.get('obs_lat'),
            'surface_obs_lon': surf.get('obs_lon'),
            'surface_distance_km': surf.get('distance_km'),
            'temperature_0m': r['temperature_0m'],
            'temperature_50m': r['temperature_50m'],
            'temperature_100m': r['temperature_100m'],
            'temperature_200m': r['temperature_200m'],
            'temperature_500m': r['temperature_500m'],
            'temperature_1000m': r['temperature_1000m']
        }
        rows.append(rec)
    if not rows:
        logger.warning('No matched rows produced')
        return None
    out_df = pd.DataFrame(rows)
    out_path = DATASET_DIR / 'train_dataset.parquet'
    out_df.to_parquet(out_path, index=False)
    logger.info(f"Saved matched dataset to {out_path} ({len(out_df)} rows)")
    return out_path


if __name__ == '__main__':
    pp = PROCESSED_DIR / 'argo_profiles.parquet'
    if pp.exists():
        match_surface_features(pp)
    else:
        print('No processed profiles found; run data/argo_preprocess.py first')
