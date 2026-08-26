"""
Surface data fetcher (SST, SSH, SSS, winds). Uses xarray/OPeNDAP endpoints when available.
Provides a function to query nearest surface fields for a lat/lon/time and a small spatial-temporal window.
"""
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
SURF_DIR = BASE / 'raw' / 'surface'
SURF_DIR.mkdir(parents=True, exist_ok=True)

# Read endpoint URLs from environment to avoid hard-coded placeholders
import os
SST_OPENDAP = os.getenv('OCEAN_SST_OPENDAP')
SSH_OPENDAP = os.getenv('OCEAN_SSH_OPENDAP')
SSS_OPENDAP = os.getenv('OCEAN_SSS_OPENDAP')
WINDS_OPENDAP = os.getenv('OCEAN_WINDS_OPENDAP')


def _haversine_km(lat1, lon1, lat2, lon2):
    # simple haversine
    from math import radians, sin, cos, atan2, sqrt
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def fetch_nearest_surface(lat, lon, dt=None, time_window_hours=24, dataset_url=None):
    """Return canonical surface variables (sst, ssh, sss, wind_u, wind_v, wind_speed) near lat/lon and within dt +/- time_window_hours.
    Uses environment-configured endpoints. Returns dict with keys and observation timestamp when found, or None.
    """
    import xarray as xr
    import numpy as np
    from datetime import timedelta

    # Choose URL: explicit dataset_url or fallback to SST env; prefer combined dataset if provided
    url = dataset_url or SST_OPENDAP
    if not url:
        logger.warning('No surface dataset URL configured (set OCEAN_SST_OPENDAP or pass dataset_url)')
        return None

    try:
        ds = xr.open_dataset(url)
    except Exception as e:
        logger.warning(f"Failed to open surface dataset at {url}: {e}")
        return None

    # If a time window is requested, select a slice; else use nearest time if available
    try:
        if dt is not None and 'time' in ds.coords:
            start = np.datetime64(dt - timedelta(hours=time_window_hours))
            end = np.datetime64(dt + timedelta(hours=time_window_hours))
            sel = ds.sel(time=slice(start, end))
            if sel.sizes.get('time', 0) == 0:
                # no samples in window
                return None
            # pick nearest time to dt
            if sel.sizes.get('time', 0) > 1:
                # compute distance in time
                times = sel['time'].values
                time_deltas = np.abs(times - np.datetime64(dt))
                nearest_idx = int(np.argmin(time_deltas))
                sel_point = sel.isel(time=nearest_idx).sel(latitude=lat, longitude=lon, method='nearest')
                obs_time = pd.to_datetime(times[nearest_idx])
            else:
                sel_point = sel.sel(latitude=lat, longitude=lon, method='nearest')
                obs_time = pd.to_datetime(sel['time'].values.ravel()[0])
        else:
            # use nearest in space/time
            if 'time' in ds.coords:
                sel_point = ds.sel(time=ds['time'].values.ravel()[0]).sel(latitude=lat, longitude=lon, method='nearest')
                obs_time = pd.to_datetime(ds['time'].values.ravel()[0])
            else:
                sel_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
                obs_time = None
    except Exception as e:
        logger.warning(f"Surface selection failed: {e}")
        return None

    out = {}
    # normalize variable names to canonical keys
    mapping = {
        'sst': ['sst', 'SST', 'sea_surface_temperature', 'analysed_sst'],
        'ssh': ['zos', 'ssh', 'sea_surface_height'],
        'sss': ['sss', 'SSS', 'sea_surface_salinity'],
        'wind_u': ['u10', 'w_u', 'wind_u'],
        'wind_v': ['v10', 'w_v', 'wind_v']
    }

    for key, candidates in mapping.items():
        val = None
        for var in candidates:
            if var in sel_point.variables:
                try:
                    val = float(sel_point[var].values)
                    break
                except Exception:
                    val = None
        out[key] = val

    # compute wind speed if components are present
    if out.get('wind_u') is not None and out.get('wind_v') is not None:
        out['wind_speed'] = float((out['wind_u']**2 + out['wind_v']**2)**0.5)
    else:
        out['wind_speed'] = None

    # also return observation metadata: location of grid point and obs_time
    try:
        obs_lat = float(sel_point['latitude'].values)
        obs_lon = float(sel_point['longitude'].values)
        out['obs_lat'] = obs_lat
        out['obs_lon'] = obs_lon
        out['distance_km'] = _haversine_km(lat, lon, obs_lat, obs_lon)
    except Exception:
        out['obs_lat'] = None
        out['obs_lon'] = None
        out['distance_km'] = None

    out['obs_time'] = pd.to_datetime(obs_time) if obs_time is not None else None
    return out


if __name__ == '__main__':
    import pandas as pd
    res = fetch_nearest_surface(15.0, 65.0)
    print(res)
