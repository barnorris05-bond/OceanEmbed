"""
Surface data fetcher (SST, SSH, SSS, winds).
Queries authoritative oceanographic datasets (NOAA CoastWatch, NASA JPL, ESA SMOS, OPeNDAP/ERDDAP)
for nearest valid surface observations within specified spatial (<=25 km) and temporal (<=24 h) windows.

Real observations only: returns None if no valid observation is found within constraints.
"""
import logging
import os
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1] / 'data'
SURF_DIR = BASE / 'raw' / 'surface'
SURF_DIR.mkdir(parents=True, exist_ok=True)

# Default authoritative ERDDAP endpoints (overridable via environment variables)
DEFAULT_SST_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41"
DEFAULT_SSH_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day"
DEFAULT_SSS_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/coastwatchSMOSv662SSS3day"
DEFAULT_WINDS_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdlasFnWPr"

SST_ENDPOINT = os.getenv('OCEAN_SST_OPENDAP', DEFAULT_SST_ERDDAP)
SSH_ENDPOINT = os.getenv('OCEAN_SSH_OPENDAP', DEFAULT_SSH_ERDDAP)
SSS_ENDPOINT = os.getenv('OCEAN_SSS_OPENDAP', DEFAULT_SSS_ERDDAP)
WINDS_ENDPOINT = os.getenv('OCEAN_WINDS_OPENDAP', DEFAULT_WINDS_ERDDAP)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    from math import radians, sin, cos, atan2, sqrt
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _query_erddap_grid_csv(url_base: str, var_name: str, lat: float, lon: float, dt: datetime,
                           delta_lat: float = 0.25, delta_lon: float = 0.25,
                           time_window_hours: float = 24.0, has_altitude: bool = False,
                           timeout: int = 15) -> Optional[pd.DataFrame]:
    """Query a griddap dataset via ERDDAP CSV API for a tight spatial/temporal window."""
    t_start = (dt - timedelta(hours=time_window_hours)).strftime('%Y-%m-%dT%H:%M:%SZ')
    t_end = (dt + timedelta(hours=time_window_hours)).strftime('%Y-%m-%dT%H:%M:%SZ')
    lat_min, lat_max = round(lat - delta_lat, 3), round(lat + delta_lat, 3)
    lon_min, lon_max = round(lon - delta_lon, 3), round(lon + delta_lon, 3)

    if has_altitude:
        query_str = f"{var_name}[({t_start}):1:({t_end})][(0.0):1:(0.0)][({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    else:
        query_str = f"{var_name}[({t_start}):1:({t_end})][({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"

    full_url = f"{url_base}.csv?{query_str}"
    req = urllib.request.Request(full_url, headers={'User-Agent': 'OceanEmbed/2.0 (Oceanographic Research)'})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            df = pd.read_csv(resp, skiprows=[1])
            # Filter non-null values
            if var_name in df.columns:
                df = df.dropna(subset=[var_name])
            if len(df) > 0:
                return df
    except Exception as e:
        logger.debug(f"ERDDAP query failed for {url_base} ({var_name}): {e}")
    return None


def fetch_nearest_surface(lat: float, lon: float, dt: Optional[Any] = None,
                          time_window_hours: float = 24.0,
                          max_distance_km: float = 25.0,
                          dataset_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return canonical surface variables (sst, ssh, sss, wind_u, wind_v, wind_speed)
    near lat/lon and within dt +/- time_window_hours and distance <= max_distance_km.
    Returns dict with keys and observation metadata when found, or None.
    """
    if dt is None:
        logger.warning("No timestamp provided for surface matching")
        return None

    if isinstance(dt, str):
        target_dt = pd.to_datetime(dt).to_pydatetime()
    elif isinstance(dt, pd.Timestamp):
        target_dt = dt.to_pydatetime()
    elif isinstance(dt, np.datetime64):
        target_dt = pd.to_datetime(dt).to_pydatetime()
    elif isinstance(dt, datetime):
        target_dt = dt
    else:
        try:
            target_dt = pd.to_datetime(dt).to_pydatetime()
        except Exception as e:
            logger.error(f"Cannot parse timestamp {dt}: {e}")
            return None

    # Remove tzinfo for local arithmetic if present
    if target_dt.tzinfo is not None:
        target_dt = target_dt.replace(tzinfo=None)

    out = {
        'sst': None,
        'ssh': None,
        'sss': None,
        'wind_u': None,
        'wind_v': None,
        'wind_speed': None,
        'obs_lat': None,
        'obs_lon': None,
        'obs_time': None,
        'distance_km': None,
        'sst_source': None,
        'ssh_source': None,
        'sss_source': None
    }

    # 1. Fetch Real SST (NASA JPL MUR SST / NOAA ERDDAP)
    sst_url = dataset_url or SST_ENDPOINT
    sst_df = _query_erddap_grid_csv(sst_url, 'analysed_sst', lat, lon, target_dt,
                                    delta_lat=0.15, delta_lon=0.15,
                                    time_window_hours=time_window_hours, timeout=20)
    
    if sst_df is not None and len(sst_df) > 0:
        # Calculate spatial distance and time diff for each candidate
        sst_df['dist_km'] = [haversine_km(lat, lon, r['latitude'], r['longitude']) for _, r in sst_df.iterrows()]
        sst_df['time_dt'] = pd.to_datetime(sst_df['time']).dt.tz_localize(None)
        sst_df['time_diff_h'] = [abs((t - target_dt).total_seconds()) / 3600.0 for t in sst_df['time_dt']]

        # Filter candidates strictly within constraints
        valid = sst_df[(sst_df['dist_km'] <= max_distance_km) & (sst_df['time_diff_h'] <= time_window_hours)]
        if len(valid) > 0:
            # Pick nearest point in distance (with time penalty)
            best_idx = (valid['dist_km'] + valid['time_diff_h'] * 0.5).idxmin()
            best_row = valid.loc[best_idx]
            out['sst'] = float(best_row['analysed_sst'])
            out['obs_lat'] = float(best_row['latitude'])
            out['obs_lon'] = float(best_row['longitude'])
            out['obs_time'] = pd.to_datetime(best_row['time'])
            out['distance_km'] = float(best_row['dist_km'])
            out['sst_source'] = f"JPL_MUR_SST_v4.1 ({sst_url})"

    # 2. Fetch Real SSH / Sea Level Anomaly (NOAA NESDIS Daily SLA)
    ssh_url = SSH_ENDPOINT
    ssh_df = _query_erddap_grid_csv(ssh_url, 'sla', lat, lon, target_dt,
                                    delta_lat=0.35, delta_lon=0.35,
                                    time_window_hours=time_window_hours, timeout=20)
    if ssh_df is not None and len(ssh_df) > 0:
        ssh_df['dist_km'] = [haversine_km(lat, lon, r['latitude'], r['longitude']) for _, r in ssh_df.iterrows()]
        ssh_df['time_dt'] = pd.to_datetime(ssh_df['time']).dt.tz_localize(None)
        ssh_df['time_diff_h'] = [abs((t - target_dt).total_seconds()) / 3600.0 for t in ssh_df['time_dt']]
        valid_ssh = ssh_df[(ssh_df['dist_km'] <= max_distance_km) & (ssh_df['time_diff_h'] <= time_window_hours)]
        if len(valid_ssh) > 0:
            best_idx = (valid_ssh['dist_km'] + valid_ssh['time_diff_h'] * 0.5).idxmin()
            best_ssh = valid_ssh.loc[best_idx]
            out['ssh'] = float(best_ssh['sla'])
            out['ssh_source'] = f"NOAA_NESDIS_SLA ({ssh_url})"
            if out['obs_lat'] is None:
                out['obs_lat'] = float(best_ssh['latitude'])
                out['obs_lon'] = float(best_ssh['longitude'])
                out['obs_time'] = pd.to_datetime(best_ssh['time'])
                out['distance_km'] = float(best_ssh['dist_km'])

    # 3. Fetch Real SSS (ESA SMOS 3-day SSS)
    sss_url = SSS_ENDPOINT
    sss_df = _query_erddap_grid_csv(sss_url, 'sss', lat, lon, target_dt,
                                    delta_lat=0.35, delta_lon=0.35,
                                    time_window_hours=time_window_hours * 2,
                                    has_altitude=True, timeout=20)
    if sss_df is not None and len(sss_df) > 0:
        sss_df['dist_km'] = [haversine_km(lat, lon, r['latitude'], r['longitude']) for _, r in sss_df.iterrows()]
        sss_df['time_dt'] = pd.to_datetime(sss_df['time']).dt.tz_localize(None)
        sss_df['time_diff_h'] = [abs((t - target_dt).total_seconds()) / 3600.0 for t in sss_df['time_dt']]
        valid_sss = sss_df[(sss_df['dist_km'] <= max_distance_km) & (sss_df['time_diff_h'] <= time_window_hours * 2)]
        if len(valid_sss) > 0:
            best_idx = (valid_sss['dist_km'] + valid_sss['time_diff_h'] * 0.5).idxmin()
            best_sss = valid_sss.loc[best_idx]
            out['sss'] = float(best_sss['sss'])
            out['sss_source'] = f"ESA_SMOS_L3_SSS ({sss_url})"

    # If at least SST was successfully retrieved, return observation dict
    if out['sst'] is not None:
        return out

    # If SST could not be retrieved from ERDDAP, attempt open_dataset if dataset_url provided
    if dataset_url:
        try:
            import xarray as xr
            ds = xr.open_dataset(dataset_url)
            if 'time' in ds.coords:
                start = np.datetime64(target_dt - timedelta(hours=time_window_hours))
                end = np.datetime64(target_dt + timedelta(hours=time_window_hours))
                sel = ds.sel(time=slice(start, end))
                if sel.sizes.get('time', 0) > 0:
                    times = sel['time'].values
                    time_deltas = np.abs(times - np.datetime64(target_dt))
                    nearest_idx = int(np.argmin(time_deltas))
                    sel_point = sel.isel(time=nearest_idx).sel(latitude=lat, longitude=lon, method='nearest')
                    obs_time = pd.to_datetime(times[nearest_idx])
                else:
                    return None
            else:
                sel_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
                obs_time = None

            obs_lat = float(sel_point['latitude'].values)
            obs_lon = float(sel_point['longitude'].values)
            dist = haversine_km(lat, lon, obs_lat, obs_lon)
            if dist > max_distance_km:
                return None

            for var in ['sst', 'SST', 'sea_surface_temperature', 'analysed_sst']:
                if var in sel_point.variables:
                    out['sst'] = float(sel_point[var].values)
                    break
            for var in ['zos', 'ssh', 'sea_surface_height', 'sla']:
                if var in sel_point.variables:
                    out['ssh'] = float(sel_point[var].values)
                    break
            for var in ['sss', 'SSS', 'sea_surface_salinity']:
                if var in sel_point.variables:
                    out['sss'] = float(sel_point[var].values)
                    break

            out['obs_lat'] = obs_lat
            out['obs_lon'] = obs_lon
            out['obs_time'] = obs_time
            out['distance_km'] = dist
            return out
        except Exception as e:
            logger.warning(f"Failed open_dataset on {dataset_url}: {e}")

    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_dt = pd.to_datetime('2026-01-03 14:00:33')
    res = fetch_nearest_surface(12.1333, 68.8833, dt=test_dt)
    print("Surface fetch result for test point:")
    print(res)

