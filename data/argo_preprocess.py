"""
Argo preprocessing: read raw NetCDFs (Argopy/xarray), apply Argo QC filtering, convert pressure->depth using TEOS-10 (gsw),
interpolate temperature onto fixed depth levels and produce a parquet table of profiles.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
RAW_DIR = BASE / 'raw' / 'argo'
PROCESSED_DIR = BASE / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS = np.array([0, 50, 100, 200, 500, 1000], dtype=float)
MIN_REQUIRED_MAX_DEPTH = 500.0
MAX_ALLOWED_MIN_DEPTH = 10.0

def pressure_to_depth(p, lat):
    try:
        import gsw
        z = gsw.z_from_p(np.asarray(p, dtype=float), float(lat))
        return -np.asarray(z, dtype=float)
    except ImportError:
        logger.warning("GSW MODULE NOT AVAILABLE: Using approximate 1 dbar ≈ 1 m")
        return np.asarray(p, dtype=float)
    except Exception as e:
        logger.warning(f"gsw.z_from_p conversion failed ({e}); falling back to 1 dbar ≈ 1 m")
        return np.asarray(p, dtype=float)

def _decode_qc_array(qc_arr):
    out = []
    for x in np.asarray(qc_arr).ravel():
        if isinstance(x, (bytes, np.bytes_)):
            try: out.append(x.decode('ascii'))
            except: out.append(str(x))
        elif pd.isna(x):
            out.append('9')
        else:
            out.append(str(int(x)) if isinstance(x, (int, float, np.integer, np.floating)) and not np.isnan(x) else str(x))
    return np.array(out)

def _extract_profile_time(ds) -> Optional[pd.Timestamp]:
    for candidate in ['TIME', 'time', 'JULD', 'juld', 'JULD_LOCATION', 'PROFILE_TIME', 'DATE']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                tval = ds[candidate].values.ravel()[0]
                ts = pd.to_datetime(tval)
                if not pd.isna(ts): return ts
            except: continue
    return None

def process_netcdf_profile(nc_path: str) -> Optional[Dict[str, Any]]:
    import xarray as xr
    p = Path(nc_path)
    if not p.exists() or p.stat().st_size < 100: return None
    try: ds = xr.open_dataset(str(p))
    except Exception as e:
        logger.warning(f"Failed to open NetCDF {p.name}: {e}")
        return None

    wmo = None; cycle = None
    for candidate in ['PLATFORM_NUMBER', 'platform_number', 'WMO', 'wmo']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                raw_wmo = ds[candidate].values.ravel()[0]
                wmo = raw_wmo.decode('utf-8').strip() if isinstance(raw_wmo, (bytes, np.bytes_)) else str(raw_wmo).strip()
                break
            except: pass
    for candidate in ['CYCLE_NUMBER', 'cycle_number', 'CYCLE', 'cyc']:
        if candidate in ds.variables or candidate in ds.coords:
            try: cycle = int(ds[candidate].values.ravel()[0]); break
            except: pass

    lat = None; lon = None
    for candidate in ['LATITUDE', 'latitude', 'lat']:
        if candidate in ds.variables or candidate in ds.coords:
            try: lat = float(ds[candidate].values.ravel()[0]); break
            except: pass
    for candidate in ['LONGITUDE', 'longitude', 'lon']:
        if candidate in ds.variables or candidate in ds.coords:
            try: lon = float(ds[candidate].values.ravel()[0]); break
            except: pass

    profile_time = _extract_profile_time(ds)
    if lat is None or lon is None or profile_time is None:
        logger.debug(f"Profile {p.name} missing coordinates or time; skipping")
        return None

    temp = None
    for candidate in ['TEMP_ADJUSTED', 'TEMP', 'temp_adjusted', 'temp', 'temperature']:
        if candidate in ds.variables:
            temp = ds[candidate].values.ravel(); break
    if temp is None: return None

    temp_qc = None
    for candidate in ['TEMP_ADJUSTED_QC', 'TEMP_QC', 'temp_adjusted_qc', 'temp_qc']:
        if candidate in ds.variables: temp_qc = ds[candidate].values.ravel(); break

    pres = None
    for candidate in ['PRES_ADJUSTED', 'PRES', 'pres_adjusted', 'pres', 'pressure']:
        if candidate in ds.variables: pres = ds[candidate].values.ravel(); break
    if pres is None: return None

    pres_qc = None
    for candidate in ['PRES_ADJUSTED_QC', 'PRES_QC', 'pres_adjusted_qc', 'pres_qc']:
        if candidate in ds.variables: pres_qc = ds[candidate].values.ravel(); break

    psal = None
    for candidate in ['PSAL_ADJUSTED', 'PSAL', 'psal_adjusted', 'psal', 'salinity']:
        if candidate in ds.variables: psal = ds[candidate].values.ravel(); break

    accepted_qc = {'1', '2'}
    if temp_qc is not None:
        t_qc_chars = _decode_qc_array(temp_qc); t_good = np.isin(t_qc_chars, list(accepted_qc))
    else: t_good = ~np.isnan(temp)
    if pres_qc is not None:
        p_qc_chars = _decode_qc_array(pres_qc); p_good = np.isin(p_qc_chars, list(accepted_qc))
    else: p_good = ~np.isnan(pres)
    valid_mask = t_good & p_good & ~np.isnan(temp) & ~np.isnan(pres) & (pres >= 0)
    if np.sum(valid_mask) < 4: return None

    temp_valid = np.array(temp, dtype=float)[valid_mask]
    pres_valid = np.array(pres, dtype=float)[valid_mask]
    depths = pressure_to_depth(pres_valid, lat)
    ok = ~np.isnan(temp_valid) & ~np.isnan(depths)
    if np.sum(ok) < 4: return None
    depths = depths[ok]; temps = temp_valid[ok]
    sort_idx = np.argsort(depths); depths = depths[sort_idx]; temps = temps[sort_idx]
    depths_unique, unique_idx = np.unique(depths, return_index=True); temps_unique = temps[unique_idx]
    min_d = float(depths_unique.min()); max_d = float(depths_unique.max())
    if min_d > MAX_ALLOWED_MIN_DEPTH or max_d < MIN_REQUIRED_MAX_DEPTH: return None

    try:
        from scipy.interpolate import interp1d
        f = interp1d(depths_unique, temps_unique, bounds_error=False, fill_value=np.nan)
        interp_temps = f(TARGET_DEPTHS)
    except:
        interp_temps = np.interp(TARGET_DEPTHS, depths_unique, temps_unique, left=np.nan, right=np.nan)
    if np.isnan(interp_temps[0]) and min_d <= MAX_ALLOWED_MIN_DEPTH:
        interp_temps[0] = temps_unique[0]

    surface_salinity = np.nan
    if psal is not None:
        psal_valid = np.array(psal, dtype=float)[valid_mask][ok][sort_idx]
        valid_psal = psal_valid[~np.isnan(psal_valid)]
        if len(valid_psal) > 0: surface_salinity = float(valid_psal[0])

    core_idxs = [1, 2, 3, 4]
    if np.isnan(interp_temps[core_idxs]).any(): return None

    rec = {
        'argo_wmo': str(wmo) if wmo is not None else p.stem.split('_')[0],
        'argo_cycle': int(cycle) if cycle is not None else 0,
        'source_file': p.name,
        'profile_time': profile_time,
        'lat': lat, 'lon': lon,
        'temperature_0m': float(interp_temps[0]),
        'temperature_50m': float(interp_temps[1]),
        'temperature_100m': float(interp_temps[2]),
        'temperature_200m': float(interp_temps[3]),
        'temperature_500m': float(interp_temps[4]),
        'temperature_1000m': float(interp_temps[5]) if not np.isnan(interp_temps[5]) else np.nan,
        'salinity_0m': surface_salinity if not np.isnan(surface_salinity) else None,
        'pressure_range_dbar': f"{float(pres_valid.min()):.1f}-{float(pres_valid.max()):.1f}",
        'max_observed_depth_m': round(max_d, 1),
        'n_qc_passed_points': int(np.sum(valid_mask)),
    }
    return rec

def build_processed_table(max_files: int = 50) -> Optional[Path]:
    nc_files = sorted(RAW_DIR.glob("*.nc"))[:max_files]
    if not nc_files:
        logger.warning(f"No NetCDF files in {RAW_DIR}")
        return None
    records = []
    for nc in nc_files:
        rec = process_netcdf_profile(str(nc))
        if rec: records.append(rec)
    if not records:
        logger.warning("No profiles survived QC processing")
        return None
    df = pd.DataFrame(records)
    out = PROCESSED_DIR / 'argo_profiles.parquet'
    df.to_parquet(out, index=False)
    logger.info(f"Processed {len(records)}/{len(nc_files)} profiles -> {out}")
    return out

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build_processed_table()