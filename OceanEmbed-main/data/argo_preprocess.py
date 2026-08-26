"""
Argo preprocessing: read raw NetCDFs (Argopy/xarray), apply Argo QC filtering, convert pressure->depth using gsw,
interpolate temperature onto fixed depth levels and produce a parquet table of profiles.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
RAW_DIR = BASE / 'raw' / 'argo'
PROCESSED_DIR = BASE / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS = np.array([0, 50, 100, 200, 500, 1000], dtype=float)
# Require profile to cover at least surface and 500m for inclusion. 1000m is recorded when present.
MIN_REQUIRED_MAX_DEPTH = 500.0


def pressure_to_depth(p, lat):
    """Convert pressure (dbar) to depth (m) using gsw when available, else simple approximation.
    p: array-like pressures
    lat: scalar latitude
    """
    try:
        import gsw
        z = gsw.z_from_p(p, lat)  # returns negative z
        return -z
    except Exception:
        # fallback with warning
        logger.debug("gsw not available; using 1 dbar ~= 1 m approximation for pressure->depth")
        return np.array(p, dtype=float)


def _decode_qc_array(qc_arr):
    """Normalize QC array entries to single-char strings like '0'..'9'"""
    out = []
    for x in qc_arr:
        if isinstance(x, (bytes, np.bytes_)):
            try:
                out.append(x.decode('ascii'))
            except Exception:
                out.append(str(x))
        else:
            out.append(str(x))
    return np.array(out)


def _extract_profile_time(ds):
    """Try common Argo time variables and convert to pandas.Timestamp when possible."""
    for candidate in ['JULD', 'juld', 'JULD_LOCATION', 'TIME', 'time', 'PROFILE_TIME', 'DATE']:
        if candidate in ds.variables:
            try:
                tval = ds[candidate].values.ravel()[0]
                # xarray/cftime may produce numpy.datetime64 or cftime; use pandas to convert
                import pandas as pd
                return pd.to_datetime(tval)
            except Exception:
                continue
    # try global attribute or return None
    if 'time' in ds.coords:
        try:
            import pandas as pd
            return pd.to_datetime(ds['time'].values.ravel()[0])
        except Exception:
            return None
    return None


def process_netcdf_profile(nc_path):
    """Read one NetCDF profile and return dict with metadata + interpolated temps or None if fails QC."""
    import xarray as xr
    p = Path(nc_path)
    try:
        ds = xr.open_dataset(str(p))
    except Exception as e:
        logger.warning(f"Failed to open {p}: {e}")
        return None

    # Temperature variable selection (prefer adjusted measurements)
    if 'TEMP_ADJUSTED' in ds:
        temp = ds['TEMP_ADJUSTED'].values
    elif 'TEMP' in ds:
        temp = ds['TEMP'].values
    elif 'temperature' in ds:
        temp = ds['temperature'].values
    elif 'temp' in ds:
        temp = ds['temp'].values
    else:
        logger.warning(f"No temperature variable found in {p}")
        return None

    # QC flags (attempt multiple names)
    qc = None
    if 'TEMP_QC' in ds:
        qc = ds['TEMP_QC'].values
    elif 'TEMP_ADJUSTED_QC' in ds:
        qc = ds['TEMP_ADJUSTED_QC'].values

    # Pressure variable
    pres = None
    for candidate in ['PRES', 'pres', 'pressure']:
        if candidate in ds.variables:
            pres = ds[candidate].values
            break
    if pres is None:
        logger.warning(f"No pressure variable found in {p}")
        return None

    # lat/lon
    lat = None
    lon = None
    if 'LATITUDE' in ds:
        lat = float(ds['LATITUDE'].values.ravel()[0])
    elif 'latitude' in ds:
        lat = float(ds['latitude'].values.ravel()[0])
    if 'LONGITUDE' in ds:
        lon = float(ds['LONGITUDE'].values.ravel()[0])
    elif 'longitude' in ds:
        lon = float(ds['longitude'].values.ravel()[0])

    # profile time
    profile_time = _extract_profile_time(ds)

    # Apply QC: decode and accept flags '0','1','2' (adjustable)
    if qc is not None:
        qchars = _decode_qc_array(qc)
        accepted = set(['0', '1', '2'])
        good_mask = np.isin(qchars, list(accepted))
    else:
        # no QC available: treat non-nan temps as usable but log
        good_mask = ~np.isnan(temp)

    temp_valid = np.array(temp)[good_mask]
    pres_valid = np.array(pres)[good_mask]

    if len(temp_valid) < 4:
        logger.debug(f"Profile {p} has too few valid points after QC: {len(temp_valid)}")
        return None

    depths = pressure_to_depth(pres_valid, lat)
    ok = ~np.isnan(temp_valid) & ~np.isnan(depths)
    if ok.sum() < 4:
        return None

    depths = depths[ok]
    temps = temp_valid[ok]

    idx = np.argsort(depths)
    depths = depths[idx]
    temps = temps[idx]

    # Check coverage: require surface (~0-5m) and at least ~500m max depth to include profile
    if depths.min() > 5.0 or depths.max() < (MIN_REQUIRED_MAX_DEPTH - 50.0):
        logger.debug(f"Profile {p} depth coverage insufficient: {depths.min():.1f}-{depths.max():.1f} m")
        return None

    # Interpolate without extrapolation: points outside observed range become NaN
    try:
        from scipy.interpolate import interp1d
        f = interp1d(depths, temps, bounds_error=False, fill_value=np.nan)
        interp_temps = f(TARGET_DEPTHS)
    except Exception:
        # fallback to numpy.interp but warn
        logger.warning("scipy.interpolate not available; using numpy.interp fallback (may extrapolate)")
        interp_temps = np.interp(TARGET_DEPTHS, depths, temps, left=np.nan, right=np.nan)

    # Require that the core targets (0,50,100,200,500) have no more than 1 NaN among them
    core_idxs = [0, 1, 2, 3, 4]
    core_nan = np.isnan(interp_temps[core_idxs]).sum()
    if core_nan > 1:
        logger.debug(f"Profile {p} missing core target depths: {core_nan} NaNs")
        return None

    rec = {
        'source_file': str(p.name),
        'lat': float(lat),
        'lon': float(lon),
        'profile_time': profile_time,
        'temperature_0m': float(interp_temps[0]) if not np.isnan(interp_temps[0]) else np.nan,
        'temperature_50m': float(interp_temps[1]) if not np.isnan(interp_temps[1]) else np.nan,
        'temperature_100m': float(interp_temps[2]) if not np.isnan(interp_temps[2]) else np.nan,
        'temperature_200m': float(interp_temps[3]) if not np.isnan(interp_temps[3]) else np.nan,
        'temperature_500m': float(interp_temps[4]) if not np.isnan(interp_temps[4]) else np.nan,
        'temperature_1000m': float(interp_temps[5]) if not np.isnan(interp_temps[5]) else np.nan
    }
    return rec


def build_processed_table(max_files=1000):
    """Process files found in data/raw/argo and write data/processed/argo_profiles.parquet"""
    files = list(RAW_DIR.glob('*'))[:max_files]
    rows = []
    for f in files:
        out = process_netcdf_profile(str(f))
        if out is not None:
            rows.append(out)
    if not rows:
        logger.warning("No processed profiles produced")
        return None
    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / 'argo_profiles.parquet'
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved processed profiles to {out_path} ({len(df)} rows)")
    return out_path


if __name__ == '__main__':
    build_processed_table(max_files=200)
