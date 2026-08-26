"""
Argo fetch helper.
Attempts to use Argopy to download canonical Argo profile NetCDFs into data/raw/argo/.
On failure the function returns an empty list and logs a clear error (no silent JSON fallbacks).
"""
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)
RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'argo'
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_profiles_bbox(start_date, end_date, min_lat=8.0, max_lat=24.0, min_lon=60.0, max_lon=77.0, max_profiles=1000):
    """Fetch Argo profiles for a bounding box and date range.
    Returns list of local NetCDF file paths. On any error or if Argopy is unavailable, returns an empty list
    and logs reasons. Does NOT write synthetic or JSON fallback files into the raw folder.
    """
    try:
        import argopy
    except Exception as e:
        logger.error(f"argopy is not available: {e}")
        return []

    try:
        # Use argopy DataFetcher in a defensive way
        from argopy import DataFetcher
        client = DataFetcher()
        # region expects [west,east,south,north]
        ds = client.region([min_lon, max_lon, min_lat, max_lat], start_date, end_date)
        profiles = []
        # ds.profile is an iterable of profile objects in argopy; be defensive about attributes
        for i, p in enumerate(ds.profile):
            if i >= max_profiles:
                break
            try:
                wmo = getattr(p, 'wmo', None) or getattr(p, 'WMO', None) or i
                out = RAW_DIR / f"profile_{wmo}.nc"
                # p.ds is the xarray Dataset for this profile
                if hasattr(p, 'ds') and p.ds is not None:
                    p.ds.to_netcdf(out)
                    profiles.append(str(out))
                else:
                    logger.warning(f"Profile {i} has no .ds dataset in argopy object; skipping")
            except Exception as ee:
                logger.warning(f"Failed to save profile {i}: {ee}")
        logger.info(f"Argopy fetched {len(profiles)} profiles")
        if len(profiles) == 0:
            logger.warning("Argopy returned zero profiles for the requested bbox/date range")
        return profiles
    except Exception as e:
        logger.error(f"Argopy fetch failed: {e}")
        return []


if __name__ == '__main__':
    import datetime
    profiles = fetch_profiles_bbox((datetime.date.today() - datetime.timedelta(days=365)).isoformat(), datetime.date.today().isoformat(), max_profiles=50)
    print(f"Fetched {len(profiles)} profiles")
