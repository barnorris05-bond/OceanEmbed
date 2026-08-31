"""
Argo fetch helper.
Uses Argopy to discover and download canonical Argo profile NetCDFs into data/raw/argo/.
Downloads genuine Argo observations (PRES, TEMP, PSAL, QC flags, WMO, CYCLE, TIME, LAT, LON)
via authoritative GDAC / ERDDAP servers.
"""
from pathlib import Path
import logging
import os
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)
RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'argo'
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_profiles_bbox(start_date: str, end_date: str,
                        min_lat: float = 8.0, max_lat: float = 24.0,
                        min_lon: float = 60.0, max_lon: float = 77.0,
                        max_profiles: int = 50) -> List[str]:
    """Fetch Argo profiles for a bounding box and date range using the Argopy API."""
    try:
        import argopy
        from argopy import DataFetcher
    except Exception as e:
        logger.error(f"argopy is not available: {e}")
        return []

    logger.info(f"Querying Argo index for box: Lon [{min_lon}, {max_lon}], Lat [{min_lat}, {max_lat}], Dates: {start_date} to {end_date}")
    box = [min_lon, max_lon, min_lat, max_lat, 0.0, 1000.0, start_date, end_date]
    try:
        fetcher = DataFetcher(src='erddap')
        idx_df = fetcher.region(box).to_index()
    except Exception as e:
        logger.error(f"Failed to fetch Argo index for region: {e}")
        return []

    if idx_df is None or len(idx_df) == 0:
        logger.warning("Argopy returned zero profiles for the requested bbox/date range")
        return []

    idx_df.columns = [c.lower() for c in idx_df.columns]
    total_discovered = len(idx_df)
    unique_wmos_disc = idx_df['wmo'].nunique() if 'wmo' in idx_df else 0
    logger.info(f"Discovered {total_discovered} real Argo profiles across {unique_wmos_disc} unique floats in region.")

    if len(idx_df) > max_profiles:
        selected_rows = []
        grouped = idx_df.groupby('wmo')
        wmo_list = list(grouped.groups.keys())
        wmo_idx = 0
        cycle_indices = {w: 0 for w in wmo_list}
        while len(selected_rows) < max_profiles:
            w = wmo_list[wmo_idx % len(wmo_list)]
            w_group = grouped.get_group(w)
            c_idx = cycle_indices[w]
            if c_idx < len(w_group):
                selected_rows.append(w_group.iloc[c_idx])
                cycle_indices[w] += 1
            wmo_idx += 1
            if all(cycle_indices[w] >= len(grouped.get_group(w)) for w in wmo_list):
                break
        selected_df = pd.DataFrame(selected_rows)
    else:
        selected_df = idx_df.copy()

    logger.info(f"Selected {len(selected_df)} profiles across {selected_df['wmo'].nunique()} floats for download.")
    saved_profiles = []
    wmos_fetched = set()
    cycles_fetched = set()

    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        try:
            wmo = int(row['wmo'])
            cyc = int(row['cyc'])
            out_file = RAW_DIR / f"WMO{wmo}_CYC{cyc}.nc"
            if out_file.exists() and out_file.stat().st_size > 500:
                logger.debug(f"Profile WMO {wmo} Cyc {cyc} already cached at {out_file.name}")
                saved_profiles.append(str(out_file))
                wmos_fetched.add(wmo)
                cycles_fetched.add((wmo, cyc))
                continue
            logger.info(f"[{i}/{len(selected_df)}] Fetching Argo profile WMO {wmo} Cycle {cyc}...")
            p_fetcher = DataFetcher(src='erddap')
            p_ds = p_fetcher.profile(wmo, cyc).to_xarray()
            if p_ds is not None and 'PRES' in p_ds and 'TEMP' in p_ds:
                p_ds.to_netcdf(out_file)
                saved_profiles.append(str(out_file))
                wmos_fetched.add(wmo)
                cycles_fetched.add((wmo, cyc))
                logger.info(f"Saved {out_file.name} ({out_file.stat().st_size / 1024:.1f} KB)")
            else:
                logger.warning(f"Profile WMO {wmo} Cycle {cyc} missing required variables; skipped")
        except Exception as e:
            logger.warning(f"Failed to fetch profile WMO {row.get('wmo')} Cycle {row.get('cyc')}: {e}")

    logger.info("=" * 60)
    logger.info(f"ARGO FETCH COMPLETE:")
    logger.info(f"  Profiles discovered: {total_discovered}")
    logger.info(f"  Profiles selected: {len(selected_df)}")
    logger.info(f"  Profiles saved: {len(saved_profiles)}")
    logger.info(f"  Unique WMO floats: {len(wmos_fetched)}")
    logger.info(f"  Unique cycles: {len(cycles_fetched)}")
    logger.info(f"  Saved directory: {RAW_DIR}")
    logger.info("=" * 60)
    return saved_profiles

if __name__ == '__main__':
    import datetime
    logging.basicConfig(level=logging.INFO)
    profiles = fetch_profiles_bbox(
        '2026-01-01', '2026-02-01',
        min_lon=68.0, max_lon=72.0,
        min_lat=12.0, max_lat=16.0,
        max_profiles=5
    )
    print(f"Fetched {len(profiles)} profiles")