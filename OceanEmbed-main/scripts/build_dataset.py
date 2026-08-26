"""
Orchestration script: run argo fetch (optional), preprocess, surface matching, and write final dataset.
Usage: python scripts/build_dataset.py --use-raw
"""
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('build_dataset')

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / 'argo'


def main(use_raw=False, max_profiles=500):
    # 1. fetch raw argo (optional)
    if use_raw:
        from data.argo_fetch import fetch_profiles_bbox
        import datetime
        profiles = fetch_profiles_bbox((datetime.date.today() - datetime.timedelta(days=365)).isoformat(), datetime.date.today().isoformat(), max_profiles=max_profiles)
        logger.info(f"Fetched {len(profiles)} raw profiles")

    # 2. preprocess
    from data.argo_preprocess import build_processed_table
    processed = build_processed_table(max_files=max_profiles)
    if not processed:
        logger.error('No processed profiles; aborting')
        return

    # 3. match surfaces
    from data.match_surface_to_argo import match_surface_features
    dataset = match_surface_features(processed)
    if not dataset:
        logger.error('Surface matching produced no dataset')
        return

    logger.info('Dataset build complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-raw', action='store_true', help='Try to fetch raw Argo profiles first')
    parser.add_argument('--max-profiles', type=int, default=500, help='Maximum number of raw profiles to process')
    args = parser.parse_args()
    main(use_raw=args.use_raw, max_profiles=args.max_profiles)
