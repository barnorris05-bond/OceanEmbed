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


def main(use_raw=False, max_profiles=500, skip_validation=False):
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
    
    # 4. VALIDATE DATASET (critical for real-data pipeline)
    if not skip_validation:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING DATA VALIDATION (required before training)")
        logger.info("=" * 60)
        from scripts.validate_dataset import generate_validation_report
        from pathlib import Path
        dataset_path = ROOT / 'data' / 'dataset' / 'train_dataset.parquet'
        output_path = ROOT / 'data' / 'dataset' / 'validation_report.txt'
        
        is_valid = generate_validation_report(dataset_path, output_path)
        
        if not is_valid:
            logger.error("")
            logger.error("❌ DATASET VALIDATION FAILED")
            logger.error("Synthetic data detected in training dataset.")
            logger.error("BLOCKING: You cannot train model.pkl on synthetic/fabricated data.")
            logger.error(f"See {output_path} for details.")
            raise RuntimeError("Synthetic data detected. Training blocked.")
        else:
            logger.info("")
            logger.info("✅ DATASET VALIDATION PASSED")
            logger.info("Dataset contains real observations. Safe to proceed to training.")
            logger.info(f"Validation report: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-raw', action='store_true', help='Try to fetch raw Argo profiles first')
    parser.add_argument('--max-profiles', type=int, default=500, help='Maximum number of raw profiles to process')
    parser.add_argument('--skip-validation', action='store_true', help='Skip data validation (NOT recommended for real data)')
    args = parser.parse_args()
    main(use_raw=args.use_raw, max_profiles=args.max_profiles, skip_validation=args.skip_validation)
