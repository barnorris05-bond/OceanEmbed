"""
OceanEmbed End-to-End Real Data Pipeline Orchestrator.
Usage: python -m scripts.build_dataset --use-raw --max-profiles 50
"""
import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('build_dataset')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / 'data' / 'dataset'
DOCS_DIR = ROOT / 'docs'
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def generate_datasheet(dataset_path, output_md, is_valid):
    df = pd.read_parquet(dataset_path)
    md = f"""# OceanEmbed Real Oceanographic Dataset Datasheet
> Scientific Integrity Certification: Every value computed from actual observations.
## Dataset Overview
- Rows: {len(df)}
- Unique Argo Floats: {df['argo_wmo'].nunique() if 'argo_wmo' in df.columns else 0}
- Validation: {'PASS' if is_valid else 'FAIL'}
## Surface Inputs
- SST: {df['sst'].min():.3f} – {df['sst'].max():.3f} °C
- SSH: {df['ssh'].min():.3f} – {df['ssh'].max():.3f} m
- SSS: {df['sss'].min():.3f} – {df['sss'].max():.3f} PSU
"""
    output_md.write_text(md, encoding='utf-8')
    logger.info(f"Generated datasheet: {output_md}")

def main(use_raw=False, max_profiles=50, start_date='2025-01-01', end_date='2026-02-01'):
    logger.info("STARTING OCEANEMBED REAL DATA PIPELINE")
    if use_raw:
        from data.argo_fetch import fetch_profiles_bbox
        logger.info(f"Step 1: Fetching {max_profiles} Argo profiles...")
        profiles = fetch_profiles_bbox(start_date, end_date, max_profiles=max_profiles)
        logger.info(f"Step 1 Complete: {len(profiles)} raw files.")

    from data.argo_preprocess import build_processed_table
    logger.info("Step 2: Preprocessing Argo profiles...")
    processed = build_processed_table(max_files=max_profiles)
    if not processed: logger.error("Step 2 Failed"); return

    from data.match_surface_to_argo import match_surface_features
    logger.info("Step 3: Matching surface observations...")
    dataset = match_surface_features(processed)
    if not dataset: logger.error("Step 3 Failed"); return

    logger.info("Step 4: Running validation hard-gate...")
    from scripts.validate_dataset import generate_validation_report
    is_valid = generate_validation_report(dataset, DATASET_DIR / 'validation_report.txt')
    if not is_valid:
        raise RuntimeError("Validation FAILED. Training blocked.")

    generate_datasheet(dataset, DOCS_DIR / 'REAL_DATA_DATASHEET.md', is_valid)
    logger.info("PIPELINE COMPLETE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-raw', action='store_true')
    parser.add_argument('--max-profiles', type=int, default=50)
    parser.add_argument('--start-date', default='2025-01-01')
    parser.add_argument('--end-date', default='2026-02-01')
    args = parser.parse_args()
    main(args.use_raw, args.max_profiles, args.start_date, args.end_date)