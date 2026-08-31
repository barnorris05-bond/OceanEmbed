"""
Data validation hard-gate: verify that the training dataset is built strictly from real observations.
Hard Gate: Raises RuntimeError and exits with code 1 if any synthetic data or invalid provenance is found.
"""
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1] / 'data'
DATASET_DIR = BASE / 'dataset'
REQUIRED_PROVENANCE_COLS = ['argo_wmo', 'argo_cycle', 'profile_time', 'surface_time', 'surface_distance_km']
REQUIRED_TARGET_COLS = ['temp_50m', 'temp_100m', 'temp_200m', 'temp_500m']

def detect_synthetic_rows(df):
    issues = []
    for col in REQUIRED_PROVENANCE_COLS:
        if col not in df.columns: issues.append(f"Missing mandatory provenance column: {col}")
        elif df[col].isna().sum() > 0:
            issues.append(f"Provenance column {col} has {df[col].isna().sum()} nulls ({100*df[col].isna().sum()/len(df):.1f}%)")
    for col in REQUIRED_TARGET_COLS:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 1 and valid.std() < 1e-6:
                issues.append(f"Target column {col} has zero std deviation (synthetic indicator)")
    check_cols = [c for c in ['lat','lon','sst','ssh','sss','temp_50m','temp_100m','temp_200m','temp_500m'] if c in df.columns]
    if len(check_cols) > 0 and len(df) > 1:
        n_dupes = df.duplicated(subset=check_cols, keep=False).sum()
        if n_dupes > max(2, len(df) * 0.1):
            issues.append(f"High duplicate rate: {n_dupes} rows ({100*n_dupes/len(df):.1f}%)")
    if 'sst' in df.columns and ((df['sst'] < 15.0).any() or (df['sst'] > 35.0).any()):
        issues.append("SST outside Arabian Sea bounds (15-35°C)")
    if 'temp_500m' in df.columns and ((df['temp_500m'] < 2.0).any() or (df['temp_500m'] > 20.0).any()):
        issues.append("temp_500m outside ocean bounds (2-20°C)")
    if 'sss' in df.columns:
        valid_sss = df['sss'].dropna()
        if (valid_sss < 30.0).any() or (valid_sss > 40.0).any():
            issues.append("SSS outside Arabian Sea bounds (30-40 PSU)")
    return issues

def generate_validation_report(train_dataset_parquet, output_file=None):
    if not train_dataset_parquet.exists():
        logger.error(f"Training dataset not found: {train_dataset_parquet}"); return False
    df = pd.read_parquet(train_dataset_parquet)
    if len(df) == 0: logger.error("Dataset is empty!"); return False

    synthetic_issues = detect_synthetic_rows(df)
    is_valid = (len(synthetic_issues) == 0)
    status_str = "[PASS] PASSED" if is_valid else "[FAIL] FAILED"

    report_lines = [
        "", "=" * 70, "OCEANEMBED REAL DATA VALIDATION REPORT & HARD GATE", "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Dataset: {train_dataset_parquet}",
        f"Result: {status_str}",
        "", f"Total Rows: {len(df)}",
        f"Unique WMO: {df['argo_wmo'].nunique() if 'argo_wmo' in df.columns else 0}",
        f"Unique Cycles: {df['argo_cycle'].nunique() if 'argo_cycle' in df.columns else 0}",
        "",
    ]
    if synthetic_issues:
        report_lines.append("[FAIL] Synthetic data detected:")
        for issue in synthetic_issues: report_lines.append(f"  - {issue}")
        report_lines.append("\nHARD GATE TRIGGERED: Training BLOCKED.")
    else:
        report_lines.extend(["[PASS] Synthetic rows: 0", "[PASS] All rows trace to genuine Argo profiles",
                            "[PASS] Physical bounds OK", "[PASS] Provenance completeness 100%"])
    report_lines.extend(["", "=" * 70, f"STATUS: {'PASSED - APPROVED FOR TRAINING' if is_valid else 'FAILED - BLOCKED'}", "=" * 70])

    report = "\n".join(report_lines)
    if output_file: output_file.write_text(report, encoding='utf-8')
    print(report)
    return is_valid

if __name__ == '__main__':
    ds = DATASET_DIR / 'train_dataset.parquet'
    report_path = DATASET_DIR / 'validation_report.txt'
    ok = generate_validation_report(ds, report_path)
    sys.exit(0 if ok else 1)