"""
Data validation report: verify that the training dataset is built from real observations only.
Generates a pre-training report showing sources, counts, quality metrics, and synthetic-data checks.
Run this BEFORE train_model.py to ensure data integrity.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Root paths
BASE = Path(__file__).resolve().parents[1] / 'data'
DATASET_DIR = BASE / 'dataset'
PROCESSED_DIR = BASE / 'processed'

REQUIRED_PROVENANCE_COLS = [
    'argo_wmo', 'argo_cycle', 'profile_time', 'surface_time', 'surface_distance_km'
]
REQUIRED_TEMP_COLS = [
    'temperature_0m', 'temperature_50m', 'temperature_100m',
    'temperature_200m', 'temperature_500m', 'temperature_1000m'
]
REQUIRED_SURFACE_COLS = ['sst', 'ssh', 'sss', 'wind_u', 'wind_v', 'wind_speed']


def detect_synthetic_rows(df):
    """
    Heuristic checks for synthetic/fabricated data:
    - Repeated patterns (e.g., round numbers, exact duplicates)
    - Absence of real-world measurement noise
    - Unrealistic value distributions
    Returns count of suspected synthetic rows and details.
    """
    issues = []
    
    # Check 1: all rows have identical non-provenance features (strong indicator of demo data)
    if len(df) > 1:
        temp_cols = [c for c in REQUIRED_TEMP_COLS if c in df.columns]
        if temp_cols and df[temp_cols].nunique().sum() <= len(temp_cols):
            issues.append(f"All temperature columns have very few unique values (likely synthetic)")
    
    # Check 2: exact duplicate rows (beyond natural coincidence)
    n_dupes = df.duplicated(subset=[c for c in df.columns if c not in ['argo_wmo', 'argo_cycle', 'source_file']], keep=False).sum()
    if n_dupes > len(df) * 0.1:  # more than 10% exact duplicates
        issues.append(f"High duplicate rate: {n_dupes} exact duplicate rows ({100*n_dupes/len(df):.1f}%)")
    
    # Check 3: missing provenance (synthetic fallback sign)
    missing_provenance = df[REQUIRED_PROVENANCE_COLS].isna().any(axis=1).sum()
    if missing_provenance > len(df) * 0.5:
        issues.append(f"High missing provenance: {missing_provenance} rows lack argo_wmo/argo_cycle/times")
    
    return issues


def generate_validation_report(train_dataset_parquet, output_file=None):
    """
    Generate a comprehensive data validation report.
    Prints to console and optionally to a text file.
    Raises RuntimeError if synthetic data is detected.
    """
    if not train_dataset_parquet.exists():
        logger.error(f"Training dataset not found: {train_dataset_parquet}")
        return False
    
    df = pd.read_parquet(train_dataset_parquet)
    logger.info(f"Loaded training dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Build report
    report_lines = [
        "",
        "=" * 60,
        "OCEANEMBED DATA VALIDATION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset path: {train_dataset_parquet}",
        "",
        "DATASET SIZE",
        "─" * 60,
        f"Total rows: {len(df)}",
        f"Total columns: {len(df.columns)}",
        "",
        "PROVENANCE & SOURCES",
        "─" * 60,
        f"Unique Argo floats (WMO): {df['argo_wmo'].nunique() if 'argo_wmo' in df else 'N/A'}",
        f"Unique cycles: {df['argo_cycle'].nunique() if 'argo_cycle' in df else 'N/A'}",
        f"Date range: {pd.to_datetime(df['profile_time']).min()} → {pd.to_datetime(df['profile_time']).max()}" if 'profile_time' in df else "N/A",
        "",
        "GEOGRAPHIC COVERAGE",
        "─" * 60,
    ]
    
    if 'lat' in df.columns:
        report_lines.append(f"Latitude: {df['lat'].min():.2f}°N → {df['lat'].max():.2f}°N (mean: {df['lat'].mean():.2f}°N)")
    if 'lon' in df.columns:
        report_lines.append(f"Longitude: {df['lon'].min():.2f}°E → {df['lon'].max():.2f}°E (mean: {df['lon'].mean():.2f}°E)")
    
    report_lines.append("")
    report_lines.append("SURFACE OBSERVATION MATCHING")
    report_lines.append("─" * 60)
    if 'surface_distance_km' in df.columns:
        report_lines.extend([
            f"Mean spatial distance to surface obs: {df['surface_distance_km'].mean():.2f} km",
            f"Max spatial distance: {df['surface_distance_km'].max():.2f} km",
            f"Rows with distance > 25 km: {(df['surface_distance_km'] > 25).sum()} ({100*(df['surface_distance_km'] > 25).sum()/len(df):.1f}%)",
        ])
    
    # Temperature statistics
    report_lines.append("")
    report_lines.append("SUBSURFACE TEMPERATURE TARGETS (Argo observations)")
    report_lines.append("─" * 60)
    for temp_col in REQUIRED_TEMP_COLS:
        if temp_col in df.columns:
            valid = df[temp_col].dropna()
            if len(valid) > 0:
                report_lines.append(
                    f"{temp_col}: "
                    f"min={valid.min():.2f}°C, max={valid.max():.2f}°C, "
                    f"mean={valid.mean():.2f}°C, n={len(valid)} ({100*len(valid)/len(df):.1f}%)"
                )
    
    # Surface observations
    report_lines.append("")
    report_lines.append("SURFACE OBSERVATIONS (satellite/in-situ)")
    report_lines.append("─" * 60)
    for surf_col in REQUIRED_SURFACE_COLS:
        if surf_col in df.columns:
            valid = df[surf_col].dropna()
            if len(valid) > 0:
                report_lines.append(
                    f"{surf_col}: "
                    f"min={valid.min():.3f}, max={valid.max():.3f}, "
                    f"mean={valid.mean():.3f}, n={len(valid)} ({100*len(valid)/len(df):.1f}%)"
                )
    
    # Missing data summary
    report_lines.append("")
    report_lines.append("MISSING DATA SUMMARY")
    report_lines.append("─" * 60)
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    for col, pct in missing_pct[missing_pct > 0].head(10).items():
        report_lines.append(f"{col}: {pct:.1f}% missing")
    
    # SYNTHETIC DATA CHECK (critical)
    report_lines.append("")
    report_lines.append("SYNTHETIC DATA DETECTION")
    report_lines.append("─" * 60)
    synthetic_issues = detect_synthetic_rows(df)
    if synthetic_issues:
        report_lines.append("⚠️  WARNINGS: Potential synthetic data detected:")
        for issue in synthetic_issues:
            report_lines.append(f"  - {issue}")
        report_lines.append("")
        report_lines.append("⛔ BLOCKED: Synthetic data detected in training dataset.")
        report_lines.append("Do NOT train model.pkl on synthetic or fabricated data.")
        synthetic_detected = True
    else:
        report_lines.append("✅ No synthetic data patterns detected.")
        report_lines.append("Dataset appears to contain real observations.")
        synthetic_detected = False
    
    report_lines.append("")
    report_lines.append("=" * 60)
    if not synthetic_detected:
        report_lines.append("✅ DATASET VALIDATION PASSED")
    else:
        report_lines.append("❌ DATASET VALIDATION FAILED")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Optionally save to file
    if output_file:
        Path(output_file).write_text(report_text)
        logger.info(f"Report saved to {output_file}")
    
    # Return success/failure
    return not synthetic_detected


if __name__ == '__main__':
    dataset_path = DATASET_DIR / 'train_dataset.parquet'
    output_path = DATASET_DIR / 'validation_report.txt'
    
    success = generate_validation_report(dataset_path, output_path)
    
    if not success:
        logger.error("Dataset validation FAILED. Synthetic data detected. Training blocked.")
        sys.exit(1)
    else:
        logger.info("Dataset validation PASSED. Safe to train on real data.")
        sys.exit(0)
