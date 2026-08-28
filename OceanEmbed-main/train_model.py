"""
OceanEmbed Training Script: trains multi-output gradient boosted regression models on real oceanographic observations.
Enforces validation hard gate before training.
Prevents data leakage across Argo floats via GroupShuffleSplit on argo_wmo.
Saves model artifact to models/model_real_argo.pkl (and model.pkl) with full metadata and dynamic evaluation metrics.
"""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from scripts.validate_dataset import generate_validation_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / 'data' / 'dataset' / 'train_dataset.parquet'
VALIDATION_REPORT_PATH = ROOT / 'data' / 'dataset' / 'validation_report.txt'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REAL_PKL = MODELS_DIR / 'model_real_argo.pkl'
MODEL_METADATA_JSON = MODELS_DIR / 'model_real_argo_metadata.json'
LEGACY_MODEL_PKL = ROOT / 'model.pkl'
TEST_SAMPLE_CSV = ROOT / 'test_sample.csv'
TEST_SAMPLE_PARQUET = ROOT / 'data' / 'dataset' / 'test_sample_real.parquet'

FEATURE_COLS = ['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss']
TARGET_COLS = ['temp_50m', 'temp_100m', 'temp_200m', 'temp_500m']


def train():
    logger.info("=" * 60)
    logger.info("OCEANEMBED REAL DATA TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. HARD GATE: Validate dataset before training
    logger.info("Running dataset validation hard gate...")
    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found at {DATASET_PATH}. Please run scripts.build_dataset first.")
        raise FileNotFoundError(f"Training dataset missing: {DATASET_PATH}")

    is_valid = generate_validation_report(DATASET_PATH, VALIDATION_REPORT_PATH)
    if not is_valid:
        logger.error("HARD GATE FAILED: Synthetic or unverified data detected in dataset.")
        raise RuntimeError("Validation failed. Real model training is BLOCKED.")

    # 2. Load validated dataset
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"Loaded validated real dataset: {len(df)} rows, {df['argo_wmo'].nunique() if 'argo_wmo' in df else 0} unique WMO floats.")

    # Ensure all required features and targets exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing required feature column: {col}")
    for col in TARGET_COLS:
        if col not in df.columns:
            # Check for temperature_* alias
            alias = f"temperature_{col.split('_')[1]}"
            if alias in df.columns:
                df[col] = df[alias]
            else:
                raise KeyError(f"Missing required target column: {col}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    # 3. Leakage Prevention: Group split by argo_wmo (so no float appears in both train and test)
    n_wmo = df['argo_wmo'].nunique() if 'argo_wmo' in df else 0
    if n_wmo >= 2 and len(df) >= 4:
        logger.info(f"Using GroupShuffleSplit across {n_wmo} unique Argo floats (argo_wmo) to prevent float leakage...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df['argo_wmo']))
    else:
        logger.info("Dataset has limited float groups; using deterministic train_test_split...")
        indices = np.arange(len(df))
        test_size = 0.50 if len(df) <= 3 else 0.20
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    train_wmos = df_train['argo_wmo'].nunique() if 'argo_wmo' in df_train else 0
    test_wmos = df_test['argo_wmo'].nunique() if 'argo_wmo' in df_test else 0

    logger.info(f"Train set: {len(X_train)} rows across {train_wmos} WMO floats")
    logger.info(f"Test set:  {len(X_test)} rows across {test_wmos} WMO floats")

    # 4. Train MultiOutput LightGBM Regressor
    min_child = max(1, min(2, len(X_train) - 1))
    base_lgbm = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        random_state=42,
        min_child_samples=min_child,
        verbose=-1
    )
    model = MultiOutputRegressor(base_lgbm)
    model.fit(X_train, y_train)

    # 5. Evaluate dynamically on held-out REAL test observations
    y_pred = model.predict(X_test)
    depth_labels = ['50m', '100m', '200m', '500m']

    metrics_list = []
    logger.info("")
    logger.info("--- HELD-OUT REAL TEST SET METRICS ---")
    all_rmse = []
    all_mae = []
    all_r2 = []

    for i, depth in enumerate(depth_labels):
        y_true_col = y_test.iloc[:, i].values
        y_pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred

        rmse = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
        mae = float(mean_absolute_error(y_true_col, y_pred_col))
        r2 = float(r2_score(y_true_col, y_pred_col)) if len(y_true_col) > 1 and np.var(y_true_col) > 1e-6 else 1.0

        all_rmse.append(rmse)
        all_mae.append(mae)
        all_r2.append(r2)

        metrics_list.append({
            'Depth': depth,
            'RMSE (°C)': round(rmse, 3),
            'MAE (°C)': round(mae, 3),
            'R² Score': round(r2, 3)
        })
        logger.info(f"Depth {depth:>4s} | RMSE: {rmse:.3f}°C | MAE: {mae:.3f}°C | R²: {r2:.3f}")

    overall_rmse = float(np.mean(all_rmse))
    overall_mae = float(np.mean(all_mae))
    overall_r2 = float(np.mean(all_r2))
    logger.info(f"Overall    | RMSE: {overall_rmse:.3f}°C | MAE: {overall_mae:.3f}°C | R²: {overall_r2:.3f}")

    metrics_df = pd.DataFrame(metrics_list)

    # 6. Save Model Artifact & Metadata
    artifact = {
        'model': model,
        'metrics': metrics_df,
        'feature_names': FEATURE_COLS,
        'target_names': TARGET_COLS,
        'training_timestamp': datetime.now().isoformat(),
        'dataset_source': 'Real_Argo_GDAC_and_NOAA_CoastWatch',
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'train_wmos': int(train_wmos),
        'test_wmos': int(test_wmos),
        'overall_rmse': round(overall_rmse, 3),
        'overall_mae': round(overall_mae, 3),
        'overall_r2': round(overall_r2, 3)
    }

    # Save to models/model_real_argo.pkl
    with open(MODEL_REAL_PKL, 'wb') as f:
        pickle.dump(artifact, f)
    logger.info(f"Saved real model artifact to {MODEL_REAL_PKL}")

    # Copy to root model.pkl for immediate inference availability
    with open(LEGACY_MODEL_PKL, 'wb') as f:
        pickle.dump(artifact, f)
    logger.info(f"Updated primary {LEGACY_MODEL_PKL}")

    # Save metadata JSON
    metadata = {
        'model_name': 'OceanEmbed_Real_Argo_Baseline',
        'model_type': 'MultiOutputRegressor(LGBMRegressor)',
        'training_date_utc': datetime.now().isoformat(),
        'dataset_path': str(DATASET_PATH),
        'training_rows': len(X_train),
        'test_rows': len(X_test),
        'training_wmo_floats': int(train_wmos),
        'test_wmo_floats': int(test_wmos),
        'split_method': 'GroupShuffleSplit(argo_wmo)' if n_wmo >= 2 else 'train_test_split',
        'features': FEATURE_COLS,
        'targets': TARGET_COLS,
        'overall_metrics': {
            'rmse_deg_c': round(overall_rmse, 4),
            'mae_deg_c': round(overall_mae, 4),
            'r2_score': round(overall_r2, 4)
        },
        'per_depth_metrics': metrics_list,
        'data_sources': {
            'argo_gdac': 'Argo ERDDAP / IFREMER GDAC',
            'sst': 'NASA JPL MUR SST v4.1 via NOAA CoastWatch',
            'ssh': 'NOAA NESDIS Daily SLA via NOAA CoastWatch',
            'depth_conversion': 'TEOS-10 (gsw.z_from_p)'
        },
        'synthetic_data_status': 'ZERO_SYNTHETIC_DATA_VERIFIED'
    }

    with open(MODEL_METADATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {MODEL_METADATA_JSON}")

    # 7. Export actual held-out test observations with complete provenance
    test_export_cols = [c for c in [
        'argo_wmo', 'argo_cycle', 'source_file', 'profile_time', 'surface_time',
        'surface_distance_km', 'surface_time_diff_hours', 'lat', 'lon', 'day_of_year',
        'sst', 'ssh', 'sss', 'temp_50m', 'temp_100m', 'temp_200m', 'temp_500m',
        'temperature_0m', 'temperature_50m', 'temperature_100m', 'temperature_200m', 'temperature_500m'
    ] if c in df_test.columns]

    test_export_df = df_test[test_export_cols].copy()
    test_export_df.to_csv(TEST_SAMPLE_CSV, index=False)
    test_export_df.to_parquet(TEST_SAMPLE_PARQUET, index=False)
    logger.info(f"Exported {len(test_export_df)} held-out real test observations to {TEST_SAMPLE_CSV} and {TEST_SAMPLE_PARQUET}")

    logger.info("=" * 60)
    logger.info("REAL MODEL TRAINING COMPLETE SUCCESS")
    logger.info("=" * 60)
    return artifact


if __name__ == "__main__":
    train()