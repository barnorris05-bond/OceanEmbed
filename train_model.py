"""
OceanEmbed model training.

DATA INTEGRITY RULE (see docs/DATA_INTEGRITY.md):
  This script will NOT silently fall back to synthetic/demo data.
  By default it requires the validated real dataset produced by the
  real-data pipeline:

      python scripts/build_dataset.py --use-raw --max-profiles 200
      # -> data/dataset/train_dataset.parquet  (only exists if
      #    scripts/validate_dataset.py passed the hard gate)

  If that file is missing, training stops with instructions instead
  of quietly using fabricated numbers.

  For local development/UI testing ONLY, you may explicitly pass
  --demo to train on synthetic data. This writes a SEPARATELY NAMED
  artifact (model_demo.pkl) so it can never be mistaken for, or
  silently overwrite, a model trained on real observations.

Usage:
    python train_model.py                # real data (default, safe)
    python train_model.py --demo         # explicit synthetic/demo model
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent
REAL_DATASET = ROOT / "data" / "dataset" / "train_dataset.parquet"
DEMO_DATASET = ROOT / "data" / "demo" / "ocean_data_synthetic.csv"

FEATURES = ["lat", "lon", "day_of_year", "sst", "ssh", "sss"]
TARGETS = ["temp_50m", "temp_100m", "temp_200m", "temp_500m"]


def _fit_and_report(X_train, X_test, y_train, y_test):
    # LightGBM's default min_child_samples (20) silently refuses to split
    # further once a leaf has fewer than 20 rows -- fine for a large
    # dataset, but with a small real-observation dataset (a handful of
    # Argo floats) it means the model barely learns anything and every
    # split gets skipped. Scale it down to the training set size instead,
    # same fix used for the small real dataset elsewhere in this project.
    min_child = max(1, min(2, len(X_train) - 1))
    base = LGBMRegressor(n_estimators=200, learning_rate=0.03, random_state=42,
                          min_child_samples=min_child, verbose=-1)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = []
    print("\n--- Model Test Metrics (held-out set) ---")
    for i, depth in enumerate(["50m", "100m", "200m", "500m"]):
        rmse = float(np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_test.iloc[:, i], y_pred[:, i]))
        metrics.append({"Depth": depth, "RMSE (°C)": round(rmse, 3), "R² Score": round(r2, 3)})
        print(f"Depth {depth:>4s} | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")
    return model, pd.DataFrame(metrics)


def train_real():
    if not REAL_DATASET.exists():
        print("=" * 70)
        print("TRAINING BLOCKED: no validated real dataset found.")
        print(f"Expected: {REAL_DATASET}")
        print()
        print("Run the real-data pipeline first (requires internet access):")
        print("    python scripts/build_dataset.py --use-raw --max-profiles 200")
        print()
        print("That command fetches genuine Argo profiles + matched satellite")
        print("surface observations, runs the hard-gate validator, and only")
        print("writes train_dataset.parquet if it passes (see")
        print("docs/DATA_INTEGRITY.md). This script refuses to substitute")
        print("synthetic data for it.")
        print()
        print("For local UI/dev testing only, run: python train_model.py --demo")
        print("=" * 70)
        sys.exit(1)

    df = pd.read_parquet(REAL_DATASET)
    print(f"Loaded {len(df)} validated real observation rows from {REAL_DATASET.name}")

    X = df[FEATURES]
    y = df[TARGETS]

    # Group split by Argo float (argo_wmo) so the same float never appears
    # in both train and test -- prevents leakage between nearby profiles
    # from the same instrument.
    if "argo_wmo" in df.columns and df["argo_wmo"].nunique() > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df["argo_wmo"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, metrics_df = _fit_and_report(X_train, X_test, y_train, y_test)

    artifact = {
        "model": model,
        "metrics": metrics_df,
        "feature_names": FEATURES,
        "data_mode": "real",
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "source_dataset": str(REAL_DATASET.name),
    }
    with open(ROOT / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    test_export = pd.concat([X_test, y_test], axis=1)
    # carry provenance columns through if present, so the app can show them
    for col in ["argo_wmo", "argo_cycle", "surface_distance_km", "surface_time_diff_hours"]:
        if col in df.columns:
            test_export[col] = df.loc[X_test.index, col]
    test_export.to_csv(ROOT / "test_sample.csv", index=False)

    print(f"\nSaved model.pkl (data_mode=real) and test_sample.csv ({len(test_export)} rows).")


def train_demo():
    if not DEMO_DATASET.exists():
        print("No demo dataset found. Generating one now (clearly-labeled synthetic data)...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_demo_data.py")], check=True)

    df = pd.read_csv(DEMO_DATASET)
    print(f"Loaded {len(df)} SYNTHETIC demo rows from {DEMO_DATASET.name}")
    print("⚠️  This model is for UI/development testing only. It is NOT trained")
    print("    on real observations and must never be presented as such.")

    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, metrics_df = _fit_and_report(X_train, X_test, y_train, y_test)

    artifact = {
        "model": model,
        "metrics": metrics_df,
        "feature_names": FEATURES,
        "data_mode": "demo_synthetic",
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "source_dataset": str(DEMO_DATASET.name),
    }
    # Deliberately different filenames so a demo model can NEVER be
    # mistaken for, or silently overwrite, a real one.
    with open(ROOT / "model_demo.pkl", "wb") as f:
        pickle.dump(artifact, f)

    test_export = pd.concat([X_test, y_test], axis=1)
    test_export.to_csv(ROOT / "test_sample_demo.csv", index=False)
    print(f"\nSaved model_demo.pkl and test_sample_demo.csv ({len(test_export)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OceanEmbed subsurface temperature model")
    parser.add_argument("--demo", action="store_true", help="Train on synthetic demo data (writes model_demo.pkl, never model.pkl)")
    args = parser.parse_args()
    train_demo() if args.demo else train_real()
