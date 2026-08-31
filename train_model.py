"""
OceanEmbed model training.
DATA INTEGRITY: will NOT silently fall back to synthetic data.
Default: requires validated real dataset from scripts/build_dataset.py --use-raw
--demo flag: trains on synthetic data (writes model_demo.pkl, never model.pkl).
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
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
    min_child = max(1, min(2, len(X_train) - 1))
    base = LGBMRegressor(n_estimators=200, learning_rate=0.03, random_state=42,
                         min_child_samples=min_child, verbose=-1)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = []
    print("\n--- Model Test Metrics ---")
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
        print("\nRun: python scripts/build_dataset.py --use-raw --max-profiles 200")
        print("\nFor UI dev only: python train_model.py --demo")
        print("=" * 70)
        sys.exit(1)
    df = pd.read_parquet(REAL_DATASET)
    print(f"Loaded {len(df)} validated real rows from {REAL_DATASET.name}")
    X, y = df[FEATURES], df[TARGETS]
    if "argo_wmo" in df.columns and df["argo_wmo"].nunique() > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df["argo_wmo"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, metrics_df = _fit_and_report(X_train, X_test, y_train, y_test)
    artifact = {"model": model, "metrics": metrics_df, "feature_names": FEATURES,
                "data_mode": "real", "n_train_rows": len(X_train), "n_test_rows": len(X_test),
                "source_dataset": str(REAL_DATASET.name)}
    with open(ROOT / "model.pkl", "wb") as f: pickle.dump(artifact, f)
    test_export = pd.concat([X_test, y_test], axis=1)
    for col in ["argo_wmo", "argo_cycle", "surface_distance_km", "surface_time_diff_hours"]:
        if col in df.columns: test_export[col] = df.loc[X_test.index, col]
    test_export.to_csv(ROOT / "test_sample.csv", index=False)
    print(f"\nSaved model.pkl (data_mode=real) and test_sample.csv ({len(test_export)} rows).")

def train_demo():
    if not DEMO_DATASET.exists():
        print("Generating synthetic demo data...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_demo_data.py")], check=True)
    df = pd.read_csv(DEMO_DATASET)
    print(f"Loaded {len(df)} SYNTHETIC demo rows from {DEMO_DATASET.name}")
    print("⚠️  This model is for UI/dev testing ONLY. NOT real observations.")
    X, y = df[FEATURES], df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, metrics_df = _fit_and_report(X_train, X_test, y_train, y_test)
    artifact = {"model": model, "metrics": metrics_df, "feature_names": FEATURES,
                "data_mode": "demo_synthetic", "n_train_rows": len(X_train), "n_test_rows": len(X_test),
                "source_dataset": str(DEMO_DATASET.name)}
    with open(ROOT / "model_demo.pkl", "wb") as f: pickle.dump(artifact, f)
    test_export = pd.concat([X_test, y_test], axis=1)
    test_export.to_csv(ROOT / "test_sample_demo.csv", index=False)
    print(f"\nSaved model_demo.pkl and test_sample_demo.csv ({len(test_export)} rows).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Train on synthetic data (writes model_demo.pkl)")
    args = parser.parse_args()
    train_demo() if args.demo else train_real()