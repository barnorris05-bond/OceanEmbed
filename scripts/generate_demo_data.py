"""
Generate CLEARLY-LABELED synthetic demo data for UI/development testing.

This replaces the old generate_data.py, which tried a live OPeNDAP fetch
and then SILENTLY fell back to fabricated numbers under the same filename
(ocean_data.csv) that the app trained on -- with no indication to the user
that a fallback had happened. That is exactly the failure mode
docs/DATA_INTEGRITY.md warns against, so this script does the opposite:

  - It never attempts to pass synthetic values off as observations.
  - It never touches the real pipeline's output files.
  - It always writes to data/demo/ocean_data_synthetic.csv, a path whose
    name makes its status unambiguous.

For real, validated data use scripts/build_dataset.py --use-raw instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ocean_data_synthetic.csv"


def generate(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Physics-informed but fully synthetic Arabian Sea profiles.

    Values follow plausible seasonal/latitudinal trends so the UI has
    something realistic-looking to render, but every number here is a
    formula output, not a measurement.
    """
    rng = np.random.default_rng(seed)
    lats = rng.uniform(8.0, 24.0, n_samples)
    lons = rng.uniform(60.0, 77.0, n_samples)
    days = rng.integers(1, 366, n_samples)

    sst = 29.0 - (lats - 8.0) * 0.15 + np.sin(2 * np.pi * days / 365.0) * 1.2 + rng.normal(0, 0.3, n_samples)
    ssh = 0.65 + (sst - 26.0) * 0.03 + rng.normal(0, 0.04, n_samples)
    sss = 35.2 - (lons - 60.0) * 0.04 + rng.normal(0, 0.1, n_samples)

    t50 = sst - 0.8 - (24.0 - lats) * 0.02 + rng.normal(0, 0.2, n_samples)
    t100 = t50 - 2.5 - (ssh * 0.5) + rng.normal(0, 0.3, n_samples)
    t200 = t100 - 4.1 + rng.normal(0, 0.4, n_samples)
    t500 = t200 - 5.5 + rng.normal(0, 0.3, n_samples)

    return pd.DataFrame({
        "lat": np.round(lats, 2), "lon": np.round(lons, 2), "day_of_year": days,
        "sst": np.round(sst, 2), "ssh": np.round(ssh, 3), "sss": np.round(sss, 2),
        "temp_50m": np.round(t50, 2), "temp_100m": np.round(t100, 2),
        "temp_200m": np.round(t200, 2), "temp_500m": np.round(t500, 2),
        "is_synthetic": True,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} SYNTHETIC demo rows to {OUT_PATH}")
    print("This file is for UI/development use only and is never read by")
    print("train_model.py's default (real) mode or by the validation hard gate.")
