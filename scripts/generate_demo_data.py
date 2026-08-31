from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo" / "ocean_data_synthetic.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    lats = rng.uniform(8.0, 24.0, n); lons = rng.uniform(60.0, 77.0, n)
    days = rng.integers(1, 366, n)
    sst = 29.0-(lats-8)*0.15+np.sin(2*np.pi*days/365)*1.2+rng.normal(0,0.3,n)
    ssh = 0.65+(sst-26)*0.03+rng.normal(0,0.04,n)
    sss = 35.2-(lons-60)*0.04+rng.normal(0,0.1,n)
    t50 = sst-0.8-(24-lats)*0.02+rng.normal(0,0.2,n)
    t100 = t50-2.5-ssh*0.5+rng.normal(0,0.3,n)
    t200 = t100-4.1+rng.normal(0,0.4,n)
    t500 = t200-5.5+rng.normal(0,0.3,n)
    return pd.DataFrame({
        "lat":np.round(lats,2),"lon":np.round(lons,2),"day_of_year":days,
        "sst":np.round(sst,2),"ssh":np.round(ssh,3),"sss":np.round(sss,2),
        "temp_50m":np.round(t50,2),"temp_100m":np.round(t100,2),
        "temp_200m":np.round(t200,2),"temp_500m":np.round(t500,2),"is_synthetic":True,
    })

if __name__=="__main__":
    df = generate(); df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {OUT}")
