"""
Simulated small build using existing ocean_data.csv as source.
Creates data/processed/argo_profiles.parquet and data/dataset/train_dataset.parquet for 10 rows.
This avoids network and heavy deps; useful to validate file paths, schema, and downstream matching code.
"""
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / 'ocean_data.csv'
PROCESSED_DIR = ROOT / 'data' / 'processed'
DATASET_DIR = ROOT / 'data' / 'dataset'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def build_simulated(n=10):
    # Read CSV lines directly to avoid pandas dependency
    with open(DATA_CSV, 'r', encoding='utf-8') as f:
        lines = f.read().strip().splitlines()
    header = lines[0].split(',')
    rows = [line.split(',') for line in lines[1:1+n]]

    processed_lines = ["source_file,lat,lon,profile_time,temperature_0m,temperature_50m,temperature_100m,temperature_200m,temperature_500m,temperature_1000m"]
    dataset_lines = ["profile_id,lat,lon,profile_time,sst,ssh,sss,wind_u,wind_v,wind_speed,match_count,surface_obs_time,surface_obs_lat,surface_obs_lon,surface_distance_km,temperature_0m,temperature_50m,temperature_100m,temperature_200m,temperature_500m,temperature_1000m"]

    for i, r in enumerate(rows):
        row = dict(zip(header, r))
        day = int(float(row['day_of_year']))
        profile_time = (datetime(2020,1,1) + timedelta(days=day-1)).isoformat()
        src_file = f"sim_row_{i}.nc"
        temp_1000 = ''  # leave empty to simulate missing 1000m
        proc_line = f"{src_file},{row['lat']},{row['lon']},{profile_time},{row['sst']},{row['temp_50m']},{row['temp_100m']},{row['temp_200m']},{row['temp_500m']},{temp_1000}"
        processed_lines.append(proc_line)

        ds_line = f"{src_file},{row['lat']},{row['lon']},{profile_time},{row['sst']},{row['ssh']},{row['sss']},,,,{1},{profile_time},{row['lat']},{row['lon']},0.0,{row['sst']},{row['temp_50m']},{row['temp_100m']},{row['temp_200m']},{row['temp_500m']},{temp_1000}"
        dataset_lines.append(ds_line)

    processed_path = PROCESSED_DIR / 'argo_profiles.csv'
    dataset_path = DATASET_DIR / 'train_dataset.csv'
    processed_path.write_text('\n'.join(processed_lines), encoding='utf-8')
    dataset_path.write_text('\n'.join(dataset_lines), encoding='utf-8')
    print(f"Wrote simulated processed CSV to {processed_path}")
    print(f"Wrote simulated matched CSV to {dataset_path} ({len(dataset_lines)-1} rows)")
    return processed_path, dataset_path


if __name__ == '__main__':
    build_simulated(10)
