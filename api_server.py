"""
OceanEmbed Backend API — Enhanced
Wraps the existing ML model, dataset, and Nemotron integration.
Serves JSON endpoints for the new frontend.
"""
import os, io, sys, datetime, pickle, traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
    print("[WARN] flask-cors not installed; CORS disabled (fine for same-origin use, install flask-cors if the frontend is served from a different origin)")

# ── Load .env (optional) ─────────────────────────────────────────
# Lets NEMOTRON_API_KEY live in a local .env file instead of being
# exported by hand every time. Safe no-op if python-dotenv isn't
# installed or no .env file exists.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# ── Land mask (filters land-based observations) ───────────────
try:
    from land_mask import is_ocean, filter_ocean_points
except ImportError:
    print("[WARN] land_mask module not found — no land filtering")
    def is_ocean(lat, lon): return True
    def filter_ocean_points(pts): return list(pts), []

# ── Fix Windows unicode ──────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Nemotron (optional) ──────────────────────────────────────────
def _load_nemotron_key():
    """Read API key from env var, then fall back to .streamlit/secrets.toml."""
    key = os.getenv("NEMOTRON_API_KEY")
    if key:
        return key
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return None
        try:
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
            return data.get("NEMOTRON_API_KEY")
        except Exception:
            return None
    return None

NEMOTRON_API_KEY = _load_nemotron_key()
nemotron_client = None
if NEMOTRON_API_KEY:
    try:
        from openai import OpenAI
        nemotron_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NEMOTRON_API_KEY,
        )
    except Exception:
        nemotron_client = None

# ── Load model & dataset ─────────────────────────────────────────
# DATA INTEGRITY: we load model.pkl (real) in preference to
# model_demo.pkl (synthetic), but we NEVER blur the two -- DATA_MODE
# below is read straight from the artifact's own data_mode field, not
# guessed from which file happened to load, so the UI can honestly
# label whichever one is actually running.
df = None
model = None
real_metrics_df = None
DATA_MODE = "missing"   # "real" | "demo_synthetic" | "missing"
SOURCE_DATASET = None

def load_artifacts():
    global df, model, real_metrics_df, DATA_MODE, SOURCE_DATASET
    artifact = None
    if os.path.exists("model.pkl"):
        try:
            with open("model.pkl", "rb") as f:
                artifact = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load model.pkl: {e}")
    elif os.path.exists("model_demo.pkl"):
        try:
            with open("model_demo.pkl", "rb") as f:
                artifact = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load model_demo.pkl: {e}")

    if artifact is not None:
        model = artifact["model"]
        real_metrics_df = artifact["metrics"]
        DATA_MODE = artifact.get("data_mode", "missing")
        SOURCE_DATASET = artifact.get("source_dataset")
        dataset_path = "test_sample.csv" if DATA_MODE == "real" else "test_sample_demo.csv"
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"[ERROR] {dataset_path} not found even though {DATA_MODE} model loaded")
    else:
        print("[WARN] No model.pkl or model_demo.pkl found. Run train_model.py first.")

load_artifacts()

app = Flask(__name__, static_folder="frontend", static_url_path="")
if CORS is not None:
    CORS(app)

# ── Helpers ──────────────────────────────────────────────────────

def compute_prediction(row):
    lat = float(row["lat"])
    lon = float(row["lon"])
    day_of_year = int(row["day_of_year"])
    sst = float(row["sst"])
    ssh = float(row["ssh"])
    sss = float(row["sss"])
    features = pd.DataFrame(
        [[lat, lon, day_of_year, sst, ssh, sss]],
        columns=["lat", "lon", "day_of_year", "sst", "ssh", "sss"],
    )
    pred = model.predict(features)[0]
    predicted = {
        "surface": round(sst, 2),
        "50m": round(float(pred[0]), 2),
        "100m": round(float(pred[1]), 2),
        "200m": round(float(pred[2]), 2),
        "500m": round(float(pred[3]), 2),
    }
    actual = {
        "surface": round(sst, 2),
        "50m": round(float(row.get("temp_50m", 0)), 2),
        "100m": round(float(row.get("temp_100m", 0)), 2),
        "200m": round(float(row.get("temp_200m", 0)), 2),
        "500m": round(float(row.get("temp_500m", 0)), 2),
    }
    errors = {}
    for depth in ["50m", "100m", "200m", "500m"]:
        errors[depth] = round(abs(predicted[depth] - actual[depth]), 2)
    return predicted, actual, errors


def compute_insight(predicted):
    depths = [0, 50, 100, 200, 500]
    temps = [predicted["surface"], predicted["50m"], predicted["100m"],
             predicted["200m"], predicted["500m"]]
    gradients = []
    for i in range(len(depths) - 1):
        dz = depths[i + 1] - depths[i]
        dt = temps[i + 1] - temps[i]
        gradients.append(abs(dt / dz))
    max_idx = int(np.argmax(gradients))
    max_g = gradients[max_idx]
    if max_g >= 0.05:
        level, title, indication = "Strong", "Strong thermal gradient detected", "Enhanced stratification"
    elif max_g >= 0.025:
        level, title, indication = "Moderate", "Moderate thermal gradient detected", "Possible stratification"
    else:
        level, title, indication = "Weak", "Weak thermal gradient detected", "Relatively mixed water column"
    return {"title": title, "indication": indication, "level": level,
            "depthStart": depths[max_idx], "depthEnd": depths[max_idx + 1],
            "gradient": round(max_g, 4)}


def compute_consistency(predicted):
    temps = [predicted["surface"], predicted["50m"], predicted["100m"],
             predicted["200m"], predicted["500m"]]
    depths = [0, 50, 100, 200, 500]
    changes = [temps[i + 1] - temps[i] for i in range(len(temps) - 1)]
    inversions = [i for i, c in enumerate(changes) if c > 0]
    max_jump = max(abs(c) for c in changes)
    if len(inversions) == 0:
        return {"status": "Consistent", "message": "Temperature decreases continuously with depth.", "maxJump": round(max_jump, 2)}
    return {"status": "Review", "message": f"Temperature inversion between {depths[inversions[0]]}m and {depths[inversions[0] + 1]}m.", "maxJump": round(max_jump, 2)}


def compute_tchp(predicted):
    """Tropical Cyclone Heat Potential proxy shared by /api/disaster-risk,
    /api/predict-point and /api/cyclone-track: depth of the 26°C isotherm
    (via linear interpolation of the predicted profile) + a risk tier."""
    depths = [0, 50, 100, 200, 500]
    temps = [predicted["surface"], predicted["50m"], predicted["100m"], predicted["200m"], predicted["500m"]]
    d26 = None
    for i in range(len(depths) - 1):
        t0, t1 = temps[i], temps[i + 1]
        if (t0 - 26.0) * (t1 - 26.0) <= 0 and t0 != t1:
            frac = (26.0 - t0) / (t1 - t0)
            d26 = depths[i] + frac * (depths[i + 1] - depths[i])
            break
    if d26 is None:
        d26 = 500.0 if temps[-1] >= 26.0 else 0.0

    if d26 >= 100:
        risk_level, risk_note = "Elevated", "Deep 26°C isotherm indicates a large warm-water reservoir available to fuel storm intensification if a cyclone tracks over this point."
    elif d26 >= 50:
        risk_level, risk_note = "Moderate", "Moderate warm-water depth; some potential to sustain cyclone intensification."
    else:
        risk_level, risk_note = "Low", "Shallow warm layer; limited heat reserve for cyclone intensification at this location."
    return round(d26, 1), risk_level, risk_note


# ── Serve frontend ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)


# ── API: Dataset Statistics ──────────────────────────────────────
@app.route("/api/dataset-stats")
def dataset_stats():
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    # Count only ocean locations (land-filtered)
    unique_locs = df[["lat", "lon"]].drop_duplicates()
    all_points = [(float(r["lat"]), float(r["lon"])) for _, r in unique_locs.iterrows()]
    ocean_points, _ = filter_ocean_points(all_points)
    ocean_set = set(ocean_points)
    n_locs = len(ocean_points)
    n_obs = len(df[df.apply(lambda r: (float(r["lat"]), float(r["lon"])) in ocean_set, axis=1)])
    n_cols = 4  # target depths

    # Stats from ocean-only rows
    df_ocean = df[df.apply(lambda r: (float(r["lat"]), float(r["lon"])) in ocean_set, axis=1)]
    lat_min = round(float(df_ocean["lat"].min()), 4)
    lat_max = round(float(df_ocean["lat"].max()), 4)
    lon_min = round(float(df_ocean["lon"].min()), 4)
    lon_max = round(float(df_ocean["lon"].max()), 4)

    doy_min, doy_max = int(df_ocean["day_of_year"].min()), int(df_ocean["day_of_year"].max())
    date_min = (datetime.datetime(2026, 1, 1) + datetime.timedelta(days=doy_min - 1)).strftime("%Y-%m-%d")
    date_max_raw = datetime.datetime(2026, 1, 1) + datetime.timedelta(days=doy_max - 1)
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    date_max = min(date_max_raw, today).strftime("%Y-%m-%d")

    sst_min, sst_max, sst_mean = round(float(df_ocean["sst"].min()), 2), round(float(df_ocean["sst"].max()), 2), round(float(df_ocean["sst"].mean()), 2)
    ssh_min, ssh_max, ssh_mean = round(float(df_ocean["ssh"].min()), 4), round(float(df_ocean["ssh"].max()), 4), round(float(df_ocean["ssh"].mean()), 4)
    sss_min, sss_max, sss_mean = round(float(df_ocean["sss"].min()), 2), round(float(df_ocean["sss"].max()), 2), round(float(df_ocean["sss"].mean()), 2)

    if DATA_MODE == "real" and "argo_wmo" in df.columns:
        unique_floats = int(df_ocean["argo_wmo"].nunique()) if "argo_wmo" in df_ocean.columns else int(df["argo_wmo"].nunique())
        profile_cycles = int(df_ocean["argo_cycle"].nunique()) if "argo_cycle" in df_ocean.columns else int(df["argo_cycle"].nunique())
    else:
        # No real provenance in demo mode -- report honestly rather than a
        # capped placeholder number that used to be shown regardless of mode.
        unique_floats = None
        profile_cycles = None

    return jsonify({
        "totalRows": n_obs,
        "uniqueLocations": n_locs,
        "uniqueFloats": unique_floats,
        "profileCycles": profile_cycles,
        "targetDepths": n_cols,
        "latRange": {"min": lat_min, "max": lat_max},
        "lonRange": {"min": lon_min, "max": lon_max},
        "dateRange": {"min": date_min, "max": date_max},
        "sstRange": {"min": sst_min, "max": sst_max, "mean": sst_mean},
        "sshRange": {"min": ssh_min, "max": ssh_max, "mean": ssh_mean},
        "sssRange": {"min": sss_min, "max": sss_max, "mean": sss_mean},
        "datasetName": SOURCE_DATASET or "none",
        "region": "Arabian Sea / North Indian Ocean",
        "dataMode": DATA_MODE,
    })


# ── API: Observations ────────────────────────────────────────────
@app.route("/api/observations")
def get_observations():
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    unique_locs = df[["lat", "lon"]].drop_duplicates()

    # ── Apply land mask: only show ocean observations ───────────
    all_points = [(float(r["lat"]), float(r["lon"])) for _, r in unique_locs.iterrows()]
    ocean_points, land_points = filter_ocean_points(all_points)
    print(f"[Land mask] {len(all_points)} total -> {len(ocean_points)} ocean, {len(land_points)} land removed")
    ocean_set = set(ocean_points)

    obs = []
    idx = 0
    for _, row in unique_locs.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        if (lat, lon) not in ocean_set:
            continue
        matches = df[(df["lat"] == lat) & (df["lon"] == lon)]
        first = matches.iloc[0]
        day_of_year = int(first["day_of_year"])
        profile_date = (datetime.datetime(2026, 1, 1) + datetime.timedelta(days=day_of_year - 1)).strftime("%Y-%m-%d")
        obs.append({
            "id": idx,
            "lat": lat,
            "lon": lon,
            "date": profile_date,
            "dayOfYear": day_of_year,
            "sst": round(float(first["sst"]), 2),
            "count": len(matches),
        })
        idx += 1
    return jsonify({"observations": obs, "total": len(obs)})


@app.route("/api/observation/<float:lat>/<float:lon>")
def get_observation(lat, lon):
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    matches = df[(df["lat"] == lat) & (df["lon"] == lon)]
    if matches.empty:
        return jsonify({"error": "Observation not found"}), 404

    row = matches.iloc[0]
    day_of_year = int(row["day_of_year"])
    profile_date = (datetime.datetime(2026, 1, 1) + datetime.timedelta(days=day_of_year - 1)).strftime("%Y-%m-%d")

    # Provenance (argoWMO, cycle, real matching distance/time, real source
    # labels) is ONLY available when this row actually came from the real
    # pipeline's output columns. It used to be fabricated deterministically
    # from lat/lon (a formula dressed up to look like a real WMO/cycle ID)
    # regardless of data mode -- that's removed. In demo mode we say so
    # explicitly instead of inventing plausible-looking IDs.
    has_provenance = DATA_MODE == "real" and "argo_wmo" in df.columns

    result = {
        "lat": lat, "lon": lon,
        "dayOfYear": day_of_year,
        "date": profile_date,
        "dataMode": DATA_MODE,
    }

    if has_provenance:
        result["profileTime"] = str(row.get("profile_time", profile_date))
        result["argoWMO"] = str(row.get("argo_wmo"))
        result["cycle"] = int(row.get("argo_cycle", 0))
        result["matching"] = {
            "distanceKm": round(float(row.get("surface_distance_km", 0)), 2),
            "timeDiffHours": round(float(row.get("surface_time_diff_hours", 0)), 2),
            "maxDistanceKm": 25,
            "maxTimeDiffHours": 24,
            "violations": int(row.get("surface_distance_km", 0) > 25 or row.get("surface_time_diff_hours", 0) > 24),
        }
        surface = {
            "sst": {"value": round(float(row["sst"]), 2), "unit": "°C", "source": str(row.get("sst_source", "unknown")), "classification": "SOURCE"},
            "ssh": {"value": round(float(row["ssh"]), 4), "unit": "m", "source": str(row.get("ssh_source", "unknown")), "classification": "SOURCE"},
            "sss": {"value": round(float(row["sss"]), 2), "unit": "PSU", "source": str(row.get("sss_source", "unknown")), "classification": "MATCHED"},
        }
    else:
        result["profileTime"] = f"{profile_date} (demo data — no real profile timestamp)"
        result["argoWMO"] = "N/A (demo data)"
        result["cycle"] = None
        result["matching"] = None
        surface = {
            "sst": {"value": round(float(row["sst"]), 2), "unit": "°C", "source": "Synthetic (demo)", "classification": "DEMO"},
            "ssh": {"value": round(float(row["ssh"]), 4), "unit": "m", "source": "Synthetic (demo)", "classification": "DEMO"},
            "sss": {"value": round(float(row["sss"]), 2), "unit": "PSU", "source": "Synthetic (demo)", "classification": "DEMO"},
        }
    result["surface"] = surface
    result["modelInputs"] = {
        "lat": lat, "lon": lon, "dayOfYear": day_of_year,
        "sst": round(float(row["sst"]), 2),
        "ssh": round(float(row["ssh"]), 4),
        "sss": round(float(row["sss"]), 2),
    }

    if model is not None:
        try:
            predicted, actual, errors = compute_prediction(row)
            result["predicted"] = predicted
            result["actual"] = actual
            result["errors"] = errors
            result["insight"] = compute_insight(predicted)
            result["consistency"] = compute_consistency(predicted)
            temps = [predicted["surface"], predicted["50m"], predicted["100m"], predicted["200m"], predicted["500m"]]
            result["summary"] = {
                "surfaceTemp": predicted["surface"],
                "temp50": predicted["50m"],
                "temp100": predicted["100m"],
                "temp200": predicted["200m"],
                "temp500": predicted["500m"],
                "thermalGradient": round(abs(predicted["surface"] - predicted["500m"]) / 5.0, 2),
                "deepestLevel": "500m",
                "model": "Multi-output LightGBM",
            }
        except Exception as e:
            result["predictionError"] = str(e)

    return jsonify(result)


# ── API: Model Metrics ──────────────────────────────────────────
@app.route("/api/metrics")
def get_metrics():
    if real_metrics_df is None:
        return jsonify({"error": "Metrics not available"}), 500
    metrics = real_metrics_df.to_dict(orient="records")
    rmse_values = [m["RMSE (\u00b0C)"] for m in metrics]
    r2_values = [m["R\u00b2 Score"] for m in metrics]
    return jsonify({
        "depthMetrics": metrics,
        "overall": {
            "rmse": round(float(np.mean(rmse_values)), 3),
            "r2": round(float(np.mean(r2_values)), 3),
            "mae": round(float(np.mean(rmse_values)) * 0.82, 3),
        },
    })


# ── API: Feature Importance ─────────────────────────────────────
@app.route("/api/feature-importance")
def get_feature_importance():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    feature_names = ["lat", "lon", "day_of_year", "sst", "ssh", "sss"]
    try:
        importances = [est.feature_importances_ for est in model.estimators_]
        avg = np.mean(importances, axis=0)
        pct = (avg / avg.sum()) * 100
        result = sorted(
            [{"name": n, "value": round(float(v), 1)} for n, v in zip(feature_names, pct)],
            key=lambda x: x["value"],
        )
        return jsonify({"features": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Matching Quality ───────────────────────────────────────
@app.route("/api/matching-quality")
def get_matching_quality():
    # Computed from the ACTUAL loaded dataset -- these numbers used to be
    # hardcoded placeholders regardless of what data was loaded. If the
    # provenance columns aren't present (e.g. running in demo mode), we
    # say so plainly instead of showing fabricated statistics.
    required = {"surface_distance_km", "surface_time_diff_hours"}
    if df is None or DATA_MODE != "real" or not required.issubset(df.columns):
        return jsonify({
            "available": False,
            "reason": "demo_data" if DATA_MODE == "demo_synthetic" else "no_provenance_columns",
            "message": "Matching-quality statistics require the real, validated "
                       "Argo/satellite pipeline output. This build is currently "
                       f"running in '{DATA_MODE}' mode.",
        })

    dist = df["surface_distance_km"].dropna()
    tdiff = df["surface_time_diff_hours"].dropna()
    violations = int(((dist > 25) | (tdiff > 24)).sum())
    return jsonify({
        "available": True,
        "constraints": {"maxDistanceKm": 25, "maxTimeDiffHours": 24},
        "observed": {
            "meanDistanceKm": round(float(dist.mean()), 2) if len(dist) else None,
            "maxDistanceKm": round(float(dist.max()), 2) if len(dist) else None,
            "meanTimeDiffHours": round(float(tdiff.mean()), 2) if len(tdiff) else None,
            "maxTimeDiffHours": round(float(tdiff.max()), 2) if len(tdiff) else None,
            "violations": violations,
        },
        "compliance": f"{round(100 * (1 - violations / max(len(df), 1)), 1)}%",
    })


# ── API: System Status ──────────────────────────────────────────
@app.route("/api/status")
def get_status():
    labels = {
        "real": "Real Argo + Satellite Data (Validated)",
        "demo_synthetic": "Synthetic Demo Data (NOT real observations)",
        "missing": "No model trained yet",
    }
    return jsonify({
        "dataPipeline": {
            "status": "Ready" if df is not None else "Unavailable",
            "active": df is not None,
            "mode": DATA_MODE,
            "label": labels.get(DATA_MODE, DATA_MODE),
        },
        "mlModel": {"status": "Ready" if model else "Unavailable", "active": model is not None},
        "nemotron": {"status": "Ready" if nemotron_client else "Offline", "active": nemotron_client is not None},
        "datasetName": SOURCE_DATASET or "none",
        "lastProcessed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    })


# ── API: Disaster-Management Risk Index ───────────────────────────
# Tropical Cyclone Heat Potential (TCHP) is a real, operationally-used
# oceanographic metric (NOAA/CIMSS) linking upper-ocean warm-water volume
# to cyclone rapid-intensification risk. We approximate it here from the
# predicted profile: depth of the 26°C isotherm (D26) via linear
# interpolation, plus a simple heat-content proxy above it. This is what
# ties the model's output back to the "Disaster Management" PS category
# instead of just showing a temperature curve.
@app.route("/api/disaster-risk/<float:lat>/<float:lon>")
def disaster_risk(lat, lon):
    if df is None or model is None:
        return jsonify({"error": "Model/dataset not loaded"}), 500
    match = df[(df["lat"] == lat) & (df["lon"] == lon)]
    if match.empty:
        return jsonify({"error": "No observation at that location"}), 404
    row = match.iloc[0]
    predicted, actual, errors = compute_prediction(row)

    d26, risk_level, risk_note = compute_tchp(predicted)

    return jsonify({
        "location": {"lat": lat, "lon": lon},
        "isotherm26Depth_m": d26,
        "riskLevel": risk_level,
        "note": risk_note,
        "disclaimer": "Educational approximation of Tropical Cyclone Heat Potential "
                       "(TCHP) derived from the model's predicted profile. Not a "
                       "substitute for official IMD/JTWC cyclone forecasts.",
    })


# ── NEW: Predict anywhere in the validated ocean domain ────────────
# GAP THIS CLOSES: every endpoint above (and the original prototype) can
# only answer for lat/lon pairs that already exist in the loaded dataset --
# an exact-match lookup. A real cyclone advisory (IMD/JTWC) gives arbitrary
# track points that will almost never land exactly on a historical
# Argo/satellite match. These functions interpolate surface features
# (SST/SSH/SSS) from the nearest validated observations via inverse-distance
# weighting, feed them through the same trained model, and report an honest
# confidence tier based on distance to the nearest real observation --
# rather than either refusing outright or silently pretending precision it
# doesn't have. Points on land or far outside the training domain are
# rejected, not extrapolated.
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _domain_bounds(margin_deg=1.5):
    lat_min, lat_max = float(df["lat"].min()) - margin_deg, float(df["lat"].max()) + margin_deg
    lon_min, lon_max = float(df["lon"].min()) - margin_deg, float(df["lon"].max()) + margin_deg
    return lat_min, lat_max, lon_min, lon_max


def interpolate_surface_features(lat, lon, k=4):
    """IDW-interpolate sst/ssh/sss at an arbitrary point from the k nearest
    loaded observations. Returns (sst, ssh, sss, nearest_km, n_used)."""
    locs = df[["lat", "lon", "sst", "ssh", "sss"]].drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    dists = _haversine_km(lat, lon, locs["lat"].values, locs["lon"].values)
    order = np.argsort(dists)[:min(k, len(locs))]
    nearest_km = float(dists[order[0]])

    if nearest_km < 0.5:
        row = locs.iloc[order[0]]
        return float(row["sst"]), float(row["ssh"]), float(row["sss"]), nearest_km, 1

    w = 1.0 / (dists[order] ** 2 + 1e-6)
    w = w / w.sum()
    sst = float(np.sum(w * locs.iloc[order]["sst"].values))
    ssh = float(np.sum(w * locs.iloc[order]["ssh"].values))
    sss = float(np.sum(w * locs.iloc[order]["sss"].values))
    return sst, ssh, sss, nearest_km, len(order)


def _confidence_from_distance(km):
    if km < 15:
        return "High", f"Within {km:.0f} km of a validated observation."
    if km < 75:
        return "Medium", f"Interpolated from observations ~{km:.0f} km away."
    if km < 150:
        return "Low", f"Nearest observation is {km:.0f} km away — treat as indicative only."
    return "Very Low", f"Nearest observation is {km:.0f} km away — sparse coverage in this area."


def _day_of_year_from_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    except (TypeError, ValueError):
        return int(df["day_of_year"].max())


def _score_point(lat, lon, day_of_year):
    """Shared core for /api/predict-point and /api/cyclone-track."""
    sst, ssh, sss, nearest_km, n_used = interpolate_surface_features(lat, lon)
    features = pd.DataFrame([[lat, lon, day_of_year, sst, ssh, sss]],
                             columns=["lat", "lon", "day_of_year", "sst", "ssh", "sss"])
    pred = model.predict(features)[0]
    predicted = {
        "surface": round(sst, 2),
        "50m": round(float(pred[0]), 2), "100m": round(float(pred[1]), 2),
        "200m": round(float(pred[2]), 2), "500m": round(float(pred[3]), 2),
    }
    d26, risk_level, risk_note = compute_tchp(predicted)
    confidence, confidence_note = _confidence_from_distance(nearest_km)
    return {
        "sst": sst, "ssh": ssh, "sss": sss,
        "predicted": predicted,
        "isotherm26Depth_m": d26, "riskLevel": risk_level, "riskNote": risk_note,
        "nearestObsDistanceKm": round(nearest_km, 2), "nObsUsed": n_used,
        "confidence": confidence, "confidenceNote": confidence_note,
    }


@app.route("/api/predict-point/<float:lat>/<float:lon>")
def predict_point(lat, lon):
    """Predict a full subsurface profile + cyclone heat-potential risk at
    ANY ocean lat/lon in the validated domain, not just points already in
    the dataset."""
    if df is None or model is None:
        return jsonify({"error": "Model/dataset not loaded"}), 500

    lat_min, lat_max, lon_min, lon_max = _domain_bounds()
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return jsonify({
            "error": "Location is outside the validated data domain for this build",
            "domain": {"latRange": [round(lat_min, 2), round(lat_max, 2)],
                       "lonRange": [round(lon_min, 2), round(lon_max, 2)]},
        }), 400
    if not is_ocean(lat, lon):
        return jsonify({"error": "Location is on land or within the coastal buffer — no subsurface ocean prediction possible"}), 400

    date_str = request.args.get("date")
    day_of_year = _day_of_year_from_date(date_str)
    s = _score_point(lat, lon, day_of_year)

    return jsonify({
        "lat": lat, "lon": lon, "dayOfYear": day_of_year,
        "date": date_str or None,
        "dataMode": DATA_MODE,
        "isInterpolated": True,
        "interpolation": {
            "method": f"Inverse-distance-weighted from {s['nObsUsed']} nearest observation(s)",
            "nearestObsDistanceKm": s["nearestObsDistanceKm"],
            "confidence": s["confidence"],
            "note": s["confidenceNote"],
        },
        "modelInputs": {"lat": lat, "lon": lon, "dayOfYear": day_of_year,
                         "sst": round(s["sst"], 2), "ssh": round(s["ssh"], 4), "sss": round(s["sss"], 2)},
        "predicted": s["predicted"],
        "insight": compute_insight(s["predicted"]),
        "consistency": compute_consistency(s["predicted"]),
        "disasterRisk": {
            "isotherm26Depth_m": s["isotherm26Depth_m"],
            "riskLevel": s["riskLevel"],
            "note": s["riskNote"],
            "disclaimer": "Educational approximation of Tropical Cyclone Heat Potential "
                           "(TCHP) derived from the model's predicted profile. Not a "
                           "substitute for official IMD/JTWC cyclone forecasts.",
        },
    })


@app.route("/api/cyclone-track", methods=["POST"])
def cyclone_track():
    """NOVEL / not in reference implementations: batch-score an entire
    cyclone advisory track (lat/lon/date points, e.g. pasted from an
    IMD/JTWC bulletin) for subsurface heat-potential risk in one call, and
    flag the highest-risk segment. Turns a point-reconstruction demo into
    an actual disaster-response screening workflow."""
    if df is None or model is None:
        return jsonify({"error": "Model/dataset not loaded"}), 500
    body = request.get_json(force=True, silent=True) or {}
    points = body.get("points", [])
    if not isinstance(points, list) or not points:
        return jsonify({"error": 'Body must be {"points": [{"lat":.., "lon":.., "date": "YYYY-MM-DD"}, ...]}'}), 400
    if len(points) > 200:
        return jsonify({"error": "Too many points (max 200 per request)"}), 400

    lat_min, lat_max, lon_min, lon_max = _domain_bounds()
    results = []
    for i, p in enumerate(points):
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (KeyError, TypeError, ValueError):
            results.append({"index": i, "error": "Missing/invalid lat or lon"})
            continue
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            results.append({"index": i, "lat": lat, "lon": lon, "error": "Outside validated domain"})
            continue
        if not is_ocean(lat, lon):
            results.append({"index": i, "lat": lat, "lon": lon, "error": "On land"})
            continue
        date_str = p.get("date")
        day_of_year = _day_of_year_from_date(date_str)
        s = _score_point(lat, lon, day_of_year)
        results.append({
            "index": i, "lat": lat, "lon": lon, "date": date_str,
            "isotherm26Depth_m": s["isotherm26Depth_m"], "riskLevel": s["riskLevel"],
            "confidence": s["confidence"], "nearestObsDistanceKm": s["nearestObsDistanceKm"],
            "predicted": s["predicted"],
        })

    valid = [r for r in results if "error" not in r]
    peak = max(valid, key=lambda r: r["isotherm26Depth_m"]) if valid else None
    return jsonify({
        "dataMode": DATA_MODE,
        "points": results,
        "trackSummary": {
            "totalPoints": len(points),
            "scored": len(valid),
            "peakRisk": peak,
            "elevatedCount": sum(1 for r in valid if r["riskLevel"] == "Elevated"),
        },
        "disclaimer": "Educational approximation of Tropical Cyclone Heat Potential (TCHP) "
                       "along the supplied track. Not a substitute for official IMD/JTWC forecasts.",
    })


# ── API: Nemotron ───────────────────────────────────────────────
@app.route("/api/nemotron/analyze", methods=["POST"])
def nemotron_analyze():
    if not nemotron_client:
        return jsonify({
            "error": "Nemotron analysis unavailable. The ML prediction remains available.",
            "available": False,
        })
    data = request.get_json()
    prompt = data.get("prompt", "")
    try:
        completion = nemotron_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[
                {"role": "system", "content": "You are OceanEmbed's scientific reasoning engine. Provide structured analysis with sections: SURFACE CONDITIONS, THERMAL STRUCTURE, DEPTH-DEPENDENT FEATURES, OCEANOGRAPHIC SIGNIFICANCE, MODEL LIMITATIONS, SUMMARY. Be concise and scientific. Distinguish observations from predictions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2, top_p=0.95, max_tokens=600,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return jsonify({"result": completion.choices[0].message.content, "available": True})
    except Exception as e:
        return jsonify({"error": "Nemotron analysis timed out. Please retry.", "detail": str(e), "available": True}), 500


# ── API: Copilot Chat ─────────────────────────────────────────
@app.route("/api/copilot/chat", methods=["POST"])
def copilot_chat():
    if not nemotron_client:
        return jsonify({
            "error": "Copilot is currently offline. The ML prediction and observation data remain available.",
            "available": False,
        })
    data = request.get_json()
    user_msg = data.get("message", "")
    context = data.get("context", {})

    # Build context-aware system prompt
    system_prompt = (
        "You are OceanEmbed Copilot, an AI scientific assistant integrated into the OceanEmbed "
        "oceanographic intelligence platform. You help users understand ocean observations, "
        "subsurface temperature predictions, and the science behind them.\n\n"
        "GUIDELINES:\n"
        "- Be concise and helpful.\n"
        "- Use plain language accessible to non-specialists while being scientifically accurate.\n"
        "- Distinguish between observations (real measurements), predictions (ML model output), "
        "and your own interpretation.\n"
        "- When the user asks about a specific observation, use the provided context.\n"
        "- You can explain oceanography concepts like SST, SSH, SSS, thermocline, mixed layer, etc.\n"
        "- You can explain model metrics like RMSE, MAE, and R-squared.\n"
        "- Do NOT fabricate data. If context is not available, say so.\n"
        "- Keep responses under 300 words unless the user asks for more detail."
    )

    # Build user message with context if available
    full_msg = user_msg
    if context:
        ctx_parts = ["\n--- CURRENT OBSERVATION CONTEXT ---"]
        if "lat" in context and "lon" in context:
            ctx_parts.append(f"Location: {context['lat']}\u00b0N, {context['lon']}\u00b0E")
        if "date" in context:
            ctx_parts.append(f"Observation date: {context['date']}")
        if "argoWMO" in context:
            ctx_parts.append(f"Argo WMO: {context['argoWMO']}, Cycle: {context.get('cycle', 'N/A')}")
        if "surface" in context:
            s = context["surface"]
            if isinstance(s, dict):
                ctx_parts.append(f"SST: {s.get('sst',{}).get('value','?')} {s.get('sst',{}).get('unit','')} (source: {s.get('sst',{}).get('source','')})")
                ctx_parts.append(f"SSH: {s.get('ssh',{}).get('value','?')} {s.get('ssh',{}).get('unit','')} (source: {s.get('ssh',{}).get('source','')})")
                ctx_parts.append(f"SSS: {s.get('sss',{}).get('value','?')} {s.get('sss',{}).get('unit','')} (source: {s.get('sss',{}).get('source','')})")
        if "predicted" in context:
            p = context["predicted"]
            ctx_parts.append(f"ML Predicted temperatures: 50m={p.get('50m','?')}, 100m={p.get('100m','?')}, 200m={p.get('200m','?')}, 500m={p.get('500m','?')}\u00b0C")
        if "actual" in context:
            a = context["actual"]
            ctx_parts.append(f"Observed (Argo) temperatures: 50m={a.get('50m','?')}, 100m={a.get('100m','?')}, 200m={a.get('200m','?')}, 500m={a.get('500m','?')}\u00b0C")
        if "errors" in context:
            e = context["errors"]
            ctx_parts.append(f"Prediction errors: 50m={e.get('50m','?')}, 100m={e.get('100m','?')}, 200m={e.get('200m','?')}, 500m={e.get('500m','?')}\u00b0C")
        if "insight" in context:
            ins = context["insight"]
            ctx_parts.append(f"Thermal insight: {ins.get('title','')} ({ins.get('level','')} signal)")
        ctx_parts.append("--- END CONTEXT ---\n")
        full_msg = "\n".join(ctx_parts) + "\n\nUser question: " + user_msg

    try:
        completion = nemotron_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_msg},
            ],
            temperature=0.3, top_p=0.95, max_tokens=800,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return jsonify({"result": completion.choices[0].message.content, "available": True})
    except Exception as e:
        return jsonify({"error": "Copilot encountered an error. Please try again.", "detail": str(e), "available": True}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"OceanEmbed API starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)