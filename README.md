# 🌊 OceanEmbed — Subsurface Ocean Temperature Reconstruction (SIH25066)

**Theme:** Disaster Management · **Category:** Software
Reconstructs subsurface ocean temperature (50m/100m/200m/500m) in the Arabian Sea
from surface satellite/in-situ observations (SST, SSH, SSS), and surfaces a
**Cyclone Heat Potential** indicator for disaster-management use.

> Where this idea sits in the field: satellite-to-subsurface temperature
> reconstruction is an active published research area (CNN/LightGBM, deep
> forest, ConvLSTM and graph-attention models all exist in the literature,
> some reaching R² ≈ 0.98 globally). This project's contribution isn't a new
> ML architecture — it's a **fully-traceable Indian-Ocean pipeline with a
> hard data-integrity gate** and a **direct disaster-response framing**
> (cyclone rapid-intensification risk), which most reference implementations
> don't provide.

## ✅ This build ships trained on real data

This app can run in exactly two modes, and it always tells you which one
you're in (see the header badge / `/api/status`):

| Mode | How it's produced | Model file | Use for |
|---|---|---|---|
| **Real** | `scripts/build_dataset.py --use-raw` → genuine Argo floats + NASA/NOAA/ESA satellite surface data, QC-filtered, hard-gate validated | `model.pkl` | Judging, real claims |
| **Demo** | `scripts/generate_demo_data.py` → formula-based synthetic numbers, explicitly labeled | `model_demo.pkl` | UI development only |

`train_model.py` will **refuse to train** (exit with instructions) if no
validated real dataset exists — it will never silently substitute synthetic
numbers the way the original prototype did.

**This repo ships pre-trained on real data.** `data/dataset/train_dataset.parquet`
is a genuine, hard-gate-validated dataset — 45 matched observations from 45
unique real Argo floats, paired with real NASA JPL MUR SST, NOAA NESDIS SSH,
and ESA SMOS / Argo in-situ SSS readings (see `data/dataset/validation_report.txt`
and `docs/REAL_DATA_DATASHEET.md` for the full audit trail). `model.pkl` was
trained directly on it, and `data_status.json` / `/api/status` report `mode:
real` accordingly. No demo/synthetic model ships in this build, so there is
no fallback path to accidentally demo the wrong one.

The dataset is real but still small (45 rows, 22 profile cycles) — honest
about that scale, not a limitation to hide. Growing it is just a matter of
internet access; see "Growing the real dataset further" below.

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# 0. (Optional) Enable the Copilot panel — see "AI Copilot setup" below.
#    Skip this and the app still runs fine, just with the "AI Offline" badge.

# 1. Launch — model.pkl and the real dataset are already included.
python api_server.py
# open http://localhost:5001
```

### Growing the real dataset further

More real Argo profiles = a stronger model. On a machine with internet
access:
```bash
python scripts/build_dataset.py --use-raw --max-profiles 200   # fetch more real data
python train_model.py                                           # retrain on the larger set
python scripts/write_data_status.py                             # refresh the badge
python api_server.py
```
This re-runs the same hard-gate validator behind the bundled dataset, so the
result stays genuinely real — it just becomes a bigger genuinely-real
dataset. If you want to fall back to synthetic data purely for UI
development (never for judging), that path still exists too:
```bash
python scripts/generate_demo_data.py
python train_model.py --demo
python scripts/write_data_status.py
python api_server.py
```
`api_server.py` always prefers `model.pkl` over `model_demo.pkl` automatically
if both are present — remove or rename `model.pkl` first if you deliberately
want to preview the demo model instead.

### 🤖 AI Copilot setup

The Copilot chat panel calls NVIDIA's Nemotron model and needs one API key.
Without it, everything else (map, predictions, metrics) still works — the
panel just shows "AI Offline".

1. Get a free key at [build.nvidia.com](https://build.nvidia.com) → open
   any Nemotron model page → **Get API Key**.
2. Copy `.env.example` to `.env` (same folder as `api_server.py`) and paste
   the key in:
   ```bash
   cp .env.example .env
   # then edit .env and set NEMOTRON_API_KEY=<your real key>
   ```
   (`.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` works too,
   if you'd rather use that format — `api_server.py` checks both.)
3. Restart `python api_server.py`. The Copilot badge should switch from
   "AI Offline" to live within a few seconds.

`.env` and `secrets.toml` are already git-ignored — never commit your real
key. Have a screenshot or cached response ready as a plan B in case judging
Wi-Fi is unreliable.

## 📁 Structure

```
OceanEmbed/
├── data/
│   ├── argo_fetch.py             # Real Argo float discovery/download (argopy)
│   ├── argo_preprocess.py        # QC filtering + TEOS-10 depth conversion
│   ├── surface_fetch.py          # Real satellite surface obs (SST/SSH/SSS)
│   ├── match_surface_to_argo.py  # Spatial/temporal matching, ≤25km/≤24h
│   ├── raw/argo/                 # Downloaded NetCDF profiles (gitignored)
│   ├── processed/                # QC'd profiles (parquet)
│   ├── dataset/                  # Final matched + validated training set
│   └── demo/                     # Synthetic demo data ONLY — never mixed in
├── scripts/
│   ├── build_dataset.py          # End-to-end real pipeline orchestrator
│   ├── validate_dataset.py       # Hard gate: blocks training on fake data
│   ├── generate_demo_data.py     # Clearly-labeled synthetic data for UI dev
│   └── write_data_status.py      # Single source of truth for real/demo badge
├── train_model.py                # Real by default; --demo is explicit opt-in
├── inference/predict.py
├── api_server.py                 # Flask API + serves frontend/
├── frontend/index.html           # Single dashboard UI (map, predictions,
│                                  #   provenance, cyclone heat potential, AI copilot)
├── land_mask.py                  # Filters land points from the map
├── land_data/                    # Natural Earth 110m land shapefile (bundled)
├── docs/DATA_INTEGRITY.md        # Full data-integrity policy & FAQ
└── legacy/                       # Old Streamlit app + old data generator,
                                   #   kept for reference, not used going forward
```

Only **one** presentation app is used going forward: the Flask + HTML
dashboard (`api_server.py` + `frontend/`). The original project bundled a
Streamlit app, this Flask/HTML dashboard, *and* a duplicate nested copy of
the whole repo — that's now consolidated into the structure above. The old
Streamlit app is kept in `legacy/` purely for reference.

## 🌀 Disaster-management framing

The dashboard computes an approximate **Tropical Cyclone Heat Potential**
indicator from the predicted profile — the depth at which predicted
temperature crosses 26°C (a real, operationally-used cyclone-intensification
proxy: a deep warm layer gives a storm more heat to draw on). This is what
ties the ML output to the *Disaster Management* PS category, rather than
showing an isolated temperature curve. See `/api/disaster-risk/<lat>/<lon>`.
This is an educational approximation, not an operational forecast — the UI
says so explicitly.

## 🌟 What's not available in comparable reference implementations

Published satellite-to-subsurface reconstruction work (and the original
prototype this was built on) evaluates the model **only at points that
already exist in the training dataset** — a plain lookup. That's a real
gap for a disaster-management tool: a live IMD/JTWC cyclone advisory gives
arbitrary lat/lon/time points that will essentially never land exactly on
a historical Argo-matched location.

- **`/api/predict-point/<lat>/<lon>`** — predicts a full subsurface
  profile at **any** ocean point in the validated domain, not just
  dataset points. Surface features (SST/SSH/SSS) are inverse-distance-
  weighted from the nearest real observations, and the response reports
  an honest confidence tier (High/Medium/Low) based on distance to the
  nearest real observation, rather than pretending precision it doesn't
  have. Points on land (Natural Earth 110m land mask) or far outside the
  training domain are rejected outright, not silently extrapolated.
  Click anywhere on the Observations map to try it.
- **`/api/cyclone-track`** — batch-scores an entire cyclone advisory track
  (a list of lat/lon/date points) for Tropical Cyclone Heat Potential in
  one call, and flags the highest-risk segment. See the **Cyclone Track
  Risk** tab. This turns a per-point reconstruction demo into an actual
  screening workflow a forecaster could paste a real advisory track into.

Both reuse the same trained model and the same TCHP proxy as the rest of
the app — they don't introduce a second, less-trustworthy code path.

## 🔍 What was fixed from the original prototype

1. **Silent synthetic fallback removed.** The old `generate_data.py` tried a
   live fetch and, on any failure, quietly wrote fabricated numbers to the
   same `ocean_data.csv` the model trained on — with no signal to the user.
   `train_model.py` now hard-fails instead, per the project's own
   `docs/DATA_INTEGRITY.md` policy.
2. **SSH/SSS were never real even on the "success" path.** Both were
   formula-generated proxies regardless of whether the Argo fetch succeeded.
   The real pipeline (`surface_fetch.py` + `match_surface_to_argo.py`) now
   pulls actual satellite SSH/SSS instead.
3. **`/api/matching-quality` was hardcoded** (fixed fake numbers like "0.39 km
   mean distance, 100% compliant") regardless of what dataset was loaded.
   It's now computed from the loaded dataset's real provenance columns, or
   honestly reports "not available in demo mode."
4. **`/api/status` said "Data Ready" for any dataframe**, real or fake. It
   now reports the actual mode (`real` / `demo_synthetic` / `missing`).
5. **Three overlapping UIs** (Streamlit, Flask+HTML, and a duplicated nested
   copy of the whole repo) consolidated into one.
6. **`requirements.txt`** now lists every package actually imported
   (flask, flask-cors, openai, argopy, gsw, shapely, pyshp, pyarrow were
   missing before).

## Known limitations / next steps

- The Nemotron AI copilot is a single external API-key dependency — have a
  plan B (screenshots or a cached response) in case the venue Wi-Fi or the
  API is unavailable during judging.
- The core model (LightGBM MultiOutputRegressor on 6 tabular features) is a
  reasonable baseline but a plain regressor; if you have time, a
  spatiotemporal model (ConvLSTM/graph-attention, as in recent literature)
  would likely improve accuracy at depth, especially at 500m.
- `/api/predict-point` and `/api/cyclone-track` interpolate surface
  features by simple inverse-distance weighting from the k nearest loaded
  observations. That's honest and fast, but a kriging/optimal-interpolation
  scheme (as satellite data providers use operationally) would give a
  statistically sounder uncertainty estimate than the current distance-only
  confidence tiers — worth doing if you have time before judging.
- `land_mask.py` now ships with the real Natural Earth 110m land shapefile
  in `land_data/` (previously missing, so land filtering silently did
  nothing). If that folder is ever empty, land filtering fails safe (no
  filtering) rather than crash — check the server log for the
  `[OK] Land mask loaded` / `[WARN] Land mask unavailable` line to confirm
  which mode you're actually running in.
