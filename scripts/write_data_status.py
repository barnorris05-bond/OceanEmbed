"""
Single source of truth for the real/demo badge.
Writes data_status.json which api_server.py reads.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_MODEL = ROOT / "model.pkl"
DEMO_MODEL = ROOT / "model_demo.pkl"
OUT = ROOT / "data_status.json"

def compute_status():
    if REAL_MODEL.exists():
        import pickle
        try:
            with open(REAL_MODEL, "rb") as f: artifact = pickle.load(f)
            if artifact.get("data_mode") == "real":
                return {"mode": "real", "label": "Real Argo + Satellite Data (Validated)"}
        except: pass
    if DEMO_MODEL.exists():
        return {"mode": "demo", "label": "Synthetic Demo Data — NOT real observations"}
    return {"mode": "missing", "label": "No trained model found"}

if __name__ == "__main__":
    status = compute_status()
    OUT.write_text(json.dumps(status, indent=2))
    print(f"Wrote {OUT}: {status['mode']}")