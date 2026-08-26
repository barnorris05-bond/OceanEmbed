"""
Simple inference wrapper: load saved artifact (model.pkl) and run prediction for a single lat/lon/time+surface features.
Returns predicted temps at target depths and (optionally) uncertainty via an ensemble if available.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'model.pkl'


def load_artifact(path=None):
    p = Path(path) if path else ARTIFACT
    with open(p, 'rb') as f:
        return pickle.load(f)


def predict_from_features(features_df, artifact=None):
    """features_df: pd.DataFrame. Accepts several common column names and maps them to the model's expected inputs.
    Returns: DataFrame with predicted temperatures.
    """
    if artifact is None:
        artifact = load_artifact()

    # Model artifact expected to be a dict with 'model' and 'feature_names' keys. Validate.
    if isinstance(artifact, dict) and 'model' in artifact:
        model = artifact['model']
        expected = artifact.get('feature_names', ['lat','lon','day_of_year','sst','ssh','sss'])
    else:
        model = artifact
        expected = ['lat','lon','day_of_year','sst','ssh','sss']

    # Schema adapter: try to produce a DataFrame with expected columns
    df = features_df.copy()
    # common renames
    renames = {
        'latitude': 'lat', 'longitude': 'lon', 'time': 'day_of_year', 'date': 'day_of_year'
    }
    df.rename(columns=renames, inplace=True)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing required columns: {missing}")

    X = df[expected]
    preds = model.predict(X)
    # attempt to infer number of outputs
    nouts = preds.shape[1] if hasattr(preds, 'shape') else 1
    col_names = ['pred_50m','pred_100m','pred_200m','pred_500m'][:nouts]
    out = pd.DataFrame(preds, columns=col_names)
    return out


if __name__ == '__main__':
    # quick demo using current ocean_data.csv
    import pandas as pd
    df = pd.read_csv(ROOT / 'ocean_data.csv')
    feat = df[['lat','lon','day_of_year','sst','ssh','sss']].head(3)
    print(predict_from_features(feat))
