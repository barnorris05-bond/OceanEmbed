import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

def train():
    print("🤖 Loading dataset and training OceanEmbed baseline model...")
    df = pd.read_csv('ocean_data.csv')

    X = df[['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss']]
    y = df[['temp_50m', 'temp_100m', 'temp_200m', 'temp_500m']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    base_lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.03, random_state=42)
    model = MultiOutputRegressor(base_lgbm)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    depths = ['50m', '100m', '200m', '500m']

    metrics = []
    print("\n--- Model Test Metrics ---")
    for i, depth in enumerate(depths):
        rmse = float(np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_test.iloc[:, i], y_pred[:, i]))
        metrics.append({
            'Depth': f"{depth}",
            'RMSE (°C)': round(rmse, 3),
            'R² Score': round(r2, 3)
        })
        print(f"Depth {depth:>4s} | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

    artifact = {
        'model': model,
        'metrics': pd.DataFrame(metrics),
        'feature_names': list(X.columns)
    }

    with open('model.pkl', 'wb') as f:
        pickle.dump(artifact, f)

    # Save test observations for real ground-truth comparisons in Streamlit
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('test_sample.csv', index=False)

    print("\n Saved trained model artifact to model.pkl and ground truth to test_sample.csv!")

if __name__ == "__main__":
    train()