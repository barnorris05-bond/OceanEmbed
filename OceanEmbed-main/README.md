# 🌊 OceanEmbed AI — Subsurface Ocean Temperature Estimator

An AI-driven oceanographic framework for predicting subsurface ocean temperatures (50m, 100m, 200m, 500m) in the Arabian Sea using surface ocean parameters (SST, SSH, SSS). This repository includes a real-Argo preprocessing and surface-observation matching pipeline, alongside a prototype model for development and testing.

**Note:** The framework is designed to train and validate models using quality-controlled Argo float observations paired with satellite-derived surface measurements. The current `model.pkl` is a prototype baseline. Real-data model training will proceed after validation of the data pipeline.

## 📁 Repository Structure
```text
OceanEmbed/
├── data/              # Ground truth & validation datasets
├── app.py             # Streamlit interactive UI dashboard
├── train_model.py     # LightGBM model training pipeline
├── generate_data.py  # Synthetic ocean data generation script
├── model.pkl          # Trained model artifact
├── requirements.txt   # Project dependencies
└── README.md          # Setup & project description
```

## 🚀 Quickstart
**Install Dependencies:**
`py -m pip install -r requirements.txt`

**Train Model:**
`py train_model.py`

**Launch Application:**
`py -m streamlit run app.py`
