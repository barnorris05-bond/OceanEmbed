#!/bin/bash
# Render build script for OceanEmbed
# Trains the ML model and downloads land mask data

set -e

echo "=== OceanEmbed Build ==="

# 1. Train the ML model from ocean_data.csv -> model.pkl
echo "Training ML model from ocean_data.csv..."
python train_model.py

# 2. Download Natural Earth 110m land shapefiles for the land mask
echo "Downloading Natural Earth land data..."
mkdir -p land_data
curl -sL -o /tmp/land.zip "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
unzip -o /tmp/land.zip -d land_data/ ne_110m_land.* 2>/dev/null || unzip -o /tmp/land.zip -d land_data/ 2>/dev/null
rm -f /tmp/land.zip

echo "=== Build complete ==="
