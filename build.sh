#!/bin/bash
# Vercel build script for OceanEmbed
set -e

echo "=== OceanEmbed Vercel Build ==="

# 1. Train the ML model from ocean_data.csv -> model.pkl
echo "[1/2] Training ML model..."
python train_model.py

# 2. Download Natural Earth 110m land shapefiles using Python (no unzip needed)
echo "[2/2] Downloading land mask data..."
mkdir -p land_data
python -c "
import urllib.request, zipfile, io, os
url = 'https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip'
print('Downloading from', url)
data = urllib.request.urlopen(url).read()
z = zipfile.ZipFile(io.BytesIO(data))
for name in z.namelist():
    if 'ne_110m_land' in name:
        z.extract(name, 'land_data/')
        print('  Extracted:', name)
print('Land data ready.')
"

# 3. Copy frontend to public/ (Vercel serves static from here)
mkdir -p public
cp frontend/index.html public/index.html

echo "=== Build complete ==="
