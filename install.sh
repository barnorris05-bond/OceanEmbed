#!/bin/bash
# Vercel install wrapper — minimal first to test build env
set -ex
echo "=== Python version ==="
python --version
echo "=== pip version ==="
pip --version
echo "=== Installing core packages ==="
pip install flask flask-cors gunicorn 2>&1
echo "=== Installing numpy + pandas ==="
pip install numpy pandas 2>&1
echo "=== Installing lightgbm + deps ==="
pip install lightgbm scikit-learn 2>&1
echo "=== Installing remaining ==="
pip install openai shapely pyshp 2>&1
echo "=== All installed ==="
pip list 2>&1
