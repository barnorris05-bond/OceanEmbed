# OceanEmbed Pipeline

1. python -m data.argo_fetch
2. python -m data.argo_preprocess
3. python -m data.surface_fetch
4. python -m data.match_surface_to_argo
5. python train_model.py
6. python api_server.py

Or: python scripts/build_dataset.py --use-raw --max-profiles 200
