import numpy as np
from data.argo_preprocess import pressure_to_depth, TARGET_DEPTHS


def test_pressure_to_depth_fallback():
    # fallback: input pressures -> expect approx equal depths when gsw not installed
    p = np.array([0, 10, 50, 100, 500])
    d = pressure_to_depth(p, 10.0)
    assert np.allclose(d, p, atol=1.0)


def test_target_depths_present():
    assert TARGET_DEPTHS[0] == 0
    assert TARGET_DEPTHS[-1] == 1000
