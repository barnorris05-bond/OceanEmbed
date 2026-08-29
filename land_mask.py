"""
Land mask for OceanEmbed observation filtering.

Uses Natural Earth 110m land polygons (shapefile) + shapely
to determine whether a lat/lon point is on land or in the ocean.

Points within a configurable buffer distance of land are also
treated as land to catch coastal/beach points.
"""
import os
from shapely.geometry import shape, Point
from shapely.ops import unary_union

# Path to the extracted Natural Earth shapefile
_SHAPEFILE_DIR = os.path.join(os.path.dirname(__file__), "land_data")
_SHAPEFILE_PATH = os.path.join(_SHAPEFILE_DIR, "ne_110m_land.shp")

# Buffer in degrees (~0.15 deg ≈ 15 km at these latitudes).
# Points within this distance of land are treated as land.
LAND_BUFFER_DEG = 0.15

_land_union = None
_land_buffered = None


def _load_land_polygons():
    """Load and union all Natural Earth land polygons into one geometry."""
    global _land_union, _land_buffered
    if _land_union is not None:
        return
    try:
        import shapefile
        sf = shapefile.Reader(_SHAPEFILE_PATH)
        polys = [shape(rec) for rec in sf.iterShapes()]
        _land_union = unary_union(polys)
        _land_buffered = _land_union.buffer(LAND_BUFFER_DEG)
    except Exception as e:
        print(f"[WARN] Could not load land mask: {e}")
        _land_union = None
        _land_buffered = None


def is_land(lat: float, lon: float) -> bool:
    """
    Return True if (lat, lon) is on or very near land.

    Uses the buffered land polygon so that points within
    ~15 km of the coast are conservatively treated as land.
    Returns False (assumes ocean) if the land mask failed to load.
    """
    _load_land_polygons()
    if _land_buffered is None:
        return False  # Fallback: don't filter if mask unavailable
    pt = Point(lon, lat)  # shapely uses (x, y) = (lon, lat)
    return _land_buffered.contains(pt)


def is_ocean(lat: float, lon: float) -> bool:
    """Return True if (lat, lon) is in open ocean (not on/near land)."""
    return not is_land(lat, lon)


def filter_ocean_points(points):
    """
    Filter a list of (lat, lon) tuples, returning only ocean points.

    Parameters
    ----------
    points : list of (lat, lon) tuples

    Returns
    -------
    ocean : list of (lat, lon) tuples that are in the ocean
    land  : list of (lat, lon) tuples that are on land (removed)
    """
    _load_land_polygons()
    ocean, land = [], []
    for lat, lon in points:
        if is_land(lat, lon):
            land.append((lat, lon))
        else:
            ocean.append((lat, lon))
    return ocean, land


# Module-level initialization: load once at import time
_load_land_polygons()
if _land_buffered is not None:
    print("[OK] Land mask loaded (Natural Earth 110m)")
else:
    print("[WARN] Land mask unavailable — no land filtering will be applied")
