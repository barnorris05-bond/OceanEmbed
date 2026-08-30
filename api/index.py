"""
Vercel serverless entry point for OceanEmbed.
Exposes the Flask app as a serverless function.
"""
import sys
import os

# Add project root to path so api_server.py and land_mask.py can be found
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from api_server import app

# Vercel Python runtime expects the handler variable to be named 'app'
