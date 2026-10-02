"""
API package for MIMIC-CXR and Digital Twin services.
Exports FastAPI application.
"""

from src.api.main import app

__all__ = ["app"]
