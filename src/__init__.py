"""
SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models

This package contains the core modules for the SPIRIT solar irradiance forecasting system.
"""

from .data_processing import SolarDataProcessor
from .embeddings import EmbeddingGenerator

# Import nowcasting with conditional availability
try:
    from .nowcasting import NowcastingModel
    NOWCASTING_AVAILABLE = True
except ImportError as e:
    # This could happen if XGBoost or other dependencies are missing
    NOWCASTING_AVAILABLE = False
    NowcastingModel = None

from .forecasting import ForecastingModel

__version__ = "1.0.0"
__author__ = "SPIRIT Team"
__email__ = "suryaravindra01@gmail.com"

__all__ = [
    "SolarDataProcessor",
    "EmbeddingGenerator", 
    "NowcastingModel",
    "ForecastingModel",
    "NOWCASTING_AVAILABLE"
]