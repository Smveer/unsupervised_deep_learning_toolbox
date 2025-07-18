"""Data projection utilities."""

from .pca import PCAProjection
from .tsne import TSNEProjection
from .interpolation import LatentSpaceInterpolation

__all__ = [
    "PCAProjection",
    "TSNEProjection",
    "LatentSpaceInterpolation",
]