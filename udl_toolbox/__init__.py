"""
UDL Toolbox - Unsupervised Deep Learning Toolbox

A comprehensive implementation of autoencoder architectures with custom loss functions,
data projection utilities, and visualization tools.
"""

__version__ = "0.1.0"
__author__ = "UDL Toolbox Team"

from .autoencoders.base import BaseAutoencoder
from .autoencoders.vanilla import VanillaAutoencoder
from .autoencoders.sparse import SparseAutoencoder
from .autoencoders.denoising import DenoisingAutoencoder
from .autoencoders.variational import VariationalAutoencoder
from .autoencoders.convolutional import ConvolutionalAutoencoder

from .losses import (
    MeanSquaredError,
    BinaryCrossentropy,
    KLDivergence,
    SparsityRegularization,
    VAELoss,
)

from .projections import (
    PCAProjection,
    TSNEProjection,
    LatentSpaceInterpolation,
)

from .visualization import (
    LatentSpaceVisualizer,
    ReconstructionVisualizer,
    LossVisualizer,
)

__all__ = [
    # Autoencoders
    "BaseAutoencoder",
    "VanillaAutoencoder",
    "SparseAutoencoder",
    "DenoisingAutoencoder",
    "VariationalAutoencoder",
    "ConvolutionalAutoencoder",
    # Loss functions
    "MeanSquaredError",
    "BinaryCrossentropy",
    "KLDivergence",
    "SparsityRegularization",
    "VAELoss",
    # Projections
    "PCAProjection",
    "TSNEProjection",
    "LatentSpaceInterpolation",
    # Visualization
    "LatentSpaceVisualizer",
    "ReconstructionVisualizer",
    "LossVisualizer",
]