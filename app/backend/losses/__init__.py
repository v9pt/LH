"""Loss functions for training."""

from .pixel_loss import PixelLoss, PerceptualLoss, AdversarialLoss
from .identity_loss import IdentityLoss, IdentityLossWithDetection
from .landmark_loss import LandmarkLoss, LandmarkLossWithDetection
from .zero_dce_loss import (
    SpatialConsistencyLoss,
    ExposureControlLoss,
    ColorConstancyLoss,
    IlluminationSmoothnessLoss,
    ZeroDCELoss
)

__all__ = [
    'PixelLoss',
    'PerceptualLoss',
    'AdversarialLoss',
    'IdentityLoss',
    'IdentityLossWithDetection',
    'LandmarkLoss',
    'LandmarkLossWithDetection',
    'SpatialConsistencyLoss',
    'ExposureControlLoss',
    'ColorConstancyLoss',
    'IlluminationSmoothnessLoss',
    'ZeroDCELoss',
]
