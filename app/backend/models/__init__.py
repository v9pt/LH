"""Model architectures for face hallucination."""

from .zero_dce import DCE_Net, ZeroDCE
from .rrdb_generator import RRDBGenerator, RRDBNet
from .discriminator import PatchGANDiscriminator
from .arcface_model import ArcFaceModel, SimplifiedArcFace
from .landmark_detector import LandmarkDetector, SimplifiedLandmarkDetector

__all__ = [
    'DCE_Net',
    'ZeroDCE',
    'RRDBGenerator',
    'RRDBNet',
    'PatchGANDiscriminator',
    'ArcFaceModel',
    'SimplifiedArcFace',
    'LandmarkDetector',
    'SimplifiedLandmarkDetector',
]
