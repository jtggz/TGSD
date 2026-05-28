"""
TGSD: Topology-Guided State-Space Diffusion Framework for EEG Spatial Super-Resolution

A framework that combines:
- Hierarchical Spatial Prior Encoder (HSPE): Learns topology-aware spatial priors over the complete electrode layout
- Conditional State-Space Diffusion Reconstructor (CSDR): Reconstructs missing-channel EEG through conditional reverse diffusion

Reference Paper: TGSD: Topology-Guided State-Space Diffusion Framework for EEG Spatial Super-Resolution
"""

__version__ = '1.0.0'
__author__ = 'Zijian Kang, Weiming Zeng, Yueyang Li, Shengyu Gong, Hongjie Yan, Wai Ting Siok, Nizhuan Wang'

from .model import TGSDModel, HierarchicalSpatialPriorEncoder, ConditionalStateSpaceDiffusionReconstructor
from .dataset import EEGDataset, EEGSpatialSuperResDataset
from .utils.util import calc_diffusion_hyperparams

__all__ = [
    'TGSDModel',
    'HierarchicalSpatialPriorEncoder',
    'ConditionalStateSpaceDiffusionReconstructor',
    'EEGDataset',
    'EEGSpatialSuperResDataset',
    'calc_diffusion_hyperparams'
]