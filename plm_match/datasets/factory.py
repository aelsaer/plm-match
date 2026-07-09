from __future__ import annotations

from typing import Dict

from .base import BaseDatasetAdapter
from .generic_rgbd import GenericRGBDDataset
from .scannet import ScanNetRGBDDataset
from .colmap import COLMAPLocalizationDataset
from .seven_scenes import SevenScenesRGBDDataset
from .cambridge import CambridgeLandmarksDataset


def build_dataset(root: str, cfg: Dict) -> BaseDatasetAdapter:
    name = cfg.get('type', 'generic_rgbd').lower()
    if name in ('generic_rgbd', 'generic'):
        return GenericRGBDDataset(root, cfg)
    if name in ('scannet_rgbd', 'scannet'):
        return ScanNetRGBDDataset(root, cfg)
    if name in ('seven_scenes_rgbd', '7scenes_rgbd', 'seven_scenes', '7scenes'):
        return SevenScenesRGBDDataset(root, cfg)
    if name in ('cambridge_landmarks', 'cambridge'):
        return CambridgeLandmarksDataset(root, cfg)
    if name in ('colmap_localization', 'hloc_colmap', 'colmap'):
        return COLMAPLocalizationDataset(root, cfg)
    raise ValueError(f'Unknown dataset adapter: {name}')
