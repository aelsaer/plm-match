from __future__ import annotations

from .sequence_scannet import load_scannet_sequence
from .sequence_tum import load_tum_sequence
from .sequence_manifest import load_manifest_sequence


def build_sequence_dataset(root: str, cfg: dict):
    name = cfg.get('type', 'scannet_sequence').lower()
    if name in ('scannet_sequence', 'scannet'):
        return load_scannet_sequence(root, cfg)
    if name in ('tum_rgbd_sequence', 'tum_sequence', 'tum'):
        return load_tum_sequence(root, cfg)
    if name in ('manifest_sequence', 'fourseasons_sequence', 'fourseasons'):
        return load_manifest_sequence(root, cfg)
    raise ValueError(f'Unknown sequence dataset type: {name}')
