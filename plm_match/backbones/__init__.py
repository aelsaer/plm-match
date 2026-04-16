from .base import BaseFeatureExtractor
from .mock import MockFeatureExtractor
from .dino import DINOv2FeatureExtractor
from .eupe import EUPEFeatureExtractor


def build_extractor(cfg: dict):
    name = cfg['name'].lower()
    if name == 'mock':
        return MockFeatureExtractor(
            grid_size=tuple(cfg.get('grid_size', [24, 24])),
            dim=int(cfg.get('dim', 128)),
            seed=int(cfg.get('seed', 13)),
        )
    if name == 'dinov2':
        return DINOv2FeatureExtractor(
            model_name=cfg.get('model_name', 'dinov2_vits14'),
            input_size=tuple(cfg.get('input_size', [336, 336])),
            device=cfg.get('device', 'cpu'),
        )
    if name == 'eupe':
        return EUPEFeatureExtractor(
            repo_dir=cfg['repo_dir'],
            weights_path=cfg['weights_path'],
            model_name=cfg.get('model_name', 'eupe_vits16'),
            input_size=tuple(cfg.get('input_size', [256, 256])),
            device=cfg.get('device', 'cpu'),
        )
    raise ValueError(f'Unknown extractor: {name}')
