from .base import BaseFeatureExtractor
from .mock import MockFeatureExtractor
from .dino import DINOv2FeatureExtractor, DINOv3FeatureExtractor
from .eupe import EUPEFeatureExtractor


def _parse_size(value, default):
    if value is None:
        value = default
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
        if 'x' in text.lower():
            parts = text.lower().split('x')
        else:
            parts = text.split(',')
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == 1:
            n = int(parts[0])
            return (n, n)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        raise ValueError(f'Invalid size override: {value!r}')
    seq = list(value)
    if len(seq) == 2:
        return (int(seq[0]), int(seq[1]))
    if seq and all(isinstance(x, str) for x in seq):
        return _parse_size(''.join(seq), default)
    raise ValueError(f'Invalid size override: {value!r}')


def build_extractor(cfg: dict):
    name = cfg['name'].lower()
    if name == 'mock':
        return MockFeatureExtractor(
            grid_size=_parse_size(cfg.get('grid_size'), [24, 24]),
            dim=int(cfg.get('dim', 128)),
            seed=int(cfg.get('seed', 13)),
        )
    if name == 'dinov2':
        return DINOv2FeatureExtractor(
            model_name=cfg.get('model_name', 'dinov2_vits14'),
            input_size=_parse_size(cfg.get('input_size'), [336, 336]),
            resize_mode=cfg.get('resize_mode', 'square'),
            device=cfg.get('device', 'cpu'),
        )
    if name == 'dinov3':
        return DINOv3FeatureExtractor(
            repo_dir=cfg.get('repo_dir'),
            weights_path=cfg.get('weights_path'),
            model_name=cfg.get('model_name', 'dinov3_vits16'),
            input_size=_parse_size(cfg.get('input_size'), [512, 512]),
            resize_mode=cfg.get('resize_mode', 'square'),
            device=cfg.get('device', 'cpu'),
        )
    if name == 'eupe':
        return EUPEFeatureExtractor(
            repo_dir=cfg['repo_dir'],
            weights_path=cfg.get('weights_path'),
            model_name=cfg.get('model_name', 'eupe_vits16'),
            input_size=_parse_size(cfg.get('input_size'), [256, 256]),
            resize_mode=cfg.get('resize_mode', 'square'),
            device=cfg.get('device', 'cpu'),
        )
    raise ValueError(f'Unknown extractor: {name}')
