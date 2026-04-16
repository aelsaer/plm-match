from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import cv2
import torch

from .base import BaseFeatureExtractor


class MockFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, grid_size: Tuple[int, int] = (24, 24), dim: int = 128, seed: int = 13):
        self.grid_h, self.grid_w = grid_size
        self._dim = int(dim)
        rng = np.random.default_rng(seed)
        in_dim = 9
        self.proj = rng.standard_normal((in_dim, self._dim), dtype=np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-8

    @property
    def device(self) -> str:
        return 'cpu'

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return 'mock'

    def extract(self, image: np.ndarray) -> Dict[str, object]:
        h, w = image.shape[:2]
        small = cv2.resize(image.astype(np.float32) / 255.0, (self.grid_w, self.grid_h), interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(small, (3, 3), 0)
        gray = cv2.cvtColor((small * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        feats = np.concatenate([
            small,
            blur,
            gray[..., None],
            grad_x[..., None],
            grad_y[..., None],
        ], axis=-1)
        tokens = feats @ self.proj
        tokens = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, self.grid_h, dtype=np.float32),
                             np.linspace(0.0, 1.0, self.grid_w, dtype=np.float32), indexing='ij')
        token_xy = np.stack([
            (xx * (w - 1)).astype(np.float32),
            (yy * (h - 1)).astype(np.float32)
        ], axis=-1)
        global_desc = tokens.mean(axis=(0, 1))
        global_desc = global_desc / (np.linalg.norm(global_desc) + 1e-8)
        return {
            'tokens': tokens.astype(np.float32),
            'token_xy': token_xy.astype(np.float32),
            'global_desc': torch.from_numpy(global_desc.astype(np.float32)),
        }
