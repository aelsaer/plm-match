from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch


class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> Dict[str, object]:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def _normalize_tokens(self, tokens: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8
        return tokens / denom
