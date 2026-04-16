from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class FrameRecord:
    frame_id: str
    image_path: Path
    intrinsics: Optional[Dict[str, float]] = None
    depth_path: Optional[Path] = None
    pose_path: Optional[Path] = None
    pose: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseDatasetAdapter(ABC):
    def __init__(self, root: str | Path, cfg: Dict[str, Any]):
        self.root = Path(root)
        self.cfg = cfg

    @property
    @abstractmethod
    def map_mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_map_frames(self) -> List[FrameRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_query_frames(self) -> List[FrameRecord]:
        raise NotImplementedError

    def get_default_intrinsics(self) -> Optional[Dict[str, float]]:
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            'name': self.__class__.__name__,
            'root': str(self.root),
            'map_mode': self.map_mode,
            'num_map_frames': len(self.get_map_frames()),
            'num_query_frames': len(self.get_query_frames()),
        }
