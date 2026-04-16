from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .base import FrameRecord


@dataclass
class SequenceDataset:
    root: Path
    name: str
    frames: List[FrameRecord]
    meta: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {'name': self.name, 'root': str(self.root), 'num_frames': len(self.frames), **self.meta}
