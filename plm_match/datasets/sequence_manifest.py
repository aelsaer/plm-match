from __future__ import annotations

from pathlib import Path
import csv
import json

from .base import FrameRecord
from .sequence_base import SequenceDataset
from plm_match.utils.io import read_intrinsics_txt


def _load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return list(data.get('frames', []))
        return list(data)
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows.extend(dict(r) for r in reader)
    return rows


def load_manifest_sequence(root: str | Path, cfg: dict) -> SequenceDataset:
    root = Path(root)
    manifest_path = root / cfg.get('manifest', 'sequence_manifest.json')
    records = _load_records(manifest_path)
    intr_global = read_intrinsics_txt(root / cfg['intrinsics_file']) if cfg.get('intrinsics_file') else None
    stride = int(cfg.get('stride', 1)); max_frames = int(cfg.get('max_frames', len(records)))
    frames = []
    for i, rec in enumerate(records[::stride][:max_frames]):
        intr = None
        if rec.get('fx') is not None:
            intr = {'fx': float(rec['fx']), 'fy': float(rec['fy']), 'cx': float(rec['cx']), 'cy': float(rec['cy']), 'width': int(rec.get('width', 0)), 'height': int(rec.get('height', 0))}
        elif intr_global is not None:
            intr = intr_global.copy()
        frames.append(FrameRecord(frame_id=str(i), image_path=root / rec['image'], intrinsics=intr, depth_path=(root / rec['depth']) if rec.get('depth') else None, pose_path=(root / rec['pose']) if rec.get('pose') else None, pose=None, meta={'manifest_index': i, 'relative_path': rec['image']}))
    return SequenceDataset(root=root, name='manifest_sequence', frames=frames, meta={'manifest': str(manifest_path)})
