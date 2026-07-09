from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .base import BaseDatasetAdapter, FrameRecord
from plm_match.utils.colmap_model import Camera, Image, Point3D, camera_to_intrinsics, image_twc, load_colmap_model


def _query_pose_key(rel: str) -> str:
    return str(rel).replace("\\", "/").lstrip("./").replace("/", "__")


def parse_query_list(path: Path, image_root: Path, default_intrinsics: Optional[Dict[str, float]] = None) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            rel = toks[0]
            intr = None
            if len(toks) >= 5:
                model = toks[1].upper()
                width = int(float(toks[2]))
                height = int(float(toks[3]))
                params = np.array([float(x) for x in toks[4:]], dtype=np.float64)
                cam = Camera(id=-1, model=model, width=width, height=height, params=params)
                intr = camera_to_intrinsics(cam)
            elif default_intrinsics is not None:
                intr = dict(default_intrinsics)
            frames.append(FrameRecord(frame_id=Path(rel).stem, image_path=image_root / rel, intrinsics=intr, meta={'relative_path': rel}))
    return frames


class COLMAPLocalizationDataset(BaseDatasetAdapter):
    @property
    def map_mode(self) -> str:
        return 'colmap'

    def __init__(self, root: str | Path, cfg: Dict):
        super().__init__(root, cfg)
        self.image_root = self.root / cfg.get('image_root', '.')
        self.model_path = self.root / cfg.get('model_path', 'sfm')
        self.cameras, self.images, self.points3d = load_colmap_model(self.model_path)
        self._map_frames = self._make_map_frames()
        self._query_frames = self._make_query_frames()

    def _make_map_frames(self) -> List[FrameRecord]:
        map_frames: List[FrameRecord] = []
        prefixes = self.cfg.get('db_image_prefixes', None)
        allowed_prefixes = tuple(prefixes) if prefixes else None
        max_frames = self.cfg.get('max_map_frames', None)
        names_file = self.cfg.get('db_image_names_file', None)
        allowed_names = None
        if names_file is not None:
            p = Path(names_file)
            if not p.is_absolute():
                p = self.root / p
            with open(p, 'r') as f:
                allowed_names = {line.strip() for line in f if line.strip()}
        for img_id in sorted(self.images):
            im = self.images[img_id]
            if allowed_prefixes is not None and not im.name.startswith(allowed_prefixes):
                continue
            if allowed_names is not None and im.name not in allowed_names:
                continue
            intr = camera_to_intrinsics(self.cameras[im.camera_id])
            map_frames.append(
                FrameRecord(
                    frame_id=Path(im.name).stem,
                    image_path=self.image_root / im.name,
                    intrinsics=intr,
                    pose=image_twc(im),
                    meta={'image_id': img_id, 'relative_path': im.name, 'xys': im.xys, 'point3D_ids': im.point3D_ids},
                )
            )
        if max_frames is not None:
            map_frames = map_frames[: int(max_frames)]
        return map_frames

    def _make_query_frames(self) -> List[FrameRecord]:
        query_list = self.cfg.get('query_list', None)
        if query_list is not None:
            default_intr = None
            if self.cfg.get('default_query_camera_from_first_map', True) and self._map_frames:
                default_intr = self._map_frames[0].intrinsics
            ql_path = Path(query_list)
            if not ql_path.is_absolute():
                ql_path = self.root / ql_path
            frames = parse_query_list(ql_path, self.image_root, default_intrinsics=default_intr)
        else:
            query_dir = self.root / self.cfg.get('query_image_dir', 'query')
            default_intr = self._map_frames[0].intrinsics if self._map_frames else None
            frames = [
                FrameRecord(frame_id=p.stem, image_path=p, intrinsics=default_intr, meta={'relative_path': str(p.relative_to(self.image_root)) if p.is_relative_to(self.image_root) else p.name})
                for p in sorted(query_dir.glob('*')) if p.is_file()
            ]
        max_q = self.cfg.get('max_query_frames', None)
        if max_q is not None:
            frames = frames[: int(max_q)]
        gt_dir = self.cfg.get('query_gt_pose_dir', None)
        if gt_dir is not None:
            gt_root = self.root / gt_dir
            for fr in frames:
                rel = str(fr.meta.get('relative_path', fr.image_path.name))
                candidates = [str(fr.frame_id), _query_pose_key(rel)]
                seen = set()
                for key in candidates:
                    if key in seen:
                        continue
                    seen.add(key)
                    p = gt_root / f'{key}.txt'
                    if p.exists():
                        fr.pose_path = p
                        break
        return frames

    def get_map_frames(self) -> List[FrameRecord]:
        return self._map_frames

    def get_query_frames(self) -> List[FrameRecord]:
        return self._query_frames

    def get_default_intrinsics(self) -> Optional[Dict[str, float]]:
        return self._map_frames[0].intrinsics if self._map_frames else None
