from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .base import BaseDatasetAdapter, FrameRecord
from plm_match.utils.colmap_model import camera_to_intrinsics, image_twc, load_colmap_model


def _read_name_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CambridgeLandmarksDataset(BaseDatasetAdapter):
    @property
    def map_mode(self) -> str:
        return "colmap"

    def __init__(self, root: str | Path, cfg: Dict):
        super().__init__(root, cfg)
        self.image_root = self._resolve_path(cfg.get("image_root", "."))
        self.sfm_dir = self._resolve_path(cfg.get("sfm_dir", "."))
        self.model_path = self._resolve_path(cfg.get("model_path", "model_train"), base=self.sfm_dir)
        self.query_model_path = self._resolve_path(cfg.get("query_model_path", "empty_all"), base=self.sfm_dir)
        self.db_list_path = self._resolve_path(cfg.get("db_list", "list_db.txt"), base=self.sfm_dir)
        self.query_list_path = self._resolve_path(cfg.get("query_list", "list_query.txt"), base=self.sfm_dir)

        self.cameras, self.images, self.points3d = load_colmap_model(self.model_path)
        self.query_cameras, self.query_images, _ = load_colmap_model(self.query_model_path)
        self._map_frames = self._make_map_frames()
        self._query_frames = self._make_query_frames()

    def _resolve_path(self, value: str | Path, *, base: Path | None = None) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (base or self.root) / path

    def _scaled_intrinsics_and_xys(self, camera, image_name: str, xys=None):
        intr = camera_to_intrinsics(camera)
        image_path = self.image_root / image_name
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read Cambridge image: {image_path}")
        h_orig, w_orig = img.shape[:2]
        sx = float(w_orig) / float(camera.width)
        sy = float(h_orig) / float(camera.height)
        if abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9:
            intr["width"] = int(w_orig)
            intr["height"] = int(h_orig)
            intr["fx"] = float(intr["fx"]) * sx
            intr["fy"] = float(intr["fy"]) * sy
            intr["cx"] = float(intr["cx"]) * sx
            intr["cy"] = float(intr["cy"]) * sy
            params = list(intr.get("params", []))
            if camera.model.upper() == "SIMPLE_RADIAL" and len(params) >= 3:
                if abs(sx - sy) > 1e-6:
                    raise ValueError(f"Non-uniform Cambridge image scale for {image_name}: {sx}, {sy}")
                params[0] *= sx
                params[1] *= sx
                params[2] *= sy
                intr["params"] = params
        if xys is None:
            return intr, None
        xys_scaled = np.asarray(xys, dtype=np.float64).copy()
        if xys_scaled.size:
            xys_scaled[:, 0] *= sx
            xys_scaled[:, 1] *= sy
        return intr, xys_scaled

    def _make_map_frames(self) -> List[FrameRecord]:
        allowed = set(_read_name_list(self.db_list_path))
        by_name = {image.name: (image_id, image) for image_id, image in self.images.items()}
        frames: list[FrameRecord] = []
        missing: list[str] = []
        for name in _read_name_list(self.db_list_path):
            item = by_name.get(name)
            if item is None:
                missing.append(name)
                continue
            image_id, image = item
            intr, xys = self._scaled_intrinsics_and_xys(self.cameras[image.camera_id], image.name, image.xys)
            frames.append(
                FrameRecord(
                    frame_id=Path(image.name).stem,
                    image_path=self.image_root / image.name,
                    intrinsics=intr,
                    pose=image_twc(image),
                    meta={
                        "image_id": int(image_id),
                        "relative_path": image.name,
                        "xys": xys,
                        "point3D_ids": image.point3D_ids,
                        "split": "map",
                    },
                )
            )
        if missing:
            raise ValueError(f"{len(missing)} Cambridge DB images were not found in {self.model_path}; first: {missing[0]}")
        # Keep any model image that is listed by the official file exactly once.
        return [frame for frame in frames if str(frame.meta["relative_path"]) in allowed]

    def _make_query_frames(self) -> List[FrameRecord]:
        by_name = {image.name: image for image in self.query_images.values()}
        frames: list[FrameRecord] = []
        missing: list[str] = []
        for name in _read_name_list(self.query_list_path):
            image = by_name.get(name)
            if image is None:
                missing.append(name)
                continue
            intr, _ = self._scaled_intrinsics_and_xys(self.query_cameras[image.camera_id], image.name)
            frames.append(
                FrameRecord(
                    frame_id=Path(image.name).stem,
                    image_path=self.image_root / image.name,
                    intrinsics=intr,
                    pose=image_twc(image),
                    meta={
                        "relative_path": image.name,
                        "split": "query",
                    },
                )
            )
        if missing:
            raise ValueError(f"{len(missing)} Cambridge query images were not found in {self.query_model_path}; first: {missing[0]}")
        return frames

    def get_map_frames(self) -> List[FrameRecord]:
        return self._map_frames

    def get_query_frames(self) -> List[FrameRecord]:
        return self._query_frames

    def get_default_intrinsics(self) -> Optional[Dict[str, float]]:
        return self._map_frames[0].intrinsics if self._map_frames else None
