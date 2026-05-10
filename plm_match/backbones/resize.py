from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class ResizeInfo:
    orig_h: int
    orig_w: int
    resized_h: int
    resized_w: int
    canvas_h: int
    canvas_w: int


def _ceil_to_multiple(value: int, multiple: int) -> int:
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def resize_for_patch_backbone(
    image: np.ndarray,
    input_size: Tuple[int, int],
    *,
    patch_size: int,
    mode: str = "square",
) -> tuple[np.ndarray, ResizeInfo]:
    """Resize an image for a patch backbone and report token geometry.

    ``square`` preserves the previous behavior: force-resize to input_size.
    ``aspect`` preserves aspect ratio inside input_size and pads bottom/right to
    a patch multiple. Padding uses ImageNet mean color so it normalizes near 0.
    """

    orig_h, orig_w = image.shape[:2]
    target_h, target_w = int(input_size[0]), int(input_size[1])
    patch = max(1, int(patch_size))
    mode = str(mode or "square").lower()
    if mode in ("square", "stretch", "resize"):
        resized_h = target_h
        resized_w = target_w
        canvas_h = target_h
        canvas_w = target_w
        out = cv2.resize(image, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    elif mode in ("aspect", "keep_aspect", "preserve_aspect", "letterbox"):
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid image shape for aspect resize: {image.shape}")
        scale = min(float(target_w) / float(orig_w), float(target_h) / float(orig_h))
        resized_w = max(1, int(round(float(orig_w) * scale)))
        resized_h = max(1, int(round(float(orig_h) * scale)))
        canvas_w = min(_ceil_to_multiple(resized_w, patch), _ceil_to_multiple(target_w, patch))
        canvas_h = min(_ceil_to_multiple(resized_h, patch), _ceil_to_multiple(target_h, patch))
        resized_w = min(resized_w, canvas_w)
        resized_h = min(resized_h, canvas_h)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        mean_color = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        out = np.empty((canvas_h, canvas_w, image.shape[2]), dtype=np.float32)
        out[...] = mean_color.reshape(1, 1, -1)
        out[:resized_h, :resized_w] = resized.astype(np.float32)
        out = np.clip(out, 0, 255).astype(image.dtype, copy=False)
    else:
        raise ValueError(f"Unsupported resize_mode={mode!r}; use 'square' or 'aspect'.")
    return out, ResizeInfo(
        orig_h=int(orig_h),
        orig_w=int(orig_w),
        resized_h=int(resized_h),
        resized_w=int(resized_w),
        canvas_h=int(canvas_h),
        canvas_w=int(canvas_w),
    )


def token_xy_from_resize(info: ResizeInfo, *, patch_size: int, grid_h: int, grid_w: int) -> np.ndarray:
    patch = float(max(1, int(patch_size)))
    center_x = ((np.arange(int(grid_w), dtype=np.float32) + 0.5) * patch) - 0.5
    center_y = ((np.arange(int(grid_h), dtype=np.float32) + 0.5) * patch) - 0.5
    scale_x = float(info.orig_w - 1) / max(1.0, float(info.resized_w - 1))
    scale_y = float(info.orig_h - 1) / max(1.0, float(info.resized_h - 1))
    xx, yy = np.meshgrid(center_x * scale_x, center_y * scale_y, indexing="xy")
    return np.stack([xx, yy], axis=-1).astype(np.float32)
