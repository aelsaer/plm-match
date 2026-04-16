from __future__ import annotations

from typing import List
import numpy as np
import cv2


def perturb_images(image: np.ndarray) -> List[np.ndarray]:
    img = image.astype(np.uint8)
    gamma = 1.15
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)], dtype=np.uint8)
    bright = cv2.LUT(img, lut)
    blur = cv2.GaussianBlur(img, (5, 5), 0.8)
    return [bright, blur]
