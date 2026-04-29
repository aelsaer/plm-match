from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Dict, Tuple
import numpy as np
import torch
import cv2

from .base import BaseFeatureExtractor


def _ensure_writable_torch_hub_dir() -> None:
    hub_dir = Path(torch.hub.get_dir())
    try:
        hub_dir.mkdir(parents=True, exist_ok=True)
        probe = hub_dir / '.write_probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        return
    except OSError:
        fallback = Path(tempfile.gettempdir()) / 'torch-hub'
        fallback.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(fallback))


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_name: str = 'dinov2_vits14', input_size: Tuple[int, int] = (336, 336), device: str = 'cpu'):
        self.model_name = model_name
        self.input_size = input_size
        self._device = device
        self._use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
        if self._use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # Some sandboxed environments expose a read-only home cache. Fall back to
        # a writable temp hub dir before torch.hub tries to clone/load DINOv2.
        _ensure_writable_torch_hub_dir()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()
        patch_size = getattr(getattr(self.model, 'patch_embed', None), 'patch_size', (14, 14))
        if isinstance(patch_size, tuple):
            self.patch = int(patch_size[0])
        else:
            self.patch = int(patch_size)
        self._dim = int(getattr(self.model, 'embed_dim', 384))
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @property
    def device(self) -> str:
        return self._device

    @property
    def output_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self.model_name

    def _prep(self, image: np.ndarray) -> torch.Tensor:
        h, w = self.input_size
        img = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> Dict[str, object]:
        orig_h, orig_w = image.shape[:2]
        x = self._prep(image)
        if self._use_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = self.model.forward_features(x)
        else:
            feats = self.model.forward_features(x)
        patch_tokens = feats['x_norm_patchtokens'][0].detach().cpu().numpy()
        cls_token = feats['x_norm_clstoken'][0].detach().cpu()
        Ht = self.input_size[0] // self.patch
        Wt = self.input_size[1] // self.patch
        tokens = patch_tokens.reshape(Ht, Wt, -1).astype(np.float32)
        tokens = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
        # Patch tokens correspond to patch centers in the resized image, not to the
        # image borders. Using edge-to-edge coordinates introduces a systematic
        # offset when sampling descriptors at COLMAP pixels and when turning query
        # tokens into 2D points for PnP.
        resized_h = int(self.input_size[0])
        resized_w = int(self.input_size[1])
        center_x = ((np.arange(Wt, dtype=np.float32) + 0.5) * float(self.patch)) - 0.5
        center_y = ((np.arange(Ht, dtype=np.float32) + 0.5) * float(self.patch)) - 0.5
        scale_x = float(orig_w - 1) / max(1.0, float(resized_w - 1))
        scale_y = float(orig_h - 1) / max(1.0, float(resized_h - 1))
        xx, yy = np.meshgrid(center_x * scale_x, center_y * scale_y, indexing='xy')
        token_xy = np.stack([xx, yy], axis=-1)
        return {
            'tokens': tokens,
            'token_xy': token_xy.astype(np.float32),
            'global_desc': cls_token,
        }
