from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Dict, Tuple
import numpy as np
import torch
import cv2

from .base import BaseFeatureExtractor
from .resize import ResizeInfo, resize_for_patch_backbone, token_xy_from_resize


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


class EUPEFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        repo_dir: str,
        weights_path: str | None = None,
        model_name: str = 'eupe_vits16',
        input_size: Tuple[int, int] = (256, 256),
        resize_mode: str = 'square',
        device: str = 'cpu',
    ):
        self.repo_dir = repo_dir
        self.weights_path = weights_path
        self.model_name = model_name
        self.input_size = input_size
        self.resize_mode = str(resize_mode or 'square').lower()
        self._device = device
        self._use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
        if self._use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        _ensure_writable_torch_hub_dir()
        load_kwargs = {'source': 'local'}
        if weights_path not in (None, ''):
            load_kwargs['weights'] = weights_path
        self.model = torch.hub.load(repo_dir, model_name, **load_kwargs).to(device)
        self.model.eval()
        self.patch = 16
        self._dim = 384 if 'vits' in model_name else 192 if 'vitt' in model_name else 768
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

    def _prep(self, image: np.ndarray) -> tuple[torch.Tensor, ResizeInfo]:
        img_resized, info = resize_for_patch_backbone(
            image,
            self.input_size,
            patch_size=self.patch,
            mode=self.resize_mode,
        )
        img = img_resized.astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        x = (x - self.mean) / self.std
        return x, info

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> Dict[str, object]:
        x, resize_info = self._prep(image)
        if self._use_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = self.model.forward_features(x)
        else:
            feats = self.model.forward_features(x)
        patch_tokens = feats['x_norm_patchtokens'][0].detach().cpu().numpy()
        cls_token = feats['x_norm_clstoken'][0].detach().cpu()
        Ht = int(x.shape[-2]) // self.patch
        Wt = int(x.shape[-1]) // self.patch
        tokens = patch_tokens.reshape(Ht, Wt, -1).astype(np.float32)
        tokens = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
        token_xy = token_xy_from_resize(resize_info, patch_size=self.patch, grid_h=Ht, grid_w=Wt)
        return {
            'tokens': tokens,
            'token_xy': token_xy,
            'global_desc': cls_token,
        }
