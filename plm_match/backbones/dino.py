from __future__ import annotations

import os
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


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        input_size: Tuple[int, int] = (336, 336),
        resize_mode: str = 'square',
        device: str = 'cpu',
    ):
        self.model_name = model_name
        self.input_size = input_size
        self.resize_mode = str(resize_mode or 'square').lower()
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


class DINOv3FeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        repo_dir: str | None = None,
        weights_path: str | None = None,
        model_name: str = 'dinov3_vits16',
        input_size: Tuple[int, int] = (512, 512),
        resize_mode: str = 'square',
        device: str = 'cpu',
    ):
        repo_dir = os.path.expandvars(repo_dir) if repo_dir else None
        weights_path = os.path.expandvars(weights_path) if weights_path else None
        self.repo_dir = Path(repo_dir).expanduser() if repo_dir else None
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.model_name = model_name
        self.input_size = input_size
        self.resize_mode = str(resize_mode or 'square').lower()
        self._device = device
        self._use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
        if self._use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        _ensure_writable_torch_hub_dir()
        if self.weights_path is None:
            raise FileNotFoundError(
                'DINOv3 weights_path is required. Meta DINOv3 checkpoints are not '
                'always downloadable through torch.hub and may return HTTP 403. '
                'Set backbone.weights_path or pass --override backbone.weights_path=/path/to/dinov3_vits16.pth.'
            )
        if not self.weights_path.exists():
            raise FileNotFoundError(f'DINOv3 weights_path does not exist: {self.weights_path}')
        self._backend = 'torchhub'
        self._num_register_tokens = 0
        hf_root = self.weights_path.parent
        if self.weights_path.suffix == '.safetensors' and (hf_root / 'config.json').exists():
            from transformers import AutoConfig, AutoModel

            config = AutoConfig.from_pretrained(str(hf_root), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(hf_root), local_files_only=True).to(device)
            self._backend = 'transformers'
            self.patch = int(getattr(config, 'patch_size', 16))
            self._dim = int(getattr(config, 'hidden_size', 0))
            self._num_register_tokens = int(getattr(config, 'num_register_tokens', 0) or 0)
        else:
            hub_kwargs = {'weights': str(self.weights_path)}
            if self.repo_dir is not None:
                if not self.repo_dir.exists():
                    raise FileNotFoundError(f'DINOv3 repo_dir does not exist: {self.repo_dir}')
                self.model = torch.hub.load(str(self.repo_dir), model_name, source='local', **hub_kwargs).to(device)
            else:
                self.model = torch.hub.load('facebookresearch/dinov3', model_name, trust_repo=True, **hub_kwargs).to(device)
            patch_size = getattr(getattr(self.model, 'patch_embed', None), 'patch_size', (16, 16))
            if isinstance(patch_size, tuple):
                self.patch = int(patch_size[0])
            else:
                self.patch = int(patch_size)
            self._dim = int(getattr(self.model, 'embed_dim', 0))
        self.model.eval()
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

    def _extract_patch_and_cls(self, feats) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(feats, dict):
            patch = None
            for key in ('x_norm_patchtokens', 'x_patchtokens', 'patch_tokens', 'tokens'):
                if key in feats and feats[key] is not None:
                    patch = feats[key]
                    break
            cls = None
            for key in ('x_norm_clstoken', 'x_clstoken', 'cls_token'):
                if key in feats and feats[key] is not None:
                    cls = feats[key]
                    break
            if patch is None:
                raise KeyError(f'DINOv3 forward_features output has no patch-token key: {list(feats.keys())}')
            return patch, cls
        if torch.is_tensor(feats):
            if feats.ndim == 3:
                return feats, feats[:, 0] if feats.shape[1] > 1 else None
            return feats, None
        if isinstance(feats, (tuple, list)) and feats:
            first = feats[0]
            if torch.is_tensor(first):
                return first, first[:, 0] if first.ndim == 3 and first.shape[1] > 1 else None
        raise TypeError(f'Unsupported DINOv3 forward_features output type: {type(feats)!r}')

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> Dict[str, object]:
        x, resize_info = self._prep(image)
        if self._use_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if self._backend == 'transformers':
                    outputs = self.model(pixel_values=x)
                else:
                    feats = self.model.forward_features(x)
        else:
            if self._backend == 'transformers':
                outputs = self.model(pixel_values=x)
            else:
                feats = self.model.forward_features(x)
        if self._backend == 'transformers':
            hidden = outputs.last_hidden_state
            cls_token_t = hidden[:, 0]
            patch_start = 1 + self._num_register_tokens
            patch_tokens_t = hidden[:, patch_start:]
        else:
            patch_tokens_t, cls_token_t = self._extract_patch_and_cls(feats)
        patch_tokens = patch_tokens_t[0].detach().float().cpu().numpy()
        n_tokens = int(patch_tokens.shape[0])
        Ht = int(x.shape[-2]) // self.patch
        Wt = int(x.shape[-1]) // self.patch
        if Ht * Wt != n_tokens:
            side = int(round(np.sqrt(n_tokens)))
            if side * side != n_tokens:
                raise ValueError(
                    f'Cannot reshape {n_tokens} DINOv3 tokens into a grid. '
                    f'Expected {Ht}x{Wt} from input_size={self.input_size}, patch={self.patch}.'
                )
            Ht = side
            Wt = side
        tokens = patch_tokens.reshape(Ht, Wt, -1).astype(np.float32)
        if self._dim <= 0:
            self._dim = int(tokens.shape[-1])
        tokens = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
        token_xy = token_xy_from_resize(resize_info, patch_size=self.patch, grid_h=Ht, grid_w=Wt)
        cls_token = cls_token_t[0].detach().float().cpu() if cls_token_t is not None else torch.zeros(tokens.shape[-1])
        return {
            'tokens': tokens,
            'token_xy': token_xy,
            'global_desc': cls_token,
        }
