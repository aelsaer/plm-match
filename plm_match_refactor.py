from __future__ import annotations

"""
PLM-Match refactor utilities.

This module is intentionally self-contained so it can be dropped into an
existing research codebase with minimal wiring. It focuses on two problems
observed in the provided failure case:

1) runtime is dominated by candidate materialization and per-anchor scoring;
2) DINO-only descriptors are ambiguous on repetitive structures.

Key changes implemented here:
- fully vectorized batched retrieval and scoring in PyTorch;
- PPCA-style landmark likelihood instead of orthogonal-residual-only scoring;
- optional fine descriptor fusion for re-ranking;
- view-envelope gating and support/staticness priors;
- reverse uniqueness pruning to keep at most one anchor per landmark.

Typical usage:
    memory = LandmarkMemory(...)
    matcher = PLMMatcher(memory, device='cuda')
    out = matcher.match(query_desc, query_xy, query_view_dirs=None)

Shapes:
    M = number of landmarks
    N = number of query anchors
    D = coarse descriptor dimension (e.g. DINO token dim)
    R = low-rank basis dimension
    F = optional fine descriptor dimension
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _l2n(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


@dataclass
class LandmarkMemory:
    # Geometry
    xyz: Tensor                    # [M, 3]

    # Coarse descriptor model (e.g. DINO / FM tokens)
    mu: Tensor                     # [M, D], normalized means
    basis: Tensor                  # [M, D, R]
    eigvals: Tensor                # [M, R], retained variances
    sigma_perp2: Tensor            # [M], residual variance outside basis

    # Reliability
    staticness: Tensor             # [M], typically [0,1]
    support_count: Tensor          # [M], number of observations / track length

    # Optional view envelope, encoded as cosine range to the landmark's mean view direction.
    mean_view_dir: Optional[Tensor] = None   # [M, 3]
    min_view_cos: Optional[Tensor] = None    # [M]
    max_view_cos: Optional[Tensor] = None    # [M]

    # Optional fine descriptor for re-ranking (e.g. XFeat/SOSNet/HardNet patch descriptor)
    fine_desc: Optional[Tensor] = None       # [M, F], normalized

    # Optional landmark id bookkeeping.
    ids: Optional[Tensor] = None             # [M]

    def to(self, device: str | torch.device) -> "LandmarkMemory":
        kwargs = {}
        for field in self.__dataclass_fields__:
            val = getattr(self, field)
            kwargs[field] = val.to(device) if isinstance(val, torch.Tensor) else val
        return LandmarkMemory(**kwargs)

    @property
    def num_landmarks(self) -> int:
        return int(self.mu.shape[0])

    @property
    def dim(self) -> int:
        return int(self.mu.shape[1])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[2])


@dataclass
class MatchConfig:
    # Candidate retrieval
    topk_retrieval: int = 32
    retrieval_chunk_size: int = 16384
    anchor_batch_size: int = 256

    # Gating
    coarse_cosine_min: float = -1.0
    margin_min: float = 0.02
    score_min: float = -1e9
    mutual_keep_best_per_landmark: bool = True

    # Scoring weights
    w_coarse_cos: float = 1.0
    w_ppca_parallel: float = 0.25
    w_ppca_perp: float = 0.50
    w_staticness: float = 0.15
    w_support: float = 0.05
    w_view: float = 0.15
    w_fine_cos: float = 0.35

    # Numerical settings
    eps: float = 1e-6


@dataclass
class MatchOutput:
    anchor_indices: Tensor         # [K]
    landmark_indices: Tensor       # [K]
    scores: Tensor                 # [K]
    margins: Tensor                # [K]
    coarse_cos: Tensor             # [K]
    fine_cos: Optional[Tensor]     # [K] or None
    xy: Tensor                     # [K, 2]
    xyz: Tensor                    # [K, 3]


class PLMMatcher:
    def __init__(self, memory: LandmarkMemory, config: Optional[MatchConfig] = None, device: str | torch.device = 'cpu'):
        self.cfg = config or MatchConfig()
        self.device = torch.device(device)
        self.memory = memory.to(self.device)

        # Keep memory tensors contiguous and compact for fast gathers/matmuls.
        self.memory.mu = _l2n(self.memory.mu.contiguous())
        self.memory.basis = self.memory.basis.contiguous()
        self.memory.eigvals = self.memory.eigvals.contiguous()
        self.memory.sigma_perp2 = self.memory.sigma_perp2.contiguous().clamp_min(self.cfg.eps)
        self.memory.staticness = self.memory.staticness.contiguous()
        self.memory.support_count = self.memory.support_count.contiguous()
        if self.memory.fine_desc is not None:
            self.memory.fine_desc = _l2n(self.memory.fine_desc.contiguous())
        if self.memory.mean_view_dir is not None:
            self.memory.mean_view_dir = _l2n(self.memory.mean_view_dir.contiguous())

    @torch.no_grad()
    def match(
        self,
        query_desc: Tensor,             # [N, D]
        query_xy: Tensor,               # [N, 2]
        query_view_dirs: Optional[Tensor] = None,   # [N, 3]
        query_fine_desc: Optional[Tensor] = None,   # [N, F]
    ) -> MatchOutput:
        cfg = self.cfg
        mem = self.memory

        qd = _l2n(query_desc.to(self.device).contiguous())
        qxy = query_xy.to(self.device).contiguous()
        qv = _l2n(query_view_dirs.to(self.device).contiguous()) if query_view_dirs is not None else None
        qf = _l2n(query_fine_desc.to(self.device).contiguous()) if query_fine_desc is not None else None

        n = qd.shape[0]
        keep_anchor_idx = []
        keep_landmark_idx = []
        keep_scores = []
        keep_margins = []
        keep_coarse = []
        keep_fine = []

        # Process anchors in batches to avoid repeated Python-level candidate materialization.
        for start in range(0, n, cfg.anchor_batch_size):
            end = min(start + cfg.anchor_batch_size, n)
            qd_b = qd[start:end]  # [B, D]
            B = qd_b.shape[0]

            coarse_vals, coarse_idx = self._topk_cosine(qd_b, mem.mu, cfg.topk_retrieval)

            scores, fine_vals = self._score_candidates(
                qd_b,
                coarse_idx,
                coarse_vals,
                q_view_dirs=qv[start:end] if qv is not None else None,
                q_fine=qf[start:end] if qf is not None else None,
            )

            best_scores, best_pos = scores.max(dim=1)
            best_landmarks = coarse_idx.gather(1, best_pos[:, None]).squeeze(1)
            best_coarse = coarse_vals.gather(1, best_pos[:, None]).squeeze(1)
            best_fine = None if fine_vals is None else fine_vals.gather(1, best_pos[:, None]).squeeze(1)

            # Margin over 2nd-best candidate.
            masked = scores.clone()
            masked.scatter_(1, best_pos[:, None], float('-inf'))
            second_scores = masked.max(dim=1).values
            margins = best_scores - second_scores

            valid = (best_scores >= cfg.score_min) & (margins >= cfg.margin_min) & (best_coarse >= cfg.coarse_cosine_min)
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            keep_anchor_idx.append(valid_idx + start)
            keep_landmark_idx.append(best_landmarks[valid_idx])
            keep_scores.append(best_scores[valid_idx])
            keep_margins.append(margins[valid_idx])
            keep_coarse.append(best_coarse[valid_idx])
            if best_fine is not None:
                keep_fine.append(best_fine[valid_idx])

        if not keep_anchor_idx:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            emptyf = torch.empty(0, device=self.device)
            return MatchOutput(
                anchor_indices=empty,
                landmark_indices=empty,
                scores=emptyf,
                margins=emptyf,
                coarse_cos=emptyf,
                fine_cos=emptyf if query_fine_desc is not None and mem.fine_desc is not None else None,
                xy=qxy[:0],
                xyz=mem.xyz[:0],
            )

        anchor_indices = torch.cat(keep_anchor_idx, dim=0)
        landmark_indices = torch.cat(keep_landmark_idx, dim=0)
        scores = torch.cat(keep_scores, dim=0)
        margins = torch.cat(keep_margins, dim=0)
        coarse_cos = torch.cat(keep_coarse, dim=0)
        fine_cos = torch.cat(keep_fine, dim=0) if keep_fine else None

        if cfg.mutual_keep_best_per_landmark:
            anchor_indices, landmark_indices, scores, margins, coarse_cos, fine_cos = self._keep_best_per_landmark(
                anchor_indices, landmark_indices, scores, margins, coarse_cos, fine_cos
            )

        return MatchOutput(
            anchor_indices=anchor_indices,
            landmark_indices=landmark_indices,
            scores=scores,
            margins=margins,
            coarse_cos=coarse_cos,
            fine_cos=fine_cos,
            xy=qxy[anchor_indices],
            xyz=mem.xyz[landmark_indices],
        )

    @torch.no_grad()
    def _topk_cosine(self, qd: Tensor, mu: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
        """Chunked cosine retrieval over landmark means."""
        B, D = qd.shape
        M = mu.shape[0]
        k = min(topk, M)

        best_vals = torch.full((B, k), float('-inf'), device=qd.device)
        best_idx = torch.full((B, k), -1, device=qd.device, dtype=torch.long)

        for s in range(0, M, self.cfg.retrieval_chunk_size):
            e = min(s + self.cfg.retrieval_chunk_size, M)
            sims = qd @ mu[s:e].T  # [B, m]
            vals, idx = torch.topk(sims, k=min(k, sims.shape[1]), dim=1)
            idx = idx + s
            cat_vals = torch.cat([best_vals, vals], dim=1)
            cat_idx = torch.cat([best_idx, idx], dim=1)
            new_vals, order = torch.topk(cat_vals, k=k, dim=1)
            new_idx = cat_idx.gather(1, order)
            best_vals, best_idx = new_vals, new_idx

        return best_vals, best_idx

    @torch.no_grad()
    def _score_candidates(
        self,
        qd: Tensor,                    # [B, D]
        cand_idx: Tensor,              # [B, K]
        coarse_cos: Tensor,            # [B, K]
        q_view_dirs: Optional[Tensor],
        q_fine: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Score with a PPCA-style descriptor likelihood.

        Why this is better than residual-only scoring:
        - residual-only only penalizes distance *orthogonal* to the landmark subspace;
          descriptors very far *within* the subspace can still score well.
        - the PPCA log-likelihood penalizes both along-subspace displacement and
          orthogonal residual, scaled by the landmark's learned variances.
        """
        cfg = self.cfg
        mem = self.memory
        B, K = cand_idx.shape
        D = qd.shape[1]

        flat = cand_idx.reshape(-1)
        mu = mem.mu.index_select(0, flat).view(B, K, D)                  # [B, K, D]
        basis = mem.basis.index_select(0, flat)                          # [B*K, D, R]
        eigvals = mem.eigvals.index_select(0, flat).view(B, K, -1)       # [B, K, R]
        sigma_perp2 = mem.sigma_perp2.index_select(0, flat).view(B, K)   # [B, K]
        staticness = mem.staticness.index_select(0, flat).view(B, K)     # [B, K]
        support = mem.support_count.index_select(0, flat).view(B, K)     # [B, K]

        z = qd[:, None, :] - mu                                           # [B, K, D]
        # Projection coefficients in each landmark basis.
        coeff = torch.einsum('bkd,bkdr->bkr', z, basis.view(B, K, D, -1)) # [B, K, R]
        parallel2 = (coeff.square() / eigvals.clamp_min(cfg.eps)).sum(dim=-1)
        z2 = z.square().sum(dim=-1)
        coeff2 = coeff.square().sum(dim=-1)
        # ||z - UU^T z||^2 = ||z||^2 - ||U^T z||^2 when U is orthonormal.
        perp2 = (z2 - coeff2).clamp_min(0.0) / sigma_perp2.clamp_min(cfg.eps)

        score = (
            cfg.w_coarse_cos * coarse_cos
            - cfg.w_ppca_parallel * parallel2
            - cfg.w_ppca_perp * perp2
            + cfg.w_staticness * staticness
            + cfg.w_support * torch.log1p(support)
        )

        # Optional view envelope compatibility.
        if q_view_dirs is not None and mem.mean_view_dir is not None and mem.min_view_cos is not None and mem.max_view_cos is not None:
            mean_view = mem.mean_view_dir.index_select(0, flat).view(B, K, 3)
            min_c = mem.min_view_cos.index_select(0, flat).view(B, K)
            max_c = mem.max_view_cos.index_select(0, flat).view(B, K)
            qcos = (q_view_dirs[:, None, :] * mean_view).sum(dim=-1)
            in_range = (qcos >= min_c) & (qcos <= max_c)
            # Soft penalty outside the learned view envelope.
            lo_pen = (min_c - qcos).clamp_min(0)
            hi_pen = (qcos - max_c).clamp_min(0)
            view_score = in_range.float() - (lo_pen + hi_pen)
            score = score + cfg.w_view * view_score

        fine_vals = None
        if q_fine is not None and mem.fine_desc is not None:
            fine = mem.fine_desc.index_select(0, flat).view(B, K, -1)
            fine_vals = (q_fine[:, None, :] * fine).sum(dim=-1)
            score = score + cfg.w_fine_cos * fine_vals

        return score, fine_vals

    @torch.no_grad()
    def _keep_best_per_landmark(
        self,
        anchor_idx: Tensor,
        landmark_idx: Tensor,
        scores: Tensor,
        margins: Tensor,
        coarse_cos: Tensor,
        fine_cos: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Reverse uniqueness: keep the best-scoring anchor for each landmark."""
        order = torch.argsort(scores, descending=True)
        anchor_idx = anchor_idx[order]
        landmark_idx = landmark_idx[order]
        scores = scores[order]
        margins = margins[order]
        coarse_cos = coarse_cos[order]
        fine_cos = fine_cos[order] if fine_cos is not None else None

        unique_landmarks = {}
        keep = []
        # Simple Python loop here is fine because it runs after heavy pruning.
        for i, lid in enumerate(landmark_idx.tolist()):
            if lid not in unique_landmarks:
                unique_landmarks[lid] = True
                keep.append(i)
        keep = torch.tensor(keep, device=anchor_idx.device, dtype=torch.long)
        return (
            anchor_idx[keep],
            landmark_idx[keep],
            scores[keep],
            margins[keep],
            coarse_cos[keep],
            fine_cos[keep] if fine_cos is not None else None,
        )


@torch.no_grad()
def build_landmark_memory_from_tracks(
    xyz: Tensor,
    track_ids: Tensor,
    descriptors: Tensor,
    staticness: Optional[Tensor] = None,
    support_count: Optional[Tensor] = None,
    rank: int = 4,
    residual_floor: float = 1e-3,
) -> LandmarkMemory:
    """
    Build per-landmark PPCA summaries from repeated observations.

    Args:
        xyz: [M, 3] landmark positions in the same order as unique track ids.
        track_ids: [T] integer track id for each descriptor observation.
        descriptors: [T, D] descriptor observations for landmarks.
        staticness: [M] or None.
        support_count: [M] or None.

    Returns:
        LandmarkMemory with basis/eigenvalues/sigma_perp2 ready for matching.
    """
    device = descriptors.device
    desc = _l2n(descriptors)
    unique_ids, inv = torch.unique(track_ids, sorted=True, return_inverse=True)
    M = unique_ids.numel()
    D = desc.shape[1]
    R = min(rank, D)

    mu = torch.zeros(M, D, device=device)
    cnt = torch.zeros(M, device=device)
    mu.index_add_(0, inv, desc)
    cnt.index_add_(0, inv, torch.ones_like(inv, dtype=desc.dtype))
    mu = _l2n(mu / cnt[:, None].clamp_min(1.0))

    basis = torch.zeros(M, D, R, device=device)
    eigvals = torch.full((M, R), residual_floor, device=device)
    sigma_perp2 = torch.full((M,), residual_floor, device=device)

    # Track-wise PPCA fitting. This loop happens offline when building memory.
    for m in range(M):
        obs = desc[inv == m]
        if obs.shape[0] <= 1:
            continue
        z = obs - mu[m:m+1]
        cov = z.T @ z / max(1, obs.shape[0] - 1)
        # eigh is stable for symmetric covariances.
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.flip(0)
        evecs = evecs.flip(1)
        r = min(R, obs.shape[0] - 1, D)
        basis[m, :, :r] = evecs[:, :r]
        eigvals[m, :r] = evals[:r].clamp_min(residual_floor)
        if r < D:
            tail = evals[r:]
            sigma_perp2[m] = tail.mean().clamp_min(residual_floor) if tail.numel() > 0 else residual_floor

    if staticness is None:
        staticness = torch.ones(M, device=device)
    if support_count is None:
        support_count = cnt

    return LandmarkMemory(
        xyz=xyz,
        mu=mu,
        basis=basis,
        eigvals=eigvals,
        sigma_perp2=sigma_perp2,
        staticness=staticness,
        support_count=support_count,
        ids=unique_ids,
    )


def suggest_default_config(num_landmarks: int, gpu: bool = True) -> MatchConfig:
    """Reasonable defaults for Aachen-scale local candidate sets."""
    if gpu:
        chunk = 32768 if num_landmarks >= 32768 else 16384
        batch = 512
    else:
        chunk = 8192
        batch = 128
    return MatchConfig(
        topk_retrieval=32,
        retrieval_chunk_size=chunk,
        anchor_batch_size=batch,
        coarse_cosine_min=0.20,
        margin_min=0.03,
        w_coarse_cos=1.0,
        w_ppca_parallel=0.20,
        w_ppca_perp=0.50,
        w_staticness=0.15,
        w_support=0.05,
        w_view=0.15,
        w_fine_cos=0.35,
    )
