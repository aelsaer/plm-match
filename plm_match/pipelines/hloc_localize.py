from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from plm_match.datasets import build_dataset
from plm_match.hloc import parse_retrieval_file, retrieval_db_image_names
from plm_match.pipelines.localize_from_map import PLMMapLocalizer
from plm_match.types import LandmarkCandidateGroup, LandmarkCandidateSchedule
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, write_json
from plm_match.utils.pose import invert_pose, pose_to_quat_t


def write_hloc_results(path: str | Path, rows: list[tuple[str, object]]) -> None:
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        for name, T_wc in rows:
            T_cw = invert_pose(T_wc)
            q, t = pose_to_quat_t(T_cw)
            f.write(f"{name} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description='PLM-MATCH HLoc/COLMAP custom localizer')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--disable-map-cache', action='store_true')
    parser.add_argument('--disable-query-cache', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.disable_map_cache:
        cfg.setdefault('map', {})
        cfg['map']['cache_path'] = None
    cfg.setdefault('map', {})
    if args.disable_query_cache:
        cfg['map']['use_query_cache'] = False
    cfg['map']['require_compact_store'] = True
    dataset_root = args.dataset_root or cfg.get('dataset_root')
    out_dir = ensure_dir(args.out_dir or cfg.get('out_dir', './outputs/hloc_plm'))
    if dataset_root is None:
        raise ValueError('dataset_root must be set')

    dataset = build_dataset(dataset_root, cfg.get('dataset', {'type': 'colmap_localization'}))
    localizer = PLMMapLocalizer(cfg)
    localizer.build_map(dataset)

    retrieval_path = args.retrieval_file or cfg.get('hloc', {}).get('retrieval_file')
    if retrieval_path:
        rp = Path(retrieval_path)
        if not rp.is_absolute():
            # support paths relative to repo CWD as well as dataset root
            rp_dataset = Path(dataset_root) / rp
            rp = rp if rp.exists() else rp_dataset
        retrievals = parse_retrieval_file(rp)
    else:
        retrievals = {}
    topk_images = int(cfg.get('hloc', {}).get('topk_db_images', 20))
    max_candidate_landmarks_raw = cfg.get('hloc', {}).get('max_candidate_landmarks', 4000)
    max_candidate_landmarks = int(max_candidate_landmarks_raw) if max_candidate_landmarks_raw is not None else -1
    max_index_landmarks_per_image = cfg.get('hloc', {}).get('max_index_landmarks_per_image', None)
    verify_image_batch_size = max(1, int(cfg.get('hloc', {}).get('verify_image_batch_size', 1)))
    verify_image_schedule_cfg = cfg.get('hloc', {}).get('verify_image_schedule', [4, 10, 20])
    manifold_rank = int(cfg.get('landmarks', {}).get('manifold_rank', 0))
    include_view = float(cfg.get('matching', {}).get('lambdas', [0, 0, 0, 0, 0])[3]) != 0.0

    # Build compact HLoc candidate provider directly from the compact store if available.
    if localizer.landmark_store is None:
        raise RuntimeError(
            'This HLoc entrypoint expects a compact COLMAP landmark store. '
            'If you have an older landmarks.pkl cache, rerun once so the compact '
            'landmarks_store cache can be rebuilt.'
        )
    store = localizer.landmark_store
    map_frames = dataset.get_map_frames()
    name_to_frame_id = {str(fr.meta.get('relative_path', fr.image_path.name)): i for i, fr in enumerate(map_frames)}

    restrict_db_images = retrieval_db_image_names(retrievals, topk_images=topk_images) if retrievals else None
    restrict_frame_ids = None
    if restrict_db_images:
        restrict_frame_ids = {name_to_frame_id[name] for name in restrict_db_images if name in name_to_frame_id}
    if not store.image_to_landmarks:
        store.build_image_to_landmarks_index(
            restrict_to_frame_ids=restrict_frame_ids,
            max_landmarks_per_image=(int(max_index_landmarks_per_image) if max_index_landmarks_per_image is not None else None),
        )
    elif restrict_frame_ids is not None:
        # Rebuild a compact restricted index to avoid scanning all landmarks on large maps.
        store.build_image_to_landmarks_index(
            restrict_to_frame_ids=restrict_frame_ids,
            max_landmarks_per_image=(int(max_index_landmarks_per_image) if max_index_landmarks_per_image is not None else None),
        )

    def candidate_provider(frame):
        qname = str(frame.meta.get('relative_path', frame.image_path.name))
        db_names = retrievals.get(qname, [])[:topk_images] if retrievals else []
        groups: list[LandmarkCandidateGroup] = []
        candidate_total = 0
        if db_names:
            valid_pairs = [(name, name_to_frame_id[name]) for name in db_names if name in name_to_frame_id]
            for start in range(0, len(valid_pairs), verify_image_batch_size):
                batch = valid_pairs[start : start + verify_image_batch_size]
                image_names = tuple(name for name, _ in batch)
                frame_ids = tuple(int(fid) for _, fid in batch)
                merged_list = []
                seen: set[int] = set()
                for fid in frame_ids:
                    idxs = store.image_to_landmarks.get(int(fid), np.zeros((0,), dtype=np.int32))
                    for idx in idxs:
                        idx = int(idx)
                        if idx in seen:
                            continue
                        seen.add(idx)
                        merged_list.append(idx)
                merged = np.asarray(merged_list, dtype=np.int32) if merged_list else np.zeros((0,), dtype=np.int32)
                if max_candidate_landmarks > 0:
                    remaining = max_candidate_landmarks - candidate_total
                    if remaining <= 0:
                        break
                    if merged.shape[0] > remaining:
                        merged = merged[:remaining]
                candidate_total += int(merged.shape[0])
                groups.append(
                    LandmarkCandidateGroup(
                        image_names=image_names,
                        frame_ids=frame_ids,
                        landmark_indices=merged,
                        meta={
                            'candidate_indices': int(merged.shape[0]),
                            'batch_size': int(len(image_names)),
                        },
                        lazy_context={
                            'store': store,
                            'indices': merged.astype(np.int32, copy=False),
                            'dataset': dataset,
                            'extractor': localizer.extractor,
                            'manifold_rank': manifold_rank,
                            'feature_cache': localizer._map_feature_cache,
                            'include_view_dirs': include_view,
                            'preferred_frame_ids': frame_ids,
                        },
                    )
                )
        else:
            idxs = store.top_landmarks(max_landmarks=max_candidate_landmarks)
            candidate_total = int(idxs.shape[0])
            groups.append(
                LandmarkCandidateGroup(
                    image_names=('__global__',),
                    frame_ids=tuple(),
                    landmark_indices=idxs.astype(np.int32, copy=False),
                    meta={'candidate_indices': int(idxs.shape[0]), 'batch_size': 1},
                    lazy_context={
                        'store': store,
                        'indices': idxs.astype(np.int32, copy=False),
                        'dataset': dataset,
                        'extractor': localizer.extractor,
                        'manifold_rank': manifold_rank,
                        'feature_cache': localizer._map_feature_cache,
                        'include_view_dirs': include_view,
                        'preferred_frame_ids': tuple(),
                    },
                )
            )

        stage_limits = []
        total_images = max(1, len(db_names)) if db_names else 1
        for value in verify_image_schedule_cfg:
            try:
                limit = int(value)
            except (TypeError, ValueError):
                continue
            if limit > 0:
                stage_limits.append(min(limit, total_images))
        stage_limits.append(total_images)
        stage_limits = sorted(set(stage_limits))

        return LandmarkCandidateSchedule(
            groups=groups,
            stages=stage_limits,
            meta={
                'db_images': int(len(db_names)),
                'candidate_indices_total': int(candidate_total),
                'grouped_verification': True,
                'verify_image_batch_size': int(verify_image_batch_size),
            },
        )

    result = localizer.localize_queries(dataset, out_dir, candidate_provider=candidate_provider)
    pose_rows = []
    pred_dir = Path(out_dir) / 'pred_poses'
    for frame in dataset.get_query_frames():
        p = pred_dir / f'{localizer._query_cache_key(frame)}.txt'
        if p.exists():
            T_wc = np.loadtxt(p, dtype=np.float64).reshape(4, 4)
            pose_rows.append((str(frame.meta.get('relative_path', frame.image_path.name)), T_wc))
    write_hloc_results(Path(out_dir) / 'hloc_results.txt', pose_rows)
    summary = dict(result['summary'])
    summary['num_landmarks'] = int(store.num_landmarks)
    write_json(Path(out_dir) / 'run_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
