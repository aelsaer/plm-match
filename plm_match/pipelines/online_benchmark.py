from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np

from plm_match.datasets import build_sequence_dataset
from plm_match.eval import ate_rmse, rpe_stats
from plm_match.pipelines.localize_from_map import PLMMapLocalizer
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_depth, read_image, read_pose_txt, write_json, write_pose_txt
from plm_match.utils.pose import rotation_error_deg, translation_error


def run_online_sequence(cfg: dict, dataset_root: str | Path, out_dir: str | Path) -> dict:
    dataset = build_sequence_dataset(str(dataset_root), cfg.get('sequence_dataset', cfg.get('dataset', {'type': 'scannet_sequence'})))
    out_dir = ensure_dir(out_dir)
    pred_dir = ensure_dir(Path(out_dir) / 'pred_poses')
    model = PLMMapLocalizer(cfg)
    online_cfg = cfg.get('online', {})
    bootstrap_count = int(online_cfg.get('bootstrap_count', min(5, len(dataset.frames))))
    max_age_frames = int(online_cfg.get('max_age_frames', 20))
    max_active_landmarks = int(online_cfg.get('max_active_landmarks', 5000))
    min_staticness = float(online_cfg.get('min_staticness', 0.0))
    integrate_on_success = bool(online_cfg.get('integrate_on_success', True))
    integrate_pose_source = str(online_cfg.get('integrate_pose_source', 'pred')).lower()
    use_gt_bootstrap = bool(online_cfg.get('use_gt_bootstrap', True))
    carry_pose_prior = bool(cfg.get('matching', {}).get('carry_pose_prior', True))

    rows = []
    pred_poses, gt_poses = [], []
    prev_pose = None
    bootstrap_integrated = 0

    for frame_id, frame in enumerate(dataset.frames[:bootstrap_count]):
        image = read_image(frame.image_path)
        depth = read_depth(frame.depth_path) if frame.depth_path is not None else None
        T_wc = frame.pose if frame.pose is not None else (read_pose_txt(frame.pose_path) if frame.pose_path is not None and frame.pose_path.exists() else None)
        if use_gt_bootstrap and depth is not None and T_wc is not None:
            bootstrap_integrated += model.integrate_frame_observations(image, depth, frame.intrinsics, T_wc, frame_id=frame_id, image_name=str(frame.meta.get('relative_path', frame.image_path.name)))
        row = {'frame_id': frame.frame_id, 'query': str(frame.meta.get('relative_path', frame.image_path.name)), 'mode': 'bootstrap', 'success': bool(T_wc is not None), 'num_matches': 0, 'num_inliers': 0}
        rows.append(row)
        if T_wc is not None:
            prev_pose = T_wc
            pred_poses.append(T_wc); gt_poses.append(T_wc)
            write_pose_txt(pred_dir / f'{frame.frame_id}.txt', T_wc)
    model.memory.finalize(); model.refresh_valid_landmarks(current_frame_id=max(0, bootstrap_count-1), max_age_frames=max_age_frames, max_landmarks=max_active_landmarks, min_staticness=min_staticness)

    for frame_id, frame in enumerate(dataset.frames[bootstrap_count:], start=bootstrap_count):
        image = read_image(frame.image_path)
        intr = frame.intrinsics
        gt_pose = frame.pose if frame.pose is not None else (read_pose_txt(frame.pose_path) if frame.pose_path is not None and frame.pose_path.exists() else None)
        depth = read_depth(frame.depth_path) if frame.depth_path is not None else None
        pose_prior = prev_pose if carry_pose_prior else None
        candidate_landmarks = model.memory.filtered_landmarks(min_obs=int(cfg['landmarks'].get('min_obs', 2)), current_frame_id=frame_id-1, max_age_frames=max_age_frames, max_landmarks=max_active_landmarks, min_staticness=min_staticness)
        t0 = time.perf_counter()
        result = model.localize_image(image, intr, pose_prior=pose_prior, candidate_landmarks=candidate_landmarks)
        t1 = time.perf_counter()
        row = {'frame_id': frame.frame_id, 'query': str(frame.meta.get('relative_path', frame.image_path.name)), 'mode': 'online', 'success': bool(result['success']), 'num_matches': int(result['num_matches']), 'num_inliers': int(result['num_inliers']), 'num_candidate_landmarks': len(candidate_landmarks), 'time_total_s': float(t1 - t0)}
        if result['success'] and result['T_wc'] is not None:
            prev_pose = result['T_wc']
            write_pose_txt(pred_dir / f'{frame.frame_id}.txt', result['T_wc'])
            if gt_pose is not None:
                row['rot_err_deg'] = float(rotation_error_deg(result['T_wc'], gt_pose))
                row['trans_err_m'] = float(translation_error(result['T_wc'], gt_pose))
                pred_poses.append(result['T_wc']); gt_poses.append(gt_pose)
            if integrate_on_success and depth is not None:
                pose_for_integration = gt_pose if integrate_pose_source == 'gt' and gt_pose is not None else result['T_wc']
                if pose_for_integration is not None:
                    row['num_integrated_obs'] = model.integrate_frame_observations(image, depth, intr, pose_for_integration, frame_id=frame_id, image_name=str(frame.meta.get('relative_path', frame.image_path.name)))
                    model.memory.finalize()
                    model.memory.prune(current_frame_id=frame_id, max_age_frames=max_age_frames, min_obs=int(cfg['landmarks'].get('min_obs', 2)), min_staticness=min_staticness, max_landmarks=max_active_landmarks)
                    model.refresh_valid_landmarks(current_frame_id=frame_id, max_age_frames=max_age_frames, max_landmarks=max_active_landmarks, min_staticness=min_staticness)
        else:
            prev_pose = None
        rows.append(row)

    summary = model.summarize_metrics(rows)
    summary['bootstrap_count'] = bootstrap_count
    summary['bootstrap_integrated_obs'] = bootstrap_integrated
    summary['final_num_landmarks'] = len(model.valid_landmarks)
    online_rows = [r for r in rows if r.get('mode') == 'online']
    if online_rows:
        summary['mean_time_total_s'] = float(np.mean([r['time_total_s'] for r in online_rows]))
        summary['median_time_total_s'] = float(np.median([r['time_total_s'] for r in online_rows]))
    if pred_poses and gt_poses and len(pred_poses) == len(gt_poses):
        ate = ate_rmse(pred_poses, gt_poses)
        if ate is not None:
            summary['ate_rmse_m'] = ate
        summary.update({k: v for k, v in rpe_stats(pred_poses, gt_poses).items() if v is not None})
    payload = {'dataset': dataset.describe(), 'frames': rows, 'summary': summary}
    write_json(Path(out_dir) / 'metrics_online.json', payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='PLM-MATCH online sliding-window benchmark runner')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset_root = args.dataset_root or cfg.get('dataset_root')
    out_dir = args.out_dir or cfg.get('out_dir', './outputs/online_run')
    if dataset_root is None:
        raise ValueError('dataset_root must be set')
    result = run_online_sequence(cfg, dataset_root, out_dir)
    print(result['summary'])


if __name__ == '__main__':
    main()
