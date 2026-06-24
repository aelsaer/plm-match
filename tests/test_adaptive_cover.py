import unittest

import numpy as np

from plm_match.pipelines.lifted_nn_localize import (
    _select_point_observation_indices,
    adaptive_cover_select,
    compute_obs_weights,
)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class AdaptiveCoverSelectionTest(unittest.TestCase):
    def test_compact_descriptors_stop_near_minimum(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        descs = _norm(base + 1e-4 * np.arange(8, dtype=np.float32).reshape(-1, 1))
        selected = adaptive_cover_select(descs, k_min=1, k_max=8, min_gain=0.005)
        self.assertGreaterEqual(selected.shape[0], 1)
        self.assertLessEqual(selected.shape[0], 2)

    def test_diverse_descriptors_expand_until_budget(self) -> None:
        descs = np.eye(6, dtype=np.float32)
        selected = adaptive_cover_select(descs, k_min=1, k_max=4, min_gain=0.005)
        self.assertEqual(selected.shape[0], 4)
        self.assertEqual(len(set(selected.tolist())), 4)

    def test_view_diversity_can_expand_identical_descriptors(self) -> None:
        descs = _norm(np.repeat(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), repeats=4, axis=0))
        view_dirs = _norm(
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        selected_no_view = adaptive_cover_select(
            descs,
            view_dirs=view_dirs,
            k_min=1,
            k_max=4,
            min_gain=0.005,
            view_weight=0.0,
        )
        selected_view = adaptive_cover_select(
            descs,
            view_dirs=view_dirs,
            k_min=1,
            k_max=4,
            min_gain=0.005,
            view_weight=1.0,
        )
        self.assertEqual(selected_no_view.shape[0], 1)
        self.assertEqual(selected_view.shape[0], 4)

    def test_adaptive_mode_uses_view_directions_from_frame_centers(self) -> None:
        descs = _norm(np.repeat(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), repeats=4, axis=0))
        selected = _select_point_observation_indices(
            descs,
            0,
            4,
            max_obs=0,
            obs_select="adaptive_cover",
            adaptive_k_min=1,
            adaptive_k_max=4,
            adaptive_min_gain=0.005,
            point_xyz=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            obs_frame_ids=np.asarray([10, 11, 12, 13], dtype=np.int32),
            frame_centers_by_frame_id={
                10: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                11: np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                12: np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
                13: np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
            },
            adaptive_view_weight=1.0,
        )
        self.assertEqual(selected.shape[0], 4)

    def test_weights_prefer_reliable_observation(self) -> None:
        weights = compute_obs_weights(
            np.asarray([0.1, 1.0, 0.5], dtype=np.float32),
            attach_dist=np.asarray([5.0, 0.0, 2.0], dtype=np.float32),
            reproj_error=np.asarray([8.0, 0.0, 1.0], dtype=np.float32),
            sigma_attach=2.0,
            sigma_reproj=4.0,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertEqual(int(np.argmax(weights)), 1)

    def test_existing_selection_modes_still_work(self) -> None:
        descs = _norm(np.eye(6, dtype=np.float32))
        for mode in ("all", "first", "uniform", "random", "diverse_desc", "fixed_fps"):
            selected = _select_point_observation_indices(
                descs,
                0,
                descs.shape[0],
                max_obs=3,
                obs_select=mode,
            )
            if mode == "all":
                self.assertEqual(selected.shape[0], descs.shape[0])
            else:
                self.assertEqual(selected.shape[0], 3)
            self.assertEqual(len(set(selected.tolist())), selected.shape[0])

    def test_adaptive_mode_returns_global_indices(self) -> None:
        descs = _norm(np.eye(8, dtype=np.float32))
        selected = _select_point_observation_indices(
            descs,
            2,
            7,
            max_obs=0,
            obs_select="adaptive_cover",
            adaptive_k_min=2,
            adaptive_k_max=4,
            adaptive_min_gain=0.0,
        )
        self.assertEqual(selected.shape[0], 4)
        self.assertTrue(np.all(selected >= 2))
        self.assertTrue(np.all(selected < 7))


if __name__ == "__main__":
    unittest.main()
