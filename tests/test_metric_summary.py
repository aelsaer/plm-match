import unittest

from plm_match.pipelines.lifted_nn_localize import _summarize_metrics


class MetricSummaryTest(unittest.TestCase):
    def test_reprojection_error_summary_uses_successful_poses(self):
        summary = _summarize_metrics(
            [
                {"success": True, "reproj_error": 1.0, "num_inliers": 10},
                {"success": True, "reproj_error": 3.0, "num_inliers": 30},
                {"success": False, "reproj_error": 100.0, "num_inliers": 1000},
                {"success": True, "reproj_error": None, "num_inliers": 20},
            ]
        )

        self.assertEqual(summary["num_queries"], 4)
        self.assertEqual(summary["num_success"], 3)
        self.assertAlmostEqual(summary["mean_reproj_error"], 2.0)
        self.assertAlmostEqual(summary["median_reproj_error"], 2.0)
        self.assertAlmostEqual(summary["p90_reproj_error"], 2.8)
        self.assertAlmostEqual(summary["p95_reproj_error"], 2.9)
        self.assertAlmostEqual(summary["inlier_weighted_mean_reproj_error"], 2.5)


if __name__ == "__main__":
    unittest.main()
