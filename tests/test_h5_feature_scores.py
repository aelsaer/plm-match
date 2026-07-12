import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from plm_match.fine_features import LocalPatchDescriptor


class H5FeatureScoreTest(unittest.TestCase):
    def test_hloc_keypoint_scores_are_loaded_and_used_for_topk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.h5"
            with h5py.File(path, "w") as fd:
                group = fd.create_group("db/example.jpg")
                group.create_dataset("keypoints", data=np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float32))
                group.create_dataset(
                    "descriptors",
                    data=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32).T,
                )
                group.create_dataset("keypoint_scores", data=np.asarray([0.2, 0.9, 0.5], dtype=np.float32))

            extractor = LocalPatchDescriptor(method="aliked_h5", db_features_path=str(path))
            try:
                keypoints, scores, descriptors = extractor.extract_keypoints("db/example.jpg", topk=2)
            finally:
                extractor.close()

            np.testing.assert_allclose(scores, np.asarray([0.9, 0.5], dtype=np.float32))
            np.testing.assert_allclose(keypoints, np.asarray([[2, 2], [3, 3]], dtype=np.float32))
            np.testing.assert_allclose(descriptors, np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
