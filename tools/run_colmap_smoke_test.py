from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the synthetic COLMAP smoke test for PLM-MATCH')
    parser.add_argument('--project_root', type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root or Path(__file__).resolve().parents[1])
    rgbd_root = root / 'synthetic_demo'
    colmap_root = root / 'synthetic_colmap_demo'

    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_dataset.py'), '--out_root', str(rgbd_root)])
    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_colmap_dataset.py'), '--src_root', str(rgbd_root), '--out_root', str(colmap_root)])
    subprocess.check_call([
        sys.executable,
        '-m', 'plm_match.pipelines.localize_from_map',
        '--config', str(root / 'configs' / 'mock_colmap.yaml'),
        '--dataset_root', str(colmap_root),
        '--out_dir', str(root / 'outputs' / 'mock_colmap'),
    ])


if __name__ == '__main__':
    main()
