from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import subprocess
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic data and run PLM-MATCH smoke test')
    parser.add_argument('--project_root', type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root or Path(__file__).resolve().parents[1])
    data_root = root / 'synthetic_demo'
    cfg = root / 'configs' / 'mock_synthetic.yaml'

    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_dataset.py'), '--out_root', str(data_root)])
    subprocess.check_call([sys.executable, '-m', 'plm_match.pipelines.localize_from_map', '--config', str(cfg), '--dataset_root', str(data_root), '--out_dir', str(root / 'outputs' / 'mock_synthetic')])


if __name__ == '__main__':
    main()
