from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess


def main() -> None:
    root = ROOT
    data_root = root / 'synthetic_tum'
    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_tum_rgbd.py'), '--out_root', str(data_root)])
    subprocess.check_call([sys.executable, '-m', 'plm_match.pipelines.online_benchmark', '--config', str(root / 'configs' / 'mock_tum_online.yaml'), '--dataset_root', str(data_root), '--out_dir', str(root / 'outputs' / 'mock_tum_online')])

if __name__ == '__main__':
    main()
