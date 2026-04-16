from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess


def main() -> None:
    root = ROOT
    src_root = root / 'synthetic_demo'
    data_root = root / 'synthetic_colmap_demo'
    # ensure source RGB-D synthetic exists for COLMAP conversion
    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_dataset.py'), '--out_root', str(src_root)])
    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_colmap_dataset.py'), '--src_root', str(src_root), '--out_root', str(data_root)])
    subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_hloc_retrievals.py'), '--dataset_root', str(data_root)])
    subprocess.check_call([
        sys.executable, '-m', 'plm_match.pipelines.hloc_localize',
        '--config', str(root / 'configs' / 'mock_hloc.yaml'),
        '--dataset_root', str(data_root),
        '--out_dir', str(root / 'outputs' / 'mock_hloc')
    ])

if __name__ == '__main__':
    main()
