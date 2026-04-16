from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess


def main() -> None:
    root = ROOT
    data_root = root / 'synthetic_scannet'
    # reuse existing scene if present
    if not (data_root / 'scene0000_00').exists():
        subprocess.check_call([sys.executable, str(root / 'tools' / 'make_synthetic_scannet_scene.py'), '--out_root', str(data_root)])
    subprocess.check_call([sys.executable, '-m', 'plm_match.pipelines.online_benchmark', '--config', str(root / 'configs' / 'mock_scannet_online.yaml'), '--dataset_root', str(data_root), '--out_dir', str(root / 'outputs' / 'mock_scannet_online')])

if __name__ == '__main__':
    main()
