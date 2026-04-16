from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess


def main() -> None:
    cmds = [
        [sys.executable, str(ROOT / 'tools' / 'run_smoke_test.py')],
        [sys.executable, str(ROOT / 'tools' / 'run_colmap_smoke_test.py')],
        [sys.executable, str(ROOT / 'tools' / 'run_hloc_smoke_test.py')],
        [sys.executable, str(ROOT / 'tools' / 'run_online_scannet_smoke_test.py')],
        [sys.executable, str(ROOT / 'tools' / 'run_online_tum_smoke_test.py')],
    ]
    for cmd in cmds:
        print('RUN', ' '.join(str(x) for x in cmd))
        subprocess.check_call(cmd)
    print('Completed all PLM-MATCH smoke tests successfully.')

if __name__ == '__main__':
    main()
