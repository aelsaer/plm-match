from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Create simple HLoc-style retrieval pairs from synthetic COLMAP demo')
    parser.add_argument('--dataset_root', required=True, type=str)
    parser.add_argument('--out_file', type=str, default='pairs-query-db.txt')
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    query_list = root / 'queries' / 'queries_with_intrinsics.txt'
    db_dir = root / 'db'
    db_names = sorted([p.relative_to(root).as_posix() for p in db_dir.glob('*') if p.is_file()])[:max(1, args.topk)]
    out_path = root / args.out_file
    with open(query_list, 'r', encoding='utf-8') as f_in, open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            q = line.split()[0]
            for db in db_names:
                f_out.write(f'{q} {db}\n')
    print(f'Wrote retrievals to {out_path}')

if __name__ == '__main__':
    main()
