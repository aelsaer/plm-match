#!/usr/bin/env python3
"""Build a spatially coherent minitest subset from an Aachen-style dataset.

Picks N queries, then collects their top retrieved DB images so the map
actually covers the same sub-scene as the queries.

Usage:
    python tools/make_minitest.py \
        --dataset_root /path/to/aachen_v1_1 \
        --out_dir ./minitest \
        --n_queries 10 \
        --n_db 100 \
        --retrieval_file pairs-query-netvlad50.txt \
        --query_list day_time_queries_with_intrinsics.txt
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_retrieval(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]].append(parts[1])
    return dict(out)


def parse_query_list(path: Path) -> list[str]:
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--out_dir', default='./minitest_files')
    parser.add_argument('--n_queries', type=int, default=10)
    parser.add_argument('--n_db', type=int, default=100)
    parser.add_argument('--retrieval_file', default='pairs-query-netvlad50.txt')
    parser.add_argument('--query_list', default='day_time_queries_with_intrinsics.txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    retrievals = parse_retrieval(root / args.retrieval_file)
    query_lines = parse_query_list(root / args.query_list)

    # Index query lines by image name
    query_by_name: dict[str, str] = {}
    for line in query_lines:
        name = line.split()[0]
        query_by_name[name] = line

    # Only keep queries that have retrieval pairs
    eligible = [q for q in query_by_name if q in retrievals]
    if len(eligible) < args.n_queries:
        print(f'Warning: only {len(eligible)} queries have retrieval pairs')
    chosen_queries = eligible[:args.n_queries]

    # Collect DB images: for each chosen query take top retrieved images
    db_set: list[str] = []
    seen: set[str] = set()
    for q in chosen_queries:
        for db in retrievals[q]:
            if db not in seen:
                db_set.append(db)
                seen.add(db)
            if len(db_set) >= args.n_db:
                break
        if len(db_set) >= args.n_db:
            break
    db_set = db_set[:args.n_db]

    print(f'Chosen {len(chosen_queries)} queries, {len(db_set)} DB images')
    print('Queries:')
    for q in chosen_queries:
        print(f'  {q}')

    # Write minitest_db_names.txt
    db_file = out / 'minitest_db_names.txt'
    db_file.write_text('\n'.join(db_set) + '\n')
    print(f'Written: {db_file}')

    # Write minitest_queries.txt (with intrinsics)
    q_file = out / 'minitest_queries.txt'
    q_file.write_text('\n'.join(query_by_name[q] for q in chosen_queries) + '\n')
    print(f'Written: {q_file}')

    # Write minitest_pairs.txt (only for chosen queries)
    pairs_lines = []
    for q in chosen_queries:
        for db in retrievals[q]:
            pairs_lines.append(f'{q} {db}')
    pairs_file = out / 'minitest_pairs.txt'
    pairs_file.write_text('\n'.join(pairs_lines) + '\n')
    print(f'Written: {pairs_file}  ({len(pairs_lines)} pairs)')

    # Print ready-to-use config snippet
    print(f"""
Add to your config (or use configs/aachen_minitest.yaml):

  dataset:
    db_image_names_file: {out}/minitest_db_names.txt
    query_list: {out}/minitest_queries.txt

  hloc:
    retrieval_file: {out}/minitest_pairs.txt
""")


if __name__ == '__main__':
    main()
