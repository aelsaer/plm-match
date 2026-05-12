#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from plm_match.hloc import parse_retrieval_file


def _norm(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./").strip()


def _build_key_map(names: list[str]) -> dict[str, str]:
    return {_norm(name): name for name in names}


def _load_attached_names(path: Path) -> list[str]:
    entries = np.load(path / "db_image_entries.npz", allow_pickle=False)
    return [str(x) for x in entries["image_names"].tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check split/index/retrieval leakage for lifted-NN runs.")
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--attached_index", required=True, type=Path)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    split = load_split(args.split_json)
    query_names = split_query_names(split)
    map_names = split_map_names(split)
    attached_names = _load_attached_names(args.attached_index)
    retrievals = parse_retrieval_file(args.retrieval_file)

    query_key_map = _build_key_map(query_names)
    map_key_map = _build_key_map(map_names)
    attached_leaks = []
    for name in attached_names:
        hit = {_norm(name)}.intersection(query_key_map.keys())
        if hit:
            attached_leaks.append({"attached": name, "query": query_key_map[sorted(hit)[0]]})

    retrieval_db_leaks = []
    self_pairs = []
    unknown_db = []
    for q, dbs in retrievals.items():
        q_norm = _norm(q)
        q_keys = {q_norm}
        for db in dbs:
            db_norm = _norm(db)
            db_keys = {db_norm}
            if q_keys.intersection(db_keys):
                self_pairs.append({"query": q, "db": db})
            hit = db_keys.intersection(query_key_map.keys())
            if hit:
                retrieval_db_leaks.append({"query": q, "db": db, "leaked_query": query_key_map[sorted(hit)[0]]})
            if not db_keys.intersection(map_key_map.keys()):
                unknown_db.append({"query": q, "db": db})

    payload = {
        "split_json": str(args.split_json),
        "attached_index": str(args.attached_index),
        "retrieval_file": str(args.retrieval_file),
        "num_queries": int(len(query_names)),
        "num_map_images": int(len(map_names)),
        "num_attached_images": int(len(attached_names)),
        "num_retrieval_queries": int(len(retrievals)),
        "num_attached_query_leaks": int(len(attached_leaks)),
        "num_retrieval_db_query_leaks": int(len(retrieval_db_leaks)),
        "num_self_pairs": int(len(self_pairs)),
        "num_unknown_retrieval_db": int(len(unknown_db)),
        "attached_query_leaks": attached_leaks[:50],
        "retrieval_db_query_leaks": retrieval_db_leaks[:50],
        "self_pairs": self_pairs[:50],
        "unknown_retrieval_db": unknown_db[:50],
        "ok": not attached_leaks and not retrieval_db_leaks and not self_pairs,
    }
    out = args.out or (args.attached_index.parent / "leakage_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out.parent / "leakage_check_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
