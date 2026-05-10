#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from loo_utils import write_split_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed Aachen DB leave-one-out split with GT poses, query list, and map image list."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=("stride", "random"), default="stride")
    parser.add_argument("--min_observations", type=int, default=100)
    parser.add_argument("--max_frame_index", type=int, default=None)
    args = parser.parse_args()

    split = write_split_files(
        config_path=args.config,
        out_dir=args.out_dir,
        dataset_root=args.dataset_root,
        num_queries=args.num_queries,
        seed=args.seed,
        selection=args.selection,
        min_observations=args.min_observations,
        max_frame_index=args.max_frame_index,
    )
    print(f"Wrote {args.out_dir / 'split.json'}")
    print(f"Wrote {split['query_list']}")
    print(f"Wrote {split['map_image_list']}")
    print(f"Wrote {split['gt_hloc_results']}")


if __name__ == "__main__":
    main()
