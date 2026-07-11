#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.utils.config import load_config
from plm_match.utils.pose import camera_center_from_Twc
from loo_utils import map_only_dataset_cfg


SUPERGLUE_URLS = {
    "outdoor": "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth",
    "indoor": "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth",
}

SUPERPOINT_URL = (
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/"
    "models/weights/superpoint_v1.pth"
)


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    return p


def _find_superglue_source(source_root: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        candidates.extend(
            [
                root,
                root / "SuperGluePretrainedNetwork",
                root / "third_party" / "SuperGluePretrainedNetwork",
            ]
        )
    candidates.extend(
        [
            ROOT / "external" / "Hierarchical-Localization" / "third_party" / "SuperGluePretrainedNetwork",
            ROOT / "external" / "SuperGluePretrainedNetwork",
        ]
    )
    candidates.extend(
        [
        Path("/home/andreas/anaconda3/envs/sam3/lib/python3.12/site-packages/imm/third_party/TopicFM/third_party/loftr/third_party/SuperGluePretrainedNetwork"),
        Path("/home/andreas/anaconda3/envs/sam3/lib/python3.12/site-packages/imm/third_party/MINIMA/third_party/LoFTR/third_party/SuperGluePretrainedNetwork"),
        Path("/home/andreas/anaconda3/envs/phd/lib/python3.10/site-packages/imm/third_party/TopicFM/third_party/loftr/third_party/SuperGluePretrainedNetwork"),
        ]
    )
    for cand in candidates:
        if (cand / "models" / "superglue.py").exists():
            return cand
    if source_root is not None:
        try:
            for p in Path(source_root).expanduser().resolve().rglob("SuperGluePretrainedNetwork/models/superglue.py"):
                return p.parent.parent
        except Exception:
            pass
    for root in (Path("/home/andreas"), Path("/home/phd")):
        try:
            for p in root.rglob("SuperGluePretrainedNetwork/models/superglue.py"):
                return p.parent.parent
        except Exception:
            continue
    return None


def _prepare_superglue_shim(
    *,
    weights: str,
    weights_path: Path | None,
    download_weights: bool,
    source_root: Path | None = None,
) -> Path:
    src = _find_superglue_source(source_root)
    if src is None:
        raise RuntimeError(
            "Could not find a local SuperGluePretrainedNetwork source tree. "
            "Install HLoc with its third_party SuperGlue dependency or provide one via the imm package."
        )
    shim_parent = Path("/tmp/plm_superglue_shim")
    pkg = shim_parent / "SuperGluePretrainedNetwork"
    models_dir = pkg / "models"
    weights_dir = models_dir / "weights"
    pkg.mkdir(parents=True, exist_ok=True)
    if models_dir.is_symlink():
        models_dir.unlink()
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "superglue.py", "superpoint.py", "matching.py", "utils.py"):
        target = src / "models" / name
        link = models_dir / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target)
            except Exception:
                shutil.copy2(target, link)
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_weight = weights_dir / f"superglue_{weights}.pth"
    target_superpoint_weight = weights_dir / "superpoint_v1.pth"
    if weights_path is not None:
        wp = weights_path.expanduser().resolve()
        if not wp.exists():
            raise FileNotFoundError(f"Provided SuperGlue weights do not exist: {wp}")
        if not target_weight.exists():
            try:
                target_weight.symlink_to(wp)
            except Exception:
                shutil.copy2(wp, target_weight)
    if not target_weight.exists():
        for root in (Path("/tmp"), Path.home()):
            try:
                found = next(root.rglob(f"superglue_{weights}.pth"))
            except StopIteration:
                continue
            except Exception:
                continue
            if found.resolve() == target_weight.resolve():
                continue
            try:
                target_weight.symlink_to(found.resolve())
            except Exception:
                shutil.copy2(found, target_weight)
            break
    # HLoc's SuperPoint wrapper imports MagicLeap's SuperPoint implementation,
    # which expects the detector weights beside the SuperGlue weights. The
    # `imm` third-party source tree often ships code only, so make the weight
    # file available in the shim too.
    if not target_superpoint_weight.exists():
        for root in (Path("/tmp"), Path("/home/andreas"), Path("/home/phd")):
            try:
                found = next(root.rglob("superpoint_v1.pth"))
            except StopIteration:
                continue
            except Exception:
                continue
            if found.resolve() == target_superpoint_weight.resolve():
                continue
            try:
                target_superpoint_weight.symlink_to(found.resolve())
            except Exception:
                shutil.copy2(found, target_superpoint_weight)
            break
    if not target_weight.exists() and download_weights:
        url = SUPERGLUE_URLS.get(weights)
        if url is None:
            raise ValueError(f"No download URL known for SuperGlue weights {weights!r}")
        print(f"Downloading SuperGlue {weights} weights to {target_weight}")
        urllib.request.urlretrieve(url, target_weight)
    if not target_superpoint_weight.exists() and download_weights:
        print(f"Downloading SuperPoint weights to {target_superpoint_weight}")
        urllib.request.urlretrieve(SUPERPOINT_URL, target_superpoint_weight)
    if not target_weight.exists():
        raise FileNotFoundError(
            f"Missing SuperGlue weights: {target_weight}\n"
            "Run this tool with `--download_weights`, or provide `--weights_path /path/to/superglue_outdoor.pth`."
        )
    if not target_superpoint_weight.exists():
        raise FileNotFoundError(
            f"Missing SuperPoint weights: {target_superpoint_weight}\n"
            "Run this tool with `--download_weights`, or download "
            "`superpoint_v1.pth` from MagicLeap/SuperGluePretrainedNetwork into that directory."
        )
    if str(shim_parent) not in sys.path:
        sys.path.insert(0, str(shim_parent))
    return shim_parent


def _selectable_count(split: dict, all_count: int) -> int:
    n = int(split.get("num_map_frames", 0)) + int(split.get("num_queries", 0))
    if n <= 0:
        return all_count
    return min(all_count, n)


def _write_pairs(
    *,
    cfg: dict,
    split_path: Path,
    pairs_path: Path,
    topk: int,
) -> int:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = build_dataset(cfg["dataset_root"], map_only_dataset_cfg(cfg))
    all_frames = dataset.get_map_frames()
    count = _selectable_count(split, len(all_frames))
    selectable = all_frames[:count]
    heldout_indices = [int(item["original_index"]) for item in split.get("heldout", [])]
    heldout_set = set(heldout_indices)
    map_indices = [i for i in range(len(selectable)) if i not in heldout_set]
    map_centers = np.stack([camera_center_from_Twc(selectable[i].pose).astype(np.float64) for i in map_indices], axis=0)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    with open(pairs_path, "w", encoding="utf-8") as f:
        for q_idx in heldout_indices:
            q_frame = selectable[q_idx]
            q_name = str(q_frame.meta.get("relative_path", q_frame.image_path.name))
            q_center = camera_center_from_Twc(q_frame.pose).astype(np.float64)
            dists = np.linalg.norm(map_centers - q_center[None, :], axis=1)
            order = np.argsort(dists)[: int(topk)]
            for pos in order:
                db_idx = map_indices[int(pos)]
                db_frame = selectable[db_idx]
                db_name = str(db_frame.meta.get("relative_path", db_frame.image_path.name))
                f.write(f"{q_name} {db_name}\n")
                num_pairs += 1
    return num_pairs


def _write_pairs_from_retrieval(
    *,
    split_path: Path,
    retrieval_file: Path,
    pairs_path: Path,
    topk: int,
) -> int:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    query_names = {str(item["name"]) for item in split.get("queries", split.get("heldout", []))}
    map_names = {str(item["name"]) for item in split.get("map_images", [])}
    per_query: dict[str, list[str]] = {}
    with open(retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            q, db = parts
            if q not in query_names:
                continue
            if map_names and db not in map_names:
                continue
            bucket = per_query.setdefault(q, [])
            if len(bucket) < int(topk):
                bucket.append(db)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    with open(pairs_path, "w", encoding="utf-8") as f:
        for q in sorted(query_names):
            for db in per_query.get(q, [])[: int(topk)]:
                f.write(f"{q} {db}\n")
                num_pairs += 1
    return num_pairs


def _read_feature_group_names(path: Path) -> set[str]:
    names: set[str] = set()
    with h5py.File(path, "r") as hfile:
        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Group) and "keypoints" in obj:
                names.add(str(name))

        hfile.visititems(visitor)
    return names


def _validate_pair_features(
    *,
    pairs_path: Path,
    query_features: Path,
    db_features: Path,
    max_examples: int = 8,
) -> None:
    query_names = _read_feature_group_names(query_features)
    db_names = _read_feature_group_names(db_features)
    missing_q: list[str] = []
    missing_db: list[str] = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            q, db = parts
            if q not in query_names and len(missing_q) < max_examples:
                missing_q.append(q)
            if db not in db_names and len(missing_db) < max_examples:
                missing_db.append(db)
    if missing_q or missing_db:
        msg = ["Feature H5 files do not cover the generated SuperGlue pairs."]
        if missing_q:
            msg.append(f"Missing query-side keys in {query_features}: {missing_q}")
        if missing_db:
            msg.append(f"Missing DB-side keys in {db_features}: {missing_db}")
        msg.append("For LOO, held-out DB images must be read from query_features_path and references from db_features_path.")
        raise KeyError("\n".join(msg))


def main() -> None:
    parser = argparse.ArgumentParser("Generate real SuperGlue matches for DB leave-one-out pairs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--loo_dir", required=True, type=Path, help="Existing leave-one-out output dir containing split.json")
    parser.add_argument("--topk_db_images", type=int, default=5)
    parser.add_argument("--pairs_path", type=Path, default=None)
    parser.add_argument("--matches_path", type=Path, default=None)
    parser.add_argument(
        "--retrieval_file",
        type=Path,
        default=None,
        help="Use this LOO retrieval file for SuperGlue pairs instead of oracle nearest-pose pairs.",
    )
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--weights", choices=("outdoor", "indoor"), default="outdoor")
    parser.add_argument("--superglue_root", type=Path, default=None)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--download_weights", action="store_true")
    parser.add_argument("--sinkhorn_iterations", type=int, default=50)
    parser.add_argument("--match_threshold", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write_pairs_only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_path = args.loo_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    fine_cfg = cfg.get("matching", {}).get("fine_rerank", {})
    db_features = args.db_features_path or fine_cfg.get("db_features_path")
    if db_features is None:
        raise ValueError("No DB SuperPoint feature H5 path provided.")
    db_features = _resolve(ROOT, db_features)
    if not db_features.exists():
        raise FileNotFoundError(f"DB SuperPoint feature H5 not found: {db_features}")
    query_features = args.query_features_path or fine_cfg.get("query_features_path") or db_features
    query_features = _resolve(ROOT, query_features)
    if not query_features.exists():
        raise FileNotFoundError(f"Query SuperPoint feature H5 not found: {query_features}")

    pairs_path = args.pairs_path or (args.loo_dir / "superglue_loo_pairs.txt")
    matches_path = args.matches_path or (args.loo_dir / "superglue_loo_matches.h5")
    if args.retrieval_file is not None:
        n_pairs = _write_pairs_from_retrieval(
            split_path=split_path,
            retrieval_file=args.retrieval_file,
            pairs_path=pairs_path,
            topk=args.topk_db_images,
        )
    else:
        n_pairs = _write_pairs(cfg=cfg, split_path=split_path, pairs_path=pairs_path, topk=args.topk_db_images)
    print(f"Wrote {n_pairs} LOO SuperGlue pairs to {pairs_path}")
    _validate_pair_features(
        pairs_path=pairs_path,
        query_features=query_features,
        db_features=db_features,
    )
    if args.write_pairs_only:
        return

    _prepare_superglue_shim(
        weights=args.weights,
        weights_path=args.weights_path,
        download_weights=bool(args.download_weights),
        source_root=args.superglue_root,
    )
    from hloc import match_features

    conf = {
        "output": "matches-superglue-loo",
        "model": {
            "name": "superglue",
            "weights": args.weights,
            "sinkhorn_iterations": int(args.sinkhorn_iterations),
            "match_threshold": float(args.match_threshold),
        },
    }
    match_features.main(
        conf,
        pairs_path,
        features=query_features,
        features_ref=db_features,
        matches=matches_path,
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote LOO SuperGlue matches to {matches_path}")
    print("Rerun LOO; tools/run_db_leave_one_out.py will auto-use this match file if it is under the same out_dir.")


if __name__ == "__main__":
    main()
