#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile


def _sequence_name(value: str) -> str:
    item = str(value).strip()
    if not item:
        raise ValueError("Empty 7Scenes sequence name")
    if item.startswith("seq-"):
        return item
    if item.startswith("sequence"):
        return f"seq-{int(item.removeprefix('sequence')):02d}"
    if item.isdigit():
        return f"seq-{int(item):02d}"
    return item


def _read_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    seqs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seqs.append(_sequence_name(line))
    return seqs


def _split_sequences(root: Path, split_files: list[str]) -> list[str]:
    seen: set[str] = set()
    seqs: list[str] = []
    for split_file in split_files:
        for seq in _read_split(root / split_file):
            if seq not in seen:
                seen.add(seq)
                seqs.append(seq)
    return seqs


def _member_parts(name: str) -> list[str]:
    return [part for part in name.replace("\\", "/").split("/") if part and part != "."]


def _requested_sequences(root: Path, args: argparse.Namespace) -> list[str]:
    if args.sequences:
        raw = args.sequences.replace(",", " ").split()
        return [_sequence_name(item) for item in raw]
    seqs = _split_sequences(root, list(args.split_file))
    if seqs:
        return seqs
    return sorted(path.stem for path in root.glob("seq-*.zip"))


def _validate_zip_members(zip_path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name:
                continue
            parts = _member_parts(name)
            if name.startswith(("/", "\\")) or ".." in parts:
                raise ValueError(f"Unsafe archive member in {zip_path}: {name}")
            target = (root.joinpath(*parts)).resolve()
            if root_resolved != target and root_resolved not in target.parents:
                raise ValueError(f"Archive member escapes root in {zip_path}: {name}")


def _is_ignored_member(name: str) -> bool:
    parts = _member_parts(name)
    if not parts:
        return False
    if "__MACOSX" in parts:
        return True
    return parts[-1].lower() in {"thumbs.db", ".ds_store"}


def _extract_archive(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or _is_ignored_member(name):
                continue
            target = root.joinpath(*_member_parts(name))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _extract_sequence(root: Path, seq: str, *, dry_run: bool) -> dict[str, str]:
    seq_dir = root / seq
    zip_path = root / f"{seq}.zip"
    if seq_dir.is_dir():
        return {"sequence": seq, "status": "present", "path": str(seq_dir)}
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {seq_dir} and {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a zip archive: {zip_path}")
    if dry_run:
        return {"sequence": seq, "status": "would_extract", "path": str(zip_path)}

    _validate_zip_members(zip_path, root)
    _extract_archive(zip_path, root)
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Extracted {zip_path}, but {seq_dir} was not created")
    return {"sequence": seq, "status": "extracted", "path": str(seq_dir)}


def extract_missing_sequences(
    root: Path,
    *,
    sequences: list[str] | None = None,
    split_files: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    if sequences is None:
        seqs = _split_sequences(root, split_files or ["TrainSplit.txt", "TestSplit.txt"])
        if not seqs:
            seqs = sorted(path.stem for path in root.glob("seq-*.zip"))
    else:
        seqs = [_sequence_name(seq) for seq in sequences]
    if not seqs:
        raise ValueError(f"No sequences found in {root}")
    return [_extract_sequence(root, seq, dry_run=bool(dry_run)) for seq in seqs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract missing official 7-Scenes seq-*.zip archives.")
    parser.add_argument("--root", required=True, type=Path, help="Scene root containing TrainSplit.txt and seq-*.zip files.")
    parser.add_argument(
        "--split_file",
        action="append",
        default=["TrainSplit.txt", "TestSplit.txt"],
        help="Split file to read sequence IDs from. Can be passed multiple times.",
    )
    parser.add_argument("--sequences", default="", help="Optional comma/space-separated sequence IDs to extract.")
    parser.add_argument("--dry_run", action="store_true", help="Report what would be extracted without writing files.")
    args = parser.parse_args()

    root = args.root
    sequences = _requested_sequences(root, args)
    rows = extract_missing_sequences(root, sequences=sequences, dry_run=bool(args.dry_run))
    summary = {
        "root": str(root),
        "dry_run": bool(args.dry_run),
        "num_sequences": len(rows),
        "statuses": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
