#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SCENE_IDS = {
    "KingsCollege": "251342",
    "OldHospital": "251340",
    "StMarysChurch": "251294",
    "ShopFacade": "251336",
    "GreatCourt": "251291",
}
DEFAULT_SCENES = ("GreatCourt", "KingsCollege", "OldHospital", "StMarysChurch")
MODEL_FILE_ID = "1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15"
MODEL_DIR_NAME = "CambridgeLandmarks_Colmap_Retriangulated_1024px"


def _scene_url(scene: str) -> str:
    return f"https://www.repository.cam.ac.uk/bitstream/handle/1810/{SCENE_IDS[scene]}/{scene}.zip"


def _has_scene(root: Path, scene: str) -> bool:
    path = root / scene
    return (
        path.is_dir()
        and (path / "dataset_train.txt").exists()
        and (path / "dataset_test.txt").exists()
        and (path / "reconstruction.nvm").exists()
    )


def _has_models(root: Path) -> bool:
    model_root = root / MODEL_DIR_NAME
    return all((model_root / scene / "model_train").exists() for scene in SCENE_IDS)


def _download(url: str, out_path: Path, *, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        if zipfile.is_zipfile(out_path):
            print(f"Reusing archive: {out_path}")
            return
        raise ValueError(f"Existing archive is not a valid zip file: {out_path}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Downloading {url}")
    print(f"  -> {out_path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(out_path)
    if not zipfile.is_zipfile(out_path):
        size = out_path.stat().st_size if out_path.exists() else 0
        raise ValueError(f"Downloaded file is not a zip archive ({size} bytes): {out_path}")


def _extract_zip(archive: Path, root: Path, *, overwrite: bool) -> None:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a valid zip archive: {archive}")
    print(f"Extracting {archive} -> {root}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = root / member.filename
            if target.exists() and not overwrite:
                continue
            zf.extract(member, root)


def _download_scene(root: Path, scene: str, *, overwrite: bool, keep_archives: bool, dry_run: bool) -> None:
    archive = root / f"{scene}.zip"
    url = _scene_url(scene)
    if _has_scene(root, scene) and not overwrite:
        print(f"Scene already present: {root / scene}")
        return
    if dry_run:
        print(f"Would download {scene}: {url}")
        return
    _download(url, archive, overwrite=overwrite)
    _extract_zip(archive, root, overwrite=overwrite)
    if not _has_scene(root, scene):
        raise RuntimeError(f"Scene extraction did not produce the expected Cambridge layout: {root / scene}")
    if not keep_archives:
        archive.unlink(missing_ok=True)


def _download_models(root: Path, *, overwrite: bool, keep_archives: bool, dry_run: bool) -> None:
    archive = root / f"{MODEL_DIR_NAME}.zip"
    if _has_models(root) and not overwrite:
        print(f"COLMAP models already present: {root / MODEL_DIR_NAME}")
        return
    if dry_run:
        print(f"Would download COLMAP models from Google Drive id {MODEL_FILE_ID}")
        return

    if archive.exists() and not overwrite and zipfile.is_zipfile(archive):
        print(f"Reusing archive: {archive}")
    else:
        print(f"Downloading COLMAP model archive with gdown id {MODEL_FILE_ID}")
        cmd = [sys.executable, "-m", "gdown", MODEL_FILE_ID, "-O", str(archive)]
        subprocess.run(cmd, check=True)
        if not zipfile.is_zipfile(archive):
            size = archive.stat().st_size if archive.exists() else 0
            raise ValueError(
                f"gdown output is not a zip archive ({size} bytes): {archive}. "
                "This usually means Google Drive returned an HTML confirmation page."
            )

    _extract_zip(archive, root, overwrite=overwrite)
    if not _has_models(root):
        raise RuntimeError(f"Model extraction did not produce all expected scene models under {root / MODEL_DIR_NAME}")
    if not keep_archives:
        archive.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Cambridge Landmarks scenes in the layout expected by HLoc and PLMLoc."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/d/private/pairs/cambridge_landmarks"),
        help="Directory that will contain scene folders such as GreatCourt and KingsCollege.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENE_IDS),
        default=list(DEFAULT_SCENES),
        help="Scenes to download. Defaults to the four non-ShopFacade Cambridge scenes.",
    )
    parser.add_argument("--download_models", action="store_true", help="Also download the retriangulated COLMAP models.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_archives", action="store_true", help="Keep zip archives after successful extraction.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        _download_scene(
            args.root,
            scene,
            overwrite=bool(args.overwrite),
            keep_archives=bool(args.keep_archives),
            dry_run=bool(args.dry_run),
        )
    if args.download_models:
        _download_models(
            args.root,
            overwrite=bool(args.overwrite),
            keep_archives=bool(args.keep_archives),
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()
