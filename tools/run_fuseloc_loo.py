#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loo_utils import evaluate_results, load_split, split_map_names, split_query_names
from plm_match.utils.config import load_config
from plm_match.utils.io import write_json


def _require_module(name: str) -> None:
    try:
        __import__(name)
    except Exception as exc:
        raise RuntimeError(
            f"FuseLoc dependency {name!r} is not importable in this environment. "
            "Install the official FuseLoc environment/dependencies first; this runner "
            "intentionally does not reimplement FuseLoc."
        ) from exc


def _model_name(camera_model) -> str:
    return str(getattr(camera_model, "name", camera_model))


def _result_file_name(
    *,
    using_global: bool,
    local_name: str,
    global_name: str,
    global_dim: int,
    lambda_val: float,
    convert: bool,
    order: str,
) -> str:
    if using_global:
        return (
            f"Aachen_v1_1_eval_{local_name}_{global_name}_{global_dim}_"
            f"{lambda_val}_{convert}_{order}.txt"
        )
    return f"Aachen_v1_1_eval_{local_name}.txt"


def _force_torch_cpu(torch_module) -> None:
    """Force third-party FuseLoc code to stay on CPU.

    The official FuseLoc code contains hard-coded `.cuda()` calls in
    trainer.py. That is fine on supported GPUs, but the pixi environment can
    ship a PyTorch build that does not contain kernels for newer cards. Rather
    than patching the external checkout, monkey-patch CUDA transfers into no-op
    CPU transfers inside this runner.
    """
    torch_module.cuda.is_available = lambda: False

    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        return self

    def _module_cuda(self, device=None):
        return self

    torch_module.Tensor.cuda = _tensor_cuda
    torch_module.nn.Module.cuda = _module_cuda


def _install_low_memory_fuseloc_patches(BaseTrainer, dd_utils, torch_module) -> None:
    """Patch official FuseLoc at runtime for this constrained LOO runner.

    The upstream trainer uses NumPy's default float64 for global descriptor
    buffers and lets PyTorch keep CUDA cache between thousands of single-image
    forwards. On a 16GB WSL setup that can trigger transient OOMs even when
    monitors show a small steady-state footprint. These patches keep the same
    algorithm but use float32 buffers and release cached CUDA blocks after each
    image forward.
    """
    import os as _os
    import pickle as _pickle

    import h5py as _h5py
    from tqdm import tqdm as _tqdm

    original_produce_image_descriptor = BaseTrainer.produce_image_descriptor
    original_produce_local_descriptors = BaseTrainer.produce_local_descriptors
    original_collect_image_descriptors = BaseTrainer.collect_image_descriptors
    original_load_local_features = BaseTrainer.load_local_features
    original_load_selected_local_features = BaseTrainer.load_selected_local_features
    original_detect_local_features_on_test_set = BaseTrainer.detect_local_features_on_test_set

    def _maybe_empty_cuda_cache() -> None:
        if torch_module.cuda.is_available():
            try:
                torch_module.cuda.synchronize()
                torch_module.cuda.empty_cache()
            except Exception:
                pass

    def _repair_or_remove_stale_h5(path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        repaired = False
        try:
            result = subprocess.run(
                ["h5clear", "-s", str(p)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            repaired = result.returncode == 0
            if repaired:
                print(f"Cleared stale HDF5 consistency flag: {p}")
        except FileNotFoundError:
            repaired = False
        if not repaired:
            # If h5clear is unavailable or fails, the safest option is to
            # discard the partial feature file and let FuseLoc regenerate it.
            p.unlink(missing_ok=True)
            print(f"Removed stale/incomplete HDF5 file: {p}")

    def _h5_key_for_image_name(name: str) -> str:
        return "/".join(str(name).split("/")[-2:])

    def _expected_feature_keys(ds) -> list[str]:
        keys: list[str] = []
        for example in ds:
            if example is None:
                continue
            keys.append(_h5_key_for_image_name(example[1]))
        return keys

    def _remove_if_h5_incomplete(path: str, expected_keys: list[str], label: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            with _h5py.File(str(p), "r") as fd:
                missing = [key for key in expected_keys if key not in fd]
        except OSError:
            _repair_or_remove_stale_h5(str(p))
            return True
        if missing:
            p.unlink(missing_ok=True)
            preview = ", ".join(missing[:3])
            print(
                f"Removed incomplete FuseLoc {label} HDF5: {p} "
                f"({len(missing)}/{len(expected_keys)} missing; first: {preview})"
            )
            return True
        return False

    def _retry_after_h5_repair(self, fn, paths: list[str]):
        try:
            return fn(self)
        except OSError as exc:
            msg = str(exc)
            if "already open for write" not in msg and "file consistency flags" not in msg:
                raise
            for path in paths:
                _repair_or_remove_stale_h5(path)
            return fn(self)

    def produce_image_descriptor_lowmem(self, name):
        try:
            return original_produce_image_descriptor(self, name)
        finally:
            _maybe_empty_cuda_cache()

    def produce_local_descriptors_lowmem(self, name, fd=None):
        try:
            return original_produce_local_descriptors(self, name, fd)
        except RuntimeError:
            if fd is not None:
                try:
                    fd.flush()
                    fd.close()
                    print("Closed FuseLoc train-feature HDF5 after CUDA local-feature failure")
                except Exception:
                    pass
            raise
        finally:
            _maybe_empty_cuda_cache()

    def load_local_features_lowmem(self):
        path = f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        expected = _expected_feature_keys(self.dataset)
        removed = _remove_if_h5_incomplete(path, expected, "train features")
        if removed:
            # Selected features depend on the train feature H5 indices. If the
            # train H5 was partial, the selected H5 is stale too.
            selected_path = f"output/{self.ds_name}/{self.local_desc_model_name}_selected_features.h5"
            Path(selected_path).unlink(missing_ok=True)
        return _retry_after_h5_repair(self, original_load_local_features, [path])

    def load_selected_local_features_lowmem(self, all_features_h5):
        path = f"output/{self.ds_name}/{self.local_desc_model_name}_selected_features.h5"
        expected = _expected_feature_keys(self.dataset)
        _remove_if_h5_incomplete(path, expected, "selected train features")
        try:
            return original_load_selected_local_features(self, all_features_h5)
        except OSError as exc:
            msg = str(exc)
            if "already open for write" not in msg and "file consistency flags" not in msg:
                raise
            _repair_or_remove_stale_h5(path)
            return original_load_selected_local_features(self, all_features_h5)

    def detect_local_features_on_test_set_lowmem(self):
        path = f"output/{self.ds_name}/{self.local_desc_model_name}_features_test.h5"
        expected = _expected_feature_keys(self.test_dataset)
        _remove_if_h5_incomplete(path, expected, "test features")
        return _retry_after_h5_repair(self, original_detect_local_features_on_test_set, [path])

    def _select_global_indices(self, all_desc: np.ndarray) -> np.ndarray:
        if not self.use_rand_indices:
            return all_desc
        if "random" in self.order:
            np.random.seed(0)
            indices = np.arange(self.global_feature_dim)
            np.random.shuffle(indices)
            indices = indices[: self.feature_dim]
        elif self.order == "center":
            n = self.global_feature_dim
            m = self.feature_dim
            middle_index = n // 2
            start_index = max(middle_index - (m // 2), 0)
            end_index = min(middle_index + (m // 2) + (m % 2), n)
            if end_index - start_index < m:
                start_index = max(end_index - m, 0)
            indices = np.arange(start_index, end_index)
        elif self.order == "first":
            indices = np.arange(0, self.feature_dim)
        elif self.order == "last":
            start_index = max(self.global_feature_dim - self.feature_dim, 0)
            indices = np.arange(start_index, self.global_feature_dim)
        else:
            # Keep uncommon PCA/Gaussian paths exactly upstream.
            return original_collect_image_descriptors(self)
        self.global_rand_indices = indices
        return all_desc

    def collect_image_descriptors_lowmem(self):
        file_name1 = f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}.npy"
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}.npy"
        if _os.path.isfile(file_name1):
            all_desc = np.load(file_name1).astype(np.float32, copy=False)
            with open(file_name2, "rb") as afile:
                all_names = _pickle.load(afile)
        else:
            print(f"Cannot find {file_name1}")
            all_desc = np.zeros((len(self.dataset), self.global_feature_dim), dtype=np.float32)
            all_names = []
            idx = 0
            with torch_module.no_grad():
                for example in _tqdm(self.dataset, desc="Collecting image descriptors"):
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[1])
                    all_desc[idx] = np.asarray(image_descriptor, dtype=np.float32).reshape(-1)
                    all_names.append(example[1])
                    idx += 1
            np.save(file_name1, all_desc)
            with open(file_name2, "wb") as handle:
                _pickle.dump(all_names, handle, protocol=_pickle.HIGHEST_PROTOCOL)

        self.all_names = all_names
        # Upstream copies this full matrix. Keeping a single float32 matrix is
        # enough for FAISS DB-conversion lookup and halves peak CPU RAM.
        self.all_image_desc_for_db_conversion = all_desc.astype(np.float32, copy=False)
        print(f"Processes image descriptors of shape {all_desc.shape}")
        selected_desc = _select_global_indices(self, all_desc)
        if isinstance(selected_desc, dict):
            return selected_desc
        image2desc = {}
        for idx, name in enumerate(all_names):
            image2desc[name] = selected_desc[idx]
        return image2desc

    def collect_image_descriptors_for_test_set_lowmem(self):
        global_descriptors_path = f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}_test.h5"
        if not _os.path.isfile(global_descriptors_path):
            all_desc = np.zeros((len(self.test_dataset), self.global_feature_dim), dtype=np.float32)
            all_names = []
            idx = 0
            with torch_module.no_grad():
                for example in _tqdm(self.test_dataset, desc="Collecting global descriptors for test set"):
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[1])
                    all_desc[idx] = np.asarray(image_descriptor, dtype=np.float32).reshape(-1)
                    all_names.append(example[1])
                    idx += 1

            global_features_h5 = _h5py.File(str(global_descriptors_path), "a", libver="latest")
            for idx, name in enumerate(all_names):
                dd_utils.write_to_h5_file(
                    global_features_h5,
                    name,
                    {"global_descriptor": all_desc[idx]},
                )
            global_features_h5.close()
        return global_descriptors_path

    BaseTrainer.produce_image_descriptor = produce_image_descriptor_lowmem
    BaseTrainer.produce_local_descriptors = produce_local_descriptors_lowmem
    BaseTrainer.collect_image_descriptors = collect_image_descriptors_lowmem
    BaseTrainer.collect_image_descriptors_for_test_set = collect_image_descriptors_for_test_set_lowmem
    BaseTrainer.load_local_features = load_local_features_lowmem
    BaseTrainer.load_selected_local_features = load_selected_local_features_lowmem
    BaseTrainer.detect_local_features_on_test_set = detect_local_features_on_test_set_lowmem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official FuseLoc implementation on an Aachen DB leave-one-out split. "
            "This imports code from --fuseloc_root; it does not implement a FuseLoc-style surrogate."
        )
    )
    parser.add_argument("--fuseloc_root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--local_desc", type=str, default="d2net")
    parser.add_argument("--local_desc_dim", type=int, default=512)
    parser.add_argument("--global_desc", type=str, default="megaloc")
    parser.add_argument("--global_desc_dim", type=int, default=8448)
    parser.add_argument("--use_global", type=int, default=1)
    parser.add_argument("--convert", type=int, default=1)
    parser.add_argument("--lambda_val", type=float, default=0.4)
    parser.add_argument("--order", type=str, default="random-0")
    parser.add_argument(
        "--local_resize_max",
        type=int,
        default=None,
        help="Override FuseLoc local feature resize_max. Use 1024 if D2Net at 1600 is unstable/OOM.",
    )
    parser.add_argument(
        "--global_resize_max",
        type=int,
        default=None,
        help="Override FuseLoc global descriptor resize_max.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use cpu to avoid unsupported CUDA kernels in the official FuseLoc environment.",
    )
    parser.add_argument(
        "--disable_low_memory_patches",
        action="store_true",
        help="Disable runner-side float32/CUDA-cache memory patches for official FuseLoc.",
    )
    parser.add_argument("--clean_work_dir", action="store_true")
    args = parser.parse_args()

    fuseloc_root = args.fuseloc_root.resolve()
    if not (fuseloc_root / "main_aachen.py").exists():
        raise FileNotFoundError(f"Not a FuseLoc checkout: {fuseloc_root}")
    _require_module("faiss")
    _require_module("pykdtree")

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root = (args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])).resolve()
    query_names = split_query_names(split)
    map_names = split_map_names(split)
    ds_type = f"aachen_loo_{len(query_names)}_{Path(args.split_json).parent.name}"

    sys.path.insert(0, str(fuseloc_root))
    third_party = fuseloc_root / "third_party"
    for child in ("CricaVPR", "salad", "MixVPR"):
        path = third_party / child
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    import torch  # type: ignore
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _force_torch_cpu(torch)
        print("Forcing FuseLoc runner to CPU mode")
    elif args.device == "auto" and torch.cuda.is_available():
        try:
            torch.empty((1,), device="cuda").mul_(1.0)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"CUDA smoke test failed ({exc}); forcing FuseLoc runner to CPU mode")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            _force_torch_cpu(torch)
    import pycolmap  # type: ignore
    from torch.utils.data import Dataset  # type: ignore

    import dd_utils  # type: ignore
    from dataset import AachenDataset  # type: ignore
    from trainer import BaseTrainer  # type: ignore
    if not args.disable_low_memory_patches:
        _install_low_memory_fuseloc_patches(BaseTrainer, dd_utils, torch)
        print("Installed low-memory FuseLoc runtime patches")

    class FuseLocLOOTrainDataset(AachenDataset):
        def __init__(self, ds_dir: str, keep_names: list[str], ds_type_name: str):
            super().__init__(ds_dir=ds_dir, train=True)
            missing = [name for name in keep_names if name not in self.image_name2id]
            if missing:
                raise KeyError(f"{len(missing)} map images are missing from FuseLoc AachenDataset, first: {missing[0]}")
            self.img_ids = [self.image_name2id[name] for name in keep_names]
            self.ds_type = ds_type_name

    class FuseLocLOOTestDataset(Dataset):
        def __init__(self, train_ds: FuseLocLOOTrainDataset, heldout_names: list[str], ds_type_name: str):
            self.ds_type = ds_type_name
            self.ds_dir = train_ds.ds_dir
            self.images_dir = train_ds.images_dir
            self.img_ids = list(heldout_names)
            self.recon_images = train_ds.recon_images
            self.recon_cameras = train_ds.recon_cameras
            self.image_name2id = train_ds.image_name2id
            missing = [name for name in heldout_names if name not in self.image_name2id]
            if missing:
                raise KeyError(f"{len(missing)} query images are missing from FuseLoc AachenDataset, first: {missing[0]}")

        def __len__(self):
            return len(self.img_ids)

        def __getitem__(self, idx):
            name = self.img_ids[idx]
            image_id = self.image_name2id[name]
            recon_image = self.recon_images[image_id]
            camera_raw = self.recon_cameras[recon_image.camera_id]
            camera = pycolmap.Camera(
                model=_model_name(camera_raw.model),
                width=int(camera_raw.width),
                height=int(camera_raw.height),
                params=np.asarray(camera_raw.params, dtype=float),
            )
            intrinsics = torch.eye(3)
            params = np.asarray(camera_raw.params, dtype=float).reshape(-1)
            if len(params) >= 4:
                intrinsics[0, 0] = float(params[0])
                intrinsics[1, 1] = float(params[0])
                intrinsics[0, 2] = float(params[1])
                intrinsics[1, 2] = float(params[2])
            image_name = str(self.images_dir / name)
            return (
                None,
                image_name,
                name,
                [],
                None,
                intrinsics,
                camera,
                None,
                None,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / "fuseloc_work"
    if args.clean_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        t0 = time.perf_counter()
        encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
            args.local_desc,
            args.global_desc,
            int(args.global_desc_dim),
        )
        if args.local_resize_max is not None:
            conf_ns.resize_max = int(args.local_resize_max)
            print(f"Overriding FuseLoc local resize_max={conf_ns.resize_max}")
        if args.global_resize_max is not None and conf_ns_retrieval is not None:
            conf_ns_retrieval.resize_max = int(args.global_resize_max)
            print(f"Overriding FuseLoc global resize_max={conf_ns_retrieval.resize_max}")
        train_ds = FuseLocLOOTrainDataset(str(dataset_root), map_names, ds_type)
        test_ds = FuseLocLOOTestDataset(train_ds, query_names, ds_type)
        build_t0 = time.perf_counter()
        trainer = BaseTrainer(
            train_ds,
            test_ds,
            int(args.local_desc_dim),
            int(args.global_desc_dim),
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            bool(args.use_global),
            convert_to_db_desc=bool(args.convert),
            lambda_val=float(args.lambda_val),
            order=str(args.order),
        )
        build_time = time.perf_counter() - build_t0
        eval_t0 = time.perf_counter()
        trainer.evaluate()
        eval_time = time.perf_counter() - eval_t0
        total_time = time.perf_counter() - t0
    finally:
        os.chdir(old_cwd)

    local_name = getattr(trainer, "local_desc_model_name", args.local_desc)
    global_name = getattr(trainer, "global_desc_model_name", args.global_desc)
    raw_result = work_dir / "results" / _result_file_name(
        using_global=bool(args.use_global),
        local_name=local_name,
        global_name=global_name,
        global_dim=int(args.global_desc_dim),
        lambda_val=float(args.lambda_val),
        convert=bool(args.convert),
        order=str(args.order),
    )
    if not raw_result.exists():
        candidates = sorted((work_dir / "results").glob("Aachen_v1_1_eval_*.txt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"FuseLoc did not produce a result file under {work_dir / 'results'}")
        raw_result = candidates[-1]
    results_path = args.out_dir / "hloc_results.txt"
    shutil.copyfile(raw_result, results_path)

    mean_query_time = float(eval_time / max(1, len(query_names)))
    run_summary = {
        "runner": "official_fuseloc_loo",
        "fuseloc_root": str(fuseloc_root),
        "method": "FuseLoc",
        "local_desc": args.local_desc,
        "local_desc_dim": int(args.local_desc_dim),
        "global_desc": args.global_desc,
        "global_desc_dim": int(args.global_desc_dim),
        "use_global": bool(args.use_global),
        "convert": bool(args.convert),
        "lambda_val": float(args.lambda_val),
        "order": str(args.order),
        "num_queries": int(len(query_names)),
        "build_time_s": float(build_time),
        "eval_time_s": float(eval_time),
        "total_time_s": float(total_time),
        "mean_query_time_s": mean_query_time,
        "raw_result_file": str(raw_result),
        "results_file": str(results_path),
    }
    metrics = evaluate_results(
        split=split,
        results_path=results_path,
        mean_query_time_s=mean_query_time,
        extra_summary=run_summary,
    )
    write_json(args.out_dir / "run_summary.json", run_summary)
    write_json(args.out_dir / "metrics.json", metrics)
    print(metrics["summary"])


if __name__ == "__main__":
    main()
