#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def deep_set(d: dict, dotkey: str, value) -> None:
    keys = dotkey.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def apply_overrides(base: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        deep_set(cfg, key, value)
    return cfg


def resolve_path(value: str | None, *, base: Path, fallback_base: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists() or fallback_base is None:
        return candidate
    return fallback_base / path


def compute_threshold_metric(frames: list[dict], trans_th: float, rot_th: float) -> float | None:
    valid = [f for f in frames if ("trans_err_m" in f and "rot_err_deg" in f)]
    if not valid:
        return None
    ok = sum(1 for f in valid if float(f["trans_err_m"]) <= trans_th and float(f["rot_err_deg"]) <= rot_th)
    return float(ok / len(valid))


def _normalize_score(value) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.5:
        score /= 100.0
    return max(0.0, min(1.0, score))


def read_eval_file(path: Path, thresholds: list[dict]) -> dict[str, float | None]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores = {}
        for thr in thresholds:
            value = data.get(thr["key"])
            if value is None:
                value = data.get(thr["label"])
            scores[thr["key"]] = _normalize_score(value)
        return scores

    text = path.read_text(encoding="utf-8")
    scores = {}
    for thr in thresholds:
        trans = f"{thr['trans_m']}".rstrip("0").rstrip(".")
        rot = f"{thr['rot_deg']}".rstrip("0").rstrip(".")
        pattern = re.compile(
            rf"{re.escape(trans)}\s*m.*?{re.escape(rot)}\s*(?:deg|°).*?([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        scores[thr["key"]] = _normalize_score(match.group(1) if match else None)
    return scores


def collect_metrics(run_dir: Path, thresholds: list[dict], eval_file: str | None) -> tuple[dict[str, float | None], dict]:
    scores: dict[str, float | None] = {thr["key"]: None for thr in thresholds}
    details: dict[str, object] = {}

    if eval_file is not None:
        eval_path = run_dir / eval_file
        if eval_path.exists():
            scores.update(read_eval_file(eval_path, thresholds))
            details["eval_file"] = str(eval_path)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        frames = payload.get("frames", [])
        for thr in thresholds:
            if scores[thr["key"]] is None:
                scores[thr["key"]] = compute_threshold_metric(frames, float(thr["trans_m"]), float(thr["rot_deg"]))
        details["metrics_file"] = str(metrics_path)

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            details["run_summary"] = json.load(f)
    return scores, details


def runtime_from_details(details: dict, wall_time_s: float | None = None) -> float | None:
    run_summary = details.get("run_summary", {})
    if isinstance(run_summary, dict):
        for key in ("mean_query_time_s", "avg_query_time_s"):
            if key in run_summary:
                return _normalize_runtime(run_summary[key])
        num_queries = run_summary.get("num_queries")
        if wall_time_s is not None and num_queries:
            return float(wall_time_s) / max(1, int(num_queries))
    return None


def _normalize_runtime(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_plm(
    run_spec: dict,
    dataset_spec: dict,
    defaults: dict,
    manifest_dir: Path,
    method_dir: Path,
    rerun: bool,
) -> tuple[str, float | None]:
    out_dir = method_dir
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not rerun:
        return "cached", None

    config_path = resolve_path(run_spec["config"], base=manifest_dir, fallback_base=ROOT)
    base_cfg = load_yaml(config_path)
    cfg = apply_overrides(base_cfg, run_spec.get("overrides", {}))
    cfg["out_dir"] = str(out_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="table_i_", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

    python_bin = run_spec.get("python") or defaults.get("plm_python") or sys.executable
    pipeline = run_spec.get("pipeline", "plm_match.pipelines.hloc_localize")
    dataset_root = dataset_spec.get("dataset_root")
    cmd = [
        str(python_bin),
        "-m",
        pipeline,
        "--config",
        str(tmp_path),
        "--dataset_root",
        str(dataset_root),
        "--out_dir",
        str(out_dir),
    ]
    if bool(run_spec.get("disable_map_cache", defaults.get("disable_map_cache", False))):
        cmd.append("--disable-map-cache")
    if bool(run_spec.get("disable_query_cache", defaults.get("disable_query_cache", True))):
        cmd.append("--disable-query-cache")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    tmp_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return f"failed({result.returncode})", elapsed
    return "done", elapsed


def run_hloc(
    run_spec: dict,
    dataset_name: str,
    dataset_spec: dict,
    defaults: dict,
    method_dir: Path,
    rerun: bool,
) -> tuple[str, float | None]:
    out_dir = method_dir
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists() and not rerun:
        return "cached", None

    python_bin = run_spec.get("python") or defaults.get("hloc_python") or sys.executable
    hloc_root = run_spec.get("hloc_root") or defaults.get("hloc_root")
    if not hloc_root:
        return "missing_hloc_root", None
    cmd = [
        str(python_bin),
        str(ROOT / "tools" / "run_hloc_baseline.py"),
        "--dataset",
        dataset_spec.get("hloc_dataset", dataset_name),
        "--method",
        run_spec["method"],
        "--dataset_root",
        str(dataset_spec["dataset_root"]),
        "--out_dir",
        str(out_dir),
        "--hloc_root",
        str(hloc_root),
    ]

    optional_keys = (
        "image_dir",
        "reference_sfm",
        "query_list",
        "retrieval_file",
        "db_prefix",
        "extractor_conf",
        "matcher_conf",
    )
    for key in optional_keys:
        value = run_spec.get(key) or dataset_spec.get(key)
        if value:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    for key in ("max_keypoints", "resize_max", "match_threshold", "ransac_thresh", "skip_matches"):
        value = run_spec.get(key)
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    if bool(run_spec.get("overwrite", False)):
        cmd.append("--overwrite")
    if bool(run_spec.get("covisibility_clustering", False)):
        cmd.append("--covisibility-clustering")

    env = None
    pythonpath_entries = [str(Path(hloc_root))]
    if defaults.get("extra_pythonpath"):
        pythonpath_entries.append(str(defaults["extra_pythonpath"]))
    env = dict(**__import__("os").environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(pythonpath_entries + ([existing] if existing else []))

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        return f"failed({result.returncode})", elapsed
    return "done", elapsed


def write_csv(path: Path, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[label for _, label in columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({label: row.get(key, "") for key, label in columns})


def pct_text(score: float | None) -> str:
    if score is None:
        return ""
    return f"{score * 100:.1f}"


def secs_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def print_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    widths = []
    for key, label in columns:
        width = len(label)
        for row in rows:
            width = max(width, len(str(row.get(key, ""))))
        widths.append(width)
    header = " | ".join(label.ljust(width) for (_, label), width in zip(columns, widths))
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row.get(key, "")).ljust(width) for (key, _), width in zip(columns, widths)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential runner for Table I localization baselines")
    parser.add_argument("manifest", type=str, help="Path to the benchmark manifest YAML")
    parser.add_argument("--only", nargs="+", default=None, help="Run only these method names")
    parser.add_argument("--skip", nargs="+", default=None, help="Skip these method names")
    parser.add_argument("--rerun", action="store_true", help="Force rerun even if outputs exist")
    parser.add_argument("--summary-only", action="store_true", help="Only collect outputs and print the table")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    manifest = load_yaml(manifest_path)
    defaults = manifest.get("defaults", {})
    outputs_root = resolve_path(defaults.get("outputs_root", "./outputs/table_i"), base=manifest_dir, fallback_base=ROOT)
    outputs_root.mkdir(parents=True, exist_ok=True)

    datasets = manifest["datasets"]
    methods = manifest["methods"]
    if args.only:
        methods = [method for method in methods if method["name"] in set(args.only)]
    if args.skip:
        methods = [method for method in methods if method["name"] not in set(args.skip)]

    table_cfg = manifest.get("table", {})
    runtime_dataset = table_cfg.get("runtime_dataset", next(iter(datasets.keys())))

    detail_rows = []
    display_rows = []

    for index, method in enumerate(methods, start=1):
        method_name = method["name"]
        label = method.get("label", method_name)
        method_dir = outputs_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(methods)}] {label}")

        method_detail = {
            "name": method_name,
            "label": label,
            "runs": {},
        }
        display = {"method": label}
        runtime_value = None

        for dataset_name, dataset_spec in datasets.items():
            run_spec = (method.get("runs") or {}).get(dataset_name)
            thresholds = dataset_spec.get("thresholds", [])
            for thr in thresholds:
                display[thr["key"]] = ""

            if run_spec is None:
                method_detail["runs"][dataset_name] = {"status": "not_configured"}
                continue

            dataset_root = Path(dataset_spec["dataset_root"])
            if not dataset_root.exists():
                method_detail["runs"][dataset_name] = {"status": "dataset_missing", "dataset_root": str(dataset_root)}
                continue

            run_dir = method_dir / dataset_name
            run_dir.mkdir(parents=True, exist_ok=True)
            status = "summary_only"
            wall_time_s = None
            if not args.summary_only:
                kind = run_spec["kind"]
                if kind == "plm":
                    status, wall_time_s = run_plm(run_spec, dataset_spec, defaults, manifest_dir, run_dir, args.rerun)
                elif kind == "hloc":
                    status, wall_time_s = run_hloc(run_spec, dataset_name, dataset_spec, defaults, run_dir, args.rerun)
                else:
                    status = f"unknown_kind({kind})"

            eval_file = run_spec.get("eval_file", dataset_spec.get("default_eval_file"))
            scores, metric_details = collect_metrics(run_dir, thresholds, eval_file)
            for thr in thresholds:
                display[thr["key"]] = pct_text(scores.get(thr["key"]))

            run_detail = {
                "status": status,
                "run_dir": str(run_dir),
                "scores": scores,
                "wall_time_s": wall_time_s,
                **metric_details,
            }
            method_detail["runs"][dataset_name] = run_detail

            if dataset_name == runtime_dataset:
                runtime_value = runtime_from_details(metric_details, wall_time_s=wall_time_s)

        display["runtime_per_query_s"] = secs_text(runtime_value)
        method_detail["runtime_per_query_s"] = runtime_value
        detail_rows.append(method_detail)
        display_rows.append(display)

    ordered_columns = [("method", "Method")]
    for dataset_name, dataset_spec in datasets.items():
        for thr in dataset_spec.get("thresholds", []):
            ordered_columns.append((thr["key"], thr["label"]))
    ordered_columns.append(("runtime_per_query_s", "Runtime / query (s)"))

    write_csv(outputs_root / "table_i_results.csv", display_rows, ordered_columns)
    write_json(outputs_root / "table_i_results.json", {"rows": detail_rows, "display_rows": display_rows})
    print_table(display_rows, ordered_columns)


if __name__ == "__main__":
    main()
