from __future__ import annotations

import argparse
from pathlib import Path
import json


def compute_threshold_metric(frames, trans_th, rot_th):
    valid = [f for f in frames if ('trans_err_m' in f and 'rot_err_deg' in f)]
    if not valid:
        return None
    ok = sum(1 for f in valid if f['trans_err_m'] <= trans_th and f['rot_err_deg'] <= rot_th)
    return float(ok / len(valid))


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate PLM-MATCH localization metrics.json at common thresholds')
    parser.add_argument('--metrics', required=True, type=str)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = data['frames']
    report = {
        '0.25m_2deg': compute_threshold_metric(frames, 0.25, 2.0),
        '0.5m_5deg': compute_threshold_metric(frames, 0.5, 5.0),
        '5m_10deg': compute_threshold_metric(frames, 5.0, 10.0),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
