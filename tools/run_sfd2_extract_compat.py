#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feixue94/sfd2 extractor with PyTorch 2.6 checkpoint compatibility.")
    parser.add_argument("--sfd2_root", required=True, type=Path)
    args, sfd2_args = parser.parse_known_args()

    root = args.sfd2_root.expanduser().resolve()
    script = root / "extract_localization.py"
    if not script.exists():
        raise FileNotFoundError(f"SFD2 extract_localization.py not found: {script}")

    import torch

    original_load = torch.load

    def _torch_load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = _torch_load_compat
    sys.path.insert(0, str(root))
    sys.argv = [str(script)] + [str(x) for x in sfd2_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
