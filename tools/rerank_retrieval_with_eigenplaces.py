#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval_rerank_common import add_common_args, rerank_and_write


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank an existing retrieval shortlist with EigenPlaces-style global descriptors. "
            "The output format is: query db score. Provide precomputed HLoc-style global descriptor "
            "H5 files, or a local TorchScript/torchvision descriptor model."
        )
    )
    add_common_args(parser)
    args = parser.parse_args()
    if (
        args.model is None
        or (
            args.model
            and not str(args.model).startswith("torchhub:")
            and str(args.model) not in {"eigenplaces", "resnet50", "resnet101"}
            and not Path(str(args.model)).exists()
        )
        and args.features_h5 is None
        and not (args.query_features_h5 is not None and args.db_features_h5 is not None)
    ):
        args.model = "eigenplaces"
    summary = rerank_and_write(args, method="eigenplaces")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
