#!/usr/bin/env python3
"""Minimal LayerCraft example: raw.json -> one task -> enriched json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layercraft.core.executor import TaskExecutor
from layercraft.core.navigator import DataNavigator


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    return data


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal LayerCraft pipeline: apply one normalize task on raw JSON"
    )
    parser.add_argument("--input", default="test_data/raw.json")
    parser.add_argument("--output", default="outputs/raw_minimal_enriched.json")
    args = parser.parse_args()

    data = load_json(Path(args.input))
    navigator = DataNavigator(data)


    task={
            "task_id": "nl_corr_fungi_day",
            "target_entity": "samples > fungi",
            "operation": "correlate",
            "operation_params": {"method": "spearman"},
            "scope": "per_entity",
            "data_sources": [
                {"property": "normalized_abundance", "inherited": False},
                {"property": "腐殖酸含量", "inherited": True},
            ],
            "output_property": "fungi_humic_acid_spearman",
        }


    TaskExecutor().execute(navigator, task)
    save_json(Path(args.output), navigator.data)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
