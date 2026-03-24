#!/usr/bin/env python3
"""End-to-end LayerCraft pipeline on test_data/raw.json.

This script demonstrates the core LayerCraft workflow:
1) Analyze hierarchical JSON structure and generate entities metadata.
2) Build DataNavigator with data + metadata.
3) Parse natural-language intents into task specs.
4) Execute a multi-step task pipeline.
5) Trigger auto skill generation via an alias operation.
6) Export a compact analysis summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_structure import AnalysisState, analyze_entity_attributes, build_report
from layercraft.core.executor import TaskExecutor
from layercraft.core.navigator import DataNavigator
from layercraft.llm.intent_parser import IntentParser


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    return data


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _analyze_structure(data: Dict[str, Any], max_id_examples: int = 5) -> Dict[str, Any]:
    state = AnalysisState()
    analyze_entity_attributes(
        node=data,
        current_path=(),
        current_odd_level=1,
        state=state,
        max_id_examples=max(1, max_id_examples),
    )
    return build_report(state)


def _build_tasks(parser: IntentParser) -> List[Dict[str, Any]]:
    # Natural-language driven specs
    normalize_task = parser.parse(
        "Normalize bacteria abundance per sample using sum_to_one"
    )
    normalize_task.update(
        {
            "task_id": "nl_norm_bacteria",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "per_group",
            "output_property": "norm_abundance_v2",
        }
    )

    aggregate_task = parser.parse(
        "Calculate sum of abundance for each sample in bacteria"
    )
    aggregate_task.update(
        {
            "task_id": "nl_agg_bacteria_sum",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "per_group",
            "output_property": "bacteria_total_abundance",
        }
    )

    correlate_task = parser.parse(
        "Calculate spearman correlation between abundance and day in bacteria"
    )
    correlate_task.update(
        {
            "task_id": "nl_corr_bacteria_day",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "spearman"},
            "data_sources": [
                {"property": "abundance", "inherited": False},
                {"property": "day", "inherited": True},
            ],
            "output_property": "bacteria_day_spearman",
        }
    )

    # Auto-extension: use alias op "norm" (maps to built-in normalize skill)
    auto_extend_task = {
        "task_id": "auto_norm_fungi_alias",
        "target_entity": "samples > fungi",
        "target_property": "abundance",
        "scope": "per_group",
        "operation": "norm",
        "operation_params": {"method": "min_max"},
        "output_property": "fungi_minmax_alias",
    }

    # Global aggregation example
    global_agg_task = {
        "task_id": "global_metabolite_mean",
        "target_entity": "samples > metabolites",
        "target_property": "abundance",
        "scope": "global",
        "operation": "aggregate",
        "operation_params": {"func": "mean"},
        "output_property": "global_mean_metabolite_abundance",
    }

    return [
        normalize_task,
        aggregate_task,
        correlate_task,
        auto_extend_task,
        global_agg_task,
    ]


def _collect_preview(navigator: DataNavigator, limit: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    entity_path = navigator.resolve_entity_path("samples > bacteria")
    if entity_path is None:
        return out

    for idx, (id_chain, attrs) in enumerate(navigator.iter_entity_instances(entity_path)):
        if idx >= limit:
            break
        day_val = navigator.get_property(entity_path, id_chain, "day")
        out.append(
            {
                "sample_id": id_chain[0] if len(id_chain) >= 1 else None,
                "otu_id": id_chain[1] if len(id_chain) >= 2 else None,
                "abundance": attrs.get("abundance"),
                "norm_abundance_v2": attrs.get("norm_abundance_v2"),
                "day": day_val,
            }
        )
    return out


def run_pipeline(
    input_path: Path,
    entities_out: Path,
    summary_out: Path,
    enriched_out: Path | None,
) -> None:
    data = _load_json(input_path)

    entities_meta = _analyze_structure(data)
    _save_json(entities_out, entities_meta)

    navigator = DataNavigator(data, entities_meta)
    parser = IntentParser(entities_meta=entities_meta)
    tasks = _build_tasks(parser)

    executor = TaskExecutor(auto_extend=True)
    executor.execute_pipeline(navigator, tasks)

    sample_totals: Dict[str, Any] = {}
    for sample_id, sample_attrs in navigator.data.get("samples", {}).items():
        if isinstance(sample_attrs, dict) and "bacteria_total_abundance" in sample_attrs:
            sample_totals[sample_id] = sample_attrs["bacteria_total_abundance"]

    summary: Dict[str, Any] = {
        "input": str(input_path),
        "entities_meta": str(entities_out),
        "task_count": len(tasks),
        "executed_operations": [t["operation"] for t in tasks],
        "registry_after_execution": executor.registry.list_operations(),
        "global_outputs": {
            "bacteria_day_spearman": navigator.data.get("bacteria_day_spearman"),
            "global_mean_metabolite_abundance": navigator.data.get(
                "global_mean_metabolite_abundance"
            ),
        },
        "sample_level_outputs_preview": dict(list(sample_totals.items())[:5]),
        "bacteria_instance_preview": _collect_preview(navigator, limit=8),
    }

    _save_json(summary_out, summary)

    if enriched_out is not None:
        _save_json(enriched_out, navigator.data)

    print("=== LayerCraft Pipeline Complete ===")
    print(f"Input data           : {input_path}")
    print(f"Entities metadata    : {entities_out}")
    print(f"Pipeline summary     : {summary_out}")
    if enriched_out is not None:
        print(f"Enriched output data : {enriched_out}")
    print(f"Tasks executed       : {len(tasks)}")
    print("Operations           : " + ", ".join(summary["executed_operations"]))
    print("Registered skills    : " + ", ".join(summary["registry_after_execution"]))
    print(
        "Spearman(abundance,day): "
        + str(summary["global_outputs"]["bacteria_day_spearman"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full LayerCraft analysis pipeline on raw hierarchical JSON data."
    )
    parser.add_argument(
        "--input",
        default="test_data/raw.json",
        help="Path to raw input JSON (default: test_data/raw.json)",
    )
    parser.add_argument(
        "--entities-out",
        default="outputs/entities_from_raw.json",
        help="Where to save generated entities metadata",
    )
    parser.add_argument(
        "--summary-out",
        default="outputs/pipeline_summary.json",
        help="Where to save compact pipeline summary",
    )
    parser.add_argument(
        "--enriched-out",
        default="",
        help="Optional path to save fully enriched data JSON (large file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched = Path(args.enriched_out) if args.enriched_out else None
    run_pipeline(
        input_path=Path(args.input),
        entities_out=Path(args.entities_out),
        summary_out=Path(args.summary_out),
        enriched_out=enriched,
    )


if __name__ == "__main__":
    main()
