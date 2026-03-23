#!/usr/bin/env python3
"""analyze_structure.py

Analyze an alternating hierarchical JSON file (odd property layers, even ID
layers) and produce an entities.json metadata file describing entity paths,
own/inherited properties, instance counts, and child entities.

Usage
-----
    python analyze_structure.py --input data.json --output-json entities.json
    python analyze_structure.py --input data.json --print-report
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


PathKey = Tuple[str, ...]


@dataclass
class PropertyStats:
    count: int = 0
    types: Set[str] = field(default_factory=set)


@dataclass
class EntityStats:
    path: PathKey
    parent_path: Optional[PathKey]
    collection_key_level: int
    id_level: int
    attribute_level: int
    instance_count: int = 0
    id_examples: List[str] = field(default_factory=list)
    own_scalar_properties: Dict[str, PropertyStats] = field(default_factory=dict)
    child_entities: Set[str] = field(default_factory=set)


@dataclass
class AnalysisState:
    entities: Dict[PathKey, EntityStats] = field(default_factory=dict)
    root_scalar_properties: Dict[str, PropertyStats] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze alternating hierarchical JSON (odd property layers, even ID layers) "
            "and report entity inheritance, own attributes, and child entities."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument(
        "--output-json",
        help="Optional output path for structured analysis JSON",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print human-readable report to stdout",
    )
    parser.add_argument(
        "--max-id-examples",
        type=int,
        default=5,
        help="Maximum ID examples retained for each entity (default: 5)",
    )
    return parser.parse_args()


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def path_to_name(path: PathKey) -> str:
    return "ROOT" if len(path) == 0 else " > ".join(path)


def update_property_stats(target: Dict[str, PropertyStats], key: str, value: Any) -> None:
    stats = target.setdefault(key, PropertyStats())
    stats.count += 1
    stats.types.add(type_name(value))


def ensure_entity(
    state: AnalysisState,
    path: PathKey,
    parent_path: Optional[PathKey],
    collection_key_level: int,
) -> EntityStats:
    existing = state.entities.get(path)
    if existing is not None:
        return existing

    entity = EntityStats(
        path=path,
        parent_path=parent_path,
        collection_key_level=collection_key_level,
        id_level=collection_key_level + 1,
        attribute_level=collection_key_level + 2,
    )
    state.entities[path] = entity
    return entity


def analyze_entity_attributes(
    node: Dict[str, Any],
    current_path: PathKey,
    current_odd_level: int,
    state: AnalysisState,
    max_id_examples: int,
) -> None:
    for key, value in node.items():
        if isinstance(value, dict):
            child_path = current_path + (key,)
            entity = ensure_entity(
                state=state,
                path=child_path,
                parent_path=current_path,
                collection_key_level=current_odd_level,
            )

            if len(current_path) == 0:
                pass
            else:
                parent_entity = state.entities.get(current_path)
                if parent_entity is not None:
                    parent_entity.child_entities.add(key)

            for child_id, child_payload in value.items():
                entity.instance_count += 1
                if len(entity.id_examples) < max_id_examples:
                    entity.id_examples.append(str(child_id))

                if isinstance(child_payload, dict):
                    analyze_entity_attributes(
                        node=child_payload,
                        current_path=child_path,
                        current_odd_level=current_odd_level + 2,
                        state=state,
                        max_id_examples=max_id_examples,
                    )
                else:
                    state.warnings.append(
                        (
                            f"Expected dict at even ID layer for entity '{path_to_name(child_path)}', "
                            f"ID '{child_id}', but got {type_name(child_payload)}"
                        )
                    )
        else:
            if len(current_path) == 0:
                update_property_stats(state.root_scalar_properties, key, value)
            else:
                entity = state.entities.get(current_path)
                if entity is None:
                    state.warnings.append(
                        f"Internal warning: missing entity for path '{path_to_name(current_path)}'"
                    )
                    continue
                update_property_stats(entity.own_scalar_properties, key, value)


def sorted_property_list(props: Dict[str, PropertyStats]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in sorted(props.keys()):
        stats = props[name]
        out.append(
            {
                "name": name,
                "observed_count": stats.count,
                "types": sorted(stats.types),
            }
        )
    return out


def collect_ancestor_paths(path: PathKey) -> List[PathKey]:
    ancestors: List[PathKey] = [()]
    for i in range(1, len(path)):
        ancestors.append(path[:i])
    return ancestors


def build_report(state: AnalysisState) -> Dict[str, Any]:
    entities_out: List[Dict[str, Any]] = []

    for path in sorted(state.entities.keys()):
        entity = state.entities[path]

        inherited: Dict[str, Set[str]] = {}

        for ancestor_path in collect_ancestor_paths(path):
            if len(ancestor_path) == 0:
                for prop, stat in state.root_scalar_properties.items():
                    inherited.setdefault(prop, set()).update(stat.types)
            else:
                ancestor_entity = state.entities.get(ancestor_path)
                if ancestor_entity is None:
                    continue
                for prop, stat in ancestor_entity.own_scalar_properties.items():
                    inherited.setdefault(prop, set()).update(stat.types)

        own_props = sorted_property_list(entity.own_scalar_properties)
        inherited_props = [
            {"name": name, "types": sorted(types)}
            for name, types in sorted(inherited.items(), key=lambda kv: kv[0])
        ]

        available = sorted(set([p["name"] for p in own_props] + [p["name"] for p in inherited_props]))

        entities_out.append(
            {
                "entity_path": list(path),
                "entity_name": path[-1],
                "entity_path_display": path_to_name(path),
                "parent_entity": path_to_name(entity.parent_path) if entity.parent_path is not None else None,
                "collection_key_level": entity.collection_key_level,
                "id_level": entity.id_level,
                "attribute_level": entity.attribute_level,
                "instance_count": entity.instance_count,
                "id_examples": entity.id_examples,
                "own_properties": own_props,
                "inherited_properties": inherited_props,
                "all_available_properties": available,
                "child_entities": sorted(entity.child_entities),
            }
        )

    report = {
        "summary": {
            "entity_type_count": len(state.entities),
            "root_scalar_properties": sorted_property_list(state.root_scalar_properties),
            "warning_count": len(state.warnings),
        },
        "entities": entities_out,
        "warnings": state.warnings,
    }
    return report


def format_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    summary = report["summary"]

    lines.append("=== Hierarchical JSON Structure Analysis ===")
    lines.append(f"Entity types: {summary['entity_type_count']}")
    lines.append(f"Warnings: {summary['warning_count']}")

    root_props = summary["root_scalar_properties"]
    if root_props:
        names = ", ".join([f"{p['name']}:{'/'.join(p['types'])}" for p in root_props])
        lines.append(f"Root scalar properties: {names}")
    else:
        lines.append("Root scalar properties: (none)")

    for entity in report["entities"]:
        lines.append("")
        lines.append(f"[Entity] {entity['entity_path_display']}")
        lines.append(f"Parent: {entity['parent_entity']}")
        lines.append(
            "Levels: "
            f"collection={entity['collection_key_level']}, "
            f"id={entity['id_level']}, "
            f"attributes={entity['attribute_level']}"
        )
        lines.append(f"Instances observed: {entity['instance_count']}")

        if entity["id_examples"]:
            lines.append("ID examples: " + ", ".join(entity["id_examples"]))
        else:
            lines.append("ID examples: (none)")

        own_names = [f"{p['name']}:{'/'.join(p['types'])}" for p in entity["own_properties"]]
        lines.append("Own properties: " + (", ".join(own_names) if own_names else "(none)"))

        inherited_names = [f"{p['name']}:{'/'.join(p['types'])}" for p in entity["inherited_properties"]]
        lines.append(
            "Inherited properties: "
            + (", ".join(inherited_names) if inherited_names else "(none)")
        )

        children = entity["child_entities"]
        lines.append("Child entities: " + (", ".join(children) if children else "(none)"))

    if report["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for msg in report["warnings"]:
            lines.append(f"- {msg}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")

    state = AnalysisState()
    analyze_entity_attributes(
        node=data,
        current_path=(),
        current_odd_level=1,
        state=state,
        max_id_examples=max(1, args.max_id_examples),
    )

    report = build_report(state)

    if args.output_json:
        output_path = Path(args.output_json)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if args.print_report or not args.output_json:
        print(format_text_report(report))


if __name__ == "__main__":
    main()
