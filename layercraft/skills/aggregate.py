"""Aggregation skill.

Supported functions
-------------------
``sum``      Sum of all values in the group.
``mean``     Arithmetic mean.
``count``    Number of instances.
``min``      Minimum value.
``max``      Maximum value.
``median``   Median value.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layercraft.core.navigator import DataNavigator


def aggregate_skill(navigator: DataNavigator, task_spec: Dict[str, Any]) -> None:
    """Aggregate *target_property* for every parent group.

    The result is written onto each **parent** instance attribute dict
    as *output_property*.

    Parameters
    ----------
    navigator:
        Wraps the data structure.
    task_spec:
        Must contain:

        - ``target_entity`` – display path of the leaf entity to aggregate,
          e.g. ``"samples > bacteria"``.
        - ``target_property`` – numeric property to aggregate.
        - ``output_property`` – where to write the result on the parent
          instance (e.g. on each sample dict).
        - ``operation_params.func`` – one of ``"sum"``, ``"mean"``,
          ``"count"``, ``"min"``, ``"max"``, ``"median"``
          (default: ``"sum"``).
                - ``scope`` – controls aggregation range:
                    - ``"global"``: one value across all instances, stored at
                        ``navigator.data[output_property]``.
                    - ``"per_group"`` (default): group by direct parent ID; write
                        one value to each parent instance.
                    - ``"per_entity"`` (or ``"per_entity_id"``): group by leaf
                        entity ID across parents; write one value to each matching
                        entity instance.
    """
    target_entity_display: str = task_spec["target_entity"]
    target_property: str = task_spec["target_property"]
    output_property: str = task_spec.get("output_property", f"agg_{target_property}")
    params: Dict[str, Any] = task_spec.get("operation_params") or {}
    func_name: str = params.get("func", "sum")
    scope: str = task_spec.get("scope", "per_group")

    entity_path = navigator.resolve_entity_path(target_entity_display)
    if entity_path is None:
        raise ValueError(f"Cannot resolve entity path: '{target_entity_display}'")

    # Collect (id_chain, value) pairs
    id_chains: List[List[str]] = []
    raw_values: List[float] = []

    for id_chain, attr_dict in navigator.iter_entity_instances(entity_path):
        if target_property in attr_dict:
            try:
                val = float(attr_dict[target_property])
            except (TypeError, ValueError):
                continue
            id_chains.append(id_chain)
            raw_values.append(val)

    if not id_chains:
        return

    if scope == "global":
        agg_val = _apply_func(raw_values, func_name)
        navigator.data[output_property] = agg_val
        return

    if scope in ("per_entity", "per_entity_id"):
        entity_groups: Dict[str, List[float]] = {}
        entity_chains: Dict[str, List[List[str]]] = {}
        for id_chain, val in zip(id_chains, raw_values):
            entity_id = id_chain[-1] if id_chain else "__root__"
            entity_groups.setdefault(entity_id, []).append(val)
            entity_chains.setdefault(entity_id, []).append(list(id_chain))

        for entity_id, vals in entity_groups.items():
            agg_val = _apply_func(vals, func_name)
            for chain in entity_chains[entity_id]:
                navigator.set_property(entity_path, chain, output_property, agg_val)
        return

    if scope != "per_group":
        raise ValueError(
            "Unsupported scope for aggregate: "
            f"'{scope}'. Expected one of: global, per_group, per_entity."
        )

    # Group by parent (penultimate id chain element is the parent ID,
    # and the parent entity path is entity_path[:-1]).
    parent_entity_path = entity_path[:-1]

    # Map parent_id → list of values
    groups: Dict[str, List[float]] = {}
    group_parent_chain: Dict[str, List[str]] = {}

    for id_chain, val in zip(id_chains, raw_values):
        if len(id_chain) >= 2:
            parent_id = id_chain[-2]
            parent_chain = id_chain[:-1]
        else:
            parent_id = id_chain[0] if id_chain else "__root__"
            parent_chain = []
        groups.setdefault(parent_id, []).append(val)
        group_parent_chain[parent_id] = parent_chain

    for parent_id, vals in groups.items():
        agg_val = _apply_func(vals, func_name)
        parent_chain = group_parent_chain[parent_id]
        if parent_chain and parent_entity_path:
            try:
                navigator.set_property(
                    parent_entity_path, parent_chain, output_property, agg_val
                )
            except KeyError:
                pass
        else:
            navigator.data[output_property] = agg_val


def _apply_func(values: List[float], func_name: str) -> float:
    if func_name == "sum":
        return sum(values)
    if func_name == "count":
        return float(len(values))
    if func_name == "mean":
        return sum(values) / len(values)
    if func_name == "min":
        return min(values)
    if func_name == "max":
        return max(values)
    if func_name == "median":
        s = sorted(values)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0
    raise ValueError(f"Unknown aggregation function: '{func_name}'")
