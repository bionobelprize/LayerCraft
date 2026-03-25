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

from typing import Any, Dict, List, Optional, Tuple

from layercraft.core.navigator import DataNavigator
from layercraft.skills._scope import resolve_scope, _GLOBAL, _PER_ENTITY


PathKey = Tuple[str, ...]


def aggregate_skill(navigator: DataNavigator, task_spec: Dict[str, Any]) -> None:
    """Aggregate *target_property* values and write the result to each group.

    The result is written onto each **target-level** instance.  For
    ``{"target": "samples"}`` the aggregated value is placed on each
    sample dict; for ``{"target": "root"}`` a single global value is
    stored at ``navigator.data[output_property]``.

    Parameters
    ----------
    navigator:
        Wraps the data structure.
    task_spec:
        Must contain:

        - ``target_entity`` – display path of the leaf entity to
          aggregate, e.g. ``"samples > bacteria"``.
        - ``target_property`` – numeric property to aggregate.
        - ``output_property`` – where to write the result.
        - ``operation_params.func`` – one of ``"sum"``, ``"mean"``,
          ``"count"``, ``"min"``, ``"max"``, ``"median"``
          (default: ``"sum"``).
        - ``scope`` – **required**.  Controls aggregation range:

          *New dict format*::

              {"source": "<entity_path>", "target": "<entity_path>"}

          ``target`` may be ``"root"`` (one global value) or any entity
          display path (one value per instance at that level; instances
          of the source entity are grouped by their ancestor at the
          target level and the aggregated value is written to each
          target-level instance).

          *Legacy strings (still accepted)*:

          - ``"global"``   – one value across all instances, at root.
          - ``"per_group"`` – group by direct parent; write to parent.
          - ``"per_entity"`` / ``"per_entity_id"`` – group by leaf
            entity ID across parents; write to each matching instance.
    """
    target_entity_display: str = task_spec["target_entity"]
    target_property: str = task_spec["target_property"]
    output_property: str = task_spec.get("output_property", f"agg_{target_property}")
    params: Dict[str, Any] = task_spec.get("operation_params") or {}
    func_name: str = params.get("func", "sum")
    scope = task_spec.get("scope")

    entity_path = navigator.resolve_entity_path(target_entity_display)
    if entity_path is None:
        raise ValueError(f"Cannot resolve entity path: '{target_entity_display}'")

    group_depth, target_path = resolve_scope(scope, entity_path, navigator)

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

    # ------------------------------------------------------------------
    # Global: one value for the entire dataset
    # ------------------------------------------------------------------
    if group_depth == _GLOBAL:
        agg_val = _apply_func(raw_values, func_name)
        navigator.data[output_property] = agg_val
        return

    # ------------------------------------------------------------------
    # Legacy per_entity: group by leaf entity ID across all parents
    # ------------------------------------------------------------------
    if group_depth == _PER_ENTITY:
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

    # ------------------------------------------------------------------
    # Depth-based: group by the first group_depth levels of the id_chain
    # and write to each instance at that level.  This handles arbitrary
    # ancestor depths in deep hierarchies.
    # ------------------------------------------------------------------
    groups: Dict[Any, List[float]] = {}
    group_target_chains: Dict[Any, List[str]] = {}

    for id_chain, val in zip(id_chains, raw_values):
        if group_depth <= len(id_chain):
            group_key: Any = tuple(id_chain[:group_depth])
            target_chain = list(id_chain[:group_depth])
        else:
            group_key = tuple(id_chain)
            target_chain = list(id_chain)
        groups.setdefault(group_key, []).append(val)
        group_target_chains[group_key] = target_chain

    for group_key, vals in groups.items():
        agg_val = _apply_func(vals, func_name)
        target_chain = group_target_chains[group_key]
        if target_path:
            try:
                navigator.set_property(
                    target_path, target_chain, output_property, agg_val
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
