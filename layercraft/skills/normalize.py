"""Normalization skill.

Supported methods
-----------------
``sum_to_one``
    Divide each value by the sum of all values in the group so that the
    group sums to 1 (relative abundance / proportion).
``min_max``
    Scale values to the [0, 1] range within the group.
``z_score``
    Standardize values to zero mean and unit standard deviation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from layercraft.core.navigator import DataNavigator
from layercraft.skills._scope import resolve_scope, _GLOBAL, _PER_ENTITY


def normalize_skill(navigator: DataNavigator, task_spec: Dict[str, Any]) -> None:
    """Normalize *target_property* for every instance of *target_entity*.

    Parameters
    ----------
    navigator:
        Wraps the data structure to operate on.
    task_spec:
        Must contain:

        - ``target_entity`` – display path, e.g. ``"samples > bacteria"``.
        - ``target_property`` – property name to normalize, e.g.
          ``"abundance"``.
        - ``scope`` – **required**.  Controls the normalization range:

          *New dict format*::

              {"source": "<entity_path>", "target": "<entity_path>"}

          ``target`` may be ``"root"`` (normalize globally across all
          instances) or any entity display path (normalize within each
          group at that hierarchy level).  ``source`` is optional and
          defaults to ``target_entity``.

          *Legacy strings (still accepted)*:

          - ``"global"`` – normalize across all instances.
          - ``"per_group"`` – normalize within each direct parent group.
          - ``"per_entity"`` / ``"per_entity_id"`` – normalize across
            all occurrences sharing the same leaf entity ID.

        - ``output_property`` – where to write the result (default:
          ``"normalized_<target_property>"``).
        - ``operation_params.method`` – one of ``"sum_to_one"``,
          ``"min_max"``, ``"z_score"`` (default: ``"sum_to_one"``).
    """
    target_entity_display: str = task_spec["target_entity"]
    target_property: str = task_spec["target_property"]
    output_property: str = task_spec.get("output_property", f"normalized_{target_property}")
    params: Dict[str, Any] = task_spec.get("operation_params") or {}
    method: str = params.get("method", "sum_to_one")
    scope = task_spec.get("scope")

    entity_path = navigator.resolve_entity_path(target_entity_display)
    if entity_path is None:
        raise ValueError(f"Cannot resolve entity path: '{target_entity_display}'")

    group_depth, _target_path = resolve_scope(scope, entity_path, navigator)

    # Collect all (id_chain, value) pairs
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

    # Build groups mapping group_key → list of indices into id_chains/raw_values
    groups: Dict[Any, List[int]] = {}

    if group_depth == _GLOBAL:
        groups["__all__"] = list(range(len(id_chains)))
    elif group_depth == _PER_ENTITY:
        # Legacy per_entity: group by leaf entity ID across all parent contexts.
        for idx, id_chain in enumerate(id_chains):
            group_key: Any = id_chain[-1] if id_chain else "__all__"
            groups.setdefault(group_key, []).append(idx)
    else:
        # Depth-based grouping: group by the first group_depth elements of
        # id_chain.  This supports arbitrary ancestor levels in deep hierarchies.
        for idx, id_chain in enumerate(id_chains):
            if group_depth <= len(id_chain):
                group_key = tuple(id_chain[:group_depth])
            else:
                group_key = tuple(id_chain)
            groups.setdefault(group_key, []).append(idx)

    normalized: List[float] = [0.0] * len(raw_values)

    for indices in groups.values():
        group_vals = [raw_values[i] for i in indices]
        computed = _apply_method(group_vals, method)
        for i, norm_val in zip(indices, computed):
            normalized[i] = norm_val

    for id_chain, norm_val in zip(id_chains, normalized):
        navigator.set_property(entity_path, id_chain, output_property, norm_val)


def _apply_method(values: List[float], method: str) -> List[float]:
    if method == "sum_to_one":
        total = sum(values)
        if total == 0:
            return [0.0] * len(values)
        return [v / total for v in values]

    if method == "min_max":
        lo, hi = min(values), max(values)
        rng = hi - lo
        if rng == 0:
            return [0.0] * len(values)
        return [(v - lo) / rng for v in values]

    if method == "z_score":
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance ** 0.5
        if std == 0:
            return [0.0] * len(values)
        return [(v - mean) / std for v in values]

    raise ValueError(f"Unknown normalization method: '{method}'")
