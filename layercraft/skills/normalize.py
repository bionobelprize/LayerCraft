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

from typing import Any, Dict, List

from layercraft.core.navigator import DataNavigator


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
        - ``output_property`` – where to write the result, e.g.
          ``"normalized_abundance"``.
        - ``operation_params.method`` – one of ``"sum_to_one"``,
          ``"min_max"``, ``"z_score"`` (default: ``"sum_to_one"``).
        - ``scope`` – ``"per_group"`` (normalize within each parent group,
          e.g. per sample) or ``"global"`` (normalize across all instances).
        - ``scope_key`` – (for ``per_group``) the level of the parent ID to
          group by (currently uses the first element of the id_chain as the
          group key when scope is ``"per_group"``).
    """
    target_entity_display: str = task_spec["target_entity"]
    target_property: str = task_spec["target_property"]
    output_property: str = task_spec.get("output_property", f"normalized_{target_property}")
    params: Dict[str, Any] = task_spec.get("operation_params") or {}
    method: str = params.get("method", "sum_to_one")
    scope: str = task_spec.get("scope", "per_group")

    entity_path = navigator.resolve_entity_path(target_entity_display)
    if entity_path is None:
        raise ValueError(f"Cannot resolve entity path: '{target_entity_display}'")

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

    if scope == "global":
        groups: Dict[str, List[int]] = {"__all__": list(range(len(id_chains)))}
    else:
        # Group by the parent instance ID (first element of the id_chain relative
        # to the parent entity depth).  For a two-level entity like
        # ("samples", "bacteria") the first id in the chain is the sample ID.
        groups = {}
        for idx, id_chain in enumerate(id_chains):
            group_key = id_chain[0] if id_chain else "__all__"
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
