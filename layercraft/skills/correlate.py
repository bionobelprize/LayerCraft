"""Correlation skill.

Supported methods
-----------------
``pearson``
    Pearson product-moment correlation coefficient.
``spearman``
    Spearman rank-order correlation coefficient.
``kendall``
    Kendall's tau-b correlation coefficient.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from layercraft.core.navigator import DataNavigator
from layercraft.skills._scope import resolve_scope, _GLOBAL, _PER_ENTITY


class _CorrelationGroup(NamedTuple):
    """Accumulated paired values and targets for one correlation group."""

    x_values: List[float]
    y_values: List[float]
    id_chains: List[List[str]]


def correlate_skill(navigator: DataNavigator, task_spec: Dict[str, Any]) -> None:
    """Compute correlation between two data sources for *target_entity*.

    Parameters
    ----------
    navigator:
        Wraps the data structure.
    task_spec:
        Must contain:

        - ``target_entity`` – display path of the entity whose instances
          are iterated.
        - ``operation_params.method`` – ``"pearson"``, ``"spearman"``, or
          ``"kendall"`` (default: ``"spearman"``).
        - ``scope`` – **required**.  Controls correlation range:

          *New dict format*::

              {"source": "<entity_path>", "target": "<entity_path>"}

          ``target`` may be ``"root"`` (one global coefficient stored at
          ``navigator.data[output_property]``) or any entity display path
          (one coefficient per group at that level, written back to every
          source-entity instance in the group).

          *Legacy strings (still accepted)*:

          - ``"global"``   – one coefficient across all instances.
          - ``"per_group"`` – one coefficient per direct parent group,
            written to each entity instance in that group.
          - ``"per_entity"`` / ``"per_entity_id"`` – one coefficient per
            leaf entity ID across parents.
          - Backward-compat: ``operation_params.group_by="entity_id"``
            is treated as ``scope="per_entity"`` when ``scope`` is absent.

        - ``data_sources`` – list of two source dicts, each with:

          - ``"property"`` – property name.
          - ``"inherited"`` – (optional bool) search ancestor nodes.

        - ``output_property`` – property name to write the correlation
          coefficient into.  Defaults to ``"correlation_<method>"``.
    """
    target_entity_display: str = task_spec["target_entity"]
    params: Dict[str, Any] = task_spec.get("operation_params") or {}
    method: str = params.get("method", "spearman")
    group_by: Optional[str] = params.get("group_by")
    scope = task_spec.get("scope")
    sources: List[Dict[str, Any]] = task_spec.get("data_sources", [])
    output_property: str = task_spec.get(
        "output_property", f"correlation_{method}"
    )

    if len(sources) < 2:
        raise ValueError("correlate_skill requires exactly 2 data_sources")

    entity_path = navigator.resolve_entity_path(target_entity_display)
    if entity_path is None:
        raise ValueError(f"Cannot resolve entity path: '{target_entity_display}'")

    prop_a = sources[0]["property"]
    inherited_a = sources[0].get("inherited", False)
    prop_b = sources[1]["property"]
    inherited_b = sources[1].get("inherited", False)

    # Backward compatibility: old correlate specs used group_by only.
    if scope is None and group_by == "entity_id":
        scope = "per_entity"
    # Further backward compatibility: if scope is still None, default to global.
    if scope is None:
        scope = "global"

    group_depth, target_path = resolve_scope(scope, entity_path, navigator)

    if group_depth == _GLOBAL:
        _correlate_global(
            navigator, entity_path, prop_a, inherited_a,
            prop_b, inherited_b, method, output_property,
        )
    elif group_depth == _PER_ENTITY:
        _correlate_per_entity(
            navigator, entity_path, prop_a, inherited_a,
            prop_b, inherited_b, method, output_property,
        )
    else:
        _correlate_by_depth(
            navigator, entity_path, group_depth,
            prop_a, inherited_a, prop_b, inherited_b,
            method, output_property,
        )


def _collect_pair(
    navigator: DataNavigator,
    entity_path: Tuple[str, ...],
    id_chain: List[str],
    attr_dict: Dict[str, Any],
    prop_a: str,
    inherited_a: bool,
    prop_b: str,
    inherited_b: bool,
) -> Optional[Tuple[float, float]]:
    """Return ``(float_a, float_b)`` for one instance, or ``None`` to skip.

    Parameters
    ----------
    navigator:
        The :class:`DataNavigator` used for inherited property look-ups.
    entity_path:
        Tuple identifying the entity collection (e.g. ``("samples",
        "bacteria")``).
    id_chain:
        List of IDs locating this specific instance (e.g.
        ``["S1", "OTU1"]``).
    attr_dict:
        The instance's own attribute dict as yielded by
        :meth:`DataNavigator.iter_entity_instances`.
    prop_a:
        Name of the first property to retrieve.
    inherited_a:
        When *True*, search ancestor nodes for *prop_a* if it is not in
        *attr_dict*.
    prop_b:
        Name of the second property to retrieve.
    inherited_b:
        When *True*, search ancestor nodes for *prop_b* if it is not in
        *attr_dict*.
    """
    if prop_a in attr_dict and not inherited_a:
        val_a = attr_dict[prop_a]
    else:
        val_a = navigator.get_property(entity_path, id_chain, prop_a)

    if prop_b in attr_dict and not inherited_b:
        val_b = attr_dict[prop_b]
    else:
        val_b = navigator.get_property(entity_path, id_chain, prop_b)

    if val_a is None or val_b is None:
        return None
    try:
        return float(val_a), float(val_b)
    except (TypeError, ValueError):
        return None


def _correlate_global(
    navigator: DataNavigator,
    entity_path: Tuple[str, ...],
    prop_a: str,
    inherited_a: bool,
    prop_b: str,
    inherited_b: bool,
    method: str,
    output_property: str,
) -> None:
    """Compute a single correlation across all entity instances and store it
    as a root-level attribute on the navigator data dict."""
    x_vals: List[float] = []
    y_vals: List[float] = []

    for id_chain, attr_dict in navigator.iter_entity_instances(entity_path):
        pair = _collect_pair(
            navigator, entity_path, id_chain, attr_dict,
            prop_a, inherited_a, prop_b, inherited_b,
        )
        if pair is not None:
            x_vals.append(pair[0])
            y_vals.append(pair[1])

    if len(x_vals) < 2:
        return

    coeff = _compute_correlation(x_vals, y_vals, method)

    # Write the coefficient as a property on the entity itself (root of the
    # entity path) – also useful for downstream inspection.
    # Since this is a global stat we store it as a root-level attribute
    # keyed by output_property.
    navigator.data[output_property] = coeff


def _correlate_by_depth(
    navigator: DataNavigator,
    entity_path: Tuple[str, ...],
    group_depth: int,
    prop_a: str,
    inherited_a: bool,
    prop_b: str,
    inherited_b: bool,
    method: str,
    output_property: str,
) -> None:
    """Compute one coefficient per group defined by the first *group_depth*
    id-chain elements, then write that coefficient back to every source-entity
    instance in the group.

    This generalises the old ``"per_group"`` (where *group_depth* equals the
    number of levels in the parent entity path) and supports arbitrary
    ancestor levels in deep hierarchies.

    Example for ``entity_path = ("samples", "bacteria")`` and
    ``group_depth = 1``: one coefficient per sample, written to every
    bacteria instance in that sample.
    """
    groups: Dict[Any, _CorrelationGroup] = {}

    for id_chain, attr_dict in navigator.iter_entity_instances(entity_path):
        if group_depth <= len(id_chain):
            group_key: Any = tuple(id_chain[:group_depth])
        elif id_chain:
            group_key = tuple(id_chain)
        else:
            group_key = ("__all__",)

        pair = _collect_pair(
            navigator, entity_path, id_chain, attr_dict,
            prop_a, inherited_a, prop_b, inherited_b,
        )
        if pair is None:
            continue
        if group_key not in groups:
            groups[group_key] = _CorrelationGroup([], [], [])
        groups[group_key].x_values.append(pair[0])
        groups[group_key].y_values.append(pair[1])
        groups[group_key].id_chains.append(list(id_chain))

    for group in groups.values():
        if len(group.x_values) < 2:
            continue
        coeff = _compute_correlation(group.x_values, group.y_values, method)
        for chain in group.id_chains:
            navigator.set_property(entity_path, chain, output_property, coeff)


def _correlate_per_entity(
    navigator: DataNavigator,
    entity_path: Tuple[str, ...],
    prop_a: str,
    inherited_a: bool,
    prop_b: str,
    inherited_b: bool,
    method: str,
    output_property: str,
) -> None:
    """Compute one correlation coefficient per unique entity ID (last element
    of ``id_chain``) by aggregating all instances that share that ID across
    different parent groups (e.g. the same OTU across all samples).

    The resulting coefficient is written back to **every** occurrence of the
    entity ID via :meth:`DataNavigator.set_property`, so each sample's copy
    of the OTU node carries the same per-OTU correlation value.
    """
    # First pass: collect paired values and id_chains grouped by entity ID
    groups: Dict[str, _CorrelationGroup] = {}

    for id_chain, attr_dict in navigator.iter_entity_instances(entity_path):
        entity_id = id_chain[-1]
        pair = _collect_pair(
            navigator, entity_path, id_chain, attr_dict,
            prop_a, inherited_a, prop_b, inherited_b,
        )
        if pair is None:
            continue
        if entity_id not in groups:
            groups[entity_id] = _CorrelationGroup([], [], [])
        groups[entity_id].x_values.append(pair[0])
        groups[entity_id].y_values.append(pair[1])
        groups[entity_id].id_chains.append(list(id_chain))

    # Second pass: compute correlation per group and write to every instance
    for entity_id, group in groups.items():
        if len(group.x_values) < 2:
            continue
        coeff = _compute_correlation(group.x_values, group.y_values, method)
        for chain in group.id_chains:
            navigator.set_property(entity_path, chain, output_property, coeff)


def _compute_correlation(x: List[float], y: List[float], method: str) -> float:
    if method == "pearson":
        return _pearson(x, y)
    if method == "spearman":
        rx = _ranks(x)
        ry = _ranks(y)
        return _pearson(rx, ry)
    if method == "kendall":
        return _kendall_tau(x, y)
    raise ValueError(f"Unknown correlation method: '{method}'")


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(values: List[float]) -> List[float]:
    """Return fractional ranks (average rank for ties)."""
    sorted_vals = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        # Find extent of tie
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = x[i] - x[j]
            sy = y[i] - y[j]
            if sx * sy > 0:
                concordant += 1
            elif sx * sy < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs
