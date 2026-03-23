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
from typing import Any, Dict, List, Optional, Tuple

from layercraft.core.navigator import DataNavigator


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
        - ``data_sources`` – list of two source dicts, each with:
          - ``"property"`` – property name.
          - ``"inherited"`` – (optional bool) search ancestor nodes.
          - ``"across"`` – ignored (kept for spec compatibility).
        - ``output_property`` – property name to write the correlation
          coefficient into.  Defaults to ``"correlation_<method>"``.
    """
    target_entity_display: str = task_spec["target_entity"]
    params: Dict[str, Any] = task_spec.get("operation_params") or {}
    method: str = params.get("method", "spearman")
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

    # Gather paired values across all instances
    x_vals: List[float] = []
    y_vals: List[float] = []

    for id_chain, attr_dict in navigator.iter_entity_instances(entity_path):
        # Resolve property A
        if prop_a in attr_dict and not inherited_a:
            val_a = attr_dict[prop_a]
        else:
            val_a = navigator.get_property(entity_path, id_chain, prop_a)
        # Resolve property B
        if prop_b in attr_dict and not inherited_b:
            val_b = attr_dict[prop_b]
        else:
            val_b = navigator.get_property(entity_path, id_chain, prop_b)

        if val_a is None or val_b is None:
            continue
        try:
            x_vals.append(float(val_a))
            y_vals.append(float(val_b))
        except (TypeError, ValueError):
            continue

    if len(x_vals) < 2:
        return

    coeff = _compute_correlation(x_vals, y_vals, method)

    # Write the coefficient as a property on the entity itself (root of the
    # entity path) – also useful for downstream inspection.
    # Since this is a global stat we store it as a root-level attribute
    # keyed by output_property.
    navigator.data[output_property] = coeff


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
