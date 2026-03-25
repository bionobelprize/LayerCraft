"""Scope resolution utilities shared across LayerCraft built-in skills.

Scope defines *where new attribute data comes from* and *where the result
is written*—the two sides of every "add a property" operation.

Accepted ``scope`` formats
--------------------------
**New dict format (recommended)**::

    {"source": "<entity_display_path>", "target": "<entity_display_path>"}

    # source  – entity whose instances are iterated for input values.
    #           Optional; defaults to the task's ``target_entity``.
    # target  – "root" writes one result to navigator.data (global).
    #           Any entity display path writes one result per entity
    #           instance at that level, grouping source instances by
    #           their ancestor at that level.

    # Examples
    scope = {"target": "root"}                    # global
    scope = {"target": "samples"}                 # group by samples level
    scope = {"target": "samples > bacteria"}      # group by bacteria level

**Legacy string format (backward-compatible)**:

    * ``"global"``            – same as ``{"target": "root"}``
    * ``"per_group"``         – group by the immediate parent entity
    * ``"per_entity"``        – group by leaf entity ID across all parents
    * ``"per_entity_id"``     – alias for ``"per_entity"``

Omitting ``scope`` (``None``) raises a :exc:`ValueError`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

PathKey = Tuple[str, ...]

# Sentinel for the "per_entity" legacy grouping mode.
_PER_ENTITY = -2
_GLOBAL = -1


def resolve_scope(
    scope: Any,
    entity_path: PathKey,
    navigator: Any,
) -> Tuple[int, Optional[PathKey]]:
    """Parse *scope* and return ``(group_depth, target_path)``.

    Parameters
    ----------
    scope:
        Value of ``task_spec["scope"]``.  See module docstring for
        accepted formats.  ``None`` is not permitted.
    entity_path:
        Path tuple of the *source* entity (i.e. the resolved
        ``task_spec["target_entity"]``).
    navigator:
        :class:`~layercraft.core.navigator.DataNavigator` used to
        resolve display-path strings.

    Returns
    -------
    group_depth : int
        * ``-1`` (``_GLOBAL``) – one group for all instances; write to
          ``navigator.data``.
        * ``-2`` (``_PER_ENTITY``) – legacy per-entity grouping: group
          by the *last* element of the id-chain across all parent
          contexts.
        * ``k >= 1`` – group by the first *k* elements of the id-chain;
          the group key is ``tuple(id_chain[:k])``.
    target_path : PathKey or None
        The entity path to use when writing results.  ``None`` means
        write directly to ``navigator.data`` (root level).

    Raises
    ------
    ValueError
        If *scope* is ``None`` or contains an unrecognised value.
    """
    if scope is None:
        raise ValueError(
            "'scope' is a required parameter. "
            "Pass a dict such as {'target': 'root'} for a global operation, "
            "{'target': '<entity_display_path>'} to group and write results "
            "at a specific hierarchy level, or one of the legacy strings "
            "'global', 'per_group', 'per_entity'."
        )

    # ------------------------------------------------------------------
    # New dict format
    # ------------------------------------------------------------------
    if isinstance(scope, dict):
        target_str: str = scope.get("target", "root")
        if target_str == "root":
            return (_GLOBAL, None)
        target_path = navigator.resolve_entity_path(target_str)
        return (len(target_path), target_path)

    # ------------------------------------------------------------------
    # Legacy string format
    # ------------------------------------------------------------------
    if scope == "global":
        return (_GLOBAL, None)

    if scope == "per_group":
        parent_path: PathKey = entity_path[:-1]
        if not parent_path:
            # No parent exists; treat as global.
            return (_GLOBAL, None)
        return (len(parent_path), parent_path)

    if scope in ("per_entity", "per_entity_id"):
        return (_PER_ENTITY, entity_path)

    raise ValueError(
        f"Unrecognised scope value: {scope!r}. "
        "Use a dict {'target': 'root'} or {'target': '<entity_display_path>'} "
        "(new format), or one of 'global', 'per_group', 'per_entity' "
        "(legacy strings)."
    )
