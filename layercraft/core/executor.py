"""TaskExecutor - drives execution of structured task specifications.

A *task spec* is a dict (often produced by the LLM intent parser) that
describes what to compute.  Example::

    {
        "task_id": "normalize_001",
        "target_entity": "samples > metabolites",
        "target_property": "abundance",
        "scope": "per_group",
        "scope_key": "sample_id",
        "operation": "normalize",
        "operation_params": {"method": "sum_to_one"},
        "output_property": "normalized_abundance"
    }

The executor looks up the ``"operation"`` in the :class:`SkillRegistry`,
invokes the skill, and returns the (possibly mutated) navigator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from layercraft.core.navigator import DataNavigator
from layercraft.core.registry import SkillRegistry

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Execute task specifications against a data structure.

    Parameters
    ----------
    registry:
        A :class:`~layercraft.core.registry.SkillRegistry` that maps
        operation names to skill callables.  If *None*, a new registry
        with all built-in skills is created.
    auto_extend:
        When *True*, attempt to auto-generate a skill when the operation
        is not found in the registry.
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        auto_extend: bool = False,
    ) -> None:
        if registry is None:
            registry = SkillRegistry().load_builtins()
        self._registry = registry
        self._auto_extend = auto_extend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    def execute(
        self,
        navigator: DataNavigator,
        task_spec: Dict[str, Any],
    ) -> DataNavigator:
        """Execute *task_spec* using the registered skill.

        Parameters
        ----------
        navigator:
            The data navigator wrapping the target data structure.
        task_spec:
            Structured task specification dict.

        Returns
        -------
        The same *navigator* (mutations are in-place).

        Raises
        ------
        ValueError
            When no skill is found for the requested operation and
            *auto_extend* is disabled.
        """
        operation = task_spec.get("operation")
        if not operation:
            raise ValueError("task_spec must contain an 'operation' key")

        skill = self._registry.get(operation)

        if skill is None and self._auto_extend:
            skill = self._try_auto_extend(operation, task_spec)

        if skill is None:
            raise ValueError(
                f"No skill registered for operation '{operation}'. "
                f"Available: {self._registry.list_operations()}"
            )

        task_id = task_spec.get("task_id", "<unnamed>")
        logger.info("Executing task '%s' with operation '%s'", task_id, operation)

        skill(navigator, task_spec)

        logger.info("Task '%s' completed", task_id)
        return navigator

    def execute_pipeline(
        self,
        navigator: DataNavigator,
        tasks: list,
    ) -> DataNavigator:
        """Execute a list of task specs in order.

        Parameters
        ----------
        navigator:
            The data navigator.
        tasks:
            Ordered list of task-spec dicts.

        Returns
        -------
        The navigator after all tasks have been applied.
        """
        for task_spec in tasks:
            self.execute(navigator, task_spec)
        return navigator

    # ------------------------------------------------------------------
    # Auto-extension
    # ------------------------------------------------------------------

    def _try_auto_extend(
        self, operation: str, task_spec: Dict[str, Any]
    ) -> Any:
        """Attempt to generate a skill for *operation* automatically."""
        try:
            from layercraft.auto.skill_generator import generate_skill

            skill = generate_skill(operation, task_spec)
            if skill is not None:
                self._registry.register_skill(operation, skill)
                logger.info("Auto-generated skill for operation '%s'", operation)
                return skill
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Auto-extension failed for operation '%s': %s", operation, exc
            )
        return None
