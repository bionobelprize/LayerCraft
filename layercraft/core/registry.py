"""SkillRegistry - manages registration and lookup of processing skills."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type


SkillFn = Callable[..., Any]


class SkillRegistry:
    """A registry that maps operation names to callable skill implementations.

    A *skill* is any callable with the signature::

        skill(navigator, task_spec, **kwargs) -> None

    It receives the :class:`~layercraft.core.navigator.DataNavigator` and the
    full task-spec dict and is responsible for writing results back into the
    data structure via :meth:`~layercraft.core.navigator.DataNavigator.set_property`.

    Example
    -------
    >>> registry = SkillRegistry()
    >>> @registry.register("my_op")
    ... def my_op(navigator, task_spec):
    ...     pass
    """

    def __init__(self) -> None:
        self._skills: Dict[str, SkillFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, operation: str) -> Callable[[SkillFn], SkillFn]:
        """Decorator that registers *func* under *operation*.

        Parameters
        ----------
        operation:
            The operation name (matches ``task_spec["operation"]``).
        """

        def decorator(func: SkillFn) -> SkillFn:
            self._skills[operation] = func
            return func

        return decorator

    def register_skill(self, operation: str, func: SkillFn) -> None:
        """Imperatively register *func* under *operation*."""
        self._skills[operation] = func

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, operation: str) -> Optional[SkillFn]:
        """Return the skill callable for *operation*, or *None*."""
        return self._skills.get(operation)

    def has(self, operation: str) -> bool:
        """Return *True* when a skill for *operation* is registered."""
        return operation in self._skills

    def list_operations(self) -> List[str]:
        """Return sorted list of registered operation names."""
        return sorted(self._skills.keys())

    # ------------------------------------------------------------------
    # Auto-load built-in skills
    # ------------------------------------------------------------------

    def load_builtins(self) -> "SkillRegistry":
        """Register all built-in skills and return *self*."""
        from layercraft.skills.normalize import normalize_skill
        from layercraft.skills.correlate import correlate_skill
        from layercraft.skills.aggregate import aggregate_skill

        self.register_skill("normalize", normalize_skill)
        self.register_skill("correlate", correlate_skill)
        self.register_skill("aggregate", aggregate_skill)
        return self
