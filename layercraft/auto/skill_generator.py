"""Auto skill generator.

When the :class:`~layercraft.core.executor.TaskExecutor` encounters an
operation that is not registered, it can call :func:`generate_skill` to
attempt runtime skill synthesis.

The current implementation provides template-based generation for a small
set of well-known operation patterns.  When an LLM client is available
it can be injected via :func:`set_llm_client` for open-ended generation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_llm_client: Optional[Callable[[str], str]] = None


def set_llm_client(client: Callable[[str], str]) -> None:
    """Set a global LLM client used for code generation.

    Parameters
    ----------
    client:
        Callable ``(prompt: str) -> str`` that returns Python source code
        implementing a skill function.
    """
    global _llm_client  # noqa: PLW0603
    _llm_client = client


def generate_skill(
    operation: str,
    task_spec: Dict[str, Any],
) -> Optional[Callable[..., Any]]:
    """Attempt to generate a skill callable for *operation*.

    First tries template-based generation, then falls back to LLM-based
    generation if a client has been configured.

    Parameters
    ----------
    operation:
        The operation name that has no registered skill.
    task_spec:
        The full task spec dict (may provide hints for generation).

    Returns
    -------
    A callable skill, or *None* when generation fails.
    """
    # Try template-based generation first
    skill = _template_generate(operation, task_spec)
    if skill is not None:
        logger.info("Template-generated skill for operation '%s'", operation)
        return skill

    # Try LLM-based generation
    if _llm_client is not None:
        skill = _llm_generate(operation, task_spec)
        if skill is not None:
            logger.info("LLM-generated skill for operation '%s'", operation)
            return skill

    return None


# ---------------------------------------------------------------------------
# Template-based generation
# ---------------------------------------------------------------------------

_OPERATION_ALIASES: Dict[str, str] = {
    "normalise": "normalize",
    "norm": "normalize",
    "corr": "correlate",
    "correlation": "correlate",
    "agg": "aggregate",
    "sum": "aggregate",
    "mean": "aggregate",
    "average": "aggregate",
}


def _template_generate(
    operation: str, task_spec: Dict[str, Any]
) -> Optional[Callable[..., Any]]:
    """Return a built-in skill aliased to *operation*, if one exists."""
    canonical = _OPERATION_ALIASES.get(operation.lower())
    if canonical is None:
        return None

    from layercraft.core.registry import SkillRegistry

    tmp = SkillRegistry().load_builtins()
    skill = tmp.get(canonical)
    if skill is None:
        return None

    # Wrap so that the aliased operation name also patches operation_params
    # if needed (e.g. "mean" implies aggregate with func=mean).
    if operation in ("mean", "average"):
        def _mean_wrapper(navigator: Any, ts: Dict[str, Any]) -> None:
            ts = dict(ts)
            params = dict(ts.get("operation_params") or {})
            params.setdefault("func", "mean")
            ts["operation_params"] = params
            skill(navigator, ts)  # type: ignore[operator]

        return _mean_wrapper

    if operation == "sum":
        def _sum_wrapper(navigator: Any, ts: Dict[str, Any]) -> None:
            ts = dict(ts)
            params = dict(ts.get("operation_params") or {})
            params.setdefault("func", "sum")
            ts["operation_params"] = params
            skill(navigator, ts)  # type: ignore[operator]

        return _sum_wrapper

    return skill


# ---------------------------------------------------------------------------
# LLM-based generation
# ---------------------------------------------------------------------------

def _llm_generate(
    operation: str, task_spec: Dict[str, Any]
) -> Optional[Callable[..., Any]]:
    """Use the configured LLM client to generate a skill function.

    .. warning::
        LLM-generated code is executed via ``exec()``.  Only use this
        feature with a trusted LLM client in a controlled environment.
        The generated code is not sandboxed and has full Python capabilities.
    """
    assert _llm_client is not None

    prompt = (
        f"Write a Python function named `{operation}_skill` that implements "
        f"the '{operation}' operation for the LayerCraft framework.\n"
        "The function signature must be:\n"
        f"    def {operation}_skill(navigator, task_spec): ...\n"
        "Where `navigator` is a DataNavigator instance and `task_spec` is a dict.\n"
        "The function should read from navigator.iter_entity_instances() and write "
        "results using navigator.set_property().\n"
        "Return ONLY the Python function definition, no imports or other code.\n"
        f"Task spec hint: {task_spec}"
    )

    try:
        code_str = _llm_client(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM code generation call failed: %s", exc)
        return None

    # Execute the generated code in a restricted namespace
    namespace: Dict[str, Any] = {}
    try:
        exec(compile(code_str, "<llm_generated>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM-generated code failed to compile/execute: %s", exc)
        return None

    func_name = f"{operation}_skill"
    skill = namespace.get(func_name)
    if callable(skill):
        return skill

    logger.warning(
        "LLM-generated code did not define expected function '%s'", func_name
    )
    return None
