"""Intent parser - converts natural language task descriptions into structured
task-spec dicts.

This module provides a :class:`IntentParser` that can optionally call an LLM
API for high-quality parsing, but also includes a keyword-based heuristic
fallback that works without any external API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntentParser:
    """Convert natural language task descriptions into structured task specs.

    Parameters
    ----------
    entities_meta:
        The output of ``analyze_structure.py`` (loaded JSON).  Used to
        resolve entity names and property names from natural language.
    llm_client:
        Optional callable ``(prompt: str) -> str`` that queries an LLM.
        When provided, it is used for parsing; otherwise the built-in
        heuristic parser is used.
    """

    def __init__(
        self,
        entities_meta: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self._meta = entities_meta or {}
        self._llm = llm_client
        self._entity_display_paths: List[str] = [
            e["entity_path_display"]
            for e in self._meta.get("entities", [])
        ]
        self._all_properties: List[str] = self._collect_all_properties()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, natural_language: str) -> Dict[str, Any]:
        """Parse *natural_language* into a task spec dict.

        Parameters
        ----------
        natural_language:
            A task description in any language (English, Chinese, etc.).

        Returns
        -------
        A task-spec dict suitable for :class:`~layercraft.core.executor.TaskExecutor`.
        """
        if self._llm is not None:
            try:
                return self._parse_with_llm(natural_language)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM parsing failed, falling back to heuristic: %s", exc)
        return self._parse_heuristic(natural_language)

    # ------------------------------------------------------------------
    # LLM-based parsing
    # ------------------------------------------------------------------

    def _parse_with_llm(self, text: str) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt()
        user_prompt = f"Parse the following task description into a task spec JSON:\n\n{text}"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._llm(full_prompt)
        # Extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("LLM response did not contain a JSON object")
        return json.loads(json_match.group())

    def _build_system_prompt(self) -> str:
        entities_summary = "\n".join(
            f"  - {e['entity_path_display']} (properties: {', '.join(e.get('all_available_properties', []))})"
            for e in self._meta.get("entities", [])
        )
        return (
            "You are a task spec generator for the LayerCraft data framework.\n"
            "Convert the user's natural language task description into a structured JSON task spec.\n"
            "The task spec must have these fields:\n"
            '  task_id, target_entity, operation, operation_params, output_property\n'
            "Available entities:\n"
            f"{entities_summary}\n"
            "Supported operations: normalize, correlate, aggregate\n"
            "Return ONLY the JSON object, no other text."
        )

    # ------------------------------------------------------------------
    # Heuristic parser
    # ------------------------------------------------------------------

    def _parse_heuristic(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        operation = self._detect_operation(text_lower)
        target_entity = self._detect_entity(text_lower)
        target_property = self._detect_property(text_lower)
        method = self._detect_method(text_lower, operation)
        scope = self._detect_scope(text_lower, operation)

        task_id = f"{operation}_{uuid.uuid4().hex[:6]}"
        output_property = self._infer_output_property(
            operation, method, target_property
        )

        spec: Dict[str, Any] = {
            "task_id": task_id,
            "target_entity": target_entity,
            "operation": operation,
            "operation_params": {"method": method} if method else {},
            "scope": scope,
            "output_property": output_property,
        }

        if operation == "normalize":
            spec["target_property"] = target_property

        if operation == "correlate":
            props = self._detect_two_properties(text_lower)
            spec["data_sources"] = [
                {"property": props[0], "inherited": False, "across": target_entity},
                {"property": props[1], "inherited": True, "across": target_entity},
            ]

        if operation == "aggregate":
            spec["target_property"] = target_property
            func = self._detect_agg_func(text_lower)
            spec["operation_params"] = {"func": func}

        return spec

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _detect_operation(self, text: str) -> str:
        normalize_kw = [
            "normaliz", "归一化", "标准化", "normalize", "normalise",
        ]
        correlate_kw = [
            "correlat", "相关", "spearman", "pearson", "kendall",
        ]
        aggregate_kw = [
            "aggregat", "sum", "average", "mean", "total", "求和", "平均", "汇总",
        ]
        for kw in correlate_kw:
            if kw in text:
                return "correlate"
        for kw in normalize_kw:
            if kw in text:
                return "normalize"
        for kw in aggregate_kw:
            if kw in text:
                return "aggregate"
        return "normalize"  # default

    def _detect_entity(self, text: str) -> str:
        # Try to match entity display paths from metadata (longest match wins)
        best: Optional[str] = None
        for display in sorted(self._entity_display_paths, key=len, reverse=True):
            lower_display = display.lower()
            if lower_display in text:
                best = display
                break
            # Also try just the last component
            last = display.split(" > ")[-1].lower()
            if last in text and best is None:
                best = display
        return best if best is not None else (self._entity_display_paths[0] if self._entity_display_paths else "samples")

    def _detect_property(self, text: str) -> str:
        for prop in sorted(self._all_properties, key=len, reverse=True):
            if prop.lower() in text:
                return prop
        return "abundance"

    def _detect_two_properties(self, text: str) -> List[str]:
        found: List[str] = []
        for prop in sorted(self._all_properties, key=len, reverse=True):
            if prop.lower() in text and prop not in found:
                found.append(prop)
            if len(found) == 2:
                break
        while len(found) < 2:
            found.append("abundance")
        return found

    def _detect_method(self, text: str, operation: str) -> str:
        if operation == "normalize":
            if "min_max" in text or "min-max" in text:
                return "min_max"
            if "z_score" in text or "z-score" in text or "zscore" in text:
                return "z_score"
            return "sum_to_one"
        if operation == "correlate":
            if "pearson" in text:
                return "pearson"
            if "kendall" in text:
                return "kendall"
            return "spearman"
        return ""

    def _detect_scope(self, text: str, operation: str) -> Dict[str, str]:
        """Return a scope dict with a ``"target"`` key.

        The dict format is the recommended scope representation.
        ``{"target": "root"}`` means global; any other target is an entity
        display path that defines the grouping level.
        """
        per_entity_kw = [
            "per entity", "by entity", "entity id", "per otu", "by otu",
            "按个体", "按实体", "按otu", "每个otu",
        ]
        per_group_kw = [
            "per group", "by group", "per sample", "each sample",
            "按组", "每组", "每个样本", "按样本",
        ]
        for kw in per_entity_kw:
            if kw in text:
                # Group by leaf entity ID across parents.
                target = self._entity_display_paths[-1] if self._entity_display_paths else "root"
                return {"target": target}
        for kw in per_group_kw:
            if kw in text:
                # Group by the immediate parent entity.
                target = self._detect_parent_entity(text)
                return {"target": target}
        if operation == "correlate":
            return {"target": "root"}
        # Default: group by the immediate parent (equivalent to legacy per_group).
        target = self._detect_parent_entity(text)
        return {"target": target}

    def _detect_parent_entity(self, text: str) -> str:
        """Return the display path of the parent entity hinted by *text*, or
        ``"root"`` when no parent can be determined."""
        # Walk detected entity paths from longest (most specific) to shortest.
        detected = self._detect_entity(text)
        # Find an entity one level up from the detected entity.
        for display in sorted(self._entity_display_paths, key=len, reverse=True):
            lower = display.lower()
            if lower in text or display.split(" > ")[-1].lower() in text:
                # Check if there is a shorter entity that is a prefix of this one.
                parts = display.split(" > ")
                if len(parts) > 1:
                    parent_display = " > ".join(parts[:-1])
                    if parent_display in self._entity_display_paths:
                        return parent_display
        # Fall back: if the detected entity has a known parent, use it.
        for entity in self._meta.get("entities", []):
            if entity.get("entity_path_display") == detected:
                parent = entity.get("parent_entity")
                if parent:
                    return parent
        return "root"

    def _detect_agg_func(self, text: str) -> str:
        if "mean" in text or "average" in text or "平均" in text:
            return "mean"
        if "max" in text or "最大" in text:
            return "max"
        if "min" in text or "最小" in text:
            return "min"
        if "count" in text or "计数" in text:
            return "count"
        if "median" in text or "中位" in text:
            return "median"
        return "sum"

    def _infer_output_property(
        self, operation: str, method: str, property_name: str
    ) -> str:
        if operation == "normalize":
            return f"normalized_{property_name}"
        if operation == "correlate":
            return f"{method}_{property_name}" if method else f"correlation_{property_name}"
        if operation == "aggregate":
            return f"agg_{property_name}"
        return f"result_{property_name}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all_properties(self) -> List[str]:
        props: List[str] = []
        seen = set()
        for entity in self._meta.get("entities", []):
            for p in entity.get("all_available_properties", []):
                if p not in seen:
                    props.append(p)
                    seen.add(p)
        return props
