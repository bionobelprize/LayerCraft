"""Tests for the LayerCraft Core Engine."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from layercraft.core.navigator import DataNavigator
from layercraft.core.registry import SkillRegistry
from layercraft.core.executor import TaskExecutor
from layercraft.llm.intent_parser import IntentParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA: Dict[str, Any] = {
    "samples": {
        "A1_1": {
            "day": 21,
            "bacteria": {
                "OTU1": {
                    "abundance": 100,
                    "taxonomy_str": "Firmicutes;Bacilli",
                },
                "OTU2": {
                    "abundance": 300,
                    "taxonomy_str": "Proteobacteria;Gamma",
                },
            },
        },
        "A1_2": {
            "day": 14,
            "bacteria": {
                "OTU1": {
                    "abundance": 50,
                    "taxonomy_str": "Firmicutes;Bacilli",
                },
                "OTU2": {
                    "abundance": 150,
                    "taxonomy_str": "Proteobacteria;Gamma",
                },
            },
        },
    }
}

ENTITIES_META: Dict[str, Any] = {
    "entities": [
        {
            "entity_path": ["samples"],
            "entity_name": "samples",
            "entity_path_display": "samples",
            "parent_entity": None,
            "collection_key_level": 1,
            "id_level": 2,
            "attribute_level": 3,
            "instance_count": 2,
            "id_examples": ["A1_1", "A1_2"],
            "own_properties": [{"name": "day", "observed_count": 2, "types": ["int"]}],
            "inherited_properties": [],
            "all_available_properties": ["day"],
            "child_entities": ["bacteria"],
        },
        {
            "entity_path": ["samples", "bacteria"],
            "entity_name": "bacteria",
            "entity_path_display": "samples > bacteria",
            "parent_entity": "samples",
            "collection_key_level": 3,
            "id_level": 4,
            "attribute_level": 5,
            "instance_count": 4,
            "id_examples": ["OTU1", "OTU2"],
            "own_properties": [
                {"name": "abundance", "observed_count": 4, "types": ["int"]},
                {"name": "taxonomy_str", "observed_count": 4, "types": ["str"]},
            ],
            "inherited_properties": [
                {"name": "day", "types": ["int"]},
            ],
            "all_available_properties": ["abundance", "day", "taxonomy_str"],
            "child_entities": [],
        },
    ]
}


@pytest.fixture()
def sample_data() -> Dict[str, Any]:
    return copy.deepcopy(SAMPLE_DATA)


@pytest.fixture()
def navigator(sample_data: Dict[str, Any]) -> DataNavigator:
    return DataNavigator(sample_data, ENTITIES_META)


# ---------------------------------------------------------------------------
# DataNavigator tests
# ---------------------------------------------------------------------------

class TestDataNavigator:
    def test_init_requires_dict(self) -> None:
        with pytest.raises(TypeError):
            DataNavigator([])  # type: ignore[arg-type]

    def test_data_property(self, navigator: DataNavigator) -> None:
        assert "samples" in navigator.data

    def test_iter_entity_instances_top_level(self, navigator: DataNavigator) -> None:
        results = list(navigator.iter_entity_instances(("samples",)))
        assert len(results) == 2
        ids = [chain[0] for chain, _ in results]
        assert set(ids) == {"A1_1", "A1_2"}

    def test_iter_entity_instances_nested(self, navigator: DataNavigator) -> None:
        results = list(navigator.iter_entity_instances(("samples", "bacteria")))
        assert len(results) == 4
        for id_chain, attr_dict in results:
            assert len(id_chain) == 2
            assert "abundance" in attr_dict

    def test_get_property_own(self, navigator: DataNavigator) -> None:
        val = navigator.get_property(("samples", "bacteria"), ["A1_1", "OTU1"], "abundance")
        assert val == 100

    def test_get_property_inherited(self, navigator: DataNavigator) -> None:
        val = navigator.get_property(("samples", "bacteria"), ["A1_1", "OTU1"], "day")
        assert val == 21

    def test_get_property_missing_returns_none(self, navigator: DataNavigator) -> None:
        val = navigator.get_property(("samples", "bacteria"), ["A1_1", "OTU1"], "nonexistent")
        assert val is None

    def test_set_property(self, navigator: DataNavigator) -> None:
        navigator.set_property(
            ("samples", "bacteria"), ["A1_1", "OTU1"], "test_prop", 42
        )
        val = navigator.data["samples"]["A1_1"]["bacteria"]["OTU1"]["test_prop"]
        assert val == 42

    def test_set_property_invalid_collection_raises(self, navigator: DataNavigator) -> None:
        with pytest.raises(KeyError):
            navigator.set_property(
                ("no_such", "bacteria"), ["X", "Y"], "prop", 1
            )

    def test_collect_property_series(self, navigator: DataNavigator) -> None:
        chains, values = navigator.collect_property_series(
            ("samples", "bacteria"), "abundance"
        )
        assert len(chains) == 4
        assert set(values) == {100, 300, 50, 150}

    def test_resolve_entity_path_from_meta(self, navigator: DataNavigator) -> None:
        path = navigator.resolve_entity_path("samples > bacteria")
        assert path == ("samples", "bacteria")

    def test_resolve_entity_path_fallback(self, navigator: DataNavigator) -> None:
        path = navigator.resolve_entity_path("samples > bacteria > unknown")
        assert path == ("samples", "bacteria", "unknown")

    def test_list_entity_paths(self, navigator: DataNavigator) -> None:
        paths = navigator.list_entity_paths()
        assert "samples" in paths
        assert "samples > bacteria" in paths


# ---------------------------------------------------------------------------
# SkillRegistry tests
# ---------------------------------------------------------------------------

class TestSkillRegistry:
    def test_register_and_get(self) -> None:
        reg = SkillRegistry()

        @reg.register("my_op")
        def my_op(nav: Any, ts: Any) -> None:
            pass

        assert reg.has("my_op")
        assert reg.get("my_op") is my_op

    def test_missing_returns_none(self) -> None:
        reg = SkillRegistry()
        assert reg.get("ghost") is None
        assert not reg.has("ghost")

    def test_load_builtins(self) -> None:
        reg = SkillRegistry().load_builtins()
        for op in ("normalize", "correlate", "aggregate"):
            assert reg.has(op), f"Missing built-in: {op}"

    def test_list_operations(self) -> None:
        reg = SkillRegistry().load_builtins()
        ops = reg.list_operations()
        assert ops == sorted(ops)


# ---------------------------------------------------------------------------
# Normalize skill tests
# ---------------------------------------------------------------------------

class TestNormalizeSkill:
    def test_sum_to_one(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        task_spec = {
            "task_id": "norm_001",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "per_group",
            "operation": "normalize",
            "operation_params": {"method": "sum_to_one"},
            "output_property": "normalized_abundance",
        }
        executor.execute(navigator, task_spec)

        # Each sample group should sum to 1
        for sample_id in ("A1_1", "A1_2"):
            bacteria = navigator.data["samples"][sample_id]["bacteria"]
            total = sum(b["normalized_abundance"] for b in bacteria.values())
            assert abs(total - 1.0) < 1e-9, f"Sample {sample_id} does not sum to 1"

    def test_min_max(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        task_spec = {
            "task_id": "norm_002",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "per_group",
            "operation": "normalize",
            "operation_params": {"method": "min_max"},
            "output_property": "mm_abundance",
        }
        executor.execute(navigator, task_spec)

        # Check that values are in [0, 1]
        for sample_id in ("A1_1", "A1_2"):
            bacteria = navigator.data["samples"][sample_id]["bacteria"]
            for otu, attrs in bacteria.items():
                val = attrs["mm_abundance"]
                assert 0.0 <= val <= 1.0, f"Out of [0,1]: {otu} in {sample_id} = {val}"

    def test_global_scope(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        task_spec = {
            "task_id": "norm_003",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "global",
            "operation": "normalize",
            "operation_params": {"method": "sum_to_one"},
            "output_property": "global_norm",
        }
        executor.execute(navigator, task_spec)

        total = 0.0
        for s in navigator.data["samples"].values():
            for otu in s["bacteria"].values():
                total += otu.get("global_norm", 0.0)
        assert abs(total - 1.0) < 1e-9

    def test_non_numeric_values_skipped(self, sample_data: Dict[str, Any]) -> None:
        # Inject a non-numeric value for one OTU
        sample_data["samples"]["A1_1"]["bacteria"]["OTU1"]["abundance"] = "N/A"
        nav = DataNavigator(sample_data, ENTITIES_META)
        executor = TaskExecutor()
        task_spec = {
            "task_id": "norm_skip",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "per_group",
            "operation": "normalize",
            "operation_params": {"method": "sum_to_one"},
            "output_property": "normalized_abundance",
        }
        # Should not raise; non-numeric OTU is silently skipped
        executor.execute(nav, task_spec)
        # OTU2 in A1_1 (abundance=300) should get normalized_abundance=1.0
        assert nav.data["samples"]["A1_1"]["bacteria"]["OTU2"]["normalized_abundance"] == 1.0


# ---------------------------------------------------------------------------
# Correlate skill tests
# ---------------------------------------------------------------------------

class TestCorrelateSkill:
    def _make_linear_data(self) -> DataNavigator:
        """Create data where abundance and day are perfectly correlated."""
        data = {
            "samples": {
                f"S{i}": {
                    "day": float(i),
                    "bacteria": {
                        "OTU1": {"abundance": float(i * 10), "normalized_abundance": float(i)},
                    },
                }
                for i in range(1, 6)
            }
        }
        meta = {
            "entities": [
                {
                    "entity_path": ["samples"],
                    "entity_name": "samples",
                    "entity_path_display": "samples",
                    "parent_entity": None,
                    "collection_key_level": 1,
                    "id_level": 2,
                    "attribute_level": 3,
                    "instance_count": 5,
                    "id_examples": [f"S{i}" for i in range(1, 6)],
                    "own_properties": [{"name": "day", "observed_count": 5, "types": ["float"]}],
                    "inherited_properties": [],
                    "all_available_properties": ["day"],
                    "child_entities": ["bacteria"],
                },
                {
                    "entity_path": ["samples", "bacteria"],
                    "entity_name": "bacteria",
                    "entity_path_display": "samples > bacteria",
                    "parent_entity": "samples",
                    "collection_key_level": 3,
                    "id_level": 4,
                    "attribute_level": 5,
                    "instance_count": 5,
                    "id_examples": ["OTU1"],
                    "own_properties": [
                        {"name": "abundance", "observed_count": 5, "types": ["float"]},
                        {"name": "normalized_abundance", "observed_count": 5, "types": ["float"]},
                    ],
                    "inherited_properties": [{"name": "day", "types": ["float"]}],
                    "all_available_properties": ["abundance", "day", "normalized_abundance"],
                    "child_entities": [],
                },
            ]
        }
        return DataNavigator(data, meta)

    def test_spearman_perfect_correlation(self) -> None:
        nav = self._make_linear_data()
        executor = TaskExecutor()
        task_spec = {
            "task_id": "corr_001",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "spearman"},
            "data_sources": [
                {"property": "abundance", "inherited": False, "across": "samples"},
                {"property": "day", "inherited": True, "across": "samples"},
            ],
            "output_property": "spearman_corr",
        }
        executor.execute(nav, task_spec)
        assert abs(nav.data["spearman_corr"] - 1.0) < 1e-9

    def test_pearson_negative_correlation(self) -> None:
        data = {
            "samples": {
                f"S{i}": {
                    "day": float(i),
                    "bacteria": {
                        "OTU1": {"abundance": float(5 - i)},
                    },
                }
                for i in range(1, 6)
            }
        }
        meta = {
            "entities": [
                {
                    "entity_path": ["samples"],
                    "entity_name": "samples",
                    "entity_path_display": "samples",
                    "parent_entity": None,
                    "collection_key_level": 1,
                    "id_level": 2,
                    "attribute_level": 3,
                    "instance_count": 5,
                    "id_examples": [],
                    "own_properties": [{"name": "day", "observed_count": 5, "types": ["float"]}],
                    "inherited_properties": [],
                    "all_available_properties": ["day"],
                    "child_entities": ["bacteria"],
                },
                {
                    "entity_path": ["samples", "bacteria"],
                    "entity_name": "bacteria",
                    "entity_path_display": "samples > bacteria",
                    "parent_entity": "samples",
                    "collection_key_level": 3,
                    "id_level": 4,
                    "attribute_level": 5,
                    "instance_count": 5,
                    "id_examples": [],
                    "own_properties": [{"name": "abundance", "observed_count": 5, "types": ["float"]}],
                    "inherited_properties": [{"name": "day", "types": ["float"]}],
                    "all_available_properties": ["abundance", "day"],
                    "child_entities": [],
                },
            ]
        }
        nav = DataNavigator(data, meta)
        executor = TaskExecutor()
        task_spec = {
            "task_id": "corr_002",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "pearson"},
            "data_sources": [
                {"property": "abundance", "inherited": False},
                {"property": "day", "inherited": True},
            ],
            "output_property": "pearson_corr",
        }
        executor.execute(nav, task_spec)
        assert nav.data["pearson_corr"] < -0.9


# ---------------------------------------------------------------------------
# Correlate skill – group_by="entity_id" tests
# ---------------------------------------------------------------------------

class TestCorrelateSkillGroupByEntityId:
    """Tests for per-OTU (group_by='entity_id') correlation."""

    def _make_multi_otu_data(self) -> DataNavigator:
        """Create data with two OTUs across five samples.

        OTU1: normalized_abundance == day  → perfect positive correlation
        OTU2: normalized_abundance == -day → perfect negative correlation
        """
        data = {
            "samples": {
                f"S{i}": {
                    "day": float(i),
                    "bacteria": {
                        "OTU1": {"normalized_abundance": float(i)},
                        "OTU2": {"normalized_abundance": float(-i)},
                    },
                }
                for i in range(1, 6)
            }
        }
        meta = {
            "entities": [
                {
                    "entity_path": ["samples"],
                    "entity_name": "samples",
                    "entity_path_display": "samples",
                    "parent_entity": None,
                    "collection_key_level": 1,
                    "id_level": 2,
                    "attribute_level": 3,
                    "instance_count": 5,
                    "id_examples": [f"S{i}" for i in range(1, 6)],
                    "own_properties": [
                        {"name": "day", "observed_count": 5, "types": ["float"]}
                    ],
                    "inherited_properties": [],
                    "all_available_properties": ["day"],
                    "child_entities": ["bacteria"],
                },
                {
                    "entity_path": ["samples", "bacteria"],
                    "entity_name": "bacteria",
                    "entity_path_display": "samples > bacteria",
                    "parent_entity": "samples",
                    "collection_key_level": 3,
                    "id_level": 4,
                    "attribute_level": 5,
                    "instance_count": 10,
                    "id_examples": ["OTU1", "OTU2"],
                    "own_properties": [
                        {
                            "name": "normalized_abundance",
                            "observed_count": 10,
                            "types": ["float"],
                        }
                    ],
                    "inherited_properties": [{"name": "day", "types": ["float"]}],
                    "all_available_properties": ["day", "normalized_abundance"],
                    "child_entities": [],
                },
            ]
        }
        return DataNavigator(data, meta)

    def test_per_otu_spearman_coefficient_values(self) -> None:
        """OTU1 should have correlation +1, OTU2 should have correlation -1."""
        nav = self._make_multi_otu_data()
        executor = TaskExecutor()
        task_spec = {
            "task_id": "corr_per_otu",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "spearman", "group_by": "entity_id"},
            "data_sources": [
                {"property": "normalized_abundance", "inherited": False},
                {"property": "day", "inherited": True},
            ],
            "output_property": "day_correlation",
        }
        executor.execute(nav, task_spec)

        # All five sample nodes for OTU1 should carry coefficient ≈ +1.0
        for i in range(1, 6):
            coeff = nav.data["samples"][f"S{i}"]["bacteria"]["OTU1"]["day_correlation"]
            assert abs(coeff - 1.0) < 1e-9, f"OTU1 S{i}: expected 1.0, got {coeff}"

        # All five sample nodes for OTU2 should carry coefficient ≈ -1.0
        for i in range(1, 6):
            coeff = nav.data["samples"][f"S{i}"]["bacteria"]["OTU2"]["day_correlation"]
            assert abs(coeff + 1.0) < 1e-9, f"OTU2 S{i}: expected -1.0, got {coeff}"

    def test_per_otu_result_not_stored_at_root(self) -> None:
        """With group_by='entity_id' the root-level key must NOT be set."""
        nav = self._make_multi_otu_data()
        executor = TaskExecutor()
        task_spec = {
            "task_id": "corr_per_otu_root_check",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "spearman", "group_by": "entity_id"},
            "data_sources": [
                {"property": "normalized_abundance", "inherited": False},
                {"property": "day", "inherited": True},
            ],
            "output_property": "day_correlation",
        }
        executor.execute(nav, task_spec)
        assert "day_correlation" not in nav.data

    def test_per_otu_pearson(self) -> None:
        """Same data should yield the same coefficients with Pearson."""
        nav = self._make_multi_otu_data()
        executor = TaskExecutor()
        task_spec = {
            "task_id": "corr_per_otu_pearson",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "pearson", "group_by": "entity_id"},
            "data_sources": [
                {"property": "normalized_abundance", "inherited": False},
                {"property": "day", "inherited": True},
            ],
            "output_property": "pearson_day_corr",
        }
        executor.execute(nav, task_spec)
        for i in range(1, 6):
            assert (
                abs(
                    nav.data["samples"][f"S{i}"]["bacteria"]["OTU1"][
                        "pearson_day_corr"
                    ]
                    - 1.0
                )
                < 1e-9
            )

    def test_per_otu_insufficient_data_skipped(self) -> None:
        """An OTU that appears in fewer than 2 samples must not receive the
        output property (not enough data for a correlation)."""
        data = {
            "samples": {
                "S1": {
                    "day": 1.0,
                    "bacteria": {
                        "OTU_SINGLE": {"normalized_abundance": 1.0},
                        "OTU_MULTI": {"normalized_abundance": 1.0},
                    },
                },
                "S2": {
                    "day": 2.0,
                    "bacteria": {
                        "OTU_MULTI": {"normalized_abundance": 2.0},
                    },
                },
            }
        }
        meta = {
            "entities": [
                {
                    "entity_path": ["samples"],
                    "entity_name": "samples",
                    "entity_path_display": "samples",
                    "parent_entity": None,
                    "collection_key_level": 1,
                    "id_level": 2,
                    "attribute_level": 3,
                    "instance_count": 2,
                    "id_examples": ["S1", "S2"],
                    "own_properties": [
                        {"name": "day", "observed_count": 2, "types": ["float"]}
                    ],
                    "inherited_properties": [],
                    "all_available_properties": ["day"],
                    "child_entities": ["bacteria"],
                },
                {
                    "entity_path": ["samples", "bacteria"],
                    "entity_name": "bacteria",
                    "entity_path_display": "samples > bacteria",
                    "parent_entity": "samples",
                    "collection_key_level": 3,
                    "id_level": 4,
                    "attribute_level": 5,
                    "instance_count": 3,
                    "id_examples": ["OTU_SINGLE", "OTU_MULTI"],
                    "own_properties": [
                        {
                            "name": "normalized_abundance",
                            "observed_count": 3,
                            "types": ["float"],
                        }
                    ],
                    "inherited_properties": [{"name": "day", "types": ["float"]}],
                    "all_available_properties": ["day", "normalized_abundance"],
                    "child_entities": [],
                },
            ]
        }
        nav = DataNavigator(data, meta)
        executor = TaskExecutor()
        task_spec = {
            "task_id": "corr_insufficient",
            "target_entity": "samples > bacteria",
            "operation": "correlate",
            "operation_params": {"method": "spearman", "group_by": "entity_id"},
            "data_sources": [
                {"property": "normalized_abundance", "inherited": False},
                {"property": "day", "inherited": True},
            ],
            "output_property": "day_corr",
        }
        executor.execute(nav, task_spec)

        # OTU_SINGLE only appears once → not enough data → no property written
        assert "day_corr" not in nav.data["samples"]["S1"]["bacteria"]["OTU_SINGLE"]

        # OTU_MULTI appears in both samples → coefficient written to both
        assert "day_corr" in nav.data["samples"]["S1"]["bacteria"]["OTU_MULTI"]
        assert "day_corr" in nav.data["samples"]["S2"]["bacteria"]["OTU_MULTI"]


# ---------------------------------------------------------------------------
# Aggregate skill tests
# ---------------------------------------------------------------------------

class TestAggregateSkill:
    def test_sum_per_group(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        task_spec = {
            "task_id": "agg_001",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "per_group",
            "operation": "aggregate",
            "operation_params": {"func": "sum"},
            "output_property": "total_abundance",
        }
        executor.execute(navigator, task_spec)

        assert navigator.data["samples"]["A1_1"]["total_abundance"] == 400.0
        assert navigator.data["samples"]["A1_2"]["total_abundance"] == 200.0

    def test_mean_global(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        task_spec = {
            "task_id": "agg_002",
            "target_entity": "samples > bacteria",
            "target_property": "abundance",
            "scope": "global",
            "operation": "aggregate",
            "operation_params": {"func": "mean"},
            "output_property": "mean_abundance",
        }
        executor.execute(navigator, task_spec)
        # (100 + 300 + 50 + 150) / 4 = 150
        assert navigator.data["mean_abundance"] == 150.0


# ---------------------------------------------------------------------------
# TaskExecutor tests
# ---------------------------------------------------------------------------

class TestTaskExecutor:
    def test_unknown_operation_raises(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        with pytest.raises(ValueError, match="No skill registered"):
            executor.execute(navigator, {"operation": "do_magic"})

    def test_missing_operation_key_raises(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        with pytest.raises(ValueError, match="must contain an 'operation'"):
            executor.execute(navigator, {})

    def test_pipeline(self, navigator: DataNavigator) -> None:
        executor = TaskExecutor()
        tasks = [
            {
                "task_id": "p1",
                "target_entity": "samples > bacteria",
                "target_property": "abundance",
                "scope": "per_group",
                "operation": "normalize",
                "operation_params": {"method": "sum_to_one"},
                "output_property": "norm_abund",
            },
            {
                "task_id": "p2",
                "target_entity": "samples > bacteria",
                "target_property": "norm_abund",
                "scope": "per_group",
                "operation": "aggregate",
                "operation_params": {"func": "sum"},
                "output_property": "total_norm",
            },
        ]
        executor.execute_pipeline(navigator, tasks)

        # norm_abund per sample should sum to ~1
        for sample_id in ("A1_1", "A1_2"):
            total = sum(
                b.get("norm_abund", 0.0)
                for b in navigator.data["samples"][sample_id]["bacteria"].values()
            )
            assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# IntentParser tests
# ---------------------------------------------------------------------------

class TestIntentParser:
    def _parser(self) -> IntentParser:
        return IntentParser(entities_meta=ENTITIES_META)

    def test_parse_normalize_chinese(self) -> None:
        parser = self._parser()
        spec = parser.parse("把每个样品的所有细菌的abundance进行归一化处理")
        assert spec["operation"] == "normalize"

    def test_parse_correlate(self) -> None:
        parser = self._parser()
        spec = parser.parse("计算bacteria的abundance和day的spearman相关性系数")
        assert spec["operation"] == "correlate"
        assert spec["operation_params"]["method"] == "spearman"

    def test_parse_aggregate(self) -> None:
        parser = self._parser()
        spec = parser.parse("计算每个样品所有细菌abundance的sum")
        assert spec["operation"] == "aggregate"

    def test_output_property_inferred(self) -> None:
        parser = self._parser()
        spec = parser.parse("normalize the abundance")
        assert "normalized" in spec["output_property"]


# ---------------------------------------------------------------------------
# analyze_structure.py CLI tests
# ---------------------------------------------------------------------------

class TestAnalyzeStructureCLI:
    def test_print_report(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.json"
        input_file.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "analyze_structure.py", "--input", str(input_file), "--print-report"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
        assert "Entity types:" in result.stdout
        assert "samples" in result.stdout

    def test_output_json(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.json"
        output_file = tmp_path / "entities.json"
        input_file.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "analyze_structure.py",
                "--input",
                str(input_file),
                "--output-json",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
        assert output_file.exists()

        report = json.loads(output_file.read_text(encoding="utf-8"))
        assert "entities" in report
        assert report["summary"]["entity_type_count"] >= 2

    def test_entity_properties_in_report(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.json"
        output_file = tmp_path / "entities.json"
        input_file.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "analyze_structure.py",
                "--input",
                str(input_file),
                "--output-json",
                str(output_file),
            ],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
        )

        report = json.loads(output_file.read_text(encoding="utf-8"))
        entities_by_path = {
            e["entity_path_display"]: e for e in report["entities"]
        }

        assert "samples" in entities_by_path
        bacteria_entity = entities_by_path.get("samples > bacteria")
        assert bacteria_entity is not None

        own_prop_names = [p["name"] for p in bacteria_entity["own_properties"]]
        assert "abundance" in own_prop_names

        inherited_prop_names = [p["name"] for p in bacteria_entity["inherited_properties"]]
        assert "day" in inherited_prop_names
