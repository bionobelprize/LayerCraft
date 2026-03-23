# LayerCraft

**LayerCraft** is an Intelligent Data Structure Automation Framework for processing and transforming hierarchical JSON data with alternating property/ID layer patterns.

It provides a skill-based execution engine, a natural-language task parser, and an auto-generation system so you can analyse and reshape nested scientific datasets—such as microbial abundance tables or metabolomics results—without writing boilerplate traversal code.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Hierarchical JSON Data Format](#hierarchical-json-data-format)
  - [Entity Metadata](#entity-metadata)
  - [Task Specification](#task-specification)
- [Usage](#usage)
  - [DataNavigator](#datanavigator)
  - [SkillRegistry](#skillregistry)
  - [TaskExecutor](#taskexecutor)
  - [Built-in Skills](#built-in-skills)
  - [Natural Language Parsing](#natural-language-parsing)
  - [Auto Skill Generation](#auto-skill-generation)
  - [CLI – analyze_structure.py](#cli--analyze_structurepy)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Layer-aware data navigation** – traverse alternating property/ID layers in nested JSON without hand-written recursion.
- **Skill registry** – register, discover, and execute named data-transformation operations via a simple decorator API.
- **Task pipeline execution** – chain multiple task specifications into a single sequential pipeline.
- **Built-in skills** – `normalize` (sum-to-one, min-max, z-score), `aggregate` (sum, mean, count, min, max, median), and `correlate` (Pearson, Spearman, Kendall) out of the box.
- **Natural language task parsing** – convert plain English or Chinese descriptions into structured task specs, with an optional LLM backend and a keyword-based heuristic fallback.
- **Auto skill generation** – automatically synthesise missing skills from templates or (optionally) an LLM, so unknown operations never hard-stop a pipeline.
- **CLI analyser** – inspect any hierarchical JSON file and export rich entity metadata ready for use with `DataNavigator`.

---

## Project Structure

```
LayerCraft/
├── layercraft/                 # Main package
│   ├── __init__.py
│   ├── core/
│   │   ├── executor.py         # TaskExecutor – runs task specs
│   │   ├── navigator.py        # DataNavigator – hierarchy traversal
│   │   └── registry.py         # SkillRegistry – operation lookup
│   ├── skills/
│   │   ├── normalize.py        # Normalization skill
│   │   ├── aggregate.py        # Aggregation skill
│   │   └── correlate.py        # Correlation skill
│   ├── llm/
│   │   └── intent_parser.py    # Natural language → task spec
│   └── auto/
│       └── skill_generator.py  # Template & LLM-based skill synthesis
├── tests/
│   └── test_core_engine.py     # Full test suite
├── analyze_structure.py        # CLI tool for JSON structure analysis
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python **3.9** or later
- [numpy](https://numpy.org/) >= 1.24
- [scipy](https://scipy.org/) >= 1.10

Optional (for running the test suite):

- [pytest](https://pytest.org/) >= 7.0

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bionobelprize/LayerCraft.git
cd LayerCraft

# Install in editable mode
pip install -e .

# Install with development/test dependencies
pip install -e ".[dev]"
```

---

## Quick Start

```python
from layercraft.core.navigator import DataNavigator
from layercraft.core.executor import TaskExecutor

# --- 1. Prepare your hierarchical data ---
data = {
    "samples": {                     # property layer
        "A1_1": {                    # id layer
            "day": 1,
            "bacteria": {            # property layer (nested)
                "OTU_1": {"abundance": 40},
                "OTU_2": {"abundance": 60},
            },
        },
        "A1_2": {
            "day": 2,
            "bacteria": {
                "OTU_1": {"abundance": 30},
                "OTU_2": {"abundance": 70},
            },
        },
    }
}

# --- 2. Describe the entity structure ---
entities_meta = {
    "samples": {
        "levels": {"collection": 1, "id": 2, "attributes": 3},
        "own_properties": ["day"],
        "inherited_properties": [],
        "children": ["bacteria"],
        "parent": None,
    },
    "samples > bacteria": {
        "levels": {"collection": 3, "id": 4, "attributes": 5},
        "own_properties": ["abundance"],
        "inherited_properties": ["day"],
        "children": [],
        "parent": "samples",
    },
}

# --- 3. Build executor and navigator ---
executor = TaskExecutor()   # loads built-in skills automatically
navigator = DataNavigator(data, entities_meta)

# --- 4. Define and run a task ---
task = {
    "task_id": "norm_001",
    "target_entity": "samples > bacteria",
    "target_property": "abundance",
    "scope": "per_group",
    "operation": "normalize",
    "operation_params": {"method": "sum_to_one"},
    "output_property": "normalized_abundance",
}

executor.execute(navigator, task)

# --- 5. Read the result ---
for id_chain, attrs in navigator.iter_entity_instances("samples > bacteria"):
    sample_id, otu_id = id_chain
    print(sample_id, otu_id, attrs.get("normalized_abundance"))
# A1_1 OTU_1 0.4
# A1_1 OTU_2 0.6
# A1_2 OTU_1 0.3
# A1_2 OTU_2 0.7
```

---

## Core Concepts

### Hierarchical JSON Data Format

LayerCraft works with JSON structures that alternate between **property layers** (dicts whose keys are property names or entity type names) and **ID layers** (dicts whose keys are instance identifiers).

```
data
└── samples          ← property layer  (level 1)
    └── <sample_id>  ← id layer        (level 2)
        ├── day      ← attribute
        └── bacteria ← property layer  (level 3)
            └── <otu_id>               (level 4)
                └── abundance          ← attribute
```

The level numbers in `entities_meta` tell `DataNavigator` exactly where each entity lives so it can iterate and read/write properties without hard-coded paths.

### Entity Metadata

Each entry in `entities_meta` describes one entity type:

| Key | Description |
|-----|-------------|
| `levels` | Dict with keys `collection`, `id`, `attributes` mapping to nesting depths |
| `own_properties` | Properties stored directly on this entity's attribute dict |
| `inherited_properties` | Properties accessible via ancestor attribute dicts |
| `children` | Names of child entity types nested inside this entity |
| `parent` | Name of the parent entity type, or `None` for roots |

### Task Specification

A task spec is a plain Python `dict`:

| Field | Required | Description |
|-------|----------|-------------|
| `task_id` | Yes | Unique identifier string |
| `target_entity` | Yes | Entity path, e.g. `"samples > bacteria"` |
| `target_property` | Yes | Property name to operate on |
| `operation` | Yes | Skill name, e.g. `"normalize"`, `"aggregate"`, `"correlate"` |
| `operation_params` | No | Dict of keyword arguments forwarded to the skill |
| `scope` | No | `"per_group"` (default) or `"global"` |
| `output_property` | No | Where to write results (defaults to `target_property`) |
| `source_property` | No | Second property for binary operations like `correlate` |

---

## Usage

### DataNavigator

```python
from layercraft.core.navigator import DataNavigator

nav = DataNavigator(data, entities_meta)

# Iterate all instances of an entity type
for id_chain, attrs in nav.iter_entity_instances("samples > bacteria"):
    sample_id, otu_id = id_chain
    print(sample_id, otu_id, attrs)

# Read a single property (searches ancestors if inherited)
value = nav.get_property("samples > bacteria", ("A1_1", "OTU_1"), "abundance")

# Write a property
nav.set_property("samples > bacteria", ("A1_1", "OTU_1"), "rel_abundance", 0.4)

# Batch-collect all values for a property
values = nav.collect_property_values("samples > bacteria", "abundance")
# {"('A1_1', 'OTU_1')": 40, "('A1_1', 'OTU_2')": 60, ...}
```

### SkillRegistry

```python
from layercraft.core.registry import SkillRegistry

registry = SkillRegistry().load_builtins()

# List registered operations
print(registry.list_operations())  # ['normalize', 'aggregate', 'correlate']

# Register a custom skill
@registry.register("my_transform")
def my_transform(navigator, task_spec):
    ...

# Retrieve and call a skill
skill_fn = registry.get("normalize")
skill_fn(navigator, task_spec)
```

### TaskExecutor

```python
from layercraft.core.executor import TaskExecutor

executor = TaskExecutor()   # auto-loads built-in skills

# Run a single task
executor.execute(navigator, task_spec)

# Run a pipeline of tasks in sequence
executor.execute_pipeline(navigator, [task1, task2, task3])
```

### Built-in Skills

#### normalize

Normalise numeric property values across entity instances.

```python
task = {
    "task_id": "norm_001",
    "target_entity": "samples > bacteria",
    "target_property": "abundance",
    "operation": "normalize",
    "operation_params": {"method": "sum_to_one"},  # or "min_max" / "z_score"
    "scope": "per_group",                           # or "global"
    "output_property": "norm_abundance",
}
```

| Method | Description |
|--------|-------------|
| `sum_to_one` | Divide each value by the group sum → proportions (0–1) |
| `min_max` | Scale to the [0, 1] range |
| `z_score` | Standardise to zero mean and unit variance |

#### aggregate

Collapse child-entity properties up to parent instances.

```python
task = {
    "task_id": "agg_001",
    "target_entity": "samples > bacteria",
    "target_property": "abundance",
    "operation": "aggregate",
    "operation_params": {"function": "sum"},  # sum | mean | count | min | max | median
    "output_property": "total_abundance",
}
```

Results are written onto the **parent** entity (`samples` in this example).

#### correlate

Compute pairwise correlation between two properties across entity instances.

```python
task = {
    "task_id": "corr_001",
    "target_entity": "samples > bacteria",
    "target_property": "abundance",
    "source_property": "rel_abundance",
    "operation": "correlate",
    "operation_params": {"method": "pearson"},  # pearson | spearman | kendall
    "output_property": "abundance_correlation",
}
```

### Natural Language Parsing

`IntentParser` turns plain-text descriptions into task specs. A heuristic fallback is used when no LLM client is configured.

```python
from layercraft.llm.intent_parser import IntentParser

parser = IntentParser(entities_meta)

# English
task = parser.parse("Normalize bacteria abundance by sample using sum_to_one")

# Chinese
task = parser.parse("对样品中细菌的丰度进行归一化处理")

# With an LLM client (must expose a .chat() or similar interface)
parser_llm = IntentParser(entities_meta, llm_client=my_llm_client)
task = parser_llm.parse("Calculate the z-score of OTU abundance globally")
```

### Auto Skill Generation

When the executor encounters an unregistered operation it calls `generate_skill`, which first checks a template alias map and then optionally queries an LLM.

```python
from layercraft.auto.skill_generator import generate_skill

# Template aliases (e.g. "norm" → uses the built-in normalize skill)
skill_fn = generate_skill("norm", task_spec)

# LLM-generated skill (requires LLM client in environment)
skill_fn = generate_skill("custom_transform", task_spec)
```

### CLI – analyze_structure.py

Inspect any hierarchical JSON file and produce entity metadata:

```bash
python analyze_structure.py \
    --input  data.json \
    --output-json entities_meta.json \
    --print-report
```

Example output:

```
=== Hierarchical JSON Structure Analysis ===
Entity types : 2
Warnings     : 0

[Entity] samples
  Parent             : None
  Levels             : collection=1, id=2, attributes=3
  Instances observed : 2
  ID examples        : A1_1, A1_2
  Own properties     : day:int
  Child entities     : bacteria

[Entity] samples > bacteria
  Parent             : samples
  Levels             : collection=3, id=4, attributes=5
  Instances observed : 4
  ID examples        : OTU_1, OTU_2
  Own properties     : abundance:int
  Inherited props    : day
```

The exported `entities_meta.json` can be loaded directly and passed to `DataNavigator`.

---

## Running Tests

```bash
# Run the full test suite
pytest

# Run with coverage report
pytest --cov=layercraft

# Run a specific test class
pytest tests/test_core_engine.py::TestDataNavigator -v
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install development dependencies: `pip install -e ".[dev]"`.
3. Make your changes and add tests under `tests/`.
4. Run `pytest` to confirm all tests pass.
5. Open a pull request with a clear description of the change.

---

## License

This project is released under the [MIT License](LICENSE).
