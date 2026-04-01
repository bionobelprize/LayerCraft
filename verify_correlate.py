#!/usr/bin/env python3
"""Independent verification of bacteria_fzs_spearman computed by correlate_task.

Ground-truth logic
------------------
For each OTU (bacteria):
  1. Iterate all samples.
  2. Collect (normalized_abundance from OTU node, 腐殖酸含量 from sample node).
  3. Skip samples where the OTU is absent or either value is missing/non-numeric.
  4. Compute Spearman rank correlation across those pairs.

This is then compared against the `bacteria_fzs_spearman` attribute written into
every sample's OTU node in outputs/e2.json by the pipeline.

Usage
-----
    python verify_correlate.py                        # default paths
    python verify_correlate.py --rows 30              # show top 30 OTUs
    python verify_correlate.py --mismatch-only        # only show differing rows
    python verify_correlate.py --raw test_data/raw.json --enriched outputs/e2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pure-Python Spearman (mirrors LayerCraft's _ranks + _pearson)
# ---------------------------------------------------------------------------

def _ranks(values: List[float]) -> List[float]:
    """Average fractional ranks (handles ties)."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


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


def spearman(x: List[float], y: List[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ground-truth calculation from raw.json
# ---------------------------------------------------------------------------

def compute_ground_truth(
    raw: dict,
) -> Dict[str, Tuple[float, int]]:
    """Return {otu_id: (spearman_coeff, n_pairs)} computed from scratch."""
    samples: dict = raw.get("samples", {})

    # Collect pairs per OTU across all samples
    otu_pairs: Dict[str, Tuple[List[float], List[float]]] = {}

    for sample_attrs in samples.values():
        fzs = sample_attrs.get("腐殖酸含量")
        if fzs is None:
            continue
        try:
            fzs_f = float(fzs)
        except (TypeError, ValueError):
            continue

        bacteria: dict = sample_attrs.get("bacteria", {})
        for otu_id, otu_attrs in bacteria.items():
            norm_ab = otu_attrs.get("normalized_abundance")
            if norm_ab is None:
                continue
            try:
                norm_ab_f = float(norm_ab)
            except (TypeError, ValueError):
                continue

            if otu_id not in otu_pairs:
                otu_pairs[otu_id] = ([], [])
            otu_pairs[otu_id][0].append(norm_ab_f)
            otu_pairs[otu_id][1].append(fzs_f)

    results: Dict[str, Tuple[float, int]] = {}
    for otu_id, (x_vals, y_vals) in otu_pairs.items():
        n = len(x_vals)
        if n < 2:
            results[otu_id] = (float("nan"), n)
        else:
            results[otu_id] = (spearman(x_vals, y_vals), n)

    return results


# ---------------------------------------------------------------------------
# Extract pipeline results from e2.json
# ---------------------------------------------------------------------------

def extract_pipeline_results(enriched: dict) -> Dict[str, Optional[float]]:
    """Return {otu_id: bacteria_fzs_spearman} from the first sample that has
    the attribute.  All samples should carry the same value for the same OTU.
    """
    samples: dict = enriched.get("samples", {})
    otu_values: Dict[str, Optional[float]] = {}

    for sample_attrs in samples.values():
        bacteria: dict = sample_attrs.get("bacteria", {})
        for otu_id, otu_attrs in bacteria.items():
            if otu_id not in otu_values:
                val = otu_attrs.get("bacteria_fzs_spearman")
                otu_values[otu_id] = float(val) if val is not None else None

    return otu_values


def check_pipeline_consistency(enriched: dict) -> Dict[str, bool]:
    """Verify that bacteria_fzs_spearman is identical across all samples for
    every OTU.  Returns {otu_id: is_consistent}."""
    samples: dict = enriched.get("samples", {})
    seen: Dict[str, float] = {}
    consistent: Dict[str, bool] = {}

    for sample_attrs in samples.values():
        bacteria: dict = sample_attrs.get("bacteria", {})
        for otu_id, otu_attrs in bacteria.items():
            val = otu_attrs.get("bacteria_fzs_spearman")
            v = float(val) if val is not None else None
            if otu_id not in seen:
                seen[otu_id] = v
                consistent[otu_id] = True
            else:
                if seen[otu_id] != v:
                    consistent[otu_id] = False

    return consistent


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

COL_OTU   = 14
COL_N     = 5
COL_COEFF = 14
COL_MATCH = 7


def _hdr() -> str:
    return (
        f"{'OTU':<{COL_OTU}} "
        f"{'N':>{COL_N}} "
        f"{'Ground-Truth':>{COL_COEFF}} "
        f"{'Pipeline':>{COL_COEFF}} "
        f"{'|Δ|':>{COL_COEFF}} "
        f"{'Match':<{COL_MATCH}}"
    )


def _sep() -> str:
    return "-" * (COL_OTU + COL_N + COL_COEFF * 3 + COL_MATCH + 5)


def _row(
    otu_id: str,
    n: int,
    gt: float,
    pl: Optional[float],
    tol: float,
) -> str:
    gt_str = f"{gt:.8f}" if not math.isnan(gt) else "     n/a    "
    pl_str = f"{pl:.8f}" if pl is not None else "   missing  "

    if math.isnan(gt) or pl is None:
        delta_str = "     n/a    "
        match_str = "n/a"
    else:
        delta = abs(gt - pl)
        delta_str = f"{delta:.2e}"
        match_str = "OK" if delta <= tol else "DIFF"

    return (
        f"{otu_id:<{COL_OTU}} "
        f"{n:>{COL_N}} "
        f"{gt_str:>{COL_COEFF}} "
        f"{pl_str:>{COL_COEFF}} "
        f"{delta_str:>{COL_COEFF}} "
        f"{match_str:<{COL_MATCH}}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",      default="test_data/raw.json",
                        help="Path to raw input JSON")
    parser.add_argument("--enriched", default="outputs/e2.json",
                        help="Path to enriched output JSON (with bacteria_fzs_spearman)")
    parser.add_argument("--rows",     type=int, default=20,
                        help="Max OTU rows to display in the table (0 = all)")
    parser.add_argument("--mismatch-only", action="store_true",
                        help="Only display rows where ground-truth ≠ pipeline")
    parser.add_argument("--tol",      type=float, default=1e-9,
                        help="Float tolerance for match check (default 1e-9)")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    enriched_path = Path(args.enriched)

    if not raw_path.exists():
        sys.exit(f"ERROR: raw JSON not found: {raw_path}")
    if not enriched_path.exists():
        sys.exit(f"ERROR: enriched JSON not found: {enriched_path}")

    print(f"Loading {raw_path} ...", flush=True)
    raw = load_json(raw_path)
    print(f"Loading {enriched_path} ...", flush=True)
    enriched = load_json(enriched_path)

    print("\nComputing ground-truth Spearman correlations ...", flush=True)
    gt_results = compute_ground_truth(raw)

    print("Extracting pipeline results from enriched JSON ...", flush=True)
    pl_results = extract_pipeline_results(enriched)

    print("Checking cross-sample consistency of pipeline values ...", flush=True)
    consistency = check_pipeline_consistency(enriched)
    inconsistent_otus = [oid for oid, ok in consistency.items() if not ok]
    if inconsistent_otus:
        print(f"  WARNING: {len(inconsistent_otus)} OTU(s) have INCONSISTENT "
              f"bacteria_fzs_spearman across samples:")
        for oid in inconsistent_otus[:10]:
            print(f"    {oid}")
        if len(inconsistent_otus) > 10:
            print(f"    ... and {len(inconsistent_otus) - 10} more")
    else:
        print("  OK – bacteria_fzs_spearman is identical across all samples for every OTU.")

    # ------------------------------------------------------------------
    # Build comparison table
    # ------------------------------------------------------------------
    all_otus = sorted(set(gt_results) | set(pl_results))
    tol = args.tol

    n_ok      = 0
    n_diff    = 0
    n_na      = 0
    n_missing = 0

    rows_to_show: list = []
    for otu_id in all_otus:
        gt_coeff, gt_n = gt_results.get(otu_id, (float("nan"), 0))
        pl_val         = pl_results.get(otu_id)

        if math.isnan(gt_coeff) or pl_val is None:
            n_na += 1 if math.isnan(gt_coeff) else 0
            n_missing += 1 if pl_val is None else 0
            status = "na_or_missing"
        else:
            delta = abs(gt_coeff - pl_val)
            if delta <= tol:
                n_ok += 1
                status = "ok"
            else:
                n_diff += 1
                status = "diff"

        rows_to_show.append((otu_id, gt_n, gt_coeff, pl_val, status))

    # ------------------------------------------------------------------
    # Print tables
    # ------------------------------------------------------------------
    print()
    print("=" * len(_sep()))
    print("COMPARISON TABLE  (Ground-Truth vs Pipeline)")
    print(f"  tolerance = {tol:.0e}")
    print("=" * len(_sep()))
    print(_hdr())
    print(_sep())

    limit = args.rows if args.rows > 0 else len(rows_to_show)
    shown = 0
    for otu_id, gt_n, gt_coeff, pl_val, status in rows_to_show:
        if args.mismatch_only and status not in ("diff",):
            continue
        if shown >= limit:
            break
        print(_row(otu_id, gt_n, gt_coeff, pl_val, tol))
        shown += 1

    if shown < len(rows_to_show) and not args.mismatch_only:
        remaining = len(rows_to_show) - shown
        print(f"  ... {remaining} more rows (use --rows 0 to show all)")

    print(_sep())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(all_otus)
    print()
    print("SUMMARY")
    print(f"  Total OTUs compared    : {total}")
    print(f"  Match (|Δ| ≤ {tol:.0e})    : {n_ok}  ({100*n_ok/max(total,1):.1f}%)")
    print(f"  Differ (|Δ| > {tol:.0e})   : {n_diff}")
    print(f"  Ground-truth n<2 (n/a) : {n_na}")
    print(f"  Missing in pipeline    : {n_missing}")

    if n_diff > 0:
        print()
        print("DIFFERING OTUs (all):")
        print(_hdr())
        print(_sep())
        for otu_id, gt_n, gt_coeff, pl_val, status in rows_to_show:
            if status == "diff":
                print(_row(otu_id, gt_n, gt_coeff, pl_val, tol))

    print()
    if n_diff == 0 and n_missing == 0:
        print("RESULT: PASS – pipeline results match ground-truth for all computable OTUs.")
    elif n_diff == 0:
        print(f"RESULT: PARTIAL – {n_missing} OTU(s) missing in pipeline output, "
              "but all present values match.")
    else:
        print(f"RESULT: FAIL – {n_diff} OTU(s) have differing values.")


if __name__ == "__main__":
    main()
