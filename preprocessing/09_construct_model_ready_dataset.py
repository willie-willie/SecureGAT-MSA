#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 02:03:50 2026

@author: Willie
"""


"""
===============================================================================
SecureGAT-Agent: Model-Ready Dataset Construction and Final Pre-Training Audit
===============================================================================

Purpose
-------
Construct immutable model-ready split manifests contract-level split.

===============================================================================
"""


import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(
    "datasets/annotated_graph_dataset"
)

SPLIT_CSV = (
    ROOT /
    "contract_level_splits.csv"
)

OUTPUT_ROOT = (
    ROOT /
    "model_ready"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT /
    "model_ready_manifest.csv"
)

OUTPUT_REPORT_JSON = (
    OUTPUT_ROOT /
    "model_ready_report.json"
)

OUTPUT_REPORT_TXT = (
    OUTPUT_ROOT /
    "model_ready_audit.txt"
)

EXPECTED_VULNERABILITIES = [
    "Overflow-Underflow",
    "Re-entrancy",
    "TOD",
    "Timestamp-Dependency",
    "Unchecked-Send",
    "Unhandled-Exceptions",
    "tx.origin",
]

EXPECTED_SPLITS = {
    "train": 245,
    "validation": 49,
    "test": 56,
}

EXPECTED_CONTRACTS = {
    "train": 35,
    "validation": 7,
    "test": 8,
}

REQUIRED_GRAPH_FIELDS = [
    "x",
    "edge_index",
    "injection_label",
    "injection_id",
    "injection_count",
    "num_injections",
    "num_matched_injections",
    "num_unmatched_injections",
]


# =============================================================================
# DISPLAY
# =============================================================================

def banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# LOAD FROZEN SPLIT
# =============================================================================

def load_split():

    if not SPLIT_CSV.exists():

        raise FileNotFoundError(
            f"Frozen Step 08 split not found:\n"
            f"{SPLIT_CSV}"
        )

    records = []

    with SPLIT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        required_columns = {
            "split",
            "vulnerability",
            "contract_id",
            "graph",
            "graph_key",
            "graph_path",
        }

        missing = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing:

            raise RuntimeError(
                "Step 08 split is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            row["contract_id"] = int(
                row["contract_id"]
            )

            records.append(row)

    return records


# =============================================================================
# GRAPH LOADING
# =============================================================================

def load_graph(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Graph does not exist: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
    )


# =============================================================================
# ATTRIBUTE CHECK
# =============================================================================

def has_field(graph, field: str) -> bool:

    return hasattr(
        graph,
        field,
    )


# =============================================================================
# GRAPH VALIDATION
# =============================================================================

def validate_graph(
    graph,
    record,
):

    errors = []

    vulnerability = (
        record["vulnerability"]
    )

    contract_id = (
        record["contract_id"]
    )

    graph_name = (
        record["graph"]
    )

    # -------------------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------------------

    for field in REQUIRED_GRAPH_FIELDS:

        if not has_field(
            graph,
            field,
        ):

            errors.append(
                f"missing_field:{field}"
            )

    if errors:
        return errors

    # -------------------------------------------------------------------------
    # Node count
    # -------------------------------------------------------------------------

    try:

        num_nodes = int(
            graph.x.shape[0]
        )

    except Exception:

        errors.append(
            "invalid_x_shape"
        )

        return errors

    # -------------------------------------------------------------------------
    # Edge count
    # -------------------------------------------------------------------------

    try:

        edge_index = (
            graph.edge_index
        )

        if edge_index.dim() != 2:

            errors.append(
                "invalid_edge_index_rank"
            )

        elif edge_index.shape[0] != 2:

            errors.append(
                "invalid_edge_index_shape"
            )

        num_edges = int(
            edge_index.shape[1]
        )

    except Exception:

        errors.append(
            "invalid_edge_index"
        )

        num_edges = -1

    # -------------------------------------------------------------------------
    # Annotation lengths
    # -------------------------------------------------------------------------

    annotations = [
        (
            "injection_label",
            graph.injection_label,
        ),
        (
            "injection_id",
            graph.injection_id,
        ),
        (
            "injection_count",
            graph.injection_count,
        ),
    ]

    for name, tensor in annotations:

        try:

            length = len(tensor)

        except Exception:

            errors.append(
                f"invalid_annotation:{name}"
            )

            continue

        if length != num_nodes:

            errors.append(
                f"{name}_length:{length}!={num_nodes}"
            )

    # -------------------------------------------------------------------------
    # Label values
    # -------------------------------------------------------------------------

    try:

        labels = (
            graph.injection_label
            .detach()
            .cpu()
            .tolist()
        )

        invalid_labels = [
            value
            for value in labels
            if value not in (0, 1)
        ]

        if invalid_labels:

            errors.append(
                "invalid_label_values"
            )

    except Exception:

        errors.append(
            "invalid_label_tensor"
        )

    # -------------------------------------------------------------------------
    # Injection counts
    # -------------------------------------------------------------------------

    try:

        counts = (
            graph.injection_count
            .detach()
            .cpu()
            .tolist()
        )

        if any(
            value < 0
            for value in counts
        ):

            errors.append(
                "negative_injection_count"
            )

    except Exception:

        errors.append(
            "invalid_injection_count_tensor"
        )

    # -------------------------------------------------------------------------
    # Metadata consistency
    # -------------------------------------------------------------------------

    try:

        num_injections = int(
            graph.num_injections
        )

        matched = int(
            graph.num_matched_injections
        )

        unmatched = int(
            graph.num_unmatched_injections
        )

        if (
            matched + unmatched
            != num_injections
        ):

            errors.append(
                "injection_metadata_mismatch"
            )

        if unmatched != 0:

            errors.append(
                "unmatched_injections"
            )

    except Exception:

        errors.append(
            "invalid_injection_metadata"
        )

    # -------------------------------------------------------------------------
    # Metadata vulnerability
    # -------------------------------------------------------------------------

    if hasattr(
        graph,
        "vulnerability",
    ):

        metadata_vulnerability = (
            graph.vulnerability
        )

        if (
            metadata_vulnerability
            != vulnerability
        ):

            errors.append(
                "vulnerability_mismatch"
            )

    # -------------------------------------------------------------------------
    # Metadata contract ID
    # -------------------------------------------------------------------------

    if hasattr(
        graph,
        "contract_id",
    ):

        metadata_contract_id = int(
            graph.contract_id
        )

        if (
            metadata_contract_id
            != contract_id
        ):

            errors.append(
                "contract_id_mismatch"
            )

    # -------------------------------------------------------------------------
    # Positive node consistency
    # -------------------------------------------------------------------------

    try:

        positive_count = sum(
            value == 1
            for value in labels
        )

        positive_from_count = sum(
            value > 0
            for value in counts
        )

        if (
            positive_count
            != positive_from_count
        ):

            errors.append(
                "label_count_positive_mismatch"
            )

    except Exception:

        pass

    # -------------------------------------------------------------------------
    # Graph must contain nodes
    # -------------------------------------------------------------------------

    if num_nodes <= 0:

        errors.append(
            "empty_graph"
        )

    if num_edges < 0:

        errors.append(
            "invalid_edge_count"
        )

    return errors


# =============================================================================
# WRITE MANIFEST
# =============================================================================

def write_manifest(
    records,
    validation_results,
):

    fields = [
        "split",
        "vulnerability",
        "contract_id",
        "graph",
        "graph_key",
        "graph_path",
        "num_nodes",
        "num_edges",
        "num_injections",
        "num_matched_injections",
        "num_unmatched_injections",
        "positive_nodes",
        "negative_nodes",
        "positive_ratio",
        "status",
    ]

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:

            key = (
                record["graph_key"]
            )

            result = (
                validation_results[key]
            )

            writer.writerow(
                {
                    "split":
                        record["split"],

                    "vulnerability":
                        record["vulnerability"],

                    "contract_id":
                        record["contract_id"],

                    "graph":
                        record["graph"],

                    "graph_key":
                        record["graph_key"],

                    "graph_path":
                        record["graph_path"],

                    "num_nodes":
                        result["num_nodes"],

                    "num_edges":
                        result["num_edges"],

                    "num_injections":
                        result["num_injections"],

                    "num_matched_injections":
                        result[
                            "num_matched_injections"
                        ],

                    "num_unmatched_injections":
                        result[
                            "num_unmatched_injections"
                        ],

                    "positive_nodes":
                        result["positive_nodes"],

                    "negative_nodes":
                        result["negative_nodes"],

                    "positive_ratio":
                        result["positive_ratio"],

                    "status":
                        result["status"],
                }
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "SecureGAT-Agent Step 09\n"
        "Model-Ready Dataset Construction "
        "and Final Pre-Training Audit"
    )

    print()

    print(
        "Execution mode: READ-ONLY GRAPH VALIDATION"
    )

    print(
        "Step 08 split is the source of truth."
    )

    print(
        "Graph annotations will NOT be modified."
    )

    print()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD SPLIT
    # =========================================================================

    records = load_split()

    print(
        f"Frozen split records: {len(records)}"
    )

    if len(records) != 350:

        raise RuntimeError(
            "Expected exactly 350 split records."
        )

    # =========================================================================
    # SPLIT COUNTS
    # =========================================================================

    split_counts = Counter(
        record["split"]
        for record in records
    )

    contract_sets = defaultdict(set)

    for record in records:

        contract_sets[
            record["split"]
        ].add(
            record["contract_id"]
        )

    print()

    print(
        "Frozen split:"
    )

    for split in (
        "train",
        "validation",
        "test",
    ):

        print(
            f"  {split:<12}: "
            f"{split_counts[split]} graphs / "
            f"{len(contract_sets[split])} contracts"
        )

        if (
            split_counts[split]
            != EXPECTED_SPLITS[split]
        ):

            raise RuntimeError(
                f"{split} graph count mismatch."
            )

        if (
            len(contract_sets[split])
            != EXPECTED_CONTRACTS[split]
        ):

            raise RuntimeError(
                f"{split} contract count mismatch."
            )

    # =========================================================================
    # CONTRACT LEAKAGE
    # =========================================================================

    contract_to_splits = defaultdict(set)

    for record in records:

        contract_to_splits[
            record["contract_id"]
        ].add(
            record["split"]
        )

    leakage = {
        contract_id: sorted(splits)
        for contract_id, splits
        in contract_to_splits.items()
        if len(splits) > 1
    }

    if leakage:

        raise RuntimeError(
            f"Contract leakage detected: "
            f"{leakage}"
        )

    print()
    print(
        "Contract leakage: 0"
    )

    # =========================================================================
    # GRAPH IDENTITY
    # =========================================================================

    graph_keys = [
        record["graph_key"]
        for record in records
    ]

    duplicate_graph_keys = [
        key
        for key, count
        in Counter(graph_keys).items()
        if count > 1
    ]

    if duplicate_graph_keys:

        raise RuntimeError(
            "Duplicate graph keys detected: "
            f"{duplicate_graph_keys}"
        )

    print(
        "Duplicate graph records: 0"
    )

    # =========================================================================
    # VALIDATE GRAPHS
    # =========================================================================

    validation_results = {}

    total_nodes = 0
    total_edges = 0
    total_positive = 0
    total_negative = 0
    total_injections = 0
    total_matched = 0
    total_unmatched = 0

    errors = []

    print()
    print(
        "Validating graph annotations..."
    )

    for index, record in enumerate(
        records,
        start=1,
    ):

        path = Path(
            record["graph_path"]
        )

        try:

            graph = load_graph(
                path
            )

            graph_errors = (
                validate_graph(
                    graph,
                    record,
                )
            )

            num_nodes = int(
                graph.x.shape[0]
            )

            num_edges = int(
                graph.edge_index.shape[1]
            )

            labels = (
                graph.injection_label
                .detach()
                .cpu()
                .tolist()
            )

            positive_nodes = sum(
                value == 1
                for value in labels
            )

            negative_nodes = (
                num_nodes
                - positive_nodes
            )

            num_injections = int(
                graph.num_injections
            )

            matched = int(
                graph.num_matched_injections
            )

            unmatched = int(
                graph.num_unmatched_injections
            )

            positive_ratio = (
                positive_nodes / num_nodes
                if num_nodes
                else 0.0
            )

            status = (
                "PASS"
                if not graph_errors
                else "FAIL"
            )

            validation_results[
                record["graph_key"]
            ] = {

                "num_nodes":
                    num_nodes,

                "num_edges":
                    num_edges,

                "num_injections":
                    num_injections,

                "num_matched_injections":
                    matched,

                "num_unmatched_injections":
                    unmatched,

                "positive_nodes":
                    positive_nodes,

                "negative_nodes":
                    negative_nodes,

                "positive_ratio":
                    positive_ratio,

                "status":
                    status,

                "errors":
                    graph_errors,
            }

            if graph_errors:

                errors.append(
                    {
                        "graph_key":
                            record[
                                "graph_key"
                            ],

                        "errors":
                            graph_errors,
                    }
                )

            total_nodes += (
                num_nodes
            )

            total_edges += (
                num_edges
            )

            total_positive += (
                positive_nodes
            )

            total_negative += (
                negative_nodes
            )

            total_injections += (
                num_injections
            )

            total_matched += (
                matched
            )

            total_unmatched += (
                unmatched
            )

        except Exception as exc:

            errors.append(
                {
                    "graph_key":
                        record["graph_key"],

                    "errors": [
                        f"load_error:{exc}"
                    ],
                }
            )

            validation_results[
                record["graph_key"]
            ] = {

                "num_nodes": 0,
                "num_edges": 0,
                "num_injections": 0,
                "num_matched_injections": 0,
                "num_unmatched_injections": 0,
                "positive_nodes": 0,
                "negative_nodes": 0,
                "positive_ratio": 0.0,
                "status": "FAIL",
                "errors": [
                    f"load_error:{exc}"
                ],
            }

        if (
            index % 50 == 0
            or index == len(records)
        ):

            print(
                f"  Validated "
                f"{index}/{len(records)}"
            )

    # =========================================================================
    # WRITE MANIFEST
    # =========================================================================

    write_manifest(
        records,
        validation_results,
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    passed_graphs = sum(
        result["status"] == "PASS"
        for result
        in validation_results.values()
    )

    failed_graphs = (
        len(records)
        - passed_graphs
    )

    total_ratio = (
        total_positive / total_nodes
        if total_nodes
        else 0.0
    )

    report = {

        "dataset":
            "SolidiFI-benchmark",

        "step":
            "09_model_ready_dataset_construction",

        "read_only":
            True,

        "source_split":
            str(SPLIT_CSV),

        "graph_records":
            len(records),

        "contracts":
            len(contract_to_splits),

        "split_counts":
            dict(split_counts),

        "contract_counts": {
            split: len(
                contract_sets[split]
            )
            for split in (
                "train",
                "validation",
                "test",
            )
        },

        "contract_leakage":
            len(leakage),

        "duplicate_graph_keys":
            len(duplicate_graph_keys),

        "graphs_passed":
            passed_graphs,

        "graphs_failed":
            failed_graphs,

        "total_nodes":
            total_nodes,

        "total_edges":
            total_edges,

        "total_positive_nodes":
            total_positive,

        "total_negative_nodes":
            total_negative,

        "positive_node_ratio":
            total_ratio,

        "total_injections":
            total_injections,

        "total_matched_injections":
            total_matched,

        "total_unmatched_injections":
            total_unmatched,

        "errors":
            errors,

        "status":
            (
                "PASS"
                if (
                    not errors
                    and not leakage
                    and not duplicate_graph_keys
                )
                else "REVIEW REQUIRED"
            ),
    }

    with OUTPUT_REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    # =========================================================================
    # TEXT REPORT
    # =========================================================================

    lines = []

    lines.append("=" * 78)
    lines.append(
        "SecureGAT-Agent Step 09"
    )
    lines.append(
        "Model-Ready Dataset Construction "
        "and Final Pre-Training Audit"
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        "READ-ONLY"
    )
    lines.append(
        "Step 08 contract-level split preserved."
    )
    lines.append(
        "Graph annotations were not modified."
    )
    lines.append("")

    lines.append(
        "DATASET"
    )
    lines.append("-" * 78)

    lines.append(
        f"Graph records: {len(records)}"
    )

    lines.append(
        f"Contracts: {len(contract_to_splits)}"
    )

    lines.append(
        f"Train graphs: "
        f"{split_counts['train']}"
    )

    lines.append(
        f"Validation graphs: "
        f"{split_counts['validation']}"
    )

    lines.append(
        f"Test graphs: "
        f"{split_counts['test']}"
    )

    lines.append("")

    lines.append(
        "ANNOTATION AUDIT"
    )
    lines.append("-" * 78)

    lines.append(
        f"Graphs passed: {passed_graphs}"
    )

    lines.append(
        f"Graphs failed: {failed_graphs}"
    )

    lines.append(
        f"Total injections: "
        f"{total_injections}"
    )

    lines.append(
        f"Matched injections: "
        f"{total_matched}"
    )

    lines.append(
        f"Unmatched injections: "
        f"{total_unmatched}"
    )

    lines.append("")

    lines.append(
        "GRAPH STATISTICS"
    )
    lines.append("-" * 78)

    lines.append(
        f"AST nodes: {total_nodes}"
    )

    lines.append(
        f"AST edges: {total_edges}"
    )

    lines.append(
        f"Positive nodes: {total_positive}"
    )

    lines.append(
        f"Negative nodes: {total_negative}"
    )

    lines.append(
        f"Positive-node ratio: "
        f"{total_ratio:.6f}"
    )

    lines.append("")

    lines.append(
        "LEAKAGE"
    )
    lines.append("-" * 78)

    lines.append(
        f"Contract leakage: "
        f"{len(leakage)}"
    )

    lines.append(
        f"Duplicate graph keys: "
        f"{len(duplicate_graph_keys)}"
    )

    lines.append("")

    if report["status"] == "PASS":

        lines.append(
            "STATUS: PASS"
        )

    else:

        lines.append(
            "STATUS: REVIEW REQUIRED"
        )

        lines.append("")

        for item in errors:

            lines.append(
                f"{item['graph_key']}: "
                f"{item['errors']}"
            )

    lines.append("")
    lines.append("=" * 78)

    with OUTPUT_REPORT_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(lines)
        )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print()
    banner(
        "STEP 09 SUMMARY"
    )

    print(
        f"Graph records: {len(records)}"
    )

    print(
        f"Contracts: {len(contract_to_splits)}"
    )

    print(
        f"Train graphs: "
        f"{split_counts['train']}"
    )

    print(
        f"Validation graphs: "
        f"{split_counts['validation']}"
    )

    print(
        f"Test graphs: "
        f"{split_counts['test']}"
    )

    print()

    print(
        f"Graphs passed: {passed_graphs}"
    )

    print(
        f"Graphs failed: {failed_graphs}"
    )

    print(
        f"Total injections: "
        f"{total_injections}"
    )

    print(
        f"Matched injections: "
        f"{total_matched}"
    )

    print(
        f"Unmatched injections: "
        f"{total_unmatched}"
    )

    print()

    print(
        f"AST nodes: {total_nodes}"
    )

    print(
        f"AST edges: {total_edges}"
    )

    print(
        f"Positive nodes: {total_positive}"
    )

    print(
        f"Negative nodes: {total_negative}"
    )

    print(
        f"Positive-node ratio: "
        f"{total_ratio:.6f}"
    )

    print()

    print(
        f"Contract leakage: "
        f"{len(leakage)}"
    )

    print(
        f"Duplicate graph keys: "
        f"{len(duplicate_graph_keys)}"
    )

    print(
        f"Audit errors: "
        f"{len(errors)}"
    )

    print()

    print(
        f"Manifest: {OUTPUT_MANIFEST}"
    )

    print(
        f"JSON report: {OUTPUT_REPORT_JSON}"
    )

    print(
        f"Text report: {OUTPUT_REPORT_TXT}"
    )

    print()

    if report["status"] == "PASS":

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: REVIEW REQUIRED"
        )

    print("=" * 78)

    if report["status"] != "PASS":

        raise RuntimeError(
            "Step 09 failed model-ready dataset audit."
        )

    print(
        "STEP 09 COMPLETE"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()