#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 00:36:09 2026

@author: Willie
"""


"""
SecureGAT-Agent: Dataset Integrity and Contract-Level Split Audit

READ-ONLY AUDIT

This script does not modify graph files. It validates the outputs and prepares contract-level split candidates.

Important annotation semantics:

    injection_label[node] == 1
        iff at least one injection covers the node.

    injection_count[node]
        number of injections covering the node.

    injection_id[node]
        representative/first injection ID, NOT a complete many-to-many
        injection membership map.

Therefore the valid label/count invariant is:

    label == 0 <=> count == 0
    label == 1 <=> count > 0

It is NOT valid to require:

    count == 1 whenever label == 1
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"
ANNOTATED_ROOT = DATASET_ROOT / "annotated_graph_dataset"

AUDIT_CSV = ANNOTATED_ROOT / "injection_audit.csv"
GRAPH_AUDIT_CSV = ANNOTATED_ROOT / "graph_integrity.csv"
SPLIT_CANDIDATES_CSV = ANNOTATED_ROOT / "split_candidates.csv"
REPORT_JSON = ANNOTATED_ROOT / "dataset_integrity_report.json"
REPORT_TXT = ANNOTATED_ROOT / "dataset_integrity_audit.txt"

EXPECTED_ANNOTATION_VERSION = "step06_v2_clean_ast"
EXPECTED_ANNOTATION_STRATEGY = "contained"

GRAPH_EXTENSIONS = {".pt"}

WIDTH = 78


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def banner(title: str) -> None:
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def section(title: str) -> None:
    print("-" * WIDTH)
    print(title)
    print("-" * WIDTH)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_graph_files() -> List[Path]:
    """
    Discover annotated graph files.

    Expected structure:

        datasets/
            annotated_graph_dataset/
                Overflow-Underflow/
                    buggy_1.pt
                    ...
                Re-entrancy/
                    buggy_1.pt
                    ...
                ...
    """

    if not ANNOTATED_ROOT.exists():
        raise FileNotFoundError(
            f"Annotated graph directory does not exist: {ANNOTATED_ROOT}"
        )

    files: List[Path] = []

    for vulnerability_dir in sorted(ANNOTATED_ROOT.iterdir()):

        if not vulnerability_dir.is_dir():
            continue

        for path in sorted(vulnerability_dir.iterdir()):

            if (
                path.is_file()
                and path.suffix.lower() in GRAPH_EXTENSIONS
            ):
                files.append(path)

    return files


# ============================================================================
# INJECTION AUDIT
# ============================================================================

REQUIRED_AUDIT_FIELDS = {
    "vulnerability",
    "contract_id",
    "graph",
    "injection_id",
    "matched",
    "matched_node_count",
}


def load_injection_audit() -> Tuple[
    List[Dict[str, str]],
    List[str],
]:

    if not AUDIT_CSV.exists():
        raise FileNotFoundError(
            f"Injection audit not found: {AUDIT_CSV}"
        )

    with AUDIT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        fieldnames = reader.fieldnames or []

        missing = sorted(
            REQUIRED_AUDIT_FIELDS
            - set(fieldnames)
        )

        if missing:
            raise ValueError(
                "Injection audit is missing required fields: "
                + ", ".join(missing)
            )

        rows = [
            dict(row)
            for row in reader
        ]

    return rows, fieldnames


def audit_key(
    row: Dict[str, str],
) -> Tuple[str, str, str]:

    return (
        str(row.get("vulnerability", "")),
        str(row.get("contract_id", "")),
        str(row.get("graph", "")),
    )


def audit_record_key(
    row: Dict[str, str],
) -> Tuple[str, str, str, str, str]:

    return (
        str(row.get("vulnerability", "")),
        str(row.get("contract_id", "")),
        str(row.get("graph", "")),
        str(row.get("injection_id", "")),
        str(row.get("buglog_row", "")),
    )


def build_audit_index(
    rows: Iterable[Dict[str, str]],
) -> Dict[
    Tuple[str, str, str],
    List[Dict[str, str]],
]:

    index: Dict[
        Tuple[str, str, str],
        List[Dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        index[audit_key(row)].append(row)

    return index


# ============================================================================
# GRAPH LOADING
# ============================================================================

def load_graph(path: Path) -> Any:
    """
    Load a PyTorch/PyG graph on CPU.
    """

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


def get_attr(
    graph: Any,
    name: str,
    default: Any = None,
) -> Any:

    if hasattr(graph, name):
        return getattr(graph, name)

    if isinstance(graph, dict):
        return graph.get(
            name,
            default,
        )

    return default


def tensor_to_int_list(
    value: Any,
) -> List[int]:

    if value is None:
        return []

    if isinstance(value, torch.Tensor):

        return [
            int(x)
            for x in value.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        ]

    if isinstance(value, (list, tuple)):

        return [
            safe_int(x)
            for x in value
        ]

    return []


def graph_num_nodes(
    graph: Any,
    labels: List[int],
) -> int:

    value = get_attr(
        graph,
        "num_nodes",
        None,
    )

    if value is not None:

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            pass

    x = get_attr(
        graph,
        "x",
        None,
    )

    if (
        isinstance(x, torch.Tensor)
        and x.ndim >= 1
    ):

        return int(x.shape[0])

    return len(labels)


def graph_num_edges(
    graph: Any,
) -> int:

    value = get_attr(
        graph,
        "num_edges",
        None,
    )

    if value is not None:

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            pass

    edge_index = get_attr(
        graph,
        "edge_index",
        None,
    )

    if isinstance(
        edge_index,
        torch.Tensor,
    ):

        if edge_index.ndim == 2:
            return int(
                edge_index.shape[1]
            )

        return int(
            edge_index.numel()
        )

    return 0


# ============================================================================
# GRAPH-LEVEL AUDIT
# ============================================================================

def graph_audit(
    graph_path: Path,
    audit_rows: List[Dict[str, str]],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:

    graph = load_graph(graph_path)

    graph_name = graph_path.name

    directory_vulnerability = (
        graph_path.parent.name
    )

    metadata_vulnerability = str(
        get_attr(
            graph,
            "vulnerability",
            directory_vulnerability,
        )
    )

    contract_id_raw = get_attr(
        graph,
        "contract_id",
        None,
    )

    contract_id = (
        ""
        if contract_id_raw is None
        else str(contract_id_raw)
    )

    # ------------------------------------------------------------------
    # Annotation tensors
    # ------------------------------------------------------------------

    labels = tensor_to_int_list(
        get_attr(
            graph,
            "injection_label",
            None,
        )
    )

    injection_ids = tensor_to_int_list(
        get_attr(
            graph,
            "injection_id",
            None,
        )
    )

    injection_counts = tensor_to_int_list(
        get_attr(
            graph,
            "injection_count",
            None,
        )
    )

    # ------------------------------------------------------------------
    # Graph dimensions
    # ------------------------------------------------------------------

    num_nodes = graph_num_nodes(
        graph,
        labels,
    )

    num_edges = graph_num_edges(
        graph,
    )

    metadata_num_nodes = safe_int(
        get_attr(
            graph,
            "num_nodes_total",
            num_nodes,
        ),
        num_nodes,
    )

    metadata_num_edges = safe_int(
        get_attr(
            graph,
            "num_edges_total",
            num_edges,
        ),
        num_edges,
    )

    # ------------------------------------------------------------------
    # Label distribution
    # ------------------------------------------------------------------

    positive_nodes = sum(
        1
        for x in labels
        if x == 1
    )

    negative_nodes = sum(
        1
        for x in labels
        if x == 0
    )

    positive_ratio = (
        positive_nodes / num_nodes
        if num_nodes > 0
        else 0.0
    )

    invalid_label_count = sum(
        1
        for x in labels
        if x not in (0, 1)
    )

    # ------------------------------------------------------------------
    # Tensor length validation
    # ------------------------------------------------------------------

    label_length_error = int(
        len(labels) != num_nodes
    )

    injection_id_length_error = int(
        len(injection_ids) != num_nodes
    )

    injection_count_length_error = int(
        len(injection_counts) != num_nodes
    )

    # ==================================================================
    # LABEL <-> INJECTION ID
    # ==================================================================

    label_injection_id_mismatch = 0

    negative_with_injection_id = 0

    positive_without_injection_id = 0

    if (
        len(labels) == num_nodes
        and
        len(injection_ids) == num_nodes
    ):

        for (
            label,
            injection_id,
        ) in zip(
            labels,
            injection_ids,
        ):

            if label in (0, 1):

                expected_label = (
                    1
                    if injection_id >= 0
                    else 0
                )

                if label != expected_label:
                    label_injection_id_mismatch += 1

            if (
                label == 0
                and
                injection_id >= 0
            ):
                negative_with_injection_id += 1

            if (
                label == 1
                and
                injection_id < 0
            ):
                positive_without_injection_id += 1

    # ==================================================================
    # LABEL <-> INJECTION COUNT
    # ==================================================================
    #
    # CRITICAL:
    #
    # injection_count is NOT binary.
    #
    # Multiple injections may cover the same AST node.
    #
    # Valid:
    #
    #     label = 0, count = 0
    #     label = 1, count = 1
    #     label = 1, count = 2
    #     label = 1, count = 3
    #
    # Invalid:
    #
    #     label = 0, count > 0
    #     label = 1, count = 0
    #
    # ==================================================================

    label_injection_count_mismatch = 0

    negative_with_positive_count = 0

    positive_with_zero_count = 0

    negative_injection_count = 0

    if (
        len(labels) == num_nodes
        and
        len(injection_counts) == num_nodes
    ):

        for (
            label,
            count,
        ) in zip(
            labels,
            injection_counts,
        ):

            if count < 0:
                negative_injection_count += 1

            expected_label = (
                1
                if count > 0
                else 0
            )

            if label != expected_label:

                label_injection_count_mismatch += 1

            if (
                label == 0
                and
                count > 0
            ):

                negative_with_positive_count += 1

            if (
                label == 1
                and
                count == 0
            ):

                positive_with_zero_count += 1

    # ==================================================================
    # GRAPH INJECTION METADATA
    # ==================================================================

    graph_num_injections = safe_int(
        get_attr(
            graph,
            "num_injections",
            -1,
        ),
        -1,
    )

    graph_num_matched = safe_int(
        get_attr(
            graph,
            "num_matched_injections",
            -1,
        ),
        -1,
    )

    graph_num_unmatched = safe_int(
        get_attr(
            graph,
            "num_unmatched_injections",
            -1,
        ),
        -1,
    )

    audit_records = len(
        audit_rows
    )

    audit_matched_records = sum(
        1
        for row in audit_rows
        if str(
            row.get(
                "matched",
                "",
            )
        ).strip().lower()
        == "true"
    )

    audit_unmatched_records = (
        audit_records
        - audit_matched_records
    )

    # ==================================================================
    # INJECTION ID DIAGNOSTICS
    # ==================================================================
    #
    # IMPORTANT:
    #
    # graph.injection_id stores only the first injection ID assigned
    # to a node.
    #
    # Therefore:
    #
    # audit ID absent from graph.injection_id
    #
    # does NOT necessarily mean an orphan injection.
    #
    # This is informational only.
    #
    # ==================================================================

    audit_injection_ids = [

        safe_int(
            row.get(
                "injection_id",
                "-1",
            ),
            -1,
        )

        for row in audit_rows
    ]

    graph_injection_ids = {

        x
        for x in injection_ids
        if x >= 0

    }

    audit_unique_injection_ids = set(
        audit_injection_ids
    )

    audit_ids_missing_in_graph = len(
        audit_unique_injection_ids
        -
        graph_injection_ids
    )

    graph_ids_missing_in_audit = len(
        graph_injection_ids
        -
        audit_unique_injection_ids
    )

    graph_unique_injection_ids = len(
        graph_injection_ids
    )

    duplicate_audit_records = (
        audit_records
        -
        len(
            {
                audit_record_key(row)
                for row in audit_rows
            }
        )
    )

    # ==================================================================
    # INJECTION-COUNT ACCOUNTING
    # ==================================================================
    #
    # This is the strong many-to-many accounting invariant:
    #
    #     sum(graph.injection_count)
    #
    # must equal
    #
    #     sum(audit.matched_node_count)
    #
    # for matched injections.
    #
    # ==================================================================

    audit_matched_node_sum = sum(

        safe_int(
            row.get(
                "matched_node_count",
                "0",
            ),
            0,
        )

        for row in audit_rows

        if str(
            row.get(
                "matched",
                "",
            )
        ).strip().lower()
        == "true"
    )

    graph_injection_count_sum = sum(
        injection_counts
    )

    injection_count_sum_mismatch = int(
        graph_injection_count_sum
        != audit_matched_node_sum
    )

    # ==================================================================
    # METADATA CONSISTENCY
    # ==================================================================

    injection_metadata_count_error = int(

        graph_num_injections >= 0

        and

        graph_num_injections
        != audit_records

    )

    matched_metadata_error = int(

        graph_num_matched >= 0

        and

        graph_num_matched
        != audit_matched_records

    )

    unmatched_metadata_error = int(

        graph_num_unmatched >= 0

        and

        graph_num_unmatched
        != audit_unmatched_records

    )

    audit_injection_count_error = (
        injection_count_sum_mismatch
    )

    # ==================================================================
    # STRUCTURAL GRAPH VALIDATION
    # ==================================================================

    node_count_error = int(
        metadata_num_nodes
        != num_nodes
    )

    edge_count_error = int(
        metadata_num_edges
        != num_edges
    )

    edge_index = get_attr(
        graph,
        "edge_index",
        None,
    )

    if isinstance(
        edge_index,
        torch.Tensor,
    ):

        if not (
            edge_index.ndim == 2
            and
            edge_index.shape[0] == 2
            and
            edge_index.shape[1] == num_edges
        ):

            edge_count_error = 1

    empty_graph = int(
        num_nodes == 0
    )

    missing_contract_id = int(
        contract_id == ""
    )

    vulnerability_mismatch = int(
        metadata_vulnerability
        != directory_vulnerability
    )

    # ==================================================================
    # ANNOTATION METADATA
    # ==================================================================

    annotation_version = str(
        get_attr(
            graph,
            "annotation_version",
            "",
        )
    )

    annotation_strategy = str(
        get_attr(
            graph,
            "annotation_strategy",
            "",
        )
    )

    missing_annotation_fields = []

    required_annotation_fields = [

        "injection_label",
        "injection_id",
        "injection_count",
        "num_injections",
        "num_matched_injections",
        "num_unmatched_injections",

    ]

    for field in required_annotation_fields:

        if get_attr(
            graph,
            field,
            None,
        ) is None:

            missing_annotation_fields.append(
                field
            )

    # ==================================================================
    # ERROR CLASSIFICATION
    # ==================================================================
    #
    # audit_ids_missing_in_graph and graph_ids_missing_in_audit are NOT
    # integrity errors because injection_id is representative.
    #
    # ==================================================================

    error_flags = {

        "missing_annotation_fields":
            int(
                bool(
                    missing_annotation_fields
                )
            ),

        "label_length":
            label_length_error,

        "injection_id_length":
            injection_id_length_error,

        "injection_count_length":
            injection_count_length_error,

        "invalid_label_count":
            int(
                invalid_label_count > 0
            ),

        "label_injection_id_mismatch":
            int(
                label_injection_id_mismatch > 0
            ),

        "label_injection_count_mismatch":
            int(
                label_injection_count_mismatch > 0
            ),

        "negative_with_injection_id":
            int(
                negative_with_injection_id > 0
            ),

        "positive_without_injection_id":
            int(
                positive_without_injection_id > 0
            ),

        "negative_with_positive_count":
            int(
                negative_with_positive_count > 0
            ),

        "positive_with_zero_count":
            int(
                positive_with_zero_count > 0
            ),

        "negative_injection_count":
            int(
                negative_injection_count > 0
            ),

        "injection_metadata_count_error":
            injection_metadata_count_error,

        "matched_metadata_error":
            matched_metadata_error,

        "unmatched_metadata_error":
            unmatched_metadata_error,

        "audit_injection_count_error":
            audit_injection_count_error,

        "node_count_error":
            node_count_error,

        "edge_count_error":
            edge_count_error,

        "empty_graph":
            empty_graph,

        "missing_contract_id":
            missing_contract_id,

        "vulnerability_mismatch":
            vulnerability_mismatch,

        "duplicate_audit_records":
            int(
                duplicate_audit_records > 0
            ),
    }

    error_names = [

        name

        for name, flag
        in error_flags.items()

        if flag

    ]

    error_count = len(
        error_names
    )

    status = (
        "PASS"
        if error_count == 0
        else "REVIEW"
    )

    # ==================================================================
    # GRAPH AUDIT ROW
    # ==================================================================

    row: Dict[str, Any] = {

        "vulnerability":
            directory_vulnerability,

        "metadata_vulnerability":
            metadata_vulnerability,

        "contract_id":
            contract_id,

        "graph":
            graph_name,

        "num_nodes":
            num_nodes,

        "num_edges":
            num_edges,

        "num_nodes_metadata":
            metadata_num_nodes,

        "num_edges_metadata":
            metadata_num_edges,

        "positive_nodes":
            positive_nodes,

        "negative_nodes":
            negative_nodes,

        "positive_ratio":
            positive_ratio,

        "num_injections":
            graph_num_injections,

        "num_matched_injections":
            graph_num_matched,

        "num_unmatched_injections":
            graph_num_unmatched,

        "graph_unique_injection_ids":
            graph_unique_injection_ids,

        "audit_records":
            audit_records,

        "audit_unique_injection_ids":
            len(
                audit_unique_injection_ids
            ),

        "audit_matched_records":
            audit_matched_records,

        "audit_unmatched_records":
            audit_unmatched_records,

        "audit_ids_missing_in_graph":
            audit_ids_missing_in_graph,

        "graph_ids_missing_in_audit":
            graph_ids_missing_in_audit,

        "missing_annotation_fields":
            "|".join(
                missing_annotation_fields
            ),

        "label_length":
            len(labels),

        "injection_id_length":
            len(injection_ids),

        "injection_count_length":
            len(injection_counts),

        "invalid_label_count":
            invalid_label_count,

        "label_injection_id_mismatch":
            label_injection_id_mismatch,

        "label_injection_count_mismatch":
            label_injection_count_mismatch,

        "negative_with_injection_id":
            negative_with_injection_id,

        "positive_without_injection_id":
            positive_without_injection_id,

        "negative_with_positive_count":
            negative_with_positive_count,

        "positive_with_zero_count":
            positive_with_zero_count,

        "negative_injection_count":
            negative_injection_count,

        "injection_count_sum":
            graph_injection_count_sum,

        "audit_matched_node_sum":
            audit_matched_node_sum,

        "injection_count_sum_mismatch":
            injection_count_sum_mismatch,

        "injection_metadata_count_error":
            injection_metadata_count_error,

        "matched_metadata_error":
            matched_metadata_error,

        "unmatched_metadata_error":
            unmatched_metadata_error,

        "audit_injection_count_error":
            audit_injection_count_error,

        "node_count_error":
            node_count_error,

        "edge_count_error":
            edge_count_error,

        "empty_graph":
            empty_graph,

        "missing_contract_id":
            missing_contract_id,

        "vulnerability_mismatch":
            vulnerability_mismatch,

        "duplicate_audit_records":
            duplicate_audit_records,

        "annotation_version":
            annotation_version,

        "annotation_strategy":
            annotation_strategy,

        "error_count":
            error_count,

        "errors":
            "|".join(
                error_names
            ),

        "status":
            status,
    }

    details = {

        "graph":
            graph,

        "labels":
            labels,

        "injection_ids":
            injection_ids,

        "injection_counts":
            injection_counts,

        "audit_rows":
            audit_rows,

        "error_flags":
            error_flags,

    }

    return row, details


# ============================================================================
# SPLIT CANDIDATES
# ============================================================================

def write_split_candidates(
    rows: List[Dict[str, Any]],
) -> None:

    """
    Create contract-level split candidates.

    A candidate is uniquely identified by:

        vulnerability + contract_id

    This prevents AST graphs from the same contract identity from being
    assigned to different train/validation/test partitions.
    """

    grouped = defaultdict(list)

    for row in rows:

        grouped[
            (
                str(row["vulnerability"]),
                str(row["contract_id"]),
            )
        ].append(row)

    output_rows = []

    for (
        vulnerability,
        contract_id,
    ), group in sorted(
        grouped.items()
    ):

        graphs = sorted(
            str(x["graph"])
            for x in group
        )

        output_rows.append({

            "vulnerability":
                vulnerability,

            "contract_id":
                contract_id,

            "graph_count":
                len(graphs),

            "graphs":
                ";".join(graphs),

        })

    fieldnames = [

        "vulnerability",
        "contract_id",
        "graph_count",
        "graphs",

    ]

    with SPLIT_CANDIDATES_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )


# ============================================================================
# GRAPH AUDIT CSV
# ============================================================================

GRAPH_AUDIT_FIELDS = [

    "vulnerability",
    "metadata_vulnerability",
    "contract_id",
    "graph",

    "num_nodes",
    "num_edges",
    "num_nodes_metadata",
    "num_edges_metadata",

    "positive_nodes",
    "negative_nodes",
    "positive_ratio",

    "num_injections",
    "num_matched_injections",
    "num_unmatched_injections",

    "graph_unique_injection_ids",

    "audit_records",
    "audit_unique_injection_ids",
    "audit_matched_records",
    "audit_unmatched_records",

    "audit_ids_missing_in_graph",
    "graph_ids_missing_in_audit",

    "missing_annotation_fields",

    "label_length",
    "injection_id_length",
    "injection_count_length",

    "invalid_label_count",

    "label_injection_id_mismatch",
    "label_injection_count_mismatch",

    "negative_with_injection_id",
    "positive_without_injection_id",

    "negative_with_positive_count",
    "positive_with_zero_count",
    "negative_injection_count",

    "injection_count_sum",
    "audit_matched_node_sum",
    "injection_count_sum_mismatch",

    "injection_metadata_count_error",
    "matched_metadata_error",
    "unmatched_metadata_error",

    "audit_injection_count_error",

    "node_count_error",
    "edge_count_error",

    "empty_graph",
    "missing_contract_id",
    "vulnerability_mismatch",

    "duplicate_audit_records",

    "annotation_version",
    "annotation_strategy",

    "error_count",
    "errors",
    "status",
]


def write_graph_audit(
    rows: List[Dict[str, Any]],
) -> None:

    with GRAPH_AUDIT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=GRAPH_AUDIT_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def build_summary(
    graph_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, str]],
) -> Dict[str, Any]:

    total_graphs = len(
        graph_rows
    )

    total_injections = len(
        audit_rows
    )

    total_nodes = sum(
        safe_int(
            row["num_nodes"]
        )
        for row in graph_rows
    )

    total_edges = sum(
        safe_int(
            row["num_edges"]
        )
        for row in graph_rows
    )

    total_positive = sum(
        safe_int(
            row["positive_nodes"]
        )
        for row in graph_rows
    )

    total_negative = sum(
        safe_int(
            row["negative_nodes"]
        )
        for row in graph_rows
    )

    matched_injections = sum(
        safe_int(
            row["num_matched_injections"]
        )
        for row in graph_rows
    )

    unmatched_injections = sum(
        safe_int(
            row["num_unmatched_injections"]
        )
        for row in graph_rows
    )

    positive_ratio = (

        total_positive / total_nodes

        if total_nodes > 0

        else 0.0

    )

    vulnerability_distribution = Counter(

        str(
            row["vulnerability"]
        )

        for row in graph_rows

    )

    # ------------------------------------------------------------------
    # Integrity counters
    # ------------------------------------------------------------------

    invalid_labels = sum(
        safe_int(
            row["invalid_label_count"]
        )
        for row in graph_rows
    )

    label_length_errors = sum(

        safe_int(
            row["label_length"]
        )
        !=
        safe_int(
            row["num_nodes"]
        )

        for row in graph_rows

    )

    injection_id_length_errors = sum(

        safe_int(
            row["injection_id_length"]
        )
        !=
        safe_int(
            row["num_nodes"]
        )

        for row in graph_rows

    )

    injection_count_length_errors = sum(

        safe_int(
            row["injection_count_length"]
        )
        !=
        safe_int(
            row["num_nodes"]
        )

        for row in graph_rows

    )

    label_injection_id_mismatches = sum(

        safe_int(
            row[
                "label_injection_id_mismatch"
            ]
        )

        for row in graph_rows

    )

    label_injection_count_mismatches = sum(

        safe_int(
            row[
                "label_injection_count_mismatch"
            ]
        )

        for row in graph_rows

    )

    negative_with_injection_id = sum(

        safe_int(
            row[
                "negative_with_injection_id"
            ]
        )

        for row in graph_rows

    )

    positive_without_injection_id = sum(

        safe_int(
            row[
                "positive_without_injection_id"
            ]
        )

        for row in graph_rows

    )

    negative_with_positive_count = sum(

        safe_int(
            row[
                "negative_with_positive_count"
            ]
        )

        for row in graph_rows

    )

    positive_with_zero_count = sum(

        safe_int(
            row[
                "positive_with_zero_count"
            ]
        )

        for row in graph_rows

    )

    negative_injection_count = sum(

        safe_int(
            row[
                "negative_injection_count"
            ]
        )

        for row in graph_rows

    )

    injection_count_sum_mismatches = sum(

        safe_int(
            row[
                "injection_count_sum_mismatch"
            ]
        )

        for row in graph_rows

    )

    injection_metadata_count_errors = sum(

        safe_int(
            row[
                "injection_metadata_count_error"
            ]
        )

        for row in graph_rows

    )

    matched_metadata_errors = sum(

        safe_int(
            row[
                "matched_metadata_error"
            ]
        )

        for row in graph_rows

    )

    unmatched_metadata_errors = sum(

        safe_int(
            row[
                "unmatched_metadata_error"
            ]
        )

        for row in graph_rows

    )

    node_count_errors = sum(

        safe_int(
            row[
                "node_count_error"
            ]
        )

        for row in graph_rows

    )

    edge_count_errors = sum(

        safe_int(
            row[
                "edge_count_error"
            ]
        )

        for row in graph_rows

    )

    empty_graphs = sum(

        safe_int(
            row[
                "empty_graph"
            ]
        )

        for row in graph_rows

    )

    missing_contract_ids = sum(

        safe_int(
            row[
                "missing_contract_id"
            ]
        )

        for row in graph_rows

    )

    vulnerability_mismatches = sum(

        safe_int(
            row[
                "vulnerability_mismatch"
            ]
        )

        for row in graph_rows

    )

    duplicate_audit_records = sum(

        safe_int(
            row[
                "duplicate_audit_records"
            ]
        )

        for row in graph_rows

    )

    # ------------------------------------------------------------------
    # Informational representative-ID diagnostics
    # ------------------------------------------------------------------

    audit_ids_missing_in_graph = sum(

        safe_int(
            row[
                "audit_ids_missing_in_graph"
            ]
        )

        for row in graph_rows

    )

    graph_ids_missing_in_audit = sum(

        safe_int(
            row[
                "graph_ids_missing_in_audit"
            ]
        )

        for row in graph_rows

    )

    # ------------------------------------------------------------------
    # Graph distributions
    # ------------------------------------------------------------------

    all_positive = sum(

        1

        for row in graph_rows

        if (
            safe_int(
                row["positive_nodes"]
            )
            ==
            safe_int(
                row["num_nodes"]
            )
            and
            safe_int(
                row["num_nodes"]
            ) > 0
        )

    )

    all_negative = sum(

        1

        for row in graph_rows

        if (
            safe_int(
                row["positive_nodes"]
            )
            == 0
            and
            safe_int(
                row["num_nodes"]
            ) > 0
        )

    )

    mixed = (

        total_graphs
        -
        all_positive
        -
        all_negative

    )

    unique_groups = len({

        (
            str(
                row["vulnerability"]
            ),
            str(
                row["contract_id"]
            ),

        )

        for row in graph_rows

    })

    unique_audit_records = len({

        audit_record_key(row)

        for row in audit_rows

    })

    # ------------------------------------------------------------------
    # Graph-local errors
    # ------------------------------------------------------------------

    graph_integrity_errors = sum(

        safe_int(
            row["error_count"]
        )

        for row in graph_rows

    )

    return {

        "dataset":
            "SolidiFI-benchmark",

        "step":
            "07_dataset_integrity_and_split_audit",

        "read_only":
            True,

        "graphs":
            total_graphs,

        "injections":
            total_injections,

        "matched_injections":
            matched_injections,

        "unmatched_injections":
            unmatched_injections,

        "unique_audit_records":
            unique_audit_records,

        "duplicate_audit_records":
            duplicate_audit_records,

        "ast_nodes":
            total_nodes,

        "ast_edges":
            total_edges,

        "positive_nodes":
            total_positive,

        "negative_nodes":
            total_negative,

        "positive_node_ratio":
            positive_ratio,

        "vulnerability_distribution":
            dict(
                sorted(
                    vulnerability_distribution.items()
                )
            ),

        "integrity_checks": {

            "invalid_labels":
                invalid_labels,

            "label_length_errors":
                int(
                    label_length_errors
                ),

            "injection_id_length_errors":
                int(
                    injection_id_length_errors
                ),

            "injection_count_length_errors":
                int(
                    injection_count_length_errors
                ),

            "label_injection_id_mismatches":
                label_injection_id_mismatches,

            "label_injection_count_mismatches":
                label_injection_count_mismatches,

            "negative_with_injection_id":
                negative_with_injection_id,

            "positive_without_injection_id":
                positive_without_injection_id,

            "negative_with_positive_count":
                negative_with_positive_count,

            "positive_with_zero_count":
                positive_with_zero_count,

            "negative_injection_count":
                negative_injection_count,

            "injection_count_sum_mismatches":
                injection_count_sum_mismatches,

            "injection_metadata_count_errors":
                injection_metadata_count_errors,

            "matched_metadata_errors":
                matched_metadata_errors,

            "unmatched_metadata_errors":
                unmatched_metadata_errors,

            "node_count_errors":
                node_count_errors,

            "edge_count_errors":
                edge_count_errors,

            "empty_graphs":
                empty_graphs,

            "missing_contract_ids":
                missing_contract_ids,

            "vulnerability_mismatches":
                vulnerability_mismatches,

        },

        "informational_diagnostics": {

            "audit_ids_missing_in_graph":
                audit_ids_missing_in_graph,

            "graph_ids_missing_in_audit":
                graph_ids_missing_in_audit,

            "note":
                (
                    "graph.injection_id stores only the "
                    "first/representative injection ID per "
                    "node; therefore missing audit IDs in "
                    "this tensor are not treated as orphan "
                    "injections or integrity errors."
                ),

        },

        "special_distributions": {

            "all_positive_graphs":
                all_positive,

            "all_negative_graphs":
                all_negative,

            "mixed_graphs":
                mixed,

        },

        "unique_vulnerability_contract_groups":
            unique_groups,

        "integrity_error_count":
            graph_integrity_errors,

        "status":
            (
                "PASS"
                if graph_integrity_errors == 0
                else "REVIEW REQUIRED"
            ),

        "files": {

            "graph_audit":
                str(
                    GRAPH_AUDIT_CSV
                ),

            "split_candidates":
                str(
                    SPLIT_CANDIDATES_CSV
                ),

            "json_report":
                str(
                    REPORT_JSON
                ),

            "text_report":
                str(
                    REPORT_TXT
                ),

        },

        "split_policy":
            (
                "Train/validation/test splitting must occur "
                "at CONTRACT level. AST nodes from the same "
                "Solidity contract must never cross dataset "
                "splits."
            ),

        "expected_annotation_version":
            EXPECTED_ANNOTATION_VERSION,

        "expected_annotation_strategy":
            EXPECTED_ANNOTATION_STRATEGY,

    }


# ============================================================================
# TEXT REPORT
# ============================================================================

def write_text_report(
    summary: Dict[str, Any],
) -> None:

    checks = summary[
        "integrity_checks"
    ]

    special = summary[
        "special_distributions"
    ]

    info = summary[
        "informational_diagnostics"
    ]

    lines: List[str] = []

    lines.append(
        "=" * WIDTH
    )

    lines.append(
        "SecureGAT-Agent Step 07"
    )

    lines.append(
        "Dataset Integrity and Contract-Level Split Audit"
    )

    lines.append(
        "=" * WIDTH
    )

    lines.append("")

    lines.append(
        "READ-ONLY AUDIT"
    )

    lines.append("")

    lines.append(
        f"Graphs: {summary['graphs']}"
    )

    lines.append(
        f"Injections: {summary['injections']}"
    )

    lines.append(
        f"Matched injections: "
        f"{summary['matched_injections']}"
    )

    lines.append(
        f"Unmatched injections: "
        f"{summary['unmatched_injections']}"
    )

    lines.append(
        f"AST nodes: {summary['ast_nodes']}"
    )

    lines.append(
        f"AST edges: {summary['ast_edges']}"
    )

    lines.append(
        f"Positive nodes: "
        f"{summary['positive_nodes']}"
    )

    lines.append(
        f"Negative nodes: "
        f"{summary['negative_nodes']}"
    )

    lines.append(
        "Positive-node ratio: "
        f"{summary['positive_node_ratio']:.6f}"
    )

    lines.append("")

    lines.append(
        "VULNERABILITY DISTRIBUTION"
    )

    lines.append(
        "-" * WIDTH
    )

    for (
        name,
        count,
    ) in summary[
        "vulnerability_distribution"
    ].items():

        lines.append(
            f"{name:<30} {count}"
        )

    lines.append("")

    lines.append(
        "INTEGRITY CHECKS"
    )

    lines.append(
        "-" * WIDTH
    )

    lines.append(
        f"Invalid labels: "
        f"{checks['invalid_labels']}"
    )

    lines.append(
        f"Label length errors: "
        f"{checks['label_length_errors']}"
    )

    lines.append(
        "Injection-ID length errors: "
        f"{checks['injection_id_length_errors']}"
    )

    lines.append(
        "Injection-count length errors: "
        f"{checks['injection_count_length_errors']}"
    )

    lines.append(
        "Label/injection-ID mismatches: "
        f"{checks['label_injection_id_mismatches']}"
    )

    lines.append(
        "Label/injection-count mismatches: "
        f"{checks['label_injection_count_mismatches']}"
    )

    lines.append(
        "Negative nodes with injection ID: "
        f"{checks['negative_with_injection_id']}"
    )

    lines.append(
        "Positive nodes without injection ID: "
        f"{checks['positive_without_injection_id']}"
    )

    lines.append(
        "Negative nodes with positive injection count: "
        f"{checks['negative_with_positive_count']}"
    )

    lines.append(
        "Positive nodes with zero injection count: "
        f"{checks['positive_with_zero_count']}"
    )

    lines.append(
        "Negative injection counts: "
        f"{checks['negative_injection_count']}"
    )

    lines.append(
        "Injection-count sum mismatches: "
        f"{checks['injection_count_sum_mismatches']}"
    )

    lines.append(
        "Injection metadata count errors: "
        f"{checks['injection_metadata_count_errors']}"
    )

    lines.append(
        "Matched metadata errors: "
        f"{checks['matched_metadata_errors']}"
    )

    lines.append(
        "Unmatched metadata errors: "
        f"{checks['unmatched_metadata_errors']}"
    )

    lines.append(
        "Node count errors: "
        f"{checks['node_count_errors']}"
    )

    lines.append(
        "Edge count errors: "
        f"{checks['edge_count_errors']}"
    )

    lines.append(
        "Empty graphs: "
        f"{checks['empty_graphs']}"
    )

    lines.append(
        "Missing contract IDs: "
        f"{checks['missing_contract_ids']}"
    )

    lines.append(
        "Vulnerability mismatches: "
        f"{checks['vulnerability_mismatches']}"
    )

    lines.append("")

    lines.append(
        "INFORMATIONAL INJECTION-ID DIAGNOSTICS"
    )

    lines.append(
        "-" * WIDTH
    )

    lines.append(
        "Audit IDs missing from graph.injection_id: "
        f"{info['audit_ids_missing_in_graph']}"
    )

    lines.append(
        "Graph IDs missing from audit: "
        f"{info['graph_ids_missing_in_audit']}"
    )

    lines.append(
        "These are NOT counted as integrity errors "
        "because graph.injection_id stores only the "
        "first/representative injection ID per node."
    )

    lines.append("")

    lines.append(
        "SPECIAL DISTRIBUTIONS"
    )

    lines.append(
        "-" * WIDTH
    )

    lines.append(
        "All-positive graphs: "
        f"{special['all_positive_graphs']}"
    )

    lines.append(
        "All-negative graphs: "
        f"{special['all_negative_graphs']}"
    )

    lines.append(
        "Mixed graphs: "
        f"{special['mixed_graphs']}"
    )

    lines.append("")

    lines.append(
        "SPLIT POLICY"
    )

    lines.append(
        "-" * WIDTH
    )

    lines.append(
        "Train/validation/test splitting must occur "
        "at CONTRACT level."
    )

    lines.append(
        "AST nodes from the same Solidity contract "
        "must never cross dataset splits."
    )

    lines.append("")

    lines.append(
        f"Integrity errors: "
        f"{summary['integrity_error_count']}"
    )

    lines.append(
        f"STATUS: {summary['status']}"
    )

    lines.append("")

    lines.append(
        "=" * WIDTH
    )

    with REPORT_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(lines)
            + "\n"
        )


# ============================================================================
# JSON REPORT
# ============================================================================

def write_json_report(
    summary: Dict[str, Any],
) -> None:

    with REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    banner(
        "SecureGAT-Agent Step 07\n"
        "Dataset Integrity and Contract-Level Split Audit"
    )

    print("")

    print(
        "READ-ONLY AUDIT"
    )

    print("")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------

    try:

        graph_files = (
            discover_graph_files()
        )

        audit_rows, _ = (
            load_injection_audit()
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    print(
        f"Graph files discovered: "
        f"{len(graph_files)}"
    )

    print(
        f"Injection records loaded: "
        f"{len(audit_rows)}"
    )

    section("")

    # ------------------------------------------------------------------
    # Build audit index
    # ------------------------------------------------------------------

    audit_index = (
        build_audit_index(
            audit_rows
        )
    )

    graph_rows: List[
        Dict[str, Any]
    ] = []

    # ------------------------------------------------------------------
    # Audit every graph
    # ------------------------------------------------------------------

    for graph_path in graph_files:

        vulnerability = (
            graph_path.parent.name
        )

        graph_name = (
            graph_path.name
        )

        # Load once to obtain contract ID.
        graph = load_graph(
            graph_path
        )

        contract_id_raw = (
            get_attr(
                graph,
                "contract_id",
                None,
            )
        )

        contract_id = (

            ""

            if contract_id_raw is None

            else str(
                contract_id_raw
            )

        )

        key = (

            vulnerability,
            contract_id,
            graph_name,

        )

        rows_for_graph = (
            audit_index.get(
                key,
                [],
            )
        )

        row, _details = (
            graph_audit(
                graph_path,
                rows_for_graph,
            )
        )

        graph_rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Detect audit records with no corresponding graph identity.
    # ------------------------------------------------------------------

    discovered_keys = {

        (
            str(
                row["vulnerability"]
            ),
            str(
                row["contract_id"]
            ),
            str(
                row["graph"]
            ),

        )

        for row in graph_rows

    }

    global_orphan_audit_records = 0

    for row in audit_rows:

        if (
            audit_key(row)
            not in discovered_keys
        ):

            global_orphan_audit_records += 1

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------

    summary = build_summary(
        graph_rows,
        audit_rows,
    )

    summary[
        "global_orphan_audit_records"
    ] = (
        global_orphan_audit_records
    )

    # A truly unmatched audit record whose graph identity cannot be found
    # is an integrity error. This is different from an injection ID that is
    # absent from graph.injection_id because of representative-ID semantics.

    if (
        global_orphan_audit_records
        > 0
    ):

        summary[
            "integrity_error_count"
        ] += (
            global_orphan_audit_records
        )

        summary[
            "status"
        ] = (
            "REVIEW REQUIRED"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------

    write_graph_audit(
        graph_rows
    )

    write_split_candidates(
        graph_rows
    )

    write_json_report(
        summary
    )

    write_text_report(
        summary
    )

    # ==================================================================
    # CONSOLE SUMMARY
    # ==================================================================

    banner(
        "STEP 07 SUMMARY"
    )

    print(
        f"Graphs: "
        f"{summary['graphs']}"
    )

    print(
        f"Injections: "
        f"{summary['injections']}"
    )

    print(
        f"Matched injections: "
        f"{summary['matched_injections']}"
    )

    print(
        f"Unmatched injections: "
        f"{summary['unmatched_injections']}"
    )

    print(
        f"AST nodes: "
        f"{summary['ast_nodes']}"
    )

    print(
        f"AST edges: "
        f"{summary['ast_edges']}"
    )

    print(
        f"Positive nodes: "
        f"{summary['positive_nodes']}"
    )

    print(
        f"Negative nodes: "
        f"{summary['negative_nodes']}"
    )

    print(
        "Positive-node ratio: "
        f"{summary['positive_node_ratio']:.6f}"
    )

    print("")

    print(
        "Vulnerability distribution:"
    )

    for (
        name,
        count,
    ) in summary[
        "vulnerability_distribution"
    ].items():

        print(
            f"  {name:<28} "
            f"{count}"
        )

    print("")

    checks = summary[
        "integrity_checks"
    ]

    print(
        "Integrity checks:"
    )

    print(
        f"  Invalid labels: "
        f"{checks['invalid_labels']}"
    )

    print(
        f"  Label length errors: "
        f"{checks['label_length_errors']}"
    )

    print(
        "  Injection-ID length errors: "
        f"{checks['injection_id_length_errors']}"
    )

    print(
        "  Injection-count length errors: "
        f"{checks['injection_count_length_errors']}"
    )

    print(
        "  Label/injection-ID mismatches: "
        f"{checks['label_injection_id_mismatches']}"
    )

    print(
        "  Label/injection-count mismatches: "
        f"{checks['label_injection_count_mismatches']}"
    )

    print(
        "  Negative with injection ID: "
        f"{checks['negative_with_injection_id']}"
    )

    print(
        "  Positive without injection ID: "
        f"{checks['positive_without_injection_id']}"
    )

    print(
        "  Negative with positive count: "
        f"{checks['negative_with_positive_count']}"
    )

    print(
        "  Positive with zero count: "
        f"{checks['positive_with_zero_count']}"
    )

    print(
        "  Negative injection counts: "
        f"{checks['negative_injection_count']}"
    )

    print(
        "  Injection-count sum mismatches: "
        f"{checks['injection_count_sum_mismatches']}"
    )

    print(
        "  Injection metadata count errors: "
        f"{checks['injection_metadata_count_errors']}"
    )

    print(
        "  Matched metadata errors: "
        f"{checks['matched_metadata_errors']}"
    )

    print(
        "  Unmatched metadata errors: "
        f"{checks['unmatched_metadata_errors']}"
    )

    print(
        "  Global orphan audit records: "
        f"{global_orphan_audit_records}"
    )

    print(
        f"  Node count errors: "
        f"{checks['node_count_errors']}"
    )

    print(
        f"  Edge count errors: "
        f"{checks['edge_count_errors']}"
    )

    print(
        f"  Empty graphs: "
        f"{checks['empty_graphs']}"
    )

    print(
        f"  Missing contract IDs: "
        f"{checks['missing_contract_ids']}"
    )

    print(
        f"  Duplicate audit records: "
        f"{summary['duplicate_audit_records']}"
    )

    print("")

    special = summary[
        "special_distributions"
    ]

    print(
        "All-positive graphs: "
        f"{special['all_positive_graphs']}"
    )

    print(
        "All-negative graphs: "
        f"{special['all_negative_graphs']}"
    )

    print(
        "Mixed graphs: "
        f"{special['mixed_graphs']}"
    )

    print(
        "Unique vulnerability/contract groups: "
        f"{summary['unique_vulnerability_contract_groups']}"
    )

    print(
        "Informational audit IDs missing "
        "from graph.injection_id: "
        f"{summary['informational_diagnostics']['audit_ids_missing_in_graph']}"
    )

    print("")

    print(
        "Integrity error count: "
        f"{summary['integrity_error_count']}"
    )

    print(
        f"STATUS: "
        f"{summary['status']}"
    )

    print("")

    print(
        f"Graph audit: "
        f"{GRAPH_AUDIT_CSV}"
    )

    print(
        f"Split candidates: "
        f"{SPLIT_CANDIDATES_CSV}"
    )

    print(
        f"JSON report: "
        f"{REPORT_JSON}"
    )

    print(
        f"Text report: "
        f"{REPORT_TXT}"
    )

    print("")

    print(
        "=" * WIDTH
    )

    print(
        "STEP 07 COMPLETE"
    )

    print(
        "=" * WIDTH
    )

    return (
        0
        if summary["status"] == "PASS"
        else 2
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )