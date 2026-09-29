#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 02:17:25 2026

@author: Willie
"""


"""
SecureGAT-Agent: Model Input and Learning Protocol Definition

Purpose
-------
Freeze the model-input, target, split, metric, checkpoint-selection,
and training-history protocol before model implementation.


"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "datasets"
    / "annotated_graph_dataset"
)

MODEL_READY_ROOT = (
    DATASET_ROOT
    / "model_ready"
)

SPLIT_CSV = (
    DATASET_ROOT
    / "contract_level_splits.csv"
)

INTEGRITY_JSON = (
    DATASET_ROOT
    / "dataset_integrity_report.json"
)

SPLIT_JSON = (
    DATASET_ROOT
    / "split_audit_report.json"
)

OUTPUT_JSON = (
    MODEL_READY_ROOT
    / "model_input_learning_protocol.json"
)

OUTPUT_TXT = (
    MODEL_READY_ROOT
    / "model_input_learning_protocol.txt"
)


# ============================================================================
# FROZEN PROTOCOL
# ============================================================================

PROTOCOL_VERSION = "step10_v1"

TARGET_FIELD = "injection_label"

DIAGNOSTIC_FIELDS = [
    "injection_id",
    "injection_count",
]

REQUIRED_MODEL_INPUT_FIELDS = [
    "x",
    "edge_index",
]

FORBIDDEN_MODEL_INPUT_FIELDS = [
    "injection_label",
    "injection_id",
    "injection_count",
    "num_injections",
    "num_matched_injections",
    "num_unmatched_injections",
    "vulnerability",
    "contract_id",
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "mcc",
]

CHECKPOINT_SELECTION_METRIC = "validation_f1"

TRAINING_HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "train_accuracy",
    "validation_accuracy",
]

TRAINING_CURVES = {
    "accuracy": {
        "training": "train_accuracy",
        "validation": "validation_accuracy",
        "output": "training_accuracy.png",
    },
    "loss": {
        "training": "train_loss",
        "validation": "validation_loss",
        "output": "training_loss.png",
    },
}


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str) -> None:
    print()
    print("=" * 78)
    print("STEP 10 FAILED")
    print("=" * 78)
    print(message)
    print("=" * 78)
    raise RuntimeError(message)


def load_json(path: Path):
    if not path.exists():
        fail(f"Required JSON file not found:\n{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_records(path: Path):
    if not path.exists():
        fail(f"Split CSV not found:\n{path}")

    records = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "contract_id",
            "graph",
            "split",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            fail(
                "Split CSV is missing required fields: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            records.append(row)

    return records


def safe_torch_load(path: Path):
    """
    Load PyTorch/PyG graph while remaining compatible with
    environments where weights_only behavior differs.
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


def get_attribute(graph, name):
    """
    Supports PyTorch Geometric Data objects as well as ordinary
    objects carrying graph attributes.
    """

    if hasattr(graph, name):
        return getattr(graph, name)

    if isinstance(graph, dict):
        return graph.get(name)

    return None


def tensor_shape(value):
    if value is None:
        return None

    if hasattr(value, "shape"):
        return list(value.shape)

    return None


def tensor_dtype(value):
    if value is None:
        return None

    if hasattr(value, "dtype"):
        return str(value.dtype)

    return None


def tensor_length(value):
    if value is None:
        return None

    try:
        return len(value)
    except Exception:
        return None


# ============================================================================
# GRAPH DISCOVERY
# ============================================================================

def discover_graphs(split_records):

    graph_records = []

    for row in split_records:

        graph_name = row["graph"]
        vulnerability = row.get(
            "vulnerability",
            "",
        )

        contract_id = str(
            row["contract_id"]
        )

        split = row["split"].strip().lower()

        graph_path = (
            DATASET_ROOT
            / vulnerability
            / graph_name
        )

        if not graph_path.exists():

            # Fallback: search by graph name if the split CSV
            # does not encode vulnerability.

            matches = list(
                DATASET_ROOT.glob(
                    f"*/{graph_name}"
                )
            )

            if len(matches) == 1:
                graph_path = matches[0]

            elif len(matches) == 0:
                fail(
                    f"Graph referenced by Step 08 split does not exist:\n"
                    f"{graph_name}"
                )

            else:
                fail(
                    f"Ambiguous graph reference:\n"
                    f"{graph_name}"
                )

        graph_records.append(
            {
                "contract_id": contract_id,
                "graph": graph_name,
                "split": split,
                "vulnerability": vulnerability,
                "path": str(graph_path),
            }
        )

    return graph_records


# ============================================================================
# GRAPH SCHEMA AUDIT
# ============================================================================

def audit_graph(
    record,
):

    path = Path(
        record["path"]
    )

    graph = safe_torch_load(path)

    x = get_attribute(
        graph,
        "x",
    )

    edge_index = get_attribute(
        graph,
        "edge_index",
    )

    label = get_attribute(
        graph,
        TARGET_FIELD,
    )

    injection_id = get_attribute(
        graph,
        "injection_id",
    )

    injection_count = get_attribute(
        graph,
        "injection_count",
    )

    # ------------------------------------------------------------------------
    # Required model fields
    # ------------------------------------------------------------------------

    missing_inputs = []

    if x is None:
        missing_inputs.append("x")

    if edge_index is None:
        missing_inputs.append("edge_index")

    if label is None:
        missing_inputs.append(
            TARGET_FIELD
        )

    if missing_inputs:

        fail(
            f"Graph is missing required fields:\n"
            f"{path}\n"
            f"Missing: {missing_inputs}"
        )

    # ------------------------------------------------------------------------
    # Node count
    # ------------------------------------------------------------------------

    if len(x.shape) != 2:

        fail(
            f"Node feature matrix must be 2-dimensional:\n"
            f"{path}\n"
            f"Shape: {tuple(x.shape)}"
        )

    num_nodes = int(
        x.shape[0]
    )

    num_features = int(
        x.shape[1]
    )

    # ------------------------------------------------------------------------
    # Edge structure
    # ------------------------------------------------------------------------

    if len(edge_index.shape) != 2:

        fail(
            f"edge_index must be 2-dimensional:\n"
            f"{path}\n"
            f"Shape: {tuple(edge_index.shape)}"
        )

    if edge_index.shape[0] != 2:

        fail(
            f"edge_index must have shape [2, E]:\n"
            f"{path}\n"
            f"Shape: {tuple(edge_index.shape)}"
        )

    num_edges = int(
        edge_index.shape[1]
    )

    # ------------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------------

    if len(label) != num_nodes:

        fail(
            f"Target length does not match node count:\n"
            f"{path}\n"
            f"nodes={num_nodes}, "
            f"labels={len(label)}"
        )

    labels = [
        int(v)
        for v in label.tolist()
    ]

    invalid_labels = [
        value
        for value in labels
        if value not in (0, 1)
    ]

    if invalid_labels:

        fail(
            f"Invalid target labels detected:\n"
            f"{path}\n"
            f"Examples: {invalid_labels[:10]}"
        )

    label_counts = Counter(
        labels
    )

    # ------------------------------------------------------------------------
    # Diagnostic metadata
    # ------------------------------------------------------------------------

    if injection_id is not None:

        if len(injection_id) != num_nodes:

            fail(
                f"injection_id length mismatch:\n"
                f"{path}"
            )

    if injection_count is not None:

        if len(injection_count) != num_nodes:

            fail(
                f"injection_count length mismatch:\n"
                f"{path}"
            )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    vulnerability = get_attribute(
        graph,
        "vulnerability",
    )

    graph_contract_id = get_attribute(
        graph,
        "contract_id",
    )

    return {
        "graph": record["graph"],
        "path": str(path),
        "split": record["split"],
        "contract_id": record["contract_id"],
        "vulnerability": record[
            "vulnerability"
        ],
        "graph_vulnerability": (
            str(vulnerability)
            if vulnerability is not None
            else None
        ),
        "graph_contract_id": (
            str(graph_contract_id)
            if graph_contract_id is not None
            else None
        ),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": num_features,
        "x_shape": tensor_shape(x),
        "x_dtype": tensor_dtype(x),
        "edge_index_shape": tensor_shape(
            edge_index
        ),
        "edge_index_dtype": tensor_dtype(
            edge_index
        ),
        "label_shape": tensor_shape(
            label
        ),
        "label_dtype": tensor_dtype(
            label
        ),
        "positive_nodes": label_counts.get(
            1,
            0,
        ),
        "negative_nodes": label_counts.get(
            0,
            0,
        ),
        "has_injection_id": (
            injection_id is not None
        ),
        "has_injection_count": (
            injection_count is not None
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("SecureGAT-Agent Step 10")
    print("Model Input and Learning Protocol Definition")
    print("=" * 78)
    print()
    print("Execution mode: READ-ONLY")
    print("Graph annotations will NOT be modified.")
    print("Step 08 split remains the source of truth.")
    print()

    MODEL_READY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Load prerequisite reports
    # ------------------------------------------------------------------------

    integrity_report = load_json(
        INTEGRITY_JSON
    )

    split_report = load_json(
        SPLIT_JSON
    )

    split_records = load_split_records(
        SPLIT_CSV
    )

    print(
        f"Frozen split records: "
        f"{len(split_records)}"
    )

    # ------------------------------------------------------------------------
    # Verify split composition
    # ------------------------------------------------------------------------

    split_counter = Counter(
        row["split"].strip().lower()
        for row in split_records
    )

    contract_split = {}

    for row in split_records:

        contract = str(
            row["contract_id"]
        )

        split = row[
            "split"
        ].strip().lower()

        if contract in contract_split:

            if contract_split[
                contract
            ] != split:

                fail(
                    "Contract appears in multiple "
                    "dataset splits:\n"
                    f"contract={contract}"
                )

        contract_split[
            contract
        ] = split

    train_contracts = {
        c
        for c, s in contract_split.items()
        if s == "train"
    }

    validation_contracts = {
        c
        for c, s in contract_split.items()
        if s in {
            "validation",
            "val",
        }
    }

    test_contracts = {
        c
        for c, s in contract_split.items()
        if s == "test"
    }

    # ------------------------------------------------------------------------
    # Split leakage validation
    # ------------------------------------------------------------------------

    split_sets = {
        "train": train_contracts,
        "validation": validation_contracts,
        "test": test_contracts,
    }

    split_names = list(
        split_sets.keys()
    )

    for i in range(
        len(split_names)
    ):

        for j in range(
            i + 1,
            len(split_names),
        ):

            a = split_names[i]
            b = split_names[j]

            overlap = (
                split_sets[a]
                &
                split_sets[b]
            )

            if overlap:

                fail(
                    "Contract leakage detected:\n"
                    f"{a} ∩ {b} = "
                    f"{sorted(overlap)}"
                )

    # ------------------------------------------------------------------------
    # Discover graphs
    # ------------------------------------------------------------------------

    graph_records = discover_graphs(
        split_records
    )

    print(
        f"Graph records: "
        f"{len(graph_records)}"
    )

    # ------------------------------------------------------------------------
    # Audit graphs
    # ------------------------------------------------------------------------

    print()
    print(
        "Auditing graph schema..."
    )

    graph_audits = []

    for index, record in enumerate(
        graph_records,
        start=1,
    ):

        audit = audit_graph(
            record
        )

        graph_audits.append(
            audit
        )

        if (
            index % 50 == 0
            or index == len(graph_records)
        ):

            print(
                f"  Audited "
                f"{index}/{len(graph_records)}"
            )

    # ------------------------------------------------------------------------
    # Feature consistency
    # ------------------------------------------------------------------------

    feature_dimensions = Counter(
        audit["num_features"]
        for audit in graph_audits
    )

    x_dtypes = Counter(
        audit["x_dtype"]
        for audit in graph_audits
    )

    edge_dtypes = Counter(
        audit["edge_index_dtype"]
        for audit in graph_audits
    )

    label_dtypes = Counter(
        audit["label_dtype"]
        for audit in graph_audits
    )

    print()
    print(
        "Node feature dimensions:"
    )

    for dimension, count in sorted(
        feature_dimensions.items()
    ):

        print(
            f"  {dimension}: {count}"
        )

    if len(feature_dimensions) != 1:

        fail(
            "Inconsistent node-feature dimensions "
            "across graphs:\n"
            f"{dict(feature_dimensions)}"
        )

    # ------------------------------------------------------------------------
    # Split graph counts
    # ------------------------------------------------------------------------

    graph_split_counter = Counter(
        audit["split"]
        for audit in graph_audits
    )

    # ------------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------------

    total_nodes = sum(
        audit["num_nodes"]
        for audit in graph_audits
    )

    total_edges = sum(
        audit["num_edges"]
        for audit in graph_audits
    )

    total_positive = sum(
        audit["positive_nodes"]
        for audit in graph_audits
    )

    total_negative = sum(
        audit["negative_nodes"]
        for audit in graph_audits
    )

    # ------------------------------------------------------------------------
    # Vulnerability distribution
    # ------------------------------------------------------------------------

    vulnerability_distribution = Counter(
        audit["vulnerability"]
        for audit in graph_audits
    )

    # ------------------------------------------------------------------------
    # Protocol definition
    # ------------------------------------------------------------------------

    protocol = {
        "dataset": "SolidiFI-benchmark",

        "step": "10_model_input_learning_protocol",

        "protocol_version": PROTOCOL_VERSION,

        "read_only": True,

        "annotations_modified": False,

        "split_source_of_truth": (
            "Step 08 "
            "contract_level_splits.csv"
        ),

        "task": {
            "type": (
                "AST-node vulnerability "
                "localization"
            ),
            "input": "graph",
            "target": TARGET_FIELD,
            "target_values": {
                "0": "non-injected AST node",
                "1": "injected/vulnerable AST node",
            },
        },

        "model_inputs": {
            "node_features": "x",
            "graph_edges": "edge_index",
        },

        "diagnostic_only": DIAGNOSTIC_FIELDS,

        "forbidden_model_inputs": (
            FORBIDDEN_MODEL_INPUT_FIELDS
        ),

        "reason_for_exclusions": {
            "injection_label": (
                "training target; direct label leakage"
            ),
            "injection_id": (
                "injection identity metadata; "
                "not a semantic node feature"
            ),
            "injection_count": (
                "direct injection metadata; "
                "must not be used as an input feature"
            ),
            "num_injections": (
                "graph-level vulnerability metadata"
            ),
            "num_matched_injections": (
                "annotation metadata"
            ),
            "num_unmatched_injections": (
                "annotation metadata"
            ),
            "vulnerability": (
                "ground-truth vulnerability class metadata"
            ),
            "contract_id": (
                "dataset identity; potential split/data leakage"
            ),
        },

        "prediction_level": "node",

        "loss_target": {
            "field": TARGET_FIELD,
            "type": "binary classification",
        },

        "metrics": METRICS,

        "primary_model_selection_metric": (
            CHECKPOINT_SELECTION_METRIC
        ),

        "metric_policy": {
            "accuracy": (
                "reported but not sufficient "
                "for model selection"
            ),
            "precision": (
                "vulnerable-node precision"
            ),
            "recall": (
                "vulnerable-node recall"
            ),
            "f1": (
                "harmonic mean of precision and recall"
            ),
            "pr_auc": (
                "primary imbalance-aware ranking metric"
            ),
            "roc_auc": (
                "threshold-independent ranking metric"
            ),
            "mcc": (
                "balanced binary classification metric"
            ),
        },

        "security_metrics": [
            "vulnerable_node_precision",
            "vulnerable_node_recall",
            "vulnerable_node_f1",
            "contract_level_detection_rate",
            "per_vulnerability_precision",
            "per_vulnerability_recall",
            "per_vulnerability_f1",
            "confusion_matrix",
        ],

        "training_history": {
            "required_fields": (
                TRAINING_HISTORY_FIELDS
            ),
            "format": [
                "CSV",
                "JSON",
            ],
        },

        "training_curves": TRAINING_CURVES,

        "split": {
            "train_contracts": len(
                train_contracts
            ),
            "validation_contracts": len(
                validation_contracts
            ),
            "test_contracts": len(
                test_contracts
            ),
            "train_graphs": graph_split_counter.get(
                "train",
                0,
            ),
            "validation_graphs": (
                graph_split_counter.get(
                    "validation",
                    0,
                )
                + graph_split_counter.get(
                    "val",
                    0,
                )
            ),
            "test_graphs": graph_split_counter.get(
                "test",
                0,
            ),
            "contract_leakage": 0,
        },

        "dataset_statistics": {
            "graph_records": len(
                graph_audits
            ),
            "contracts": len(
                contract_split
            ),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "positive_nodes": total_positive,
            "negative_nodes": total_negative,
            "positive_ratio": (
                total_positive
                / total_nodes
                if total_nodes
                else 0.0
            ),
            "feature_dimensions": dict(
                feature_dimensions
            ),
            "x_dtypes": dict(
                x_dtypes
            ),
            "edge_index_dtypes": dict(
                edge_dtypes
            ),
            "label_dtypes": dict(
                label_dtypes
            ),
            "vulnerability_distribution": dict(
                vulnerability_distribution
            ),
        },

        "baseline_sequence": [
            "MLP",
            "GCN",
            "GAT",
            "GraphSAGE",
            "SecureGAT",
        ],

        "test_policy": {
            "test_labels": (
                "must not be used for "
                "checkpoint selection"
            ),
            "test_metrics": (
                "reported only after model "
                "selection is frozen"
            ),
        },

        "graph_audit": {
            "graphs_audited": len(
                graph_audits
            ),
            "graphs_passed": len(
                graph_audits
            ),
            "graphs_failed": 0,
        },

        "source_reports": {
            "integrity": str(
                INTEGRITY_JSON
            ),
            "split": str(
                SPLIT_JSON
            ),
        },
    }

    # ------------------------------------------------------------------------
    # Write JSON protocol
    # ------------------------------------------------------------------------

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            protocol,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Human-readable protocol
    # ------------------------------------------------------------------------

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "SecureGAT-Agent Step 10"
    )

    lines.append(
        "MODEL INPUT AND LEARNING PROTOCOL"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        "STATUS: FROZEN"
    )

    lines.append(
        "Execution mode: READ-ONLY"
    )

    lines.append("")

    lines.append(
        "TASK"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "AST-node vulnerability localization"
    )

    lines.append(
        "Target: injection_label"
    )

    lines.append(
        "0 = non-injected AST node"
    )

    lines.append(
        "1 = injected/vulnerable AST node"
    )

    lines.append("")

    lines.append(
        "MODEL INPUTS"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "Node features : x"
    )

    lines.append(
        "Graph edges   : edge_index"
    )

    lines.append("")

    lines.append(
        "FORBIDDEN MODEL INPUTS"
    )

    lines.append(
        "-" * 78
    )

    for field in FORBIDDEN_MODEL_INPUT_FIELDS:

        lines.append(
            f"{field}"
        )

    lines.append("")

    lines.append(
        "DIAGNOSTIC-ONLY FIELDS"
    )

    lines.append(
        "-" * 78
    )

    for field in DIAGNOSTIC_FIELDS:

        lines.append(
            field
        )

    lines.append("")

    lines.append(
        "DATASET"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Graphs    : {len(graph_audits)}"
    )

    lines.append(
        f"Contracts : {len(contract_split)}"
    )

    lines.append(
        f"Nodes     : {total_nodes}"
    )

    lines.append(
        f"Edges     : {total_edges}"
    )

    lines.append(
        f"Positive  : {total_positive}"
    )

    lines.append(
        f"Negative  : {total_negative}"
    )

    lines.append(
        f"Positive ratio: "
        f"{total_positive / total_nodes:.6f}"
    )

    lines.append("")

    lines.append(
        "SPLIT"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Train      : "
        f"{len(train_contracts)} contracts / "
        f"{graph_split_counter.get('train', 0)} graphs"
    )

    lines.append(
        f"Validation : "
        f"{len(validation_contracts)} contracts / "
        f"{graph_split_counter.get('validation', 0) + graph_split_counter.get('val', 0)} graphs"
    )

    lines.append(
        f"Test       : "
        f"{len(test_contracts)} contracts / "
        f"{graph_split_counter.get('test', 0)} graphs"
    )

    lines.append(
        "Contract leakage: 0"
    )

    lines.append("")

    lines.append(
        "EVALUATION METRICS"
    )

    lines.append(
        "-" * 78
    )

    for metric in METRICS:

        lines.append(
            metric
        )

    lines.append("")

    lines.append(
        "CHECKPOINT SELECTION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "Validation F1"
    )

    lines.append(
        "Accuracy is reported but is not the "
        "checkpoint-selection criterion."
    )

    lines.append("")

    lines.append(
        "TRAINING HISTORY"
    )

    lines.append(
        "-" * 78
    )

    for field in TRAINING_HISTORY_FIELDS:

        lines.append(
            field
        )

    lines.append("")

    lines.append(
        "TRAINING CURVES"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "training_accuracy.png"
    )

    lines.append(
        "training_loss.png"
    )

    lines.append("")

    lines.append(
        "BASELINE SEQUENCE"
    )

    lines.append(
        "-" * 78
    )

    for index, model in enumerate(
        protocol["baseline_sequence"],
        start=1,
    ):

        lines.append(
            f"{index}. {model}"
        )

    lines.append("")

    lines.append(
        "=" * 78
    )

    lines.append(
        "STEP 10 COMPLETE"
    )

    lines.append(
        "Protocol frozen for model implementation."
    )

    lines.append(
        "=" * 78
    )

    OUTPUT_TXT.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STEP 10 SUMMARY")
    print("=" * 78)

    print(
        f"Graphs audited       : "
        f"{len(graph_audits)}"
    )

    print(
        f"Contracts             : "
        f"{len(contract_split)}"
    )

    print(
        f"Node feature dimension: "
        f"{next(iter(feature_dimensions))}"
    )

    print(
        f"Total nodes           : "
        f"{total_nodes}"
    )

    print(
        f"Positive nodes        : "
        f"{total_positive}"
    )

    print(
        f"Negative nodes        : "
        f"{total_negative}"
    )

    print(
        f"Positive-node ratio   : "
        f"{total_positive / total_nodes:.6f}"
    )

    print()

    print(
        "Split:"
    )

    print(
        f"  Train      : "
        f"{len(train_contracts)} contracts / "
        f"{graph_split_counter.get('train', 0)} graphs"
    )

    print(
        f"  Validation : "
        f"{len(validation_contracts)} contracts / "
        f"{graph_split_counter.get('validation', 0) + graph_split_counter.get('val', 0)} graphs"
    )

    print(
        f"  Test       : "
        f"{len(test_contracts)} contracts / "
        f"{graph_split_counter.get('test', 0)} graphs"
    )

    print(
        "  Leakage    : 0"
    )

    print()

    print(
        "Target:"
    )

    print(
        "  injection_label"
    )

    print()

    print(
        "Model inputs:"
    )

    print(
        "  x"
    )

    print(
        "  edge_index"
    )

    print()

    print(
        "Checkpoint selection:"
    )

    print(
        "  validation F1"
    )

    print()

    print(
        "Metrics:"
    )

    for metric in METRICS:

        print(
            f"  {metric}"
        )

    print()

    print(
        "Training curves:"
    )

    print(
        "  training_accuracy.png"
    )

    print(
        "  training_loss.png"
    )

    print()

    print(
        f"Protocol JSON: {OUTPUT_JSON}"
    )

    print(
        f"Protocol text: {OUTPUT_TXT}"
    )

    print()

    print(
        "STATUS: PASS"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()