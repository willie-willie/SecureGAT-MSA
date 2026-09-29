#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 02:38:38 2026

@author: Willie
"""


"""
SecureGAT-Agent: SecureGAT Node Representation Protocol

Purpose
-------
Define and audit the frozen node representation that will be supplied
to the SecureGAT model.


Representation
--------------
AST node type:
    41 categorical node types
    embedding dimension = 32

AST node depth:
    normalized by maximum observed depth = 25
    dimension = 1

Final representation:
    32 + 1 = 33 dimensions

Model topology input:
    edge_index

Target:
    injection_label
"""

import csv
import json
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

OUTPUT_JSON = (
    MODEL_READY_ROOT
    / "securegat_node_representation_protocol.json"
)

OUTPUT_TXT = (
    MODEL_READY_ROOT
    / "securegat_node_representation_protocol.txt"
)


# ============================================================================
# FROZEN REPRESENTATION PROTOCOL
# ============================================================================

PROTOCOL_VERSION = "step12_v1"

NUM_AST_NODE_TYPES = 41

AST_NODE_ID_MIN = 0
AST_NODE_ID_MAX = 40

AST_EMBEDDING_DIM = 32

MAX_AST_DEPTH = 25

DEPTH_FEATURE_DIM = 1

FINAL_NODE_FEATURE_DIM = (
    AST_EMBEDDING_DIM
    + DEPTH_FEATURE_DIM
)

TARGET_FIELD = "injection_label"

GRAPH_STRUCTURE_FIELD = "edge_index"

NODE_TYPE_FIELD = "node_type"

NODE_TYPE_NAME_FIELD = "node_type_name"

NODE_DEPTH_FIELD = "node_depth"


FORBIDDEN_FEATURE_FIELDS = [
    "injection_label",
    "injection_id",
    "injection_count",
    "num_injections",
    "num_matched_injections",
    "num_unmatched_injections",
    "vulnerability",
    "vulnerability_label",
    "contract_id",
]


# ============================================================================
# HELPERS
# ============================================================================

def fail(message):
    print()
    print("=" * 78)
    print("STEP 12 FAILED")
    print("=" * 78)
    print(message)
    print("=" * 78)
    raise RuntimeError(message)


def safe_torch_load(path):
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
    if hasattr(graph, name):
        return getattr(graph, name)

    if isinstance(graph, dict):
        return graph.get(name)

    return None


def tensor_to_list(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


def load_split_records():

    if not SPLIT_CSV.exists():
        fail(
            "Step 08 split file not found:\n"
            f"{SPLIT_CSV}"
        )

    records = []

    with SPLIT_CSV.open(
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

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:
            fail(
                "Split CSV missing required fields: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            records.append(row)

    return records


def resolve_graph_path(graph_name):

    candidates = [
        DATASET_ROOT / graph_name,
    ]

    graph_name_path = Path(graph_name)

    if graph_name_path.is_absolute():
        candidates.insert(
            0,
            graph_name_path,
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(
        DATASET_ROOT.rglob(graph_name)
    )

    if matches:
        return matches[0]

    return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("SecureGAT-Agent Step 12")
    print()
    print("SecureGAT Node Representation Protocol")
    print("=" * 78)
    print()
    print("Execution mode: READ-ONLY")
    print()
    print("Graph annotations will NOT be modified.")
    print("Step 08 split remains the source of truth.")
    print()

    MODEL_READY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # LOAD SPLIT
    # ------------------------------------------------------------------------

    split_records = load_split_records()

    if not split_records:
        fail("Step 08 split contains no records.")

    print(
        f"Frozen split records: "
        f"{len(split_records)}"
    )

    split_counter = Counter(
        row["split"]
        for row in split_records
    )

    contract_by_split = {}

    for row in split_records:

        contract_id = str(
            row["contract_id"]
        )

        split = row["split"]

        contract_by_split.setdefault(
            split,
            set(),
        ).add(contract_id)

    print()

    print("Frozen split:")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        contracts = contract_by_split.get(
            split_name,
            set(),
        )

        print(
            f"  {split_name:<11}: "
            f"{len(contracts)} contracts / "
            f"{split_counter.get(split_name, 0)} graphs"
        )

    # ------------------------------------------------------------------------
    # LEAKAGE CHECK
    # ------------------------------------------------------------------------

    train_contracts = contract_by_split.get(
        "train",
        set(),
    )

    validation_contracts = contract_by_split.get(
        "validation",
        set(),
    )

    test_contracts = contract_by_split.get(
        "test",
        set(),
    )

    leakage = (
        train_contracts
        & validation_contracts
    ) | (
        train_contracts
        & test_contracts
    ) | (
        validation_contracts
        & test_contracts
    )

    if leakage:
        fail(
            "Contract leakage detected: "
            + ", ".join(sorted(leakage))
        )

    print()
    print("Contract leakage: 0")

    # ------------------------------------------------------------------------
    # AUDIT STATE
    # ------------------------------------------------------------------------

    graph_count = 0

    total_nodes = 0
    total_edges = 0

    node_type_counter = Counter()

    node_type_name_counter = Counter()

    depth_counter = Counter()

    feature_dimensions = Counter()

    feature_dtypes = Counter()

    errors = []

    representation_examples = []

    global_node_type_mapping = {}

    global_type_name_mapping = {}

    min_depth = None
    max_depth = None
    depth_sum = 0
    depth_count = 0

    # ------------------------------------------------------------------------
    # GRAPH AUDIT
    # ------------------------------------------------------------------------

    print()
    print("Auditing SecureGAT node representation...")
    print()

    for index, row in enumerate(
        split_records,
        start=1,
    ):

        graph_name = row["graph"]

        graph_path = resolve_graph_path(
            graph_name
        )

        if graph_path is None:

            errors.append(
                f"Missing graph: {graph_name}"
            )

            continue

        try:
            graph = safe_torch_load(
                graph_path
            )
        except Exception as exc:

            errors.append(
                f"Failed loading {graph_name}: "
                f"{exc}"
            )

            continue

        graph_count += 1

        x = get_attribute(
            graph,
            "x",
        )

        node_type = get_attribute(
            graph,
            NODE_TYPE_FIELD,
        )

        node_type_name = get_attribute(
            graph,
            NODE_TYPE_NAME_FIELD,
        )

        node_depth = get_attribute(
            graph,
            NODE_DEPTH_FIELD,
        )

        edge_index = get_attribute(
            graph,
            GRAPH_STRUCTURE_FIELD,
        )

        injection_label = get_attribute(
            graph,
            TARGET_FIELD,
        )

        # --------------------------------------------------------------------
        # BASIC FIELD CHECKS
        # --------------------------------------------------------------------

        if x is None:
            errors.append(
                f"{graph_name}: missing x"
            )
            continue

        if node_type is None:
            errors.append(
                f"{graph_name}: missing node_type"
            )
            continue

        if node_type_name is None:
            errors.append(
                f"{graph_name}: missing node_type_name"
            )
            continue

        if node_depth is None:
            errors.append(
                f"{graph_name}: missing node_depth"
            )
            continue

        if edge_index is None:
            errors.append(
                f"{graph_name}: missing edge_index"
            )
            continue

        if injection_label is None:
            errors.append(
                f"{graph_name}: missing injection_label"
            )
            continue

        # --------------------------------------------------------------------
        # TENSOR CONVERSION
        # --------------------------------------------------------------------

        try:
            x_tensor = (
                x
                if torch.is_tensor(x)
                else torch.as_tensor(x)
            )

            node_type_tensor = (
                node_type
                if torch.is_tensor(node_type)
                else torch.as_tensor(node_type)
            )

            node_depth_tensor = (
                node_depth
                if torch.is_tensor(node_depth)
                else torch.as_tensor(node_depth)
            )

            labels_tensor = (
                injection_label
                if torch.is_tensor(injection_label)
                else torch.as_tensor(injection_label)
            )

            edge_tensor = (
                edge_index
                if torch.is_tensor(edge_index)
                else torch.as_tensor(edge_index)
            )

        except Exception as exc:

            errors.append(
                f"{graph_name}: tensor conversion failed: "
                f"{exc}"
            )

            continue

        # --------------------------------------------------------------------
        # NODE COUNT
        # --------------------------------------------------------------------

        if node_type_tensor.ndim != 1:

            errors.append(
                f"{graph_name}: node_type must be 1-D, "
                f"got {tuple(node_type_tensor.shape)}"
            )

            continue

        num_nodes = (
            node_type_tensor.shape[0]
        )

        total_nodes += num_nodes

        # --------------------------------------------------------------------
        # x VALIDATION
        # --------------------------------------------------------------------

        if x_tensor.ndim != 2:

            errors.append(
                f"{graph_name}: x must be 2-D, "
                f"got {tuple(x_tensor.shape)}"
            )

        else:

            if x_tensor.shape[0] != num_nodes:

                errors.append(
                    f"{graph_name}: x/node_type "
                    f"length mismatch"
                )

            feature_dimensions[
                tuple(x_tensor.shape)
            ] += 1

            feature_dtypes[
                str(x_tensor.dtype)
            ] += 1

        # --------------------------------------------------------------------
        # x == node_type
        # --------------------------------------------------------------------

        if (
            x_tensor.ndim == 2
            and x_tensor.shape[1] == 1
            and x_tensor.shape[0] == num_nodes
        ):

            x_flat = x_tensor[:, 0].long()
            nt_flat = node_type_tensor.long()

            if not torch.equal(
                x_flat,
                nt_flat,
            ):

                errors.append(
                    f"{graph_name}: "
                    "x[:,0] != node_type"
                )

        # --------------------------------------------------------------------
        # NODE TYPE RANGE
        # --------------------------------------------------------------------

        type_values = (
            node_type_tensor
            .detach()
            .cpu()
            .long()
            .tolist()
        )

        for value in type_values:

            node_type_counter[value] += 1

            if value < AST_NODE_ID_MIN:
                errors.append(
                    f"{graph_name}: "
                    f"node type {value} "
                    "below minimum"
                )

            if value > AST_NODE_ID_MAX:
                errors.append(
                    f"{graph_name}: "
                    f"node type {value} "
                    "above maximum"
                )

        # --------------------------------------------------------------------
        # NODE TYPE NAMES
        # --------------------------------------------------------------------

        names = tensor_to_list(
            node_type_name
        )

        if len(names) != num_nodes:

            errors.append(
                f"{graph_name}: "
                "node_type_name length mismatch"
            )

        else:

            for type_id, name in zip(
                type_values,
                names,
            ):

                name = str(name)

                node_type_name_counter[
                    name
                ] += 1

                previous_name = (
                    global_node_type_mapping.get(
                        type_id
                    )
                )

                if (
                    previous_name is not None
                    and previous_name != name
                ):

                    errors.append(
                        f"{graph_name}: "
                        f"type ID {type_id} maps "
                        f"to both "
                        f"{previous_name} and {name}"
                    )

                global_node_type_mapping[
                    type_id
                ] = name

                previous_id = (
                    global_type_name_mapping.get(
                        name
                    )
                )

                if (
                    previous_id is not None
                    and previous_id != type_id
                ):

                    errors.append(
                        f"{graph_name}: "
                        f"type name {name} maps "
                        f"to both "
                        f"{previous_id} and {type_id}"
                    )

                global_type_name_mapping[
                    name
                ] = type_id

        # --------------------------------------------------------------------
        # NODE DEPTH
        # --------------------------------------------------------------------

        depth_values = (
            node_depth_tensor
            .detach()
            .cpu()
            .tolist()
        )

        if len(depth_values) != num_nodes:

            errors.append(
                f"{graph_name}: "
                "node_depth length mismatch"
            )

        else:

            for depth in depth_values:

                try:
                    depth_float = float(
                        depth
                    )
                except Exception:

                    errors.append(
                        f"{graph_name}: "
                        f"invalid depth {depth}"
                    )

                    continue

                depth_counter[
                    depth_float
                ] += 1

                depth_sum += depth_float
                depth_count += 1

                if (
                    min_depth is None
                    or depth_float < min_depth
                ):
                    min_depth = depth_float

                if (
                    max_depth is None
                    or depth_float > max_depth
                ):
                    max_depth = depth_float

                if depth_float < 0:

                    errors.append(
                        f"{graph_name}: "
                        f"negative node depth "
                        f"{depth_float}"
                    )

                if depth_float > MAX_AST_DEPTH:

                    errors.append(
                        f"{graph_name}: "
                        f"node depth "
                        f"{depth_float} exceeds "
                        f"protocol maximum "
                        f"{MAX_AST_DEPTH}"
                    )

        # --------------------------------------------------------------------
        # DEPTH NORMALIZATION TEST
        # --------------------------------------------------------------------

        if len(depth_values) == num_nodes:

            depth_tensor = (
                node_depth_tensor
                .float()
            )

            normalized_depth = (
                depth_tensor
                / float(MAX_AST_DEPTH)
            )

            if (
                torch.isnan(
                    normalized_depth
                ).any()
                or torch.isinf(
                    normalized_depth
                ).any()
            ):

                errors.append(
                    f"{graph_name}: "
                    "normalized depth contains "
                    "NaN or Inf"
                )

            if (
                normalized_depth < 0
            ).any():

                errors.append(
                    f"{graph_name}: "
                    "normalized depth below 0"
                )

            if (
                normalized_depth > 1
            ).any():

                errors.append(
                    f"{graph_name}: "
                    "normalized depth above 1"
                )

        # --------------------------------------------------------------------
        # EDGE INDEX
        # --------------------------------------------------------------------

        if edge_tensor.ndim != 2:

            errors.append(
                f"{graph_name}: "
                f"edge_index must be 2-D, "
                f"got {tuple(edge_tensor.shape)}"
            )

        else:

            if edge_tensor.shape[0] != 2:

                errors.append(
                    f"{graph_name}: "
                    "edge_index first dimension "
                    f"must be 2, got "
                    f"{edge_tensor.shape[0]}"
                )

            else:

                total_edges += (
                    edge_tensor.shape[1]
                )

                if edge_tensor.numel() > 0:

                    min_edge = int(
                        edge_tensor.min().item()
                    )

                    max_edge = int(
                        edge_tensor.max().item()
                    )

                    if min_edge < 0:

                        errors.append(
                            f"{graph_name}: "
                            f"negative edge index "
                            f"{min_edge}"
                        )

                    if max_edge >= num_nodes:

                        errors.append(
                            f"{graph_name}: "
                            f"edge index {max_edge} "
                            f">= node count "
                            f"{num_nodes}"
                        )

        # --------------------------------------------------------------------
        # LABEL SHAPE
        # --------------------------------------------------------------------

        if labels_tensor.ndim != 1:

            errors.append(
                f"{graph_name}: "
                "injection_label must be 1-D"
            )

        elif labels_tensor.shape[0] != num_nodes:

            errors.append(
                f"{graph_name}: "
                "injection_label length mismatch"
            )

        # --------------------------------------------------------------------
        # REPRESENTATION DIMENSION
        # --------------------------------------------------------------------

        expected_shape = (
            num_nodes,
            FINAL_NODE_FEATURE_DIM,
        )

        # We construct the representation conceptually
        # using a deterministic zero embedding solely to
        # validate dimensionality. No learned parameters
        # are created or persisted in Step 12.

        simulated_embedding = torch.zeros(
            num_nodes,
            AST_EMBEDDING_DIM,
            dtype=torch.float32,
        )

        if len(depth_values) == num_nodes:

            normalized_depth_column = (
                node_depth_tensor
                .float()
                .reshape(-1, 1)
                / float(MAX_AST_DEPTH)
            )

            simulated_representation = torch.cat(
                [
                    simulated_embedding,
                    normalized_depth_column,
                ],
                dim=1,
            )

            if (
                tuple(
                    simulated_representation.shape
                )
                != expected_shape
            ):

                errors.append(
                    f"{graph_name}: "
                    "final representation shape "
                    "mismatch"
                )

        # --------------------------------------------------------------------
        # STORE FIRST EXAMPLE
        # --------------------------------------------------------------------

        if len(representation_examples) < 5:

            representation_examples.append(
                {
                    "graph": graph_name,
                    "num_nodes": num_nodes,
                    "original_x_shape": list(
                        x_tensor.shape
                    ),
                    "node_type_shape": list(
                        node_type_tensor.shape
                    ),
                    "node_depth_shape": list(
                        node_depth_tensor.shape
                    ),
                    "normalized_depth_shape": [
                        num_nodes,
                        1,
                    ],
                    "final_representation_shape": [
                        num_nodes,
                        FINAL_NODE_FEATURE_DIM,
                    ],
                    "edge_index_shape": list(
                        edge_tensor.shape
                    ),
                }
            )

        # --------------------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------------------

        if (
            index % 50 == 0
            or index == len(split_records)
        ):

            print(
                f"  Audited "
                f"{index}/{len(split_records)}"
            )

    # =========================================================================
    # GLOBAL AUDIT
    # =========================================================================

    mean_depth = (
        depth_sum / depth_count
        if depth_count
        else None
    )

    # Ensure vocabulary coverage is exactly within protocol.

    observed_type_ids = set(
        node_type_counter.keys()
    )

    expected_type_ids = set(
        range(
            AST_NODE_ID_MIN,
            AST_NODE_ID_MAX + 1,
        )
    )

    missing_type_ids = sorted(
        expected_type_ids
        - observed_type_ids
    )

    unexpected_type_ids = sorted(
        observed_type_ids
        - expected_type_ids
    )

    if unexpected_type_ids:

        errors.append(
            "Unexpected node-type IDs: "
            + ", ".join(
                str(x)
                for x in unexpected_type_ids
            )
        )

    # =========================================================================
    # RESULT
    # =========================================================================

    audit_status = (
        "PASS"
        if not errors
        else "REVIEW REQUIRED"
    )

    report = {

        "dataset": "SolidiFI-benchmark",

        "step":
            "12_securegat_node_representation",

        "protocol_version":
            PROTOCOL_VERSION,

        "read_only": True,

        "graphs_audited":
            graph_count,

        "total_nodes":
            total_nodes,

        "total_edges":
            total_edges,

        "split": {

            "records":
                len(split_records),

            "train_graphs":
                split_counter.get(
                    "train",
                    0,
                ),

            "validation_graphs":
                split_counter.get(
                    "validation",
                    0,
                ),

            "test_graphs":
                split_counter.get(
                    "test",
                    0,
                ),

            "train_contracts":
                len(train_contracts),

            "validation_contracts":
                len(validation_contracts),

            "test_contracts":
                len(test_contracts),

            "contract_leakage":
                len(leakage),
        },

        "representation": {

            "node_type_field":
                NODE_TYPE_FIELD,

            "node_type_name_field":
                NODE_TYPE_NAME_FIELD,

            "num_ast_node_types":
                NUM_AST_NODE_TYPES,

            "node_type_id_min":
                AST_NODE_ID_MIN,

            "node_type_id_max":
                AST_NODE_ID_MAX,

            "embedding_dimension":
                AST_EMBEDDING_DIM,

            "depth_field":
                NODE_DEPTH_FIELD,

            "depth_normalization":
                f"depth / {MAX_AST_DEPTH}",

            "maximum_depth":
                MAX_AST_DEPTH,

            "depth_dimension":
                DEPTH_FEATURE_DIM,

            "final_node_feature_dimension":
                FINAL_NODE_FEATURE_DIM,

            "graph_structure":
                GRAPH_STRUCTURE_FIELD,

            "target":
                TARGET_FIELD,

            "forbidden_features":
                FORBIDDEN_FEATURE_FIELDS,
        },

        "feature_audit": {

            "x_dimensions":
                {
                    str(k): v
                    for k, v
                    in feature_dimensions.items()
                },

            "x_dtypes":
                dict(feature_dtypes),

            "node_type_vocabulary_observed":
                len(observed_type_ids),

            "missing_type_ids":
                missing_type_ids,

            "unexpected_type_ids":
                unexpected_type_ids,

            "node_type_frequency":
                {
                    str(k): v
                    for k, v
                    in sorted(
                        node_type_counter.items()
                    )
                },

            "node_type_mapping":
                {
                    str(k): v
                    for k, v
                    in sorted(
                        global_node_type_mapping.items()
                    )
                },
        },

        "depth_audit": {

            "minimum":
                min_depth,

            "maximum":
                max_depth,

            "mean":
                mean_depth,

            "normalization_min":
                (
                    min_depth / MAX_AST_DEPTH
                    if min_depth is not None
                    else None
                ),

            "normalization_max":
                (
                    max_depth / MAX_AST_DEPTH
                    if max_depth is not None
                    else None
                ),
        },

        "representation_examples":
            representation_examples,

        "errors":
            errors,

        "error_count":
            len(errors),

        "status":
            audit_status,
    }

    # =========================================================================
    # WRITE JSON
    # =========================================================================

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # =========================================================================
    # WRITE TEXT REPORT
    # =========================================================================

    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 78
            + "\n"
        )

        f.write(
            "SecureGAT-Agent Step 12\n"
        )

        f.write(
            "SecureGAT Node Representation Protocol\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            "READ-ONLY AUDIT\n\n"
        )

        f.write(
            f"Graphs audited: "
            f"{graph_count}\n"
        )

        f.write(
            f"Total AST nodes: "
            f"{total_nodes}\n"
        )

        f.write(
            f"Total AST edges: "
            f"{total_edges}\n"
        )

        f.write(
            f"Node-type vocabulary: "
            f"{NUM_AST_NODE_TYPES}\n"
        )

        f.write(
            f"Node-type ID range: "
            f"({AST_NODE_ID_MIN}, "
            f"{AST_NODE_ID_MAX})\n"
        )

        f.write(
            f"AST embedding dimension: "
            f"{AST_EMBEDDING_DIM}\n"
        )

        f.write(
            f"Depth feature dimension: "
            f"{DEPTH_FEATURE_DIM}\n"
        )

        f.write(
            f"Maximum AST depth: "
            f"{MAX_AST_DEPTH}\n"
        )

        f.write(
            f"Final node feature dimension: "
            f"{FINAL_NODE_FEATURE_DIM}\n\n"
        )

        f.write(
            "SPLIT\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            f"Train contracts: "
            f"{len(train_contracts)}\n"
        )

        f.write(
            f"Validation contracts: "
            f"{len(validation_contracts)}\n"
        )

        f.write(
            f"Test contracts: "
            f"{len(test_contracts)}\n"
        )

        f.write(
            f"Contract leakage: "
            f"{len(leakage)}\n\n"
        )

        f.write(
            "REPRESENTATION\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            "Node type → learnable embedding\n"
        )

        f.write(
            f"Embedding dimension: "
            f"{AST_EMBEDDING_DIM}\n"
        )

        f.write(
            "Node depth → normalized scalar\n"
        )

        f.write(
            f"Depth normalization: "
            f"depth / {MAX_AST_DEPTH}\n"
        )

        f.write(
            f"Final representation: "
            f"{AST_EMBEDDING_DIM} + "
            f"{DEPTH_FEATURE_DIM} = "
            f"{FINAL_NODE_FEATURE_DIM}\n"
        )

        f.write(
            f"Graph topology: "
            f"{GRAPH_STRUCTURE_FIELD}\n"
        )

        f.write(
            f"Target: "
            f"{TARGET_FIELD}\n\n"
        )

        f.write(
            "FEATURES EXCLUDED FROM MODEL INPUT\n"
            + "-" * 78
            + "\n"
        )

        for field in FORBIDDEN_FEATURE_FIELDS:

            f.write(
                f"  {field}\n"
            )

        f.write("\n")

        f.write(
            "DEPTH AUDIT\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            f"Minimum: {min_depth}\n"
        )

        f.write(
            f"Maximum: {max_depth}\n"
        )

        f.write(
            f"Mean: {mean_depth}\n\n"
        )

        f.write(
            "ERRORS\n"
            + "-" * 78
            + "\n"
        )

        f.write(
            f"Error count: "
            f"{len(errors)}\n"
        )

        for error in errors[:100]:

            f.write(
                f"  {error}\n"
            )

        f.write("\n")

        f.write(
            f"STATUS: {audit_status}\n"
        )

        f.write(
            "=" * 78
            + "\n"
        )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print()
    print("=" * 78)
    print("STEP 12 SUMMARY")
    print("=" * 78)

    print(
        f"Graphs audited: {graph_count}"
    )

    print(
        f"Total AST nodes: {total_nodes}"
    )

    print(
        f"Total AST edges: {total_edges}"
    )

    print(
        f"AST node types: "
        f"{NUM_AST_NODE_TYPES}"
    )

    print(
        f"Node-type ID range: "
        f"{AST_NODE_ID_MIN}-"
        f"{AST_NODE_ID_MAX}"
    )

    print(
        f"AST embedding dimension: "
        f"{AST_EMBEDDING_DIM}"
    )

    print(
        f"Depth feature dimension: "
        f"{DEPTH_FEATURE_DIM}"
    )

    print(
        f"Maximum AST depth: "
        f"{MAX_AST_DEPTH}"
    )

    print(
        f"Final node representation: "
        f"{FINAL_NODE_FEATURE_DIM} dimensions"
    )

    print()

    print("Representation:")

    print(
        "  node_type → learnable embedding"
    )

    print(
        "  node_depth → normalized depth"
    )

    print(
        "  embedding + depth → 33-D node feature"
    )

    print(
        "  edge_index → graph topology"
    )

    print()

    print(
        f"Contract leakage: "
        f"{len(leakage)}"
    )

    print(
        f"Audit errors: "
        f"{len(errors)}"
    )

    print(
        f"STATUS: {audit_status}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        f"Text: {OUTPUT_TXT}"
    )

    print("=" * 78)

    if errors:

        raise RuntimeError(
            "Step 12 failed representation audit."
        )

    print()
    print(
        "STEP 12 COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()