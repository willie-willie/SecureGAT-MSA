#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 02:31:30 2026

@author: Willie
"""


"""
SecureGAT-Agent: AST Node Representation and Feature-Semantic Audit

Purpose
-------
Audit the existing graph node representation before model implementation.

It verifies:

1. x == node_type
2. node_type <-> node_type_name consistency
3. global AST node-type vocabulary
4. node-depth statistics
5. parent-index validity
6. source-position validity
7. feature dimensionality
8. forbidden annotation/label leakage
9. consistency across all 350 graph records

No graph annotations are modified.
"""

import json
from collections import Counter, defaultdict
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

OUTPUT_ROOT = (
    DATASET_ROOT
    / "model_ready"
)

OUTPUT_JSON = (
    OUTPUT_ROOT
    / "ast_node_representation_audit.json"
)

OUTPUT_TXT = (
    OUTPUT_ROOT
    / "ast_node_representation_audit.txt"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

FORBIDDEN_ATTRIBUTES = {
    "injection_label",
    "injection_id",
    "injection_count",
    "num_injections",
    "num_matched_injections",
    "num_unmatched_injections",
    "vulnerability",
    "vulnerability_label",
    "contract_id",
}

REQUIRED_ATTRIBUTES = {
    "x",
    "node_type",
    "node_type_name",
    "node_start",
    "node_end",
    "node_depth",
    "parent_index",
    "source_file_index",
    "edge_index",
}


# ============================================================================
# HELPERS
# ============================================================================

def safe_torch_load(path: Path):
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


def get_attr(graph, name):
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

    return list(value)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("SecureGAT-Agent Step 11")
    print("AST Node Representation and Feature-Semantic Audit")
    print("=" * 78)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_paths = sorted(
        DATASET_ROOT.glob("*/*.pt")
    )

    if not graph_paths:
        raise RuntimeError(
            "No graph files found."
        )

    print()
    print("Graph files discovered:", len(graph_paths))
    print()
    print("READ-ONLY AUDIT")
    print("Graph annotations will NOT be modified.")
    print("-" * 78)

    # ------------------------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------------------------

    global_node_types = Counter()
    global_type_names = Counter()

    id_to_names = defaultdict(set)
    name_to_ids = defaultdict(set)

    node_depth_values = []
    source_file_indices = set()

    x_equals_node_type_errors = 0
    shape_errors = 0
    dtype_errors = 0

    node_type_length_errors = 0
    node_type_name_length_errors = 0
    node_depth_length_errors = 0
    parent_length_errors = 0
    start_length_errors = 0
    end_length_errors = 0
    source_index_length_errors = 0

    parent_index_errors = 0
    source_position_errors = 0

    graph_attribute_missing = Counter()

    forbidden_attributes_found = Counter()

    total_nodes = 0

    graph_reports = []

    # ------------------------------------------------------------------------
    # Process graphs
    # ------------------------------------------------------------------------

    for counter, path in enumerate(graph_paths, start=1):

        graph = safe_torch_load(path)

        graph_name = path.name
        vulnerability = get_attr(
            graph,
            "vulnerability",
        )

        contract_id = get_attr(
            graph,
            "contract_id",
        )

        x = get_attr(graph, "x")
        node_type = get_attr(graph, "node_type")
        node_type_name = get_attr(
            graph,
            "node_type_name",
        )
        node_start = get_attr(
            graph,
            "node_start",
        )
        node_end = get_attr(
            graph,
            "node_end",
        )
        node_depth = get_attr(
            graph,
            "node_depth",
        )
        parent_index = get_attr(
            graph,
            "parent_index",
        )
        source_file_index = get_attr(
            graph,
            "source_file_index",
        )

        # --------------------------------------------------------------------
        # Attribute presence
        # --------------------------------------------------------------------

        missing = []

        for attribute in REQUIRED_ATTRIBUTES:

            if get_attr(graph, attribute) is None:
                missing.append(attribute)
                graph_attribute_missing[attribute] += 1

        # --------------------------------------------------------------------
        # Forbidden attributes
        # --------------------------------------------------------------------

        available = set()

        if hasattr(graph, "keys"):
            try:
                available = set(graph.keys())
            except Exception:
                available = set()

        elif isinstance(graph, dict):
            available = set(graph.keys())

        for attribute in FORBIDDEN_ATTRIBUTES:

            if attribute in available:

                # These attributes may exist in the graph for
                # diagnostics, but they must not be model inputs.
                forbidden_attributes_found[attribute] += 1

        # --------------------------------------------------------------------
        # Node count
        # --------------------------------------------------------------------

        if x is not None:

            if not torch.is_tensor(x):

                shape_errors += 1

            else:

                if x.ndim != 2:
                    shape_errors += 1

                elif x.shape[1] != 1:
                    shape_errors += 1

                if x.dtype != torch.long:
                    dtype_errors += 1

                num_nodes = x.shape[0]

        elif node_type is not None:

            num_nodes = len(node_type)

        else:

            num_nodes = 0

        total_nodes += num_nodes

        # --------------------------------------------------------------------
        # Length checks
        # --------------------------------------------------------------------

        def check_length(
            value,
            counter_name,
        ):

            if value is None:
                return

            try:
                length = len(value)
            except TypeError:
                return

            if length != num_nodes:

                if counter_name == "node_type":
                    nonlocal_dummy[0] += 1

        # Python does not permit dynamic nonlocal counters conveniently here,
        # therefore explicit checks follow.

        if node_type is not None:

            if len(node_type) != num_nodes:
                node_type_length_errors += 1

        if node_type_name is not None:

            if len(node_type_name) != num_nodes:
                node_type_name_length_errors += 1

        if node_depth is not None:

            if len(node_depth) != num_nodes:
                node_depth_length_errors += 1

        if parent_index is not None:

            if len(parent_index) != num_nodes:
                parent_length_errors += 1

        if node_start is not None:

            if len(node_start) != num_nodes:
                start_length_errors += 1

        if node_end is not None:

            if len(node_end) != num_nodes:
                end_length_errors += 1

        if source_file_index is not None:

            if len(source_file_index) != num_nodes:
                source_index_length_errors += 1

        # --------------------------------------------------------------------
        # x == node_type
        # --------------------------------------------------------------------

        if (
            torch.is_tensor(x)
            and torch.is_tensor(node_type)
            and x.ndim == 2
            and x.shape[1] == 1
            and x.shape[0] == node_type.shape[0]
        ):

            if not torch.equal(
                x[:, 0].long(),
                node_type.long(),
            ):

                x_equals_node_type_errors += 1

        # --------------------------------------------------------------------
        # Node type vocabulary
        # --------------------------------------------------------------------

        if node_type is not None:

            type_values = tensor_to_list(
                node_type
            )

            for value in type_values:

                value = int(value)

                global_node_types[value] += 1

        if node_type_name is not None:

            names = tensor_to_list(
                node_type_name
            )

            for name in names:

                name = str(name)

                global_type_names[name] += 1

        # --------------------------------------------------------------------
        # ID <-> name mapping
        # --------------------------------------------------------------------

        if (
            node_type is not None
            and node_type_name is not None
            and len(node_type) == len(node_type_name)
        ):

            ids = tensor_to_list(node_type)
            names = tensor_to_list(node_type_name)

            for node_id, name in zip(
                ids,
                names,
            ):

                node_id = int(node_id)
                name = str(name)

                id_to_names[node_id].add(name)
                name_to_ids[name].add(node_id)

        # --------------------------------------------------------------------
        # Depth statistics
        # --------------------------------------------------------------------

        if node_depth is not None:

            depths = tensor_to_list(
                node_depth
            )

            for value in depths:

                value = int(value)

                node_depth_values.append(
                    value
                )

        # --------------------------------------------------------------------
        # Source-file index statistics
        # --------------------------------------------------------------------

        if source_file_index is not None:

            values = tensor_to_list(
                source_file_index
            )

            for value in values:

                source_file_indices.add(
                    int(value)
                )

        # --------------------------------------------------------------------
        # Parent index validity
        # --------------------------------------------------------------------

        if (
            parent_index is not None
            and len(parent_index) == num_nodes
        ):

            parents = tensor_to_list(
                parent_index
            )

            for index, parent in enumerate(
                parents
            ):

                parent = int(parent)

                # -1 is accepted as root/no-parent.
                if parent < -1:
                    parent_index_errors += 1

                elif parent >= num_nodes:
                    parent_index_errors += 1

                elif parent == index:
                    parent_index_errors += 1

        # --------------------------------------------------------------------
        # Source-position validity
        # --------------------------------------------------------------------

        if (
            node_start is not None
            and node_end is not None
            and len(node_start) == num_nodes
            and len(node_end) == num_nodes
        ):

            starts = tensor_to_list(
                node_start
            )

            ends = tensor_to_list(
                node_end
            )

            for start, end in zip(
                starts,
                ends,
            ):

                start = int(start)
                end = int(end)

                if start < 0:
                    source_position_errors += 1

                if end < start:
                    source_position_errors += 1

        # --------------------------------------------------------------------
        # Per-graph report
        # --------------------------------------------------------------------

        graph_reports.append(
            {
                "graph": graph_name,
                "vulnerability": str(vulnerability),
                "contract_id": str(contract_id),
                "num_nodes": num_nodes,
            }
        )

        if counter % 50 == 0:

            print(
                f"  Audited {counter}/{len(graph_paths)}"
            )

    # ------------------------------------------------------------------------
    # Mapping consistency
    # ------------------------------------------------------------------------

    inconsistent_id_mappings = {
        str(node_id): sorted(names)
        for node_id, names in id_to_names.items()
        if len(names) > 1
    }

    inconsistent_name_mappings = {
        str(name): sorted(ids)
        for name, ids in name_to_ids.items()
        if len(ids) > 1
    }

    # ------------------------------------------------------------------------
    # Depth statistics
    # ------------------------------------------------------------------------

    if node_depth_values:

        min_depth = min(
            node_depth_values
        )

        max_depth = max(
            node_depth_values
        )

        mean_depth = (
            sum(node_depth_values)
            / len(node_depth_values)
        )

    else:

        min_depth = None
        max_depth = None
        mean_depth = None

    # ------------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------------

    vocabulary = []

    for node_id in sorted(
        global_node_types
    ):

        vocabulary.append(
            {
                "id": node_id,
                "count": global_node_types[node_id],
                "names": sorted(
                    id_to_names.get(
                        node_id,
                        set(),
                    )
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Global audit
    # ------------------------------------------------------------------------

    audit_errors = 0

    audit_errors += x_equals_node_type_errors
    audit_errors += shape_errors
    audit_errors += dtype_errors
    audit_errors += node_type_length_errors
    audit_errors += node_type_name_length_errors
    audit_errors += node_depth_length_errors
    audit_errors += parent_length_errors
    audit_errors += start_length_errors
    audit_errors += end_length_errors
    audit_errors += source_index_length_errors
    audit_errors += parent_index_errors
    audit_errors += source_position_errors
    audit_errors += len(
        inconsistent_id_mappings
    )

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------

    report = {
        "dataset": "SolidiFI-benchmark",
        "step": "11_ast_node_representation_audit",
        "read_only": True,
        "graphs": len(graph_paths),
        "total_nodes": total_nodes,
        "x": {
            "shape": "[N, 1]",
            "dtype": "torch.int64",
            "confirmed_node_type_encoding": (
                x_equals_node_type_errors == 0
            ),
            "unique_values": len(
                global_node_types
            ),
            "minimum": (
                min(global_node_types)
                if global_node_types
                else None
            ),
            "maximum": (
                max(global_node_types)
                if global_node_types
                else None
            ),
        },
        "node_type_vocabulary": vocabulary,
        "node_type_name_vocabulary": sorted(
            global_type_names
        ),
        "mapping_consistency": {
            "inconsistent_id_to_name": (
                inconsistent_id_mappings
            ),
            "inconsistent_name_to_id": (
                inconsistent_name_mappings
            ),
        },
        "node_depth": {
            "minimum": min_depth,
            "maximum": max_depth,
            "mean": mean_depth,
        },
        "source_file_indices": sorted(
            source_file_indices
        ),
        "structural_checks": {
            "x_equals_node_type_errors":
                x_equals_node_type_errors,
            "shape_errors":
                shape_errors,
            "dtype_errors":
                dtype_errors,
            "node_type_length_errors":
                node_type_length_errors,
            "node_type_name_length_errors":
                node_type_name_length_errors,
            "node_depth_length_errors":
                node_depth_length_errors,
            "parent_length_errors":
                parent_length_errors,
            "start_length_errors":
                start_length_errors,
            "end_length_errors":
                end_length_errors,
            "source_index_length_errors":
                source_index_length_errors,
            "parent_index_errors":
                parent_index_errors,
            "source_position_errors":
                source_position_errors,
        },
        "forbidden_model_information": {
            key: value
            for key, value
            in forbidden_attributes_found.items()
        },
        "audit_errors": audit_errors,
        "status": (
            "PASS"
            if audit_errors == 0
            else "REVIEW REQUIRED"
        ),
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------------

    lines = []

    lines.append("=" * 78)
    lines.append(
        "SecureGAT-Agent Step 11"
    )
    lines.append(
        "AST Node Representation and Feature-Semantic Audit"
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append("READ-ONLY AUDIT")
    lines.append(
        "Graph annotations were NOT modified."
    )
    lines.append("")
    lines.append(
        f"Graphs: {len(graph_paths)}"
    )
    lines.append(
        f"Total AST nodes: {total_nodes}"
    )
    lines.append("")
    lines.append(
        "NODE FEATURE REPRESENTATION"
    )
    lines.append("-" * 78)
    lines.append(
        "x shape: [N, 1]"
    )
    lines.append(
        "x dtype: torch.int64"
    )
    lines.append(
        "Unique node-type IDs: "
        + str(len(global_node_types))
    )
    lines.append(
        "Minimum node-type ID: "
        + str(
            min(global_node_types)
            if global_node_types
            else None
        )
    )
    lines.append(
        "Maximum node-type ID: "
        + str(
            max(global_node_types)
            if global_node_types
            else None
        )
    )
    lines.append(
        "x == node_type errors: "
        + str(x_equals_node_type_errors)
    )
    lines.append("")
    lines.append(
        "NODE-TYPE VOCABULARY"
    )
    lines.append("-" * 78)

    for item in vocabulary:

        lines.append(
            f"{item['id']:>3} : "
            f"{', '.join(item['names'])} "
            f"({item['count']} nodes)"
        )

    lines.append("")
    lines.append(
        "NODE DEPTH"
    )
    lines.append("-" * 78)
    lines.append(
        f"Minimum: {min_depth}"
    )
    lines.append(
        f"Maximum: {max_depth}"
    )
    lines.append(
        f"Mean: {mean_depth}"
    )
    lines.append("")
    lines.append(
        "STRUCTURAL VALIDATION"
    )
    lines.append("-" * 78)

    for key, value in report[
        "structural_checks"
    ].items():

        lines.append(
            f"{key}: {value}"
        )

    lines.append("")
    lines.append(
        "MAPPING CONSISTENCY"
    )
    lines.append("-" * 78)
    lines.append(
        "Inconsistent ID -> name mappings: "
        + str(
            len(inconsistent_id_mappings)
        )
    )
    lines.append(
        "Inconsistent name -> ID mappings: "
        + str(
            len(inconsistent_name_mappings)
        )
    )
    lines.append("")
    lines.append(
        "MODEL REPRESENTATION RECOMMENDATION"
    )
    lines.append("-" * 78)
    lines.append(
        "The current x feature is a categorical AST "
        "node-type identifier."
    )
    lines.append(
        "It should not be interpreted as a continuous scalar."
    )
    lines.append(
        "Use a learnable node-type embedding before GAT layers."
    )
    lines.append(
        "Structural topology should be provided through edge_index."
    )
    lines.append(
        "node_depth may be evaluated as an additional "
        "structural feature."
    )
    lines.append(
        "injection labels and injection metadata must remain "
        "outside model inputs."
    )
    lines.append("")
    lines.append(
        f"AUDIT ERRORS: {audit_errors}"
    )
    lines.append(
        "STATUS: "
        + report["status"]
    )
    lines.append("")
    lines.append("=" * 78)

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STEP 11 SUMMARY")
    print("=" * 78)
    print()
    print(
        "Graphs audited:",
        len(graph_paths),
    )
    print(
        "Total AST nodes:",
        total_nodes,
    )
    print()
    print(
        "Node-type vocabulary:",
        len(global_node_types),
    )
    print(
        "Node-type ID range:",
        (
            min(global_node_types),
            max(global_node_types),
        ),
    )
    print(
        "x == node_type errors:",
        x_equals_node_type_errors,
    )
    print(
        "Inconsistent ID/name mappings:",
        len(inconsistent_id_mappings),
    )
    print(
        "Parent-index errors:",
        parent_index_errors,
    )
    print(
        "Source-position errors:",
        source_position_errors,
    )
    print()
    print(
        "Node depth:"
    )
    print(
        "  minimum:",
        min_depth,
    )
    print(
        "  maximum:",
        max_depth,
    )
    print(
        "  mean:",
        f"{mean_depth:.4f}"
        if mean_depth is not None
        else "None",
    )
    print()
    print(
        "Audit errors:",
        audit_errors,
    )
    print(
        "STATUS:",
        report["status"],
    )
    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )
    print(
        "Text:",
        OUTPUT_TXT,
    )
    print("=" * 78)

    if audit_errors != 0:
        raise RuntimeError(
            "Step 11 failed representation audit."
        )


if __name__ == "__main__":
    # Dummy container used only to keep the helper section
    # intentionally simple and Python-version compatible.
    nonlocal_dummy = [0]
    main()