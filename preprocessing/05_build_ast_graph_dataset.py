#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:38:55 2026

@author: Willie
"""


"""
SecureGAT-Agent: SolidiFI-specific AST version.

The SolidiFI AST representation used in this project has the structure:

    {
        "name": "SourceUnit",
        "src": "93:10166:0",
        "children": [
            {
                "name": "PragmaDirective",
                "src": "93:32:0",
                ...
            },
            {
                "name": "ContractDefinition",
                "src": "...",
                "children": [...]
            }
        ],
        "attributes": {...}
    }

IMPORTANT
---------
Only the `children` field represents AST child nodes.

Fields such as:

    attributes
    name
    member_name
    type
    typeIdentifier
    referencedDeclaration
    scope
    ...

are metadata and MUST NOT become graph nodes.

"""

import os
import json
import re
import torch

from torch_geometric.data import Data


# ======================================================================
# Configuration
# ======================================================================

AST_ROOT = "datasets/ast_dataset"

OUTPUT_ROOT = "datasets/graph_dataset"

VULNERABILITIES = [
    "Overflow-Underflow",
    "Re-entrancy",
    "TOD",
    "Timestamp-Dependency",
    "Unchecked-Send",
    "Unhandled-Exceptions",
    "tx.origin",
]


# ======================================================================
# Node vocabulary
# ======================================================================

NODE_TYPES = {}


def get_node_id(node_type):
    """
    Assign a unique integer ID to an AST node type.
    """

    if node_type not in NODE_TYPES:
        NODE_TYPES[node_type] = len(NODE_TYPES)

    return NODE_TYPES[node_type]


# ======================================================================
# Contract ID
# ======================================================================

def contract_number_from_filename(filename):
    """
    Convert:

        buggy_1_ast.json

    to:

        1
    """

    match = re.match(
        r"buggy_(\d+)_ast\.json$",
        filename
    )

    if match is None:
        return None

    return int(match.group(1))


# ======================================================================
# Source range
# ======================================================================

def extract_source_range(node):
    """
    Parse SolidiFI AST source location.

    Example:

        src = "934:15:0"

    means:

        start  = 934
        length = 15
        file   = 0

    Therefore:

        end = 934 + 15 = 949
    """

    src = node.get("src")

    if not src:
        return -1, -1, -1

    try:

        parts = str(src).split(":")

        if len(parts) < 2:
            return -1, -1, -1

        start = int(parts[0])
        length = int(parts[1])

        if len(parts) >= 3:
            file_index = int(parts[2])
        else:
            file_index = -1

        if start < 0 or length < 0:
            return -1, -1, file_index

        end = start + length

        return (
            start,
            end,
            file_index
        )

    except Exception:

        return -1, -1, -1


# ======================================================================
# AST traversal
# ======================================================================

def traverse_ast(
    node,
    nodes,
    edges,
    parent=None,
    depth=0
):
    """
    Traverse ONLY the actual SolidiFI AST hierarchy.

    CRITICAL:
    ----------
    We traverse:

        node["children"]

    only.

    We DO NOT recursively traverse arbitrary dictionaries.

    This prevents metadata such as:

        EIP20Interface
        totalSupply
        uint256
        balanceOf
        _owner
        address

    from becoming graph nodes.
    """

    if not isinstance(node, dict):
        return

    # --------------------------------------------------------------
    # AST node type
    # --------------------------------------------------------------

    node_type = node.get(
        "name",
        "Unknown"
    )

    # --------------------------------------------------------------
    # Source interval
    # --------------------------------------------------------------

    start, end, file_index = (
        extract_source_range(node)
    )

    # --------------------------------------------------------------
    # Node index
    # --------------------------------------------------------------

    current_index = len(nodes)

    # --------------------------------------------------------------
    # Store node
    # --------------------------------------------------------------

    nodes.append({

        "type": str(node_type),

        "start": start,

        "end": end,

        "file_index": file_index,

        "depth": depth,

        "parent": (
            parent
            if parent is not None
            else -1
        ),

    })

    # --------------------------------------------------------------
    # Parent -> child edge
    # --------------------------------------------------------------

    if parent is not None:

        edges.append(
            (
                parent,
                current_index
            )
        )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # SolidiFI AST children are stored under `children`.
    #
    # Do NOT traverse arbitrary dictionaries.
    # --------------------------------------------------------------

    children = node.get(
        "children",
        []
    )

    if not isinstance(
        children,
        list
    ):

        return

    # --------------------------------------------------------------
    # Traverse actual AST children
    # --------------------------------------------------------------

    for child in children:

        if not isinstance(
            child,
            dict
        ):

            continue

        traverse_ast(
            child,
            nodes,
            edges,
            parent=current_index,
            depth=depth + 1
        )


# ======================================================================
# AST -> Graph
# ======================================================================

def ast_to_graph(ast):

    nodes = []

    edges = []

    # --------------------------------------------------------------
    # Traverse
    # --------------------------------------------------------------

    traverse_ast(
        ast,
        nodes,
        edges
    )

    # --------------------------------------------------------------
    # Node data
    # --------------------------------------------------------------

    node_type_ids = []

    node_type_names = []

    node_start = []

    node_end = []

    node_depth = []

    parent_index = []

    source_file_index = []

    x_values = []

    # --------------------------------------------------------------
    # Convert nodes
    # --------------------------------------------------------------

    for node in nodes:

        node_type_name = node["type"]

        node_type_id = get_node_id(
            node_type_name
        )

        # Original node feature:
        #
        # one integer representing node type
        #

        x_values.append(
            [node_type_id]
        )

        node_type_ids.append(
            node_type_id
        )

        node_type_names.append(
            node_type_name
        )

        node_start.append(
            node["start"]
        )

        node_end.append(
            node["end"]
        )

        node_depth.append(
            node["depth"]
        )

        parent_index.append(
            node["parent"]
        )

        source_file_index.append(
            node["file_index"]
        )

    # --------------------------------------------------------------
    # Node feature tensor
    # --------------------------------------------------------------

    if len(x_values) > 0:

        x = torch.tensor(
            x_values,
            dtype=torch.long
        )

    else:

        x = torch.empty(
            (0, 1),
            dtype=torch.long
        )

    # --------------------------------------------------------------
    # Edge tensor
    # --------------------------------------------------------------

    if len(edges) > 0:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    return (
        x,
        edge_index,
        node_type_ids,
        node_type_names,
        node_start,
        node_end,
        node_depth,
        parent_index,
        source_file_index,
    )


# ======================================================================
# Main
# ======================================================================

def main():

    print()

    print("=" * 70)

    print(
        "SecureGAT-Agent AST Graph Construction"
    )

    print("=" * 70)

    print()

    print(
        "AST representation: SolidiFI `children` hierarchy"
    )

    print(
        "Execution device: CPU"
    )

    print()

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Vulnerability labels
    # --------------------------------------------------------------

    label_map = {

        vulnerability: index

        for index, vulnerability

        in enumerate(
            VULNERABILITIES
        )

    }

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    total_graphs = 0

    total_nodes = 0

    total_edges = 0

    report = {}

    # ==============================================================
    # Vulnerability loop
    # ==============================================================

    for vulnerability in VULNERABILITIES:

        print(
            vulnerability
        )

        print(
            "-" * 70
        )

        input_dir = os.path.join(
            AST_ROOT,
            vulnerability
        )

        output_dir = os.path.join(
            OUTPUT_ROOT,
            vulnerability
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        if not os.path.isdir(
            input_dir
        ):

            print(
                "WARNING: AST directory not found:",
                input_dir
            )

            continue

        # ----------------------------------------------------------
        # AST files
        # ----------------------------------------------------------

        ast_files = sorted(

            [

                filename

                for filename
                in os.listdir(input_dir)

                if filename.endswith(
                    "_ast.json"
                )

            ],

            key=lambda filename:
                contract_number_from_filename(
                    filename
                )
                or 999999

        )

        print(
            f"Contracts: {len(ast_files)}"
        )

        vulnerability_nodes = 0

        vulnerability_edges = 0

        contract_report = {}

        # ==========================================================
        # Contract loop
        # ==========================================================

        for ast_file in ast_files:

            contract_id = (
                contract_number_from_filename(
                    ast_file
                )
            )

            if contract_id is None:

                print(
                    "WARNING: cannot determine "
                    f"contract ID: {ast_file}"
                )

                continue

            ast_path = os.path.join(
                input_dir,
                ast_file
            )

            # ------------------------------------------------------
            # Load AST
            # ------------------------------------------------------

            try:

                with open(
                    ast_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    ast = json.load(f)

            except Exception as exc:

                print(
                    f"ERROR loading {ast_file}: "
                    f"{exc}"
                )

                continue

            # ------------------------------------------------------
            # Build graph
            # ------------------------------------------------------

            (
                x,
                edge_index,
                node_type_ids,
                node_type_names,
                node_start,
                node_end,
                node_depth,
                parent_index,
                source_file_index,

            ) = ast_to_graph(
                ast
            )

            num_nodes = len(
                node_type_names
            )

            num_edges = (
                edge_index.shape[1]
            )

            # ------------------------------------------------------
            # Create PyG graph
            # ------------------------------------------------------

            graph = Data(
                x=x,
                edge_index=edge_index
            )

            # ------------------------------------------------------
            # Node metadata
            # ------------------------------------------------------

            graph.node_type = torch.tensor(
                node_type_ids,
                dtype=torch.long
            )

            graph.node_type_name = (
                node_type_names
            )

            graph.node_start = torch.tensor(
                node_start,
                dtype=torch.long
            )

            graph.node_end = torch.tensor(
                node_end,
                dtype=torch.long
            )

            graph.node_depth = torch.tensor(
                node_depth,
                dtype=torch.long
            )

            graph.parent_index = torch.tensor(
                parent_index,
                dtype=torch.long
            )

            graph.source_file_index = (
                torch.tensor(
                    source_file_index,
                    dtype=torch.long
                )
            )

            # ------------------------------------------------------
            # Dataset metadata
            # ------------------------------------------------------

            graph.vulnerability = (
                vulnerability
            )

            graph.vulnerability_label = (
                label_map[vulnerability]
            )

            graph.contract_id = (
                contract_id
            )

            graph.ast_file = (
                ast_file
            )

            graph.num_nodes_total = (
                num_nodes
            )

            graph.num_edges_total = (
                num_edges
            )

            # ------------------------------------------------------
            # Save graph
            # ------------------------------------------------------

            output_path = os.path.join(
                output_dir,
                f"buggy_{contract_id}.pt"
            )

            torch.save(
                graph,
                output_path
            )

            # ------------------------------------------------------
            # Statistics
            # ------------------------------------------------------

            total_graphs += 1

            total_nodes += num_nodes

            total_edges += num_edges

            vulnerability_nodes += (
                num_nodes
            )

            vulnerability_edges += (
                num_edges
            )

            contract_report[
                str(contract_id)
            ] = {

                "ast_file":
                    ast_file,

                "graph_file":
                    f"buggy_{contract_id}.pt",

                "nodes":
                    num_nodes,

                "edges":
                    num_edges,

            }

        # ----------------------------------------------------------
        # Vulnerability report
        # ----------------------------------------------------------

        report[
            vulnerability
        ] = {

            "graphs":
                len(contract_report),

            "total_nodes":
                vulnerability_nodes,

            "total_edges":
                vulnerability_edges,

            "contracts":
                contract_report,

        }

        print()

    # ==============================================================
    # Node vocabulary
    # ==============================================================

    vocabulary = {

        node_type: node_id

        for node_type, node_id

        in NODE_TYPES.items()

    }

    vocabulary_path = os.path.join(
        OUTPUT_ROOT,
        "node_type_vocabulary.json"
    )

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            vocabulary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------------
    # Reverse vocabulary
    # --------------------------------------------------------------

    reverse_vocabulary = {

        str(node_id): node_type

        for node_type, node_id

        in NODE_TYPES.items()

    }

    reverse_vocabulary_path = os.path.join(
        OUTPUT_ROOT,
        "node_type_vocabulary_reverse.json"
    )

    with open(
        reverse_vocabulary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            reverse_vocabulary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ==============================================================
    # Global report
    # ==============================================================

    report["dataset"] = (
        "SolidiFI-benchmark"
    )

    report["total_graphs"] = (
        total_graphs
    )

    report["total_nodes"] = (
        total_nodes
    )

    report["total_edges"] = (
        total_edges
    )

    report["unique_ast_node_types"] = (
        len(NODE_TYPES)
    )

    report["execution_device"] = (
        "CPU"
    )

    report["ast_traversal"] = (
        "children_only"
    )

    report["metadata_excluded"] = True

    report_path = os.path.join(
        OUTPUT_ROOT,
        "graph_extraction_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ==============================================================
    # Final summary
    # ==============================================================

    print("=" * 70)

    print(
        "GRAPH EXTRACTION SUMMARY"
    )

    print("=" * 70)

    print(
        f"Total graphs: {total_graphs}"
    )

    print(
        f"Total AST nodes: {total_nodes}"
    )

    print(
        f"Total AST edges: {total_edges}"
    )

    print(
        f"Unique AST node types: "
        f"{len(NODE_TYPES)}"
    )

    print(
        "AST traversal: children only"
    )

    print(
        "Metadata dictionaries: excluded"
    )

    print(
        f"Saved to: {OUTPUT_ROOT}"
    )

    print(
        f"Vocabulary: {vocabulary_path}"
    )

    print(
        f"Report: {report_path}"
    )

    print("=" * 70)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    main()