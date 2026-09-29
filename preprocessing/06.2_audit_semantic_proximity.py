#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 23:50:00 2026

@author: Willie
"""


"""
======================================================================
SecureGAT-Agent: Semantic Proximity / Nearest Vulnerability AST Construct Audit
======================================================================

Purpose
-------
Audit the semantic relationship between the currently annotated
injection nodes and their AST ancestors.

IMPORTANT:
    This script determines, for every BugLog injection:

    1. Positive/selected AST nodes
    2. Ancestor chains
    3. Nearest vulnerability-relevant AST construct
    4. Distance from selected node to semantic construct
    5. Whether the semantic construct is directly selected
    6. Whether it is recovered only through ancestry

Input
-----
    datasets/annotated_graph_dataset/

Required:
    injection_audit.csv
    vulnerability/buggy_N.pt

Output
------
    datasets/annotated_graph_dataset/
        semantic_proximity_audit.csv
        semantic_proximity_report.json
        semantic_proximity_audit.txt
        semantic_proximity_examples.txt

Annotation policy
-----------------

The audit only evaluates the existing annotation protocol.

======================================================================
"""

import os
import csv
import json
import ast
from collections import Counter, defaultdict

import torch


# ======================================================================
# Configuration
# ======================================================================

ANNOTATED_ROOT = "datasets/annotated_graph_dataset"

INJECTION_AUDIT = os.path.join(
    ANNOTATED_ROOT,
    "injection_audit.csv"
)

OUTPUT_CSV = os.path.join(
    ANNOTATED_ROOT,
    "semantic_proximity_audit.csv"
)

OUTPUT_JSON = os.path.join(
    ANNOTATED_ROOT,
    "semantic_proximity_report.json"
)

OUTPUT_TXT = os.path.join(
    ANNOTATED_ROOT,
    "semantic_proximity_audit.txt"
)

OUTPUT_EXAMPLES = os.path.join(
    ANNOTATED_ROOT,
    "semantic_proximity_examples.txt"
)


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
# Vulnerability-specific semantic constructs
# ======================================================================

SEMANTIC_TYPES = {

    "Overflow-Underflow": [
        "BinaryOperation",
        "Assignment",
        "UnaryOperation",
        "ExpressionStatement",
    ],

    "Re-entrancy": [
        "FunctionCall",
        "MemberAccess",
        "Assignment",
        "BinaryOperation",
        "ExpressionStatement",
    ],

    "TOD": [
        "FunctionCall",
        "MemberAccess",
        "Assignment",
        "BinaryOperation",
        "IfStatement",
        "ExpressionStatement",
    ],


    "Timestamp-Dependency": [
        "Identifier",
        "MemberAccess",
        "FunctionCall",
        "BinaryOperation",
        "IfStatement",
        "VariableDeclaration",
        "FunctionDefinition",    
        
    ],

    "Unchecked-Send": [
        "FunctionCall",
        "MemberAccess",
        "ExpressionStatement",
        "Assignment",
    ],

    "Unhandled-Exceptions": [
        "FunctionCall",
        "MemberAccess",
        "ExpressionStatement",
    ],

    "tx.origin": [
        "MemberAccess",
        "Identifier",
        "FunctionCall",
        "BinaryOperation",
        "IfStatement",
    ],
}


# ======================================================================
# Broader contextual constructs
# ======================================================================

CONTEXT_TYPES = {
    "ExpressionStatement",
    "Block",
    "FunctionDefinition",
    "ContractDefinition",
    "SourceUnit",
    "VariableDeclarationStatement",
    "ParameterList",
    "Return",
    "IfStatement",
}


# ======================================================================
# Utilities
# ======================================================================

def safe_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def parse_node_list(value):
    """
    Parse matched_nodes column.

    Example:
        "[198, 199, 203]"

    Returns:
        [198, 199, 203]
    """

    if value is None:
        return []

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                int(x)
                for x in parsed
                if isinstance(x, (int, float))
            ]

    except Exception:
        pass

    return []


def tensor_to_list(value):
    """
    Convert tensor-like graph attribute to Python list.
    """

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, list):
        return value

    return list(value)


# ======================================================================
# Load injection audit
# ======================================================================

def load_injection_audit():

    if not os.path.exists(INJECTION_AUDIT):
        raise FileNotFoundError(
            f"Injection audit not found:\n{INJECTION_AUDIT}"
        )

    rows = []

    with open(
        INJECTION_AUDIT,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            row["contract_id"] = safe_int(
                row.get("contract_id")
            )

            row["injection_id"] = safe_int(
                row.get("injection_id")
            )

            row["buglog_row"] = safe_int(
                row.get("buglog_row")
            )

            row["loc"] = safe_int(
                row.get("loc")
            )

            row["length"] = safe_int(
                row.get("length")
            )

            row["matched_nodes_list"] = parse_node_list(
                row.get("matched_nodes")
            )

            rows.append(row)

    return rows


# ======================================================================
# Load graph
# ======================================================================

def load_graph(
    vulnerability,
    graph_filename
):

    path = os.path.join(
        ANNOTATED_ROOT,
        vulnerability,
        graph_filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu"
    )


# ======================================================================
# Build graph node information
# ======================================================================

def build_node_information(graph):

    node_types = tensor_to_list(
        graph.node_type_name
    )

    parent_index = tensor_to_list(
        graph.parent_index
    )

    node_start = tensor_to_list(
        graph.node_start
    )

    node_end = tensor_to_list(
        graph.node_end
    )

    num_nodes = len(node_types)

    nodes = {}

    for i in range(num_nodes):

        nodes[i] = {
            "index": i,
            "type": str(node_types[i]),
            "parent": safe_int(
                parent_index[i],
                -1
            ),
            "start": safe_int(
                node_start[i],
                -1
            ),
            "end": safe_int(
                node_end[i],
                -1
            ),
        }

    return nodes


# ======================================================================
# Ancestor chain
# ======================================================================

def get_ancestor_chain(
    node_id,
    nodes,
    max_depth=1000
):

    chain = []

    current = node_id
    visited = set()

    while (
        current is not None
        and current >= 0
        and current in nodes
        and current not in visited
        and len(chain) < max_depth
    ):

        visited.add(current)

        parent = nodes[current]["parent"]

        if parent < 0:
            break

        chain.append(parent)

        current = parent

    return chain


# ======================================================================
# Find semantic ancestors
# ======================================================================

def find_semantic_ancestors(
    node_id,
    nodes,
    semantic_types
):

    chain = get_ancestor_chain(
        node_id,
        nodes
    )

    results = []

    for distance, ancestor_id in enumerate(
        chain,
        start=1
    ):

        ancestor = nodes[ancestor_id]

        if ancestor["type"] in semantic_types:

            results.append({
                "node_id": ancestor_id,
                "type": ancestor["type"],
                "distance": distance,
                "start": ancestor["start"],
                "end": ancestor["end"],
            })

    return results


# ======================================================================
# Find nearest semantic construct
# ======================================================================

def nearest_semantic_construct(
    matched_nodes,
    nodes,
    semantic_types
):

    candidates = []

    for positive_node in matched_nodes:

        if positive_node not in nodes:
            continue

        ancestors = find_semantic_ancestors(
            positive_node,
            nodes,
            semantic_types
        )

        for candidate in ancestors:

            candidates.append({
                "positive_node":
                    positive_node,

                "node_id":
                    candidate["node_id"],

                "type":
                    candidate["type"],

                "distance":
                    candidate["distance"],

                "start":
                    candidate["start"],

                "end":
                    candidate["end"],
            })

    if not candidates:
        return None, []

    candidates.sort(
        key=lambda x: (
            x["distance"],
            x["node_id"]
        )
    )

    return candidates[0], candidates


# ======================================================================
# Direct semantic node check
# ======================================================================

def direct_semantic_nodes(
    matched_nodes,
    nodes,
    semantic_types
):

    result = []

    for node_id in matched_nodes:

        if node_id not in nodes:
            continue

        node = nodes[node_id]

        if node["type"] in semantic_types:

            result.append({
                "node_id":
                    node_id,

                "type":
                    node["type"],

                "start":
                    node["start"],

                "end":
                    node["end"],
            })

    return result


# ======================================================================
# Full injection analysis
# ======================================================================

def analyze_injection(
    row,
    graph,
    nodes
):

    vulnerability = row["vulnerability"]

    semantic_types = set(
        SEMANTIC_TYPES[
            vulnerability
        ]
    )

    matched_nodes = row[
        "matched_nodes_list"
    ]

    # --------------------------------------------------------------
    # Direct semantic nodes
    # --------------------------------------------------------------

    direct_nodes = direct_semantic_nodes(
        matched_nodes,
        nodes,
        semantic_types
    )

    # --------------------------------------------------------------
    # Nearest semantic ancestor
    # --------------------------------------------------------------

    nearest, all_candidates = (
        nearest_semantic_construct(
            matched_nodes,
            nodes,
            semantic_types
        )
    )

    # --------------------------------------------------------------
    # Ancestor type counts
    # --------------------------------------------------------------

    ancestor_types = Counter()

    for positive_node in matched_nodes:

        chain = get_ancestor_chain(
            positive_node,
            nodes
        )

        for ancestor_id in chain:

            ancestor_types[
                nodes[ancestor_id]["type"]
            ] += 1

    # --------------------------------------------------------------
    # All semantic constructs reached
    # --------------------------------------------------------------

    semantic_reached = Counter()

    for candidate in all_candidates:

        semantic_reached[
            candidate["type"]
        ] += 1

    # --------------------------------------------------------------
    # Determine semantic relationship
    # --------------------------------------------------------------

    if direct_nodes:

        relation = "direct"

    elif nearest is not None:

        relation = "ancestor"

    else:

        relation = "none"

    # --------------------------------------------------------------
    # Distance
    # --------------------------------------------------------------

    if nearest is not None:
        nearest_distance = nearest[
            "distance"
        ]
    else:
        nearest_distance = -1

    # --------------------------------------------------------------
    # Semantic type
    # --------------------------------------------------------------

    nearest_type = (
        nearest["type"]
        if nearest is not None
        else ""
    )

    nearest_node_id = (
        nearest["node_id"]
        if nearest is not None
        else -1
    )

    # --------------------------------------------------------------
    # Direct semantic types
    # --------------------------------------------------------------

    direct_types = sorted(
        set(
            x["type"]
            for x in direct_nodes
        )
    )

    # --------------------------------------------------------------
    # Semantic types reached
    # --------------------------------------------------------------

    semantic_types_reached = sorted(
        semantic_reached.keys()
    )

    # --------------------------------------------------------------
    # Output record
    # --------------------------------------------------------------

    result = {
        "vulnerability":
            vulnerability,

        "contract_id":
            row["contract_id"],

        "graph":
            row["graph"],

        "injection_id":
            row["injection_id"],

        "buglog_row":
            row.get("buglog_row", -1),

        "loc":
            row["loc"],

        "length":
            row["length"],

        "matched_node_count":
            len(matched_nodes),

        "direct_semantic_node_count":
            len(direct_nodes),

        "direct_semantic_types":
            json.dumps(
                direct_types
            ),

        "nearest_semantic_type":
            nearest_type,

        "nearest_semantic_node":
            nearest_node_id,

        "nearest_semantic_distance":
            nearest_distance,

        "semantic_relation":
            relation,

        "semantic_types_reached":
            json.dumps(
                semantic_types_reached
            ),

        "ancestor_types":
            json.dumps(
                dict(
                    ancestor_types
                )
            ),
    }

    return result


# ======================================================================
# Vulnerability report
# ======================================================================

def summarize_vulnerability(
    rows
):

    total = len(rows)

    direct = sum(
        1
        for row in rows
        if row["semantic_relation"]
        == "direct"
    )

    ancestor = sum(
        1
        for row in rows
        if row["semantic_relation"]
        == "ancestor"
    )

    none = sum(
        1
        for row in rows
        if row["semantic_relation"]
        == "none"
    )

    distance_counter = Counter()

    nearest_type_counter = Counter()

    for row in rows:

        distance = row[
            "nearest_semantic_distance"
        ]

        if distance >= 0:
            distance_counter[
                str(distance)
            ] += 1

        nearest_type = row[
            "nearest_semantic_type"
        ]

        if nearest_type:
            nearest_type_counter[
                nearest_type
            ] += 1

    return {
        "injections":
            total,

        "direct_semantic":
            direct,

        "ancestor_semantic":
            ancestor,

        "no_semantic_construct":
            none,

        "direct_rate":
            (
                direct / total * 100
                if total
                else 0.0
            ),

        "ancestor_rate":
            (
                ancestor / total * 100
                if total
                else 0.0
            ),

        "no_semantic_rate":
            (
                none / total * 100
                if total
                else 0.0
            ),

        "nearest_distance_distribution":
            dict(
                sorted(
                    distance_counter.items(),
                    key=lambda x:
                        int(x[0])
                )
            ),

        "nearest_semantic_type_distribution":
            dict(
                nearest_type_counter
            ),
    }


# ======================================================================
# Write CSV
# ======================================================================

def write_csv(rows):

    fieldnames = [
        "vulnerability",
        "contract_id",
        "graph",
        "injection_id",
        "buglog_row",
        "loc",
        "length",
        "matched_node_count",
        "direct_semantic_node_count",
        "direct_semantic_types",
        "nearest_semantic_type",
        "nearest_semantic_node",
        "nearest_semantic_distance",
        "semantic_relation",
        "semantic_types_reached",
        "ancestor_types",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ======================================================================
# Write text report
# ======================================================================

def write_text_report(
    global_report
):

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "SecureGAT-Agent Step 06.2\n"
        )

        f.write(
            "Semantic Proximity / Nearest AST Construct Audit\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Step 06 annotations were NOT modified.\n\n"
        )

        f.write(
            f"Total injections: "
            f"{global_report['total_injections']}\n"
        )

        f.write(
            f"Direct semantic: "
            f"{global_report['direct_semantic']}\n"
        )

        f.write(
            f"Ancestor semantic: "
            f"{global_report['ancestor_semantic']}\n"
        )

        f.write(
            f"No semantic construct: "
            f"{global_report['no_semantic_construct']}\n\n"
        )

        for vulnerability in VULNERABILITIES:

            report = global_report[
                "vulnerabilities"
            ][vulnerability]

            f.write(
                "=" * 70 + "\n"
            )

            f.write(
                vulnerability + "\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            f.write(
                f"Injections: "
                f"{report['injections']}\n"
            )

            f.write(
                f"Direct semantic: "
                f"{report['direct_semantic']} "
                f"({report['direct_rate']:.2f}%)\n"
            )

            f.write(
                f"Ancestor semantic: "
                f"{report['ancestor_semantic']} "
                f"({report['ancestor_rate']:.2f}%)\n"
            )

            f.write(
                f"No semantic construct: "
                f"{report['no_semantic_construct']} "
                f"({report['no_semantic_rate']:.2f}%)\n\n"
            )

            f.write(
                "Nearest semantic construct:\n"
            )

            for (
                semantic_type,
                count
            ) in sorted(
                report[
                    "nearest_semantic_type_distribution"
                ].items(),
                key=lambda x: (-x[1], x[0])
            ):

                f.write(
                    f"  {semantic_type:<30}"
                    f"{count}\n"
                )

            f.write("\n")

            f.write(
                "Nearest semantic distance:\n"
            )

            for (
                distance,
                count
            ) in report[
                "nearest_distance_distribution"
            ].items():

                f.write(
                    f"  distance={distance:<5}"
                    f"{count}\n"
                )

            f.write("\n")


# ======================================================================
# Write examples
# ======================================================================

def write_examples(
    rows,
    max_per_vulnerability=5
):

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["vulnerability"]
        ].append(row)

    with open(
        OUTPUT_EXAMPLES,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SecureGAT-Agent Step 06.2\n"
        )

        f.write(
            "Semantic Proximity Examples\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        for vulnerability in VULNERABILITIES:

            f.write(
                f"VULNERABILITY: "
                f"{vulnerability}\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            examples = grouped[
                vulnerability
            ][
                :max_per_vulnerability
            ]

            for row in examples:

                f.write(
                    f"Contract: "
                    f"{row['contract_id']}\n"
                )

                f.write(
                    f"Injection: "
                    f"{row['injection_id']}\n"
                )

                f.write(
                    f"BugLog location: "
                    f"{row['loc']} "
                    f"(length={row['length']})\n"
                )

                f.write(
                    f"Matched nodes: "
                    f"{row['matched_node_count']}\n"
                )

                f.write(
                    f"Direct semantic nodes: "
                    f"{row['direct_semantic_node_count']}\n"
                )

                f.write(
                    f"Direct semantic types: "
                    f"{row['direct_semantic_types']}\n"
                )

                f.write(
                    f"Nearest semantic type: "
                    f"{row['nearest_semantic_type']}\n"
                )

                f.write(
                    f"Nearest semantic node: "
                    f"{row['nearest_semantic_node']}\n"
                )

                f.write(
                    f"Nearest semantic distance: "
                    f"{row['nearest_semantic_distance']}\n"
                )

                f.write(
                    f"Relationship: "
                    f"{row['semantic_relation']}\n"
                )

                f.write(
                    "\n"
                )

            f.write("\n")


# ======================================================================
# Main
# ======================================================================

def main():

    print()
    print("=" * 70)
    print(
        "SecureGAT-Agent Step 06.2"
    )
    print(
        "Semantic Proximity / Nearest AST Construct Audit"
    )
    print("=" * 70)
    print()

    print(
        "Execution device: CPU"
    )

    print(
        "Step 06 annotations: NOT modified"
    )

    print()

    print(
        "Loading injection audit:"
    )

    print(
        INJECTION_AUDIT
    )

    rows = load_injection_audit()

    print(
        f"Injections loaded: "
        f"{len(rows)}"
    )

    print()

    # --------------------------------------------------------------
    # Graph cache
    # --------------------------------------------------------------

    graph_cache = {}

    analyzed_rows = []

    vulnerability_rows = defaultdict(list)

    # --------------------------------------------------------------
    # Process injections
    # --------------------------------------------------------------

    for index, row in enumerate(rows):

        vulnerability = row[
            "vulnerability"
        ]

        graph_filename = row[
            "graph"
        ]

        cache_key = (
            vulnerability,
            graph_filename
        )

        if cache_key not in graph_cache:

            graph = load_graph(
                vulnerability,
                graph_filename
            )

            graph_cache[
                cache_key
            ] = build_node_information(
                graph
            )

        nodes = graph_cache[
            cache_key
        ]

        result = analyze_injection(
            row,
            None,
            nodes
        )

        analyzed_rows.append(
            result
        )

        vulnerability_rows[
            vulnerability
        ].append(
            result
        )

        # Progress every 1000 injections

        if (
            (index + 1) % 1000 == 0
            or index + 1 == len(rows)
        ):

            print(
                f"Processed: "
                f"{index + 1}/"
                f"{len(rows)}"
            )

    # --------------------------------------------------------------
    # Global summary
    # --------------------------------------------------------------

    total = len(
        analyzed_rows
    )

    direct = sum(
        1
        for row in analyzed_rows
        if row["semantic_relation"]
        == "direct"
    )

    ancestor = sum(
        1
        for row in analyzed_rows
        if row["semantic_relation"]
        == "ancestor"
    )

    none = sum(
        1
        for row in analyzed_rows
        if row["semantic_relation"]
        == "none"
    )

    global_report = {

        "dataset":
            "SolidiFI-benchmark",

        "step":
            "06.2_semantic_proximity_audit",

        "annotations_modified":
            False,

        "total_injections":
            total,

        "direct_semantic":
            direct,

        "ancestor_semantic":
            ancestor,

        "no_semantic_construct":
            none,

        "direct_rate":
            (
                direct / total * 100
                if total
                else 0.0
            ),

        "ancestor_rate":
            (
                ancestor / total * 100
                if total
                else 0.0
            ),

        "no_semantic_rate":
            (
                none / total * 100
                if total
                else 0.0
            ),

        "vulnerabilities": {}
    }

    # --------------------------------------------------------------
    # Vulnerability summaries
    # --------------------------------------------------------------

    for vulnerability in VULNERABILITIES:

        vuln_rows = vulnerability_rows[
            vulnerability
        ]

        report = summarize_vulnerability(
            vuln_rows
        )

        global_report[
            "vulnerabilities"
        ][
            vulnerability
        ] = report

        print()
        print(
            vulnerability
        )
        print(
            "-" * 70
        )

        print(
            f"Injections: "
            f"{report['injections']}"
        )

        print(
            f"Direct semantic: "
            f"{report['direct_semantic']} "
            f"({report['direct_rate']:.2f}%)"
        )

        print(
            f"Ancestor semantic: "
            f"{report['ancestor_semantic']} "
            f"({report['ancestor_rate']:.2f}%)"
        )

        print(
            f"No semantic construct: "
            f"{report['no_semantic_construct']} "
            f"({report['no_semantic_rate']:.2f}%)"
        )

        print(
            "Nearest semantic construct:"
        )

        for (
            semantic_type,
            count
        ) in sorted(
            report[
                "nearest_semantic_type_distribution"
            ].items(),
            key=lambda x: (-x[1], x[0])
        ):

            print(
                f"  {semantic_type:<30}"
                f"{count}"
            )

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------

    write_csv(
        analyzed_rows
    )

    write_text_report(
        global_report
    )

    write_examples(
        analyzed_rows
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            global_report,
            f,
            indent=2
        )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 06.2 SUMMARY"
    )
    print("=" * 70)

    print(
        f"Total injections: "
        f"{total}"
    )

    print(
        f"Direct semantic: "
        f"{direct}"
    )

    print(
        f"Ancestor semantic: "
        f"{ancestor}"
    )

    print(
        f"No semantic construct: "
        f"{none}"
    )

    print(
        f"Direct rate: "
        f"{global_report['direct_rate']:.2f} %"
    )

    print(
        f"Ancestor rate: "
        f"{global_report['ancestor_rate']:.2f} %"
    )

    print(
        f"No-semantic rate: "
        f"{global_report['no_semantic_rate']:.2f} %"
    )

    print()

    print(
        f"CSV: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"JSON: "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Text report: "
        f"{OUTPUT_TXT}"
    )

    print(
        f"Examples: "
        f"{OUTPUT_EXAMPLES}"
    )

    print()

    print(
        "STEP 06.2 COMPLETE"
    )

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()