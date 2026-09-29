#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 23:15:45 2026

@author: Willie
"""


"""
======================================================================
SecureGAT-Agent: Ancestor-Aware Semantic Injection Annotation Audit
======================================================================

Purpose
-------
Audit the semantic quality of the injection annotations.


    selection_method = contained_innermost

Therefore the directly annotated nodes are often leaf/deep AST nodes
such as:

    Identifier
    Literal
    ElementaryTypeName
    ParameterList

while semantically meaningful structures may occur in their ancestor
chain:

    Identifier
        |
    BinaryOperation
        |
    ExpressionStatement
        |
    Block
        |
    FunctionDefinition

This script therefore measures BOTH:

1. Direct coverage
   ----------------
   Is the expected structural AST node itself directly annotated?

2. Ancestor coverage
   ------------------
   Does the expected structural AST node occur anywhere in the
   ancestor chain of a directly annotated node?

3. Semantic coverage
   ------------------
   Union of direct and ancestor coverage.

IMPORTANT
---------
This script does NOT modify:

    datasets/annotated_graph_dataset/**/*.pt

======================================================================
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


# ======================================================================
# Configuration
# ======================================================================

ANNOTATED_ROOT = Path(
    "datasets/annotated_graph_dataset"
)

AUDIT_CSV = (
    ANNOTATED_ROOT /
    "injection_audit.csv"
)

OUTPUT_JSON = (
    ANNOTATED_ROOT /
    "semantic_annotation_audit.json"
)

OUTPUT_TXT = (
    ANNOTATED_ROOT /
    "semantic_annotation_audit.txt"
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
# Expected semantic structures
# ======================================================================
#
# These are audit targets only.
# They DO NOT alter the Step 06 labels.
#
# ======================================================================

EXPECTED_STRUCTURES = {

    "Overflow-Underflow": [
        "BinaryOperation",
        "Assignment",
        "ExpressionStatement",
        "UnaryOperation",
    ],

    "Re-entrancy": [
        "FunctionCall",
        "MemberAccess",
        "ExpressionStatement",
        "Assignment",
        "BinaryOperation",
    ],

    "TOD": [
        "FunctionCall",
        "MemberAccess",
        "ExpressionStatement",
        "Assignment",
        "IfStatement",
    ],

    "Timestamp-Dependency": [
        "Identifier",
        "MemberAccess",
        "FunctionCall",
        "BinaryOperation",
        "IfStatement",
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
# CPU enforcement
# ======================================================================

DEVICE = torch.device("cpu")


# ======================================================================
# Utility
# ======================================================================

def safe_int(value, default=-1):
    """
    Safely convert tensor/scalar/string values to int.
    """

    try:
        if torch.is_tensor(value):
            return int(value.item())

        return int(value)

    except Exception:
        return default


# ======================================================================
# Load injection audit
# ======================================================================

def load_injection_audit():
    """
    Load Step 06 injection_audit.csv.

    Returns
    -------
    list of dictionaries
    """

    if not AUDIT_CSV.exists():
        raise FileNotFoundError(
            f"Injection audit not found: {AUDIT_CSV}"
        )

    records = []

    with open(
        AUDIT_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                vulnerability = row[
                    "vulnerability"
                ]

                contract_id = int(
                    row["contract_id"]
                )

                graph = row["graph"]

                injection_id = int(
                    row["injection_id"]
                )

                loc = int(
                    row["loc"]
                )

                length = int(
                    row["length"]
                )

                matched_nodes = json.loads(
                    row["matched_nodes"]
                )

                if not isinstance(
                    matched_nodes,
                    list
                ):
                    matched_nodes = []

            except Exception as exc:

                print(
                    "WARNING: skipping malformed "
                    f"audit row: {exc}"
                )

                continue

            records.append({
                "vulnerability":
                    vulnerability,

                "contract_id":
                    contract_id,

                "graph":
                    graph,

                "injection_id":
                    injection_id,

                "loc":
                    loc,

                "length":
                    length,

                "bug_type":
                    row.get(
                        "bug_type",
                        ""
                    ),

                "approach":
                    row.get(
                        "approach",
                        ""
                    ),

                "matched":
                    row.get(
                        "matched",
                        ""
                    ).lower() == "true",

                "selection_method":
                    row.get(
                        "selection_method",
                        ""
                    ),

                "matched_nodes":
                    [
                        int(x)
                        for x in matched_nodes
                    ],
            })

    return records


# ======================================================================
# Exact graph path
# ======================================================================

def graph_path_for_record(record):
    """
    Resolve:

        annotated_graph_dataset/
            <vulnerability>/
                buggy_N.pt
    """

    return (
        ANNOTATED_ROOT
        / record["vulnerability"]
        / record["graph"]
    )


# ======================================================================
# Load graph
# ======================================================================

def load_graph(path):
    """
    Load graph strictly on CPU.
    """

    if not path.exists():
        return None

    return torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )


# ======================================================================
# Node type extraction
# ======================================================================

def get_node_type_name(graph, node_index):
    """
    Return the human-readable AST node type.

    Preferred source:

        graph.node_type_name

    Fallback:

        graph.node_type

    """

    try:

        if hasattr(
            graph,
            "node_type_name"
        ):

            value = graph.node_type_name[
                node_index
            ]

            if isinstance(
                value,
                str
            ):
                return value

            if hasattr(
                value,
                "item"
            ):

                value = value.item()

            return str(value)

    except Exception:
        pass

    try:

        value = graph.node_type[
            node_index
        ]

        if isinstance(
            value,
            str
        ):
            return value

        if hasattr(
            value,
            "item"
        ):

            value = value.item()

        return str(value)

    except Exception:

        return "Unknown"


# ======================================================================
# Parent lookup
# ======================================================================

def get_parent_index(graph, node_index):
    """
    Return the parent AST node.

    The clean Step 05 graphs contain:

        parent_index

    Root has:

        -1
    """

    try:

        return safe_int(
            graph.parent_index[
                node_index
            ],
            default=-1
        )

    except Exception:

        return -1


# ======================================================================
# Build ancestor chain
# ======================================================================

def get_ancestor_chain(
    graph,
    node_index,
    max_depth=1000
):
    """
    Return ancestor node indices.

    Example:

        node
          ↓
        parent
          ↓
        parent
          ↓
        root

    Returned list excludes the original node.

    """

    ancestors = []

    current = node_index

    visited = set()

    for _ in range(max_depth):

        if current in visited:
            break

        visited.add(current)

        parent = get_parent_index(
            graph,
            current
        )

        if parent < 0:
            break

        ancestors.append(parent)

        current = parent

    return ancestors


# ======================================================================
# Build node metadata
# ======================================================================

def build_graph_metadata(graph):
    """
    Precompute:

        node type
        parent
        ancestors
    """

    num_nodes = int(
        graph.num_nodes
    )

    node_types = []
    parents = []
    ancestors = []

    for i in range(num_nodes):

        node_types.append(
            get_node_type_name(
                graph,
                i
            )
        )

        parents.append(
            get_parent_index(
                graph,
                i
            )
        )

    for i in range(num_nodes):

        ancestors.append(
            get_ancestor_chain(
                graph,
                i
            )
        )

    return {
        "num_nodes":
            num_nodes,

        "node_types":
            node_types,

        "parents":
            parents,

        "ancestors":
            ancestors,
    }


# ======================================================================
# Analyze one injection
# ======================================================================

def analyze_injection(
    graph_metadata,
    matched_nodes,
    expected_structures
):
    """
    Analyze one injection.

    Returns
    -------
    dict
    """

    node_types = (
        graph_metadata["node_types"]
    )

    ancestors = (
        graph_metadata["ancestors"]
    )

    direct_types = Counter()

    ancestor_types = Counter()

    all_context_types = Counter()

    direct_node_indices = []

    ancestor_node_indices = []

    # --------------------------------------------------------------
    # Direct nodes
    # --------------------------------------------------------------

    for node_index in matched_nodes:

        if (
            node_index < 0
            or node_index >= len(node_types)
        ):
            continue

        node_type = node_types[
            node_index
        ]

        direct_types[node_type] += 1

        all_context_types[
            node_type
        ] += 1

        direct_node_indices.append(
            node_index
        )

    # --------------------------------------------------------------
    # Ancestors
    # --------------------------------------------------------------

    seen_ancestor_nodes = set()

    for node_index in matched_nodes:

        if (
            node_index < 0
            or node_index >= len(ancestors)
        ):
            continue

        for ancestor_index in ancestors[
            node_index
        ]:

            if ancestor_index in seen_ancestor_nodes:
                continue

            seen_ancestor_nodes.add(
                ancestor_index
            )

            if (
                ancestor_index < 0
                or ancestor_index >= len(node_types)
            ):
                continue

            ancestor_type = node_types[
                ancestor_index
            ]

            ancestor_types[
                ancestor_type
            ] += 1

            all_context_types[
                ancestor_type
            ] += 1

            ancestor_node_indices.append(
                ancestor_index
            )

    # --------------------------------------------------------------
    # Coverage
    # --------------------------------------------------------------

    direct_coverage = {}
    ancestor_coverage = {}
    semantic_coverage = {}

    for structure in expected_structures:

        direct_hit = (
            direct_types[structure] > 0
        )

        ancestor_hit = (
            ancestor_types[structure] > 0
        )

        semantic_hit = (
            direct_hit
            or ancestor_hit
        )

        direct_coverage[
            structure
        ] = direct_hit

        ancestor_coverage[
            structure
        ] = ancestor_hit

        semantic_coverage[
            structure
        ] = semantic_hit

    return {

        "direct_types":
            dict(direct_types),

        "ancestor_types":
            dict(ancestor_types),

        "context_types":
            dict(all_context_types),

        "direct_node_indices":
            direct_node_indices,

        "ancestor_node_indices":
            sorted(
                set(
                    ancestor_node_indices
                )
            ),

        "direct_coverage":
            direct_coverage,

        "ancestor_coverage":
            ancestor_coverage,

        "semantic_coverage":
            semantic_coverage,
    }


# ======================================================================
# Initialize vulnerability statistics
# ======================================================================

def initialize_vulnerability_statistics(
    vulnerability
):

    structures = EXPECTED_STRUCTURES[
        vulnerability
    ]

    return {

        "injections":
            0,

        "matched_injections":
            0,

        "direct_coverage":
            {
                s: 0
                for s in structures
            },

        "ancestor_coverage":
            {
                s: 0
                for s in structures
            },

        "semantic_coverage":
            {
                s: 0
                for s in structures
            },

        "direct_node_type_distribution":
            Counter(),

        "ancestor_node_type_distribution":
            Counter(),

        "semantic_node_type_distribution":
            Counter(),

        "examples":
            [],

        "missing_graphs":
            [],

    }


# ======================================================================
# Analyze vulnerability
# ======================================================================

def analyze_vulnerability(
    vulnerability,
    records
):

    structures = EXPECTED_STRUCTURES[
        vulnerability
    ]

    statistics = (
        initialize_vulnerability_statistics(
            vulnerability
        )
    )

    graph_cache = {}

    vulnerability_records = [
        r
        for r in records
        if r["vulnerability"]
        == vulnerability
    ]

    for record in vulnerability_records:

        statistics["injections"] += 1

        graph_path = graph_path_for_record(
            record
        )

        cache_key = str(
            graph_path
        )

        if cache_key not in graph_cache:

            graph = load_graph(
                graph_path
            )

            if graph is None:

                graph_cache[
                    cache_key
                ] = None

                statistics[
                    "missing_graphs"
                ].append(
                    str(graph_path)
                )

            else:

                graph_cache[
                    cache_key
                ] = build_graph_metadata(
                    graph
                )

        metadata = graph_cache[
            cache_key
        ]

        if metadata is None:
            continue

        if record["matched"]:

            statistics[
                "matched_injections"
            ] += 1

        analysis = analyze_injection(
            metadata,
            record["matched_nodes"],
            structures
        )

        # ----------------------------------------------------------
        # Aggregate direct coverage
        # ----------------------------------------------------------

        for structure in structures:

            if analysis[
                "direct_coverage"
            ][structure]:

                statistics[
                    "direct_coverage"
                ][structure] += 1

        # ----------------------------------------------------------
        # Aggregate ancestor coverage
        # ----------------------------------------------------------

        for structure in structures:

            if analysis[
                "ancestor_coverage"
            ][structure]:

                statistics[
                    "ancestor_coverage"
                ][structure] += 1

        # ----------------------------------------------------------
        # Aggregate semantic coverage
        # ----------------------------------------------------------

        for structure in structures:

            if analysis[
                "semantic_coverage"
            ][structure]:

                statistics[
                    "semantic_coverage"
                ][structure] += 1

        # ----------------------------------------------------------
        # Node-type distributions
        # ----------------------------------------------------------

        statistics[
            "direct_node_type_distribution"
        ].update(
            analysis[
                "direct_types"
            ]
        )

        statistics[
            "ancestor_node_type_distribution"
        ].update(
            analysis[
                "ancestor_types"
            ]
        )

        statistics[
            "semantic_node_type_distribution"
        ].update(
            analysis[
                "context_types"
            ]
        )

        # ----------------------------------------------------------
        # Save first examples
        # ----------------------------------------------------------

        if len(
            statistics["examples"]
        ) < 10:

            statistics[
                "examples"
            ].append({

                "contract_id":
                    record[
                        "contract_id"
                    ],

                "graph":
                    record[
                        "graph"
                    ],

                "injection_id":
                    record[
                        "injection_id"
                    ],

                "loc":
                    record[
                        "loc"
                    ],

                "length":
                    record[
                        "length"
                    ],

                "selection_method":
                    record[
                        "selection_method"
                    ],

                "matched_nodes":
                    record[
                        "matched_nodes"
                    ],

                "direct_types":
                    analysis[
                        "direct_types"
                    ],

                "ancestor_types":
                    analysis[
                        "ancestor_types"
                    ],

                "direct_coverage":
                    analysis[
                        "direct_coverage"
                    ],

                "ancestor_coverage":
                    analysis[
                        "ancestor_coverage"
                    ],

                "semantic_coverage":
                    analysis[
                        "semantic_coverage"
                    ],
            })

    return statistics


# ======================================================================
# Convert statistics to JSON-safe form
# ======================================================================

def statistics_to_json(
    statistics
):

    result = {

        "injections":
            statistics[
                "injections"
            ],

        "matched_injections":
            statistics[
                "matched_injections"
            ],

        "direct_coverage":
            statistics[
                "direct_coverage"
            ],

        "ancestor_coverage":
            statistics[
                "ancestor_coverage"
            ],

        "semantic_coverage":
            statistics[
                "semantic_coverage"
            ],

        "direct_node_type_distribution":
            dict(
                statistics[
                    "direct_node_type_distribution"
                ]
            ),

        "ancestor_node_type_distribution":
            dict(
                statistics[
                    "ancestor_node_type_distribution"
                ]
            ),

        "semantic_node_type_distribution":
            dict(
                statistics[
                    "semantic_node_type_distribution"
                ]
            ),

        "examples":
            statistics[
                "examples"
            ],

        "missing_graphs":
            statistics[
                "missing_graphs"
            ],
    }

    return result


# ======================================================================
# Percentage
# ======================================================================

def percentage(
    value,
    total
):

    if total <= 0:
        return 0.0

    return (
        value /
        total *
        100.0
    )


# ======================================================================
# Print vulnerability result
# ======================================================================

def print_vulnerability_result(
    vulnerability,
    statistics
):

    structures = EXPECTED_STRUCTURES[
        vulnerability
    ]

    total = statistics[
        "injections"
    ]

    print()
    print("=" * 70)
    print(
        f"VULNERABILITY: {vulnerability}"
    )
    print("=" * 70)

    print(
        f"Injections: {total}"
    )

    print()
    print(
        "DIRECT NODE COVERAGE"
    )

    print(
        "-" * 70
    )

    for structure in structures:

        count = statistics[
            "direct_coverage"
        ][structure]

        rate = percentage(
            count,
            total
        )

        print(
            f"  {structure:30s} "
            f"{count:4d}/{total:<4d} "
            f"({rate:6.2f}%)"
        )

    print()
    print(
        "ANCESTOR STRUCTURAL COVERAGE"
    )

    print(
        "-" * 70
    )

    for structure in structures:

        count = statistics[
            "ancestor_coverage"
        ][structure]

        rate = percentage(
            count,
            total
        )

        print(
            f"  {structure:30s} "
            f"{count:4d}/{total:<4d} "
            f"({rate:6.2f}%)"
        )

    print()
    print(
        "SEMANTIC COVERAGE "
        "(DIRECT OR ANCESTOR)"
    )

    print(
        "-" * 70
    )

    for structure in structures:

        count = statistics[
            "semantic_coverage"
        ][structure]

        rate = percentage(
            count,
            total
        )

        print(
            f"  {structure:30s} "
            f"{count:4d}/{total:<4d} "
            f"({rate:6.2f}%)"
        )

    print()
    print(
        "DIRECT NODE TYPES"
    )

    print(
        "-" * 70
    )

    direct_distribution = (
        statistics[
            "direct_node_type_distribution"
        ]
    )

    for node_type, count in (
        direct_distribution
        .most_common()
    ):

        print(
            f"  {node_type:30s} "
            f"{count}"
        )

    print()
    print(
        "ANCESTOR NODE TYPES"
    )

    print(
        "-" * 70
    )

    ancestor_distribution = (
        statistics[
            "ancestor_node_type_distribution"
        ]
    )

    for node_type, count in (
        ancestor_distribution
        .most_common()
    ):

        print(
            f"  {node_type:30s} "
            f"{count}"
        )

    if statistics[
        "missing_graphs"
    ]:

        print()
        print(
            "WARNING: missing graphs: "
            f"{len(statistics['missing_graphs'])}"
        )


# ======================================================================
# Write text report
# ======================================================================

def write_text_report(
    report
):

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            "SecureGAT-Agent Step 06.1\n"
        )

        f.write(
            "Ancestor-Aware Semantic "
            "Injection Annotation Audit\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "This is an audit only. "
            "Step 06 annotations were not modified.\n"
        )

        f.write(
            "Annotation strategy: "
            "contained_innermost\n"
        )

        f.write(
            "Execution device: CPU\n\n"
        )

        for vulnerability in (
            VULNERABILITIES
        ):

            statistics = (
                report[
                    "vulnerabilities"
                ][vulnerability]
            )

            structures = (
                EXPECTED_STRUCTURES[
                    vulnerability
                ]
            )

            total = statistics[
                "injections"
            ]

            f.write(
                "=" * 70
                + "\n"
            )

            f.write(
                f"VULNERABILITY: "
                f"{vulnerability}\n"
            )

            f.write(
                "=" * 70
                + "\n"
            )

            f.write(
                f"Injections: {total}\n\n"
            )

            f.write(
                "DIRECT NODE COVERAGE\n"
            )

            f.write(
                "-" * 70
                + "\n"
            )

            for structure in structures:

                count = statistics[
                    "direct_coverage"
                ][structure]

                rate = percentage(
                    count,
                    total
                )

                f.write(
                    f"  {structure:30s} "
                    f"{count:4d}/{total:<4d} "
                    f"({rate:6.2f}%)\n"
                )

            f.write("\n")

            f.write(
                "ANCESTOR STRUCTURAL COVERAGE\n"
            )

            f.write(
                "-" * 70
                + "\n"
            )

            for structure in structures:

                count = statistics[
                    "ancestor_coverage"
                ][structure]

                rate = percentage(
                    count,
                    total
                )

                f.write(
                    f"  {structure:30s} "
                    f"{count:4d}/{total:<4d} "
                    f"({rate:6.2f}%)\n"
                )

            f.write("\n")

            f.write(
                "SEMANTIC COVERAGE "
                "(DIRECT OR ANCESTOR)\n"
            )

            f.write(
                "-" * 70
                + "\n"
            )

            for structure in structures:

                count = statistics[
                    "semantic_coverage"
                ][structure]

                rate = percentage(
                    count,
                    total
                )

                f.write(
                    f"  {structure:30s} "
                    f"{count:4d}/{total:<4d} "
                    f"({rate:6.2f}%)\n"
                )

            f.write("\n")

            f.write(
                "FIRST 10 INJECTION EXAMPLES\n"
            )

            f.write(
                "-" * 70
                + "\n"
            )

            for example in (
                statistics["examples"]
            ):

                f.write(
                    f"\nContract: "
                    f"{example['contract_id']}\n"
                )

                f.write(
                    f"Graph: "
                    f"{example['graph']}\n"
                )

                f.write(
                    f"Injection ID: "
                    f"{example['injection_id']}\n"
                )

                f.write(
                    f"BugLog location: "
                    f"line {example['loc']} "
                    f"(length={example['length']})\n"
                )

                f.write(
                    f"Selection method: "
                    f"{example['selection_method']}\n"
                )

                f.write(
                    "Direct node types:\n"
                )

                for (
                    node_type,
                    count
                ) in sorted(
                    example[
                        "direct_types"
                    ].items()
                ):

                    f.write(
                        f"  {node_type}: "
                        f"{count}\n"
                    )

                f.write(
                    "Ancestor node types:\n"
                )

                for (
                    node_type,
                    count
                ) in sorted(
                    example[
                        "ancestor_types"
                    ].items()
                ):

                    f.write(
                        f"  {node_type}: "
                        f"{count}\n"
                    )

                f.write(
                    "Semantic coverage:\n"
                )

                for (
                    structure,
                    covered
                ) in example[
                    "semantic_coverage"
                ].items():

                    f.write(
                        f"  {structure}: "
                        f"{covered}\n"
                    )

                f.write("\n")


# ======================================================================
# Main
# ======================================================================

def main():

    print()
    print("=" * 70)
    print(
        "SecureGAT-Agent Step 06.1"
    )
    print(
        "Ancestor-Aware Semantic "
        "Injection Annotation Audit"
    )
    print("=" * 70)
    print()

    print(
        "Execution device: CPU"
    )

    print(
        "Step 06 annotations: NOT modified"
    )

    print(
        f"Loading injection audit:\n"
        f"{AUDIT_CSV}"
    )

    records = load_injection_audit()

    print(
        f"Injections loaded: "
        f"{len(records)}"
    )

    print()

    # --------------------------------------------------------------
    # Analyze all vulnerabilities
    # --------------------------------------------------------------

    vulnerability_results = {}

    for vulnerability in (
        VULNERABILITIES
    ):

        result = analyze_vulnerability(
            vulnerability,
            records
        )

        vulnerability_results[
            vulnerability
        ] = statistics_to_json(
            result
        )

        print_vulnerability_result(
            vulnerability,
            result
        )

    # --------------------------------------------------------------
    # Global summary
    # --------------------------------------------------------------

    total_injections = len(
        records
    )

    total_matched = sum(
        result[
            "matched_injections"
        ]
        for result in (
            vulnerability_results
            .values()
        )
    )

    global_direct = defaultdict(int)
    global_ancestor = defaultdict(int)
    global_semantic = defaultdict(int)

    for vulnerability in (
        VULNERABILITIES
    ):

        result = vulnerability_results[
            vulnerability
        ]

        for structure in (
            EXPECTED_STRUCTURES[
                vulnerability
            ]
        ):

            global_direct[
                f"{vulnerability}:{structure}"
            ] = result[
                "direct_coverage"
            ][structure]

            global_ancestor[
                f"{vulnerability}:{structure}"
            ] = result[
                "ancestor_coverage"
            ][structure]

            global_semantic[
                f"{vulnerability}:{structure}"
            ] = result[
                "semantic_coverage"
            ][structure]

    report = {

        "dataset":
            "SolidiFI-benchmark",

        "step":
            "06.1_ancestor_aware_semantic_audit",

        "execution_device":
            "cpu",

        "annotation_strategy":
            "contained_innermost",

        "step_06_modified":
            False,

        "injections_loaded":
            total_injections,

        "matched_injections":
            total_matched,

        "matching_rate":
            percentage(
                total_matched,
                total_injections
            ),

        "methodology": {

            "direct_coverage":
                "Expected AST type is directly "
                "included among Step 06 matched nodes.",

            "ancestor_coverage":
                "Expected AST type occurs in the "
                "ancestor chain of a Step 06 matched node.",

            "semantic_coverage":
                "Union of direct and ancestor coverage.",

        },

        "vulnerabilities":
            vulnerability_results,

        "global": {

            "direct_coverage":
                dict(global_direct),

            "ancestor_coverage":
                dict(global_ancestor),

            "semantic_coverage":
                dict(global_semantic),

        },
    }

    # --------------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------------
    # Save text report
    # --------------------------------------------------------------

    write_text_report(
        report
    )

    # --------------------------------------------------------------
    # Final output
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 06.1 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Injections loaded: "
        f"{total_injections}"
    )

    print(
        f"Matched injections: "
        f"{total_matched}"
    )

    print(
        f"Matching rate: "
        f"{percentage(total_matched, total_injections):.2f} %"
    )

    print(
        f"JSON report: "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Text report: "
        f"{OUTPUT_TXT}"
    )

    print("=" * 70)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    main()