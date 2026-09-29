#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 22:01:08 2026

@author: Willie
"""


"""
======================================================================
07 - SecureGAT-Agent Injection Annotation Quality Audit
======================================================================

Purpose
-------
Audit the quality of Step 06 injection-aware graph annotations.

Checks
------
1. Every graph contains annotation tensors.
2. Number of positive nodes.
3. Positive-node ratio.
4. Injection-to-node mapping size.
5. AST node-type distribution of positive nodes.
6. Vulnerability-specific statistics.
7. Suspiciously broad injections.
8. Zero-node injections.
9. Human-readable examples.

Input
-----
datasets/annotated_graph_dataset
datasets/SolidiFI-benchmark/buggy_contracts

Output
------
datasets/annotation_audit/

Files
-----
global_audit_report.json
graph_audit.csv
injection_audit.csv
node_type_distribution.csv
manual_examples.txt
"""

import os
import csv
import json
from collections import Counter, defaultdict

import torch


# ======================================================================
# Configuration
# ======================================================================

ANNOTATED_ROOT = "datasets/annotated_graph_dataset"

BUG_ROOT = (
    "datasets/SolidiFI-benchmark/"
    "buggy_contracts"
)

OUTPUT_ROOT = "datasets/annotation_audit"

VULNERABILITIES = [
    "Overflow-Underflow",
    "Re-entrancy",
    "TOD",
    "Timestamp-Dependency",
    "Unchecked-Send",
    "Unhandled-Exceptions",
    "tx.origin",
]

# Threshold for potentially broad injection mappings.
# We do NOT automatically declare them wrong.
BROAD_INJECTION_THRESHOLD = 25

# Number of human-readable examples per vulnerability.
EXAMPLES_PER_VULNERABILITY = 5


# ======================================================================
# Utility
# ======================================================================

def contract_number_from_graph(filename):
    """
    buggy_1.pt -> 1
    buggy_50.pt -> 50
    """

    base = os.path.basename(filename)

    if not base.startswith("buggy_"):
        return None

    try:
        number = base.split("_")[1]
        number = number.split(".")[0]
        return int(number)
    except Exception:
        return None


def normalize_node_type(value):
    """
    Convert graph node_type entry into a readable string.
    """

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace"
        )

    return str(value)


def load_source(vulnerability, contract_id):
    path = os.path.join(
        BUG_ROOT,
        vulnerability,
        f"buggy_{contract_id}.sol"
    )

    with open(path, "rb") as f:
        return f.read()


def build_line_offsets(source_bytes):

    offsets = [0]

    for i, byte in enumerate(source_bytes):
        if byte == 10:
            offsets.append(i + 1)

    return offsets


def source_interval(
    loc,
    length,
    line_offsets,
    source_length
):

    try:
        loc = int(loc)
        length = int(length)
    except Exception:
        return None

    if loc < 1:
        return None

    start_line = loc - 1

    if start_line >= len(line_offsets):
        return None

    start = line_offsets[start_line]

    end_line = start_line + length

    if end_line < len(line_offsets):
        end = line_offsets[end_line]
    else:
        end = source_length

    return start, end


def read_buglog(
    vulnerability,
    contract_id
):

    path = os.path.join(
        BUG_ROOT,
        vulnerability,
        f"BugLog_{contract_id}.csv"
    )

    injections = []

    if not os.path.exists(path):
        return injections

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):

            try:
                loc = int(row["loc"])
                length = int(row["length"])
            except Exception:
                continue

            injections.append({
                "id": row_index,
                "loc": loc,
                "length": length,
                "bug_type": row.get(
                    "bug type",
                    ""
                ),
                "approach": row.get(
                    "approach",
                    ""
                ),
            })

    return injections


def get_node_interval(
    graph,
    index
):

    try:
        start = int(graph.node_start[index])
        end = int(graph.node_end[index])
    except Exception:
        return None

    if start < 0:
        return None

    if end < start:
        return None

    return start, end


def intervals_overlap(
    node_start,
    node_end,
    injection_start,
    injection_end
):

    return (
        node_start < injection_end
        and
        node_end > injection_start
    )


# ======================================================================
# Audit one graph
# ======================================================================

def audit_graph(
    vulnerability,
    graph_file
):

    contract_id = contract_number_from_graph(
        graph_file
    )

    graph_path = os.path.join(
        ANNOTATED_ROOT,
        vulnerability,
        graph_file
    )

    graph = torch.load(
        graph_path,
        map_location="cpu"
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    required_attributes = [
        "injection_label",
        "injection_id",
        "num_injections",
        "num_matched_injections",
        "num_unmatched_injections",
        "node_type",
        "node_start",
        "node_end",
    ]

    missing = [
        attr
        for attr in required_attributes
        if not hasattr(graph, attr)
    ]

    if missing:
        raise RuntimeError(
            f"{vulnerability}/{graph_file}: "
            f"missing attributes: {missing}"
        )

    num_nodes = int(graph.num_nodes)

    labels = graph.injection_label
    injection_ids = graph.injection_id

    positive_indices = (
        torch.where(labels == 1)[0]
        .tolist()
    )

    num_positive = len(
        positive_indices
    )

    positive_ratio = (
        num_positive / num_nodes
        if num_nodes > 0
        else 0.0
    )

    # --------------------------------------------------------------
    # Positive node types
    # --------------------------------------------------------------

    positive_types = Counter()

    for node_index in positive_indices:

        node_type = normalize_node_type(
            graph.node_type[node_index]
        )

        positive_types[node_type] += 1

    # --------------------------------------------------------------
    # Reconstruct injection -> node mapping
    # --------------------------------------------------------------

    source_bytes = load_source(
        vulnerability,
        contract_id
    )

    line_offsets = build_line_offsets(
        source_bytes
    )

    source_length = len(source_bytes)

    injections = read_buglog(
        vulnerability,
        contract_id
    )

    injection_records = []

    broad_injections = []

    zero_node_injections = []

    for injection in injections:

        interval = source_interval(
            injection["loc"],
            injection["length"],
            line_offsets,
            source_length
        )

        if interval is None:

            zero_node_injections.append(
                injection["id"]
            )

            injection_records.append({
                "injection_id":
                    injection["id"],
                "loc":
                    injection["loc"],
                "length":
                    injection["length"],
                "matched_nodes": [],
                "matched_node_count": 0,
                "node_types": [],
            })

            continue

        injection_start, injection_end = interval

        matched_nodes = []

        node_types = Counter()

        for node_index in range(num_nodes):

            node_interval = get_node_interval(
                graph,
                node_index
            )

            if node_interval is None:
                continue

            node_start, node_end = node_interval

            if intervals_overlap(
                node_start,
                node_end,
                injection_start,
                injection_end
            ):

                matched_nodes.append(
                    node_index
                )

                node_type = normalize_node_type(
                    graph.node_type[node_index]
                )

                node_types[node_type] += 1

        matched_count = len(
            matched_nodes
        )

        if matched_count == 0:
            zero_node_injections.append(
                injection["id"]
            )

        if (
            matched_count
            >= BROAD_INJECTION_THRESHOLD
        ):

            broad_injections.append({
                "injection_id":
                    injection["id"],
                "loc":
                    injection["loc"],
                "length":
                    injection["length"],
                "matched_nodes":
                    matched_count,
                "node_types":
                    dict(node_types),
            })

        injection_records.append({
            "injection_id":
                injection["id"],
            "loc":
                injection["loc"],
            "length":
                injection["length"],
            "matched_nodes":
                matched_nodes,
            "matched_node_count":
                matched_count,
            "node_types":
                dict(node_types),
        })

    # --------------------------------------------------------------
    # Return result
    # --------------------------------------------------------------

    return {
        "vulnerability":
            vulnerability,

        "contract_id":
            contract_id,

        "graph":
            graph_file,

        "num_nodes":
            num_nodes,

        "positive_nodes":
            num_positive,

        "positive_ratio":
            positive_ratio,

        "num_injections":
            len(injections),

        "annotated_num_injections":
            int(graph.num_injections),

        "annotated_matched_injections":
            int(graph.num_matched_injections),

        "annotated_unmatched_injections":
            int(graph.num_unmatched_injections),

        "positive_node_types":
            dict(positive_types),

        "injection_records":
            injection_records,

        "zero_node_injections":
            zero_node_injections,

        "broad_injections":
            broad_injections,
    }


# ======================================================================
# Main
# ======================================================================

def main():

    print()
    print("=" * 78)
    print(
        "SecureGAT-Agent Step 07 "
        "Injection Annotation Quality Audit"
    )
    print("=" * 78)
    print()

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    graph_rows = []
    injection_rows = []

    global_node_types = Counter()
    vulnerability_node_types = defaultdict(
        Counter
    )

    vulnerability_summary = {}

    total_graphs = 0
    total_injections = 0
    total_positive_nodes = 0
    total_nodes = 0
    total_zero_node_injections = 0
    total_broad_injections = 0

    manual_examples = []

    # ==================================================================
    # Vulnerability loop
    # ==================================================================

    for vulnerability in VULNERABILITIES:

        print(vulnerability)
        print("-" * 70)

        directory = os.path.join(
            ANNOTATED_ROOT,
            vulnerability
        )

        if not os.path.isdir(directory):

            print(
                f"WARNING: missing directory: "
                f"{directory}"
            )

            continue

        graph_files = sorted(
            [
                f
                for f in os.listdir(directory)
                if f.endswith(".pt")
            ],
            key=lambda x:
                contract_number_from_graph(x)
                or 999999
        )

        vulnerability_graphs = 0
        vulnerability_injections = 0
        vulnerability_positive_nodes = 0
        vulnerability_nodes = 0
        vulnerability_zero = 0
        vulnerability_broad = 0

        example_count = 0

        for graph_file in graph_files:

            result = audit_graph(
                vulnerability,
                graph_file
            )

            total_graphs += 1
            vulnerability_graphs += 1

            total_injections += (
                result["num_injections"]
            )

            vulnerability_injections += (
                result["num_injections"]
            )

            total_positive_nodes += (
                result["positive_nodes"]
            )

            vulnerability_positive_nodes += (
                result["positive_nodes"]
            )

            total_nodes += (
                result["num_nodes"]
            )

            vulnerability_nodes += (
                result["num_nodes"]
            )

            zero_count = len(
                result["zero_node_injections"]
            )

            broad_count = len(
                result["broad_injections"]
            )

            total_zero_node_injections += (
                zero_count
            )

            vulnerability_zero += (
                zero_count
            )

            total_broad_injections += (
                broad_count
            )

            vulnerability_broad += (
                broad_count
            )

            # ----------------------------------------------------------
            # Graph CSV row
            # ----------------------------------------------------------

            graph_rows.append({
                "vulnerability":
                    vulnerability,

                "contract_id":
                    result["contract_id"],

                "graph":
                    result["graph"],

                "num_nodes":
                    result["num_nodes"],

                "positive_nodes":
                    result["positive_nodes"],

                "positive_ratio":
                    result["positive_ratio"],

                "num_injections":
                    result["num_injections"],

                "zero_node_injections":
                    zero_count,

                "broad_injections":
                    broad_count,
            })

            # ----------------------------------------------------------
            # Node type statistics
            # ----------------------------------------------------------

            for (
                node_type,
                count
            ) in result[
                "positive_node_types"
            ].items():

                global_node_types[
                    node_type
                ] += count

                vulnerability_node_types[
                    vulnerability
                ][node_type] += count

            # ----------------------------------------------------------
            # Injection rows
            # ----------------------------------------------------------

            for record in result[
                "injection_records"
            ]:

                injection_rows.append({
                    "vulnerability":
                        vulnerability,

                    "contract_id":
                        result["contract_id"],

                    "graph":
                        result["graph"],

                    "injection_id":
                        record["injection_id"],

                    "loc":
                        record["loc"],

                    "length":
                        record["length"],

                    "matched_node_count":
                        record[
                            "matched_node_count"
                        ],

                    "node_types":
                        json.dumps(
                            record["node_types"],
                            sort_keys=True
                        ),
                })

                # ------------------------------------------------------
                # Human-readable examples
                # ------------------------------------------------------

                if (
                    example_count
                    < EXAMPLES_PER_VULNERABILITY
                    and
                    record["matched_node_count"]
                    > 0
                ):

                    manual_examples.append(
                        {
                            "vulnerability":
                                vulnerability,

                            "contract":
                                result["contract_id"],

                            "graph":
                                result["graph"],

                            "injection_id":
                                record[
                                    "injection_id"
                                ],

                            "loc":
                                record["loc"],

                            "length":
                                record["length"],

                            "matched_nodes":
                                record[
                                    "matched_nodes"
                                ],

                            "node_types":
                                record[
                                    "node_types"
                                ],
                        }
                    )

                    example_count += 1

        # --------------------------------------------------------------
        # Vulnerability summary
        # --------------------------------------------------------------

        positive_ratio = (
            vulnerability_positive_nodes
            / vulnerability_nodes
            if vulnerability_nodes > 0
            else 0.0
        )

        vulnerability_summary[
            vulnerability
        ] = {
            "graphs":
                vulnerability_graphs,

            "injections":
                vulnerability_injections,

            "positive_nodes":
                vulnerability_positive_nodes,

            "total_nodes":
                vulnerability_nodes,

            "positive_node_ratio":
                positive_ratio,

            "zero_node_injections":
                vulnerability_zero,

            "broad_injections":
                vulnerability_broad,

            "node_types":
                dict(
                    vulnerability_node_types[
                        vulnerability
                    ]
                ),
        }

        print(
            f"Graphs: {vulnerability_graphs}"
        )

        print(
            f"Injections: "
            f"{vulnerability_injections}"
        )

        print(
            f"Positive nodes: "
            f"{vulnerability_positive_nodes}"
        )

        print(
            f"Positive-node ratio: "
            f"{positive_ratio:.6f}"
        )

        print(
            f"Zero-node injections: "
            f"{vulnerability_zero}"
        )

        print(
            f"Broad injections >= "
            f"{BROAD_INJECTION_THRESHOLD} nodes: "
            f"{vulnerability_broad}"
        )

        print()

    # ==================================================================
    # Write graph CSV
    # ==================================================================

    graph_csv = os.path.join(
        OUTPUT_ROOT,
        "graph_audit.csv"
    )

    with open(
        graph_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vulnerability",
                "contract_id",
                "graph",
                "num_nodes",
                "positive_nodes",
                "positive_ratio",
                "num_injections",
                "zero_node_injections",
                "broad_injections",
            ]
        )

        writer.writeheader()
        writer.writerows(graph_rows)

    # ==================================================================
    # Write injection CSV
    # ==================================================================

    injection_csv = os.path.join(
        OUTPUT_ROOT,
        "injection_audit.csv"
    )

    with open(
        injection_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vulnerability",
                "contract_id",
                "graph",
                "injection_id",
                "loc",
                "length",
                "matched_node_count",
                "node_types",
            ]
        )

        writer.writeheader()
        writer.writerows(
            injection_rows
        )

    # ==================================================================
    # Node-type CSV
    # ==================================================================

    node_type_csv = os.path.join(
        OUTPUT_ROOT,
        "node_type_distribution.csv"
    )

    with open(
        node_type_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "vulnerability",
            "node_type",
            "count",
        ])

        for vulnerability in VULNERABILITIES:

            for (
                node_type,
                count
            ) in sorted(
                vulnerability_node_types[
                    vulnerability
                ].items(),
                key=lambda x: -x[1]
            ):

                writer.writerow([
                    vulnerability,
                    node_type,
                    count,
                ])

    # ==================================================================
    # Human-readable examples
    # ==================================================================

    examples_path = os.path.join(
        OUTPUT_ROOT,
        "manual_examples.txt"
    )

    with open(
        examples_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SecureGAT-Agent Step 07 "
            "Manual Annotation Examples\n"
        )

        f.write("=" * 78 + "\n\n")

        for example in manual_examples:

            f.write(
                f"Vulnerability: "
                f"{example['vulnerability']}\n"
            )

            f.write(
                f"Contract: "
                f"buggy_{example['contract']}\n"
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
                f"Matched nodes: "
                f"{example['matched_nodes']}\n"
            )

            f.write(
                f"Node types: "
                f"{example['node_types']}\n"
            )

            f.write("-" * 78 + "\n")

    # ==================================================================
    # Global report
    # ==================================================================

    global_positive_ratio = (
        total_positive_nodes
        / total_nodes
        if total_nodes > 0
        else 0.0
    )

    report = {

        "dataset":
            "SolidiFI-benchmark",

        "step":
            "07_annotation_quality_audit",

        "graphs":
            total_graphs,

        "total_injections":
            total_injections,

        "total_nodes":
            total_nodes,

        "total_positive_nodes":
            total_positive_nodes,

        "global_positive_node_ratio":
            global_positive_ratio,

        "zero_node_injections":
            total_zero_node_injections,

        "broad_injections":
            total_broad_injections,

        "broad_injection_threshold":
            BROAD_INJECTION_THRESHOLD,

        "global_node_type_distribution":
            dict(global_node_types),

        "vulnerabilities":
            vulnerability_summary,

        "files": {
            "graph_audit":
                "graph_audit.csv",

            "injection_audit":
                "injection_audit.csv",

            "node_type_distribution":
                "node_type_distribution.csv",

            "manual_examples":
                "manual_examples.txt",
        }
    }

    report_path = os.path.join(
        OUTPUT_ROOT,
        "global_audit_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ==================================================================
    # Final summary
    # ==================================================================

    print("=" * 78)
    print(
        "STEP 07 AUDIT SUMMARY"
    )
    print("=" * 78)

    print(
        f"Graphs: "
        f"{total_graphs}"
    )

    print(
        f"Total injections: "
        f"{total_injections}"
    )

    print(
        f"Total AST nodes: "
        f"{total_nodes}"
    )

    print(
        f"Positive AST nodes: "
        f"{total_positive_nodes}"
    )

    print(
        f"Global positive-node ratio: "
        f"{global_positive_ratio:.6f}"
    )

    print(
        f"Zero-node injections: "
        f"{total_zero_node_injections}"
    )

    print(
        f"Broad injections: "
        f"{total_broad_injections}"
    )

    print()
    print(
        f"Report: {report_path}"
    )

    print(
        f"Graph audit: {graph_csv}"
    )

    print(
        f"Injection audit: {injection_csv}"
    )

    print(
        f"Node types: {node_type_csv}"
    )

    print(
        f"Examples: {examples_path}"
    )

    print()


if __name__ == "__main__":
    main()