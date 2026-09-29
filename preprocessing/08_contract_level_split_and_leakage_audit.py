#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 01:50:13 2026

@author: Willie
"""


"""
======================================================================
SecureGAT-Agent: Contract-Level Train/Validation/Test Split and Leakage Audit
======================================================================

Purpose
-------
Construct a deterministic contract-level dataset split and verify that
no Solidity contract appears in more than one dataset partition.

Dataset structure
-----------------
350 graph records
7 vulnerability classes
50 contract IDs
7 vulnerability graphs per contract

Split
-----
35 contracts -> train
 7 contracts -> validation
 8 contracts -> test

Therefore:

245 graph records -> train
 49 graph records -> validation
 56 graph records -> test

Total = 350 graph records

IMPORTANT GRAPH IDENTITY RULE
------------------------------
A graph is identified by:

    (vulnerability, graph filename)

because buggy_1.pt exists once inside each vulnerability directory.

Therefore:

    Overflow-Underflow/buggy_1.pt
    Re-entrancy/buggy_1.pt

are different graph records.

Contract leakage is determined ONLY by contract_id.

======================================================================
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET_ROOT = Path(
    "datasets/annotated_graph_dataset"
)

OUTPUT_SPLIT_CSV = (
    DATASET_ROOT /
    "contract_level_splits.csv"
)

OUTPUT_LEAKAGE_CSV = (
    DATASET_ROOT /
    "split_leakage_audit.csv"
)

OUTPUT_JSON = (
    DATASET_ROOT /
    "split_audit_report.json"
)

OUTPUT_TEXT = (
    DATASET_ROOT /
    "split_audit_report.txt"
)

RANDOM_SEED = 20260912

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.14
TEST_RATIO = 0.16

EXPECTED_VULNERABILITIES = [
    "Overflow-Underflow",
    "Re-entrancy",
    "TOD",
    "Timestamp-Dependency",
    "Unchecked-Send",
    "Unhandled-Exceptions",
    "tx.origin",
]


# ======================================================================
# DISPLAY
# ======================================================================

def banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


# ======================================================================
# GRAPH DISCOVERY
# ======================================================================

def discover_graphs():

    graphs = []

    for vulnerability in EXPECTED_VULNERABILITIES:

        vulnerability_dir = (
            DATASET_ROOT / vulnerability
        )

        if not vulnerability_dir.exists():
            raise FileNotFoundError(
                f"Missing vulnerability directory: "
                f"{vulnerability_dir}"
            )

        for graph_path in sorted(
            vulnerability_dir.glob("*.pt")
        ):

            graphs.append(
                {
                    "vulnerability": vulnerability,
                    "graph": graph_path.name,
                    "graph_path": str(graph_path),
                    "graph_key": (
                        vulnerability,
                        graph_path.name,
                    ),
                }
            )

    return graphs


# ======================================================================
# CONTRACT ID
# ======================================================================

def extract_contract_id(
    graph_name: str
) -> int:

    stem = Path(
        graph_name
    ).stem

    if not stem.startswith("buggy_"):
        raise ValueError(
            f"Unexpected graph filename: "
            f"{graph_name}"
        )

    suffix = stem[
        len("buggy_"):
    ]

    try:
        return int(suffix)

    except ValueError as exc:

        raise ValueError(
            f"Cannot extract contract ID from "
            f"{graph_name}"
        ) from exc


# ======================================================================
# DATASET STRUCTURE AUDIT
# ======================================================================

def audit_dataset_structure(
    graphs
):

    vulnerability_counts = Counter()

    contract_vulnerability = (
        defaultdict(set)
    )

    contract_graphs = (
        defaultdict(list)
    )

    graph_keys = set()

    errors = []

    for record in graphs:

        vulnerability = (
            record["vulnerability"]
        )

        graph = record["graph"]

        graph_key = (
            vulnerability,
            graph,
        )

        contract_id = (
            extract_contract_id(graph)
        )

        record["contract_id"] = (
            contract_id
        )

        vulnerability_counts[
            vulnerability
        ] += 1

        contract_vulnerability[
            contract_id
        ].add(vulnerability)

        contract_graphs[
            contract_id
        ].append(record)

        if graph_key in graph_keys:

            errors.append(
                "Duplicate graph record: "
                f"{vulnerability}/{graph}"
            )

        graph_keys.add(
            graph_key
        )

    # --------------------------------------------------------------
    # Vulnerability counts
    # --------------------------------------------------------------

    for vulnerability in (
        EXPECTED_VULNERABILITIES
    ):

        count = (
            vulnerability_counts[
                vulnerability
            ]
        )

        if count != 50:

            errors.append(
                f"{vulnerability}: expected "
                f"50 graphs, found {count}"
            )

    # --------------------------------------------------------------
    # Contract IDs
    # --------------------------------------------------------------

    contract_ids = sorted(
        contract_vulnerability.keys()
    )

    expected_contract_ids = list(
        range(1, 51)
    )

    if contract_ids != (
        expected_contract_ids
    ):

        errors.append(
            "Expected contract IDs 1..50, "
            f"found {contract_ids}"
        )

    # --------------------------------------------------------------
    # Every contract must contain all seven
    # vulnerability variants
    # --------------------------------------------------------------

    expected_vulnerabilities = set(
        EXPECTED_VULNERABILITIES
    )

    for contract_id in contract_ids:

        observed = (
            contract_vulnerability[
                contract_id
            ]
        )

        missing = (
            expected_vulnerabilities
            - observed
        )

        extra = (
            observed
            - expected_vulnerabilities
        )

        if missing:

            errors.append(
                f"Contract {contract_id}: "
                f"missing vulnerabilities "
                f"{sorted(missing)}"
            )

        if extra:

            errors.append(
                f"Contract {contract_id}: "
                f"unexpected vulnerabilities "
                f"{sorted(extra)}"
            )

    # --------------------------------------------------------------
    # Seven graph records per contract
    # --------------------------------------------------------------

    for contract_id in contract_ids:

        count = len(
            contract_graphs[
                contract_id
            ]
        )

        if count != 7:

            errors.append(
                f"Contract {contract_id}: "
                f"expected 7 graph records, "
                f"found {count}"
            )

    return {
        "vulnerability_counts":
            dict(vulnerability_counts),

        "contract_ids":
            contract_ids,

        "contract_vulnerability": {
            str(k): sorted(v)
            for k, v in (
                contract_vulnerability.items()
            )
        },

        "contract_graphs":
            contract_graphs,

        "errors":
            errors,
    }


# ======================================================================
# CONTRACT SPLIT
# ======================================================================

def construct_split(
    contract_ids
):

    contract_ids = sorted(
        contract_ids
    )

    rng = random.Random(
        RANDOM_SEED
    )

    shuffled = list(
        contract_ids
    )

    rng.shuffle(
        shuffled
    )

    total = len(
        shuffled
    )

    train_count = int(
        round(
            total *
            TRAIN_RATIO
        )
    )

    validation_count = int(
        round(
            total *
            VALIDATION_RATIO
        )
    )

    test_count = (
        total
        - train_count
        - validation_count
    )

    if (
        train_count
        + validation_count
        + test_count
        != total
    ):

        raise RuntimeError(
            "Split counts do not sum to "
            "the total number of contracts."
        )

    train_ids = sorted(
        shuffled[
            :train_count
        ]
    )

    validation_ids = sorted(
        shuffled[
            train_count:
            train_count
            + validation_count
        ]
    )

    test_ids = sorted(
        shuffled[
            train_count
            + validation_count:
        ]
    )

    split_by_contract = {}

    for contract_id in train_ids:

        split_by_contract[
            contract_id
        ] = "train"

    for contract_id in validation_ids:

        split_by_contract[
            contract_id
        ] = "validation"

    for contract_id in test_ids:

        split_by_contract[
            contract_id
        ] = "test"

    return (
        split_by_contract,
        train_ids,
        validation_ids,
        test_ids,
    )


# ======================================================================
# ASSIGN GRAPH RECORDS
# ======================================================================

def assign_graph_splits(
    graphs,
    split_by_contract
):

    records = []

    for record in graphs:

        contract_id = (
            record["contract_id"]
        )

        if contract_id not in (
            split_by_contract
        ):

            raise RuntimeError(
                f"Contract {contract_id} "
                f"has no split assignment."
            )

        output = dict(
            record
        )

        output["split"] = (
            split_by_contract[
                contract_id
            ]
        )

        records.append(
            output
        )

    return records


# ======================================================================
# SPLIT AUDIT
# ======================================================================

def audit_splits(
    records
):

    errors = []

    # --------------------------------------------------------------
    # Split -> contracts
    # --------------------------------------------------------------

    split_contracts = (
        defaultdict(set)
    )

    # --------------------------------------------------------------
    # Split -> UNIQUE GRAPH RECORDS
    #
    # IMPORTANT:
    # graph identity is (vulnerability, graph)
    # --------------------------------------------------------------

    split_graphs = (
        defaultdict(set)
    )

    # --------------------------------------------------------------
    # Vulnerability distribution
    # --------------------------------------------------------------

    vulnerability_split_counts = (
        defaultdict(Counter)
    )

    # --------------------------------------------------------------
    # Contract -> splits
    # --------------------------------------------------------------

    contract_split_counts = (
        defaultdict(Counter)
    )

    # --------------------------------------------------------------
    # Graph identity tracking
    # --------------------------------------------------------------

    graph_seen = {}

    for record in records:

        contract_id = (
            record["contract_id"]
        )

        vulnerability = (
            record["vulnerability"]
        )

        graph = (
            record["graph"]
        )

        split = (
            record["split"]
        )

        graph_key = (
            vulnerability,
            graph,
        )

        split_contracts[
            split
        ].add(
            contract_id
        )

        split_graphs[
            split
        ].add(
            graph_key
        )

        vulnerability_split_counts[
            split
        ][
            vulnerability
        ] += 1

        contract_split_counts[
            contract_id
        ][
            split
        ] += 1

        # ----------------------------------------------------------
        # A graph record may occur only once.
        # ----------------------------------------------------------

        if graph_key in graph_seen:

            previous_split = (
                graph_seen[
                    graph_key
                ]
            )

            errors.append(
                "DUPLICATE GRAPH RECORD: "
                f"{vulnerability}/{graph} "
                f"appears in "
                f"{previous_split} and {split}"
            )

        else:

            graph_seen[
                graph_key
            ] = split

    # ==================================================================
    # CONTRACT LEAKAGE
    # ==================================================================

    contract_leakage = []

    for (
        contract_id,
        splits
    ) in contract_split_counts.items():

        observed_splits = sorted(
            splits.keys()
        )

        if len(
            observed_splits
        ) > 1:

            contract_leakage.append(
                {
                    "contract_id":
                        contract_id,

                    "splits":
                        observed_splits,
                }
            )

            errors.append(
                "CONTRACT LEAKAGE: "
                f"contract {contract_id} "
                f"appears in "
                f"{observed_splits}"
            )

    # ==================================================================
    # GRAPH DUPLICATION
    # ==================================================================

    graph_leakage = []

    graph_record_counts = Counter()

    for record in records:

        graph_key = (
            record["vulnerability"],
            record["graph"],
        )

        graph_record_counts[
            graph_key
        ] += 1

    for (
        graph_key,
        occurrences
    ) in graph_record_counts.items():

        if occurrences > 1:

            vulnerability, graph = (
                graph_key
            )

            graph_leakage.append(
                {
                    "vulnerability":
                        vulnerability,

                    "graph":
                        graph,

                    "occurrences":
                        occurrences,
                }
            )

    # ==================================================================
    # EXPECTED SPLIT COUNTS
    # ==================================================================

    expected_contract_counts = {
        "train": 35,
        "validation": 7,
        "test": 8,
    }

    expected_graph_counts = {
        "train": 245,
        "validation": 49,
        "test": 56,
    }

    actual_contract_counts = {
        split: len(
            split_contracts[
                split
            ]
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    actual_graph_counts = {
        split: len(
            split_graphs[
                split
            ]
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    for split in (
        "train",
        "validation",
        "test",
    ):

        if (
            actual_contract_counts[
                split
            ]
            != expected_contract_counts[
                split
            ]
        ):

            errors.append(
                f"{split}: expected "
                f"{expected_contract_counts[split]} "
                f"contracts, found "
                f"{actual_contract_counts[split]}"
            )

        if (
            actual_graph_counts[
                split
            ]
            != expected_graph_counts[
                split
            ]
        ):

            errors.append(
                f"{split}: expected "
                f"{expected_graph_counts[split]} "
                f"graph records, found "
                f"{actual_graph_counts[split]}"
            )

    # ==================================================================
    # VULNERABILITY BALANCE
    # ==================================================================

    vulnerability_balance_errors = []

    expected_per_vulnerability = {
        "train": 35,
        "validation": 7,
        "test": 8,
    }

    for split in (
        "train",
        "validation",
        "test",
    ):

        for vulnerability in (
            EXPECTED_VULNERABILITIES
        ):

            observed = (
                vulnerability_split_counts[
                    split
                ][
                    vulnerability
                ]
            )

            expected = (
                expected_per_vulnerability[
                    split
                ]
            )

            if observed != expected:

                vulnerability_balance_errors.append(
                    {
                        "split":
                            split,

                        "vulnerability":
                            vulnerability,

                        "expected":
                            expected,

                        "observed":
                            observed,
                    }
                )

                errors.append(
                    f"{split}/"
                    f"{vulnerability}: "
                    f"expected {expected}, "
                    f"found {observed}"
                )

    # ==================================================================
    # PAIRWISE CONTRACT OVERLAP
    # ==================================================================

    pairwise_overlap = {}

    splits = [
        "train",
        "validation",
        "test",
    ]

    for i in range(
        len(splits)
    ):

        for j in range(
            i + 1,
            len(splits)
        ):

            a = splits[i]
            b = splits[j]

            intersection = (
                split_contracts[a]
                &
                split_contracts[b]
            )

            key = (
                f"{a}_vs_{b}"
            )

            pairwise_overlap[
                key
            ] = sorted(
                intersection
            )

            if intersection:

                errors.append(
                    f"Contract overlap "
                    f"{a}/{b}: "
                    f"{sorted(intersection)}"
                )

    return {
        "errors":
            errors,

        "contract_leakage":
            contract_leakage,

        "graph_leakage":
            graph_leakage,

        "vulnerability_balance_errors":
            vulnerability_balance_errors,

        "split_contracts": {
            split: sorted(
                split_contracts[
                    split
                ]
            )
            for split in splits
        },

        "split_graph_counts":
            actual_graph_counts,

        "split_contract_counts":
            actual_contract_counts,

        "vulnerability_split_counts": {
            split: dict(
                vulnerability_split_counts[
                    split
                ]
            )
            for split in splits
        },

        "pairwise_contract_overlap":
            pairwise_overlap,
    }


# ======================================================================
# SPLIT CSV
# ======================================================================

def write_split_csv(
    records
):

    fields = [
        "split",
        "vulnerability",
        "contract_id",
        "graph",
        "graph_key",
        "graph_path",
    ]

    with OUTPUT_SPLIT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        split_order = {
            "train": 0,
            "validation": 1,
            "test": 2,
        }

        for record in sorted(
            records,
            key=lambda x: (
                split_order[
                    x["split"]
                ],
                x["contract_id"],
                x["vulnerability"],
            ),
        ):

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
                        (
                            f"{record['vulnerability']}/"
                            f"{record['graph']}"
                        ),

                    "graph_path":
                        record["graph_path"],
                }
            )


# ======================================================================
# LEAKAGE CSV
# ======================================================================

def write_leakage_csv(
    audit
):

    fields = [
        "issue_type",
        "contract_id",
        "vulnerability",
        "graph",
        "split_a",
        "split_b",
        "details",
    ]

    rows = []

    for item in (
        audit[
            "contract_leakage"
        ]
    ):

        rows.append(
            {
                "issue_type":
                    "contract_leakage",

                "contract_id":
                    item["contract_id"],

                "vulnerability":
                    "",

                "graph":
                    "",

                "split_a":
                    item["splits"][0],

                "split_b":
                    ",".join(
                        item["splits"][1:]
                    ),

                "details":
                    (
                        "Same contract appears "
                        "in multiple splits."
                    ),
            }
        )

    for item in (
        audit[
            "graph_leakage"
        ]
    ):

        rows.append(
            {
                "issue_type":
                    "graph_duplication",

                "contract_id":
                    "",

                "vulnerability":
                    item[
                        "vulnerability"
                    ],

                "graph":
                    item["graph"],

                "split_a":
                    "",

                "split_b":
                    "",

                "details":
                    (
                        f"Occurrences: "
                        f"{item['occurrences']}"
                    ),
            }
        )

    for item in (
        audit[
            "vulnerability_balance_errors"
        ]
    ):

        rows.append(
            {
                "issue_type":
                    "vulnerability_balance",

                "contract_id":
                    "",

                "vulnerability":
                    item[
                        "vulnerability"
                    ],

                "graph":
                    "",

                "split_a":
                    item["split"],

                "split_b":
                    "",

                "details":
                    (
                        f"Expected "
                        f"{item['expected']}, "
                        f"observed "
                        f"{item['observed']}"
                    ),
            }
        )

    with OUTPUT_LEAKAGE_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ======================================================================
# JSON REPORT
# ======================================================================

def write_json_report(
    structure,
    split_by_contract,
    train_ids,
    validation_ids,
    test_ids,
    audit,
    total_graphs,
):

    report = {

        "dataset":
            "SolidiFI-benchmark",

        "step":
            "08_contract_level_split_and_leakage_audit",

        "read_only":
            True,

        "random_seed":
            RANDOM_SEED,

        "graph_identity":
            "vulnerability + graph filename",

        "split_policy": {

            "contract_level":
                True,

            "train_ratio":
                TRAIN_RATIO,

            "validation_ratio":
                VALIDATION_RATIO,

            "test_ratio":
                TEST_RATIO,

            "train_contracts":
                len(train_ids),

            "validation_contracts":
                len(validation_ids),

            "test_contracts":
                len(test_ids),
        },

        "dataset": {

            "graph_records":
                total_graphs,

            "contracts":
                len(
                    structure[
                        "contract_ids"
                    ]
                ),

            "vulnerabilities":
                len(
                    EXPECTED_VULNERABILITIES
                ),
        },

        "contract_assignments": {
            str(k): v
            for k, v in sorted(
                split_by_contract.items()
            )
        },

        "train_contract_ids":
            train_ids,

        "validation_contract_ids":
            validation_ids,

        "test_contract_ids":
            test_ids,

        "audit":
            audit,

        "status":
            (
                "PASS"
                if not audit["errors"]
                else "REVIEW REQUIRED"
            ),
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )


# ======================================================================
# TEXT REPORT
# ======================================================================

def write_text_report(
    structure,
    split_by_contract,
    train_ids,
    validation_ids,
    test_ids,
    audit,
    total_graphs,
):

    lines = []

    lines.append("=" * 78)
    lines.append(
        "SecureGAT-Agent Step 08"
    )
    lines.append(
        "Contract-Level Train/Validation/Test "
        "Split and Leakage Audit"
    )
    lines.append("=" * 78)

    lines.append("")
    lines.append(
        "READ-ONLY SPLIT CONSTRUCTION"
    )
    lines.append("")

    lines.append(
        f"Graph records: {total_graphs}"
    )

    lines.append(
        f"Contracts: "
        f"{len(structure['contract_ids'])}"
    )

    lines.append(
        f"Vulnerabilities: "
        f"{len(EXPECTED_VULNERABILITIES)}"
    )

    lines.append("")

    lines.append(
        "GRAPH IDENTITY"
    )
    lines.append("-" * 78)
    lines.append(
        "Graph identity = vulnerability + graph filename"
    )
    lines.append(
        "This is required because buggy_1.pt exists "
        "under every vulnerability directory."
    )

    lines.append("")

    lines.append(
        "SPLIT POLICY"
    )
    lines.append("-" * 78)

    lines.append(
        "All vulnerability variants belonging to "
        "the same contract_id remain in one split."
    )

    lines.append(
        f"Random seed: {RANDOM_SEED}"
    )

    lines.append("")

    lines.append(
        "CONTRACT SPLITS"
    )
    lines.append("-" * 78)

    for split, ids in [
        ("TRAIN", train_ids),
        ("VALIDATION", validation_ids),
        ("TEST", test_ids),
    ]:

        lines.append(
            f"{split}: {len(ids)} contracts"
        )

        lines.append(
            f"IDs: {ids}"
        )

        lines.append("")

    lines.append(
        "GRAPH RECORD DISTRIBUTION"
    )
    lines.append("-" * 78)

    for split in (
        "train",
        "validation",
        "test",
    ):

        lines.append(
            f"{split}: "
            f"{audit['split_graph_counts'][split]}"
        )

    lines.append("")

    lines.append(
        "VULNERABILITY DISTRIBUTION"
    )
    lines.append("-" * 78)

    for split in (
        "train",
        "validation",
        "test",
    ):

        lines.append(
            split.upper()
        )

        for vulnerability in (
            EXPECTED_VULNERABILITIES
        ):

            count = (
                audit[
                    "vulnerability_split_counts"
                ][
                    split
                ].get(
                    vulnerability,
                    0,
                )
            )

            lines.append(
                f"  {vulnerability:<28}"
                f"{count}"
            )

        lines.append("")

    lines.append(
        "LEAKAGE AUDIT"
    )
    lines.append("-" * 78)

    lines.append(
        f"Contract leakage: "
        f"{len(audit['contract_leakage'])}"
    )

    lines.append(
        f"Graph duplication: "
        f"{len(audit['graph_leakage'])}"
    )

    lines.append(
        f"Vulnerability balance errors: "
        f"{len(audit['vulnerability_balance_errors'])}"
    )

    lines.append("")

    lines.append(
        "PAIRWISE CONTRACT OVERLAP"
    )
    lines.append("-" * 78)

    for (
        key,
        values
    ) in (
        audit[
            "pairwise_contract_overlap"
        ].items()
    ):

        lines.append(
            f"{key}: {values}"
        )

    lines.append("")

    lines.append(
        "FINAL STATUS"
    )
    lines.append("-" * 78)

    if audit["errors"]:

        lines.append(
            "STATUS: REVIEW REQUIRED"
        )

        lines.append("")

        for error in (
            audit["errors"]
        ):

            lines.append(
                f"ERROR: {error}"
            )

    else:

        lines.append(
            "STATUS: PASS"
        )

    lines.append("")
    lines.append("=" * 78)

    with OUTPUT_TEXT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(lines)
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "SecureGAT-Agent Step 08\n"
        "Contract-Level Train/Validation/Test "
        "Split and Leakage Audit"
    )

    print()

    print(
        "Execution mode: READ-ONLY"
    )

    print(
        "Dataset annotations will NOT be modified."
    )

    print()

    # ==================================================================
    # DISCOVER
    # ==================================================================

    graphs = discover_graphs()

    print(
        f"Graph files discovered: "
        f"{len(graphs)}"
    )

    # ==================================================================
    # STRUCTURE
    # ==================================================================

    structure = (
        audit_dataset_structure(
            graphs
        )
    )

    if structure["errors"]:

        print()
        print(
            "STRUCTURAL AUDIT FAILED"
        )
        print()

        for error in (
            structure["errors"]
        ):

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Dataset structure is not suitable "
            "for contract-level splitting."
        )

    print(
        "Structural audit: PASS"
    )

    print(
        f"Contracts: "
        f"{len(structure['contract_ids'])}"
    )

    print(
        f"Vulnerabilities: "
        f"{len(EXPECTED_VULNERABILITIES)}"
    )

    print()

    # ==================================================================
    # CONTRACT SPLIT
    # ==================================================================

    (
        split_by_contract,
        train_ids,
        validation_ids,
        test_ids,
    ) = construct_split(
        structure["contract_ids"]
    )

    print(
        "Contract-level split constructed:"
    )

    print(
        f"  Train      : "
        f"{len(train_ids)} contracts"
    )

    print(
        f"  Validation : "
        f"{len(validation_ids)} contracts"
    )

    print(
        f"  Test       : "
        f"{len(test_ids)} contracts"
    )

    print()

    # ==================================================================
    # ASSIGN
    # ==================================================================

    records = assign_graph_splits(
        graphs,
        split_by_contract
    )

    # ==================================================================
    # AUDIT
    # ==================================================================

    audit = audit_splits(
        records
    )

    print(
        "Split audit:"
    )

    print(
        f"  Contract leakage: "
        f"{len(audit['contract_leakage'])}"
    )

    print(
        f"  Graph duplication: "
        f"{len(audit['graph_leakage'])}"
    )

    print(
        f"  Vulnerability balance errors: "
        f"{len(audit['vulnerability_balance_errors'])}"
    )

    print()

    # ==================================================================
    # OUTPUTS
    # ==================================================================

    write_split_csv(
        records
    )

    write_leakage_csv(
        audit
    )

    write_json_report(
        structure,
        split_by_contract,
        train_ids,
        validation_ids,
        test_ids,
        audit,
        len(graphs),
    )

    write_text_report(
        structure,
        split_by_contract,
        train_ids,
        validation_ids,
        test_ids,
        audit,
        len(graphs),
    )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    print("=" * 78)
    print(
        "STEP 08 SUMMARY"
    )
    print("=" * 78)

    print(
        f"Graph records: "
        f"{len(graphs)}"
    )

    print(
        f"Contracts: "
        f"{len(structure['contract_ids'])}"
    )

    print(
        f"Train contracts: "
        f"{len(train_ids)}"
    )

    print(
        f"Validation contracts: "
        f"{len(validation_ids)}"
    )

    print(
        f"Test contracts: "
        f"{len(test_ids)}"
    )

    print()

    print(
        f"Train graph records: "
        f"{audit['split_graph_counts']['train']}"
    )

    print(
        f"Validation graph records: "
        f"{audit['split_graph_counts']['validation']}"
    )

    print(
        f"Test graph records: "
        f"{audit['split_graph_counts']['test']}"
    )

    print()

    print(
        f"Contract leakage: "
        f"{len(audit['contract_leakage'])}"
    )

    print(
        f"Graph duplication: "
        f"{len(audit['graph_leakage'])}"
    )

    print(
        f"Vulnerability balance errors: "
        f"{len(audit['vulnerability_balance_errors'])}"
    )

    print(
        f"Audit errors: "
        f"{len(audit['errors'])}"
    )

    print()

    if audit["errors"]:

        print(
            "STATUS: REVIEW REQUIRED"
        )

    else:

        print(
            "STATUS: PASS"
        )

    print()

    print(
        f"Split CSV: "
        f"{OUTPUT_SPLIT_CSV}"
    )

    print(
        f"Leakage audit: "
        f"{OUTPUT_LEAKAGE_CSV}"
    )

    print(
        f"JSON report: "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Text report: "
        f"{OUTPUT_TEXT}"
    )

    print("=" * 78)

    if audit["errors"]:

        raise RuntimeError(
            "Step 08 failed leakage/integrity audit."
        )

    print()
    print(
        "STEP 08 COMPLETE"
    )
    print()


if __name__ == "__main__":
    main()