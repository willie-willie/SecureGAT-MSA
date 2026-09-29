#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 00:29:36 2026

@author: mac
"""




"""
SecureGAT-Agent: Semantic Compatibility Audit

Purpose
-------
Evaluate whether the semantic AST construct identified is compatible with the expected semantic mechanism of each vulnerability.


"""

import csv
import json
import os
from collections import Counter, defaultdict


INPUT_CSV = (
    "datasets/annotated_graph_dataset/"
    "semantic_proximity_audit.csv"
)

OUTPUT_DIR = "datasets/annotated_graph_dataset"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "semantic_compatibility_audit.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "semantic_compatibility_report.json"
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "semantic_compatibility_audit.txt"
)


# ============================================================================
# Vulnerability semantic model
# ============================================================================

# PRIMARY = direct semantic constructs that strongly represent the
# vulnerability mechanism.
#
# CONTEXTUAL = structurally meaningful constructs that may legitimately
# surround the vulnerability mechanism, but are not themselves the
# vulnerability mechanism.

SEMANTIC_EXPECTATIONS = {

    "Overflow-Underflow": {
        "primary": {
            "Assignment",
            "BinaryOperation",
        },
        "contextual": {
            "ExpressionStatement",
            "VariableDeclarationStatement",
            "VariableDeclaration",
        },
    },

    "Re-entrancy": {
        "primary": {
            "FunctionCall",
            "MemberAccess",
        },
        "contextual": {
            "ExpressionStatement",
            "Assignment",
            "BinaryOperation",
            "IfStatement",
        },
    },

    "TOD": {
        "primary": {
            "FunctionCall",
            "MemberAccess",
        },
        "contextual": {
            "ExpressionStatement",
            "Assignment",
            "IfStatement",
            "BinaryOperation",
        },
    },

    "Timestamp-Dependency": {
        "primary": {
            "Identifier",
            "MemberAccess",
            "FunctionCall",
            "BinaryOperation",
            "IfStatement",
        },
        "contextual": {
            "VariableDeclaration",
            "VariableDeclarationStatement",
            "FunctionDefinition",
            "ExpressionStatement",
        },
    },

    "Unchecked-Send": {
        "primary": {
            "FunctionCall",
            "MemberAccess",
        },
        "contextual": {
            "ExpressionStatement",
            "Assignment",
        },
    },

    "Unhandled-Exceptions": {
        "primary": {
            "FunctionCall",
            "MemberAccess",
        },
        "contextual": {
            "ExpressionStatement",
        },
    },

    "tx.origin": {
        "primary": {
            "MemberAccess",
            "Identifier",
            "FunctionCall",
        },
        "contextual": {
            "BinaryOperation",
            "IfStatement",
            "ExpressionStatement",
        },
    },
}


# ============================================================================
# Helpers
# ============================================================================

def parse_json_field(value, default):
    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    try:
        return json.loads(value)
    except Exception:
        return default


def classify(row):
    vulnerability = row["vulnerability"]

    expectation = SEMANTIC_EXPECTATIONS.get(
        vulnerability
    )

    if expectation is None:
        return "UNRESOLVED"

    primary = expectation["primary"]
    contextual = expectation["contextual"]

    nearest = (
        row.get("nearest_semantic_type", "")
        or ""
    ).strip()

    relation = (
        row.get("semantic_relation", "")
        or ""
    ).strip()

    direct_types = set(
        parse_json_field(
            row.get("direct_semantic_types"),
            []
        )
    )

    reached_types = set(
        parse_json_field(
            row.get("semantic_types_reached"),
            []
        )
    )

    # ----------------------------------------------------------------------
    # 1. No semantic construct
    # ----------------------------------------------------------------------

    if relation == "none":
        return "UNRESOLVED"

    if not nearest:
        return "UNRESOLVED"

    # ----------------------------------------------------------------------
    # 2. Strong compatibility
    #
    # The nearest construct itself is one of the primary semantic
    # constructs expected for this vulnerability.
    # ----------------------------------------------------------------------

    if nearest in primary:
        return "COMPATIBLE"

    # ----------------------------------------------------------------------
    # 3. Contextual compatibility
    #
    # The nearest node is a legitimate structural/contextual construct.
    # This is particularly important for timestamp BugLog locations such
    # as:
    #
    #     address winner_tmstmp26;
    #
    # or:
    #
    #     function bug_tmstmp17()
    #
    # where the BugLog location itself is not the actual timestamp
    # operation.
    # ----------------------------------------------------------------------

    if nearest in contextual:
        return "PARTIALLY_COMPATIBLE"

    # ----------------------------------------------------------------------
    # 4. Semantic construct exists somewhere in the reached neighborhood.
    #
    # This indicates that the selected node is not itself the mechanism,
    # but the local AST neighborhood contains a vulnerability-compatible
    # construct.
    # ----------------------------------------------------------------------

    if reached_types.intersection(primary):
        return "PARTIALLY_COMPATIBLE"

    if direct_types.intersection(primary):
        return "PARTIALLY_COMPATIBLE"

    # ----------------------------------------------------------------------
    # 5. Everything else
    # ----------------------------------------------------------------------

    return "INCOMPATIBLE"


def compatibility_score(category):
    return {
        "COMPATIBLE": 1.0,
        "PARTIALLY_COMPATIBLE": 0.5,
        "INCOMPATIBLE": 0.0,
        "UNRESOLVED": 0.0,
    }.get(category, 0.0)


# ============================================================================
# Main audit
# ============================================================================

def main():

    print("=" * 70)
    print("SecureGAT-Agent Step 06.3")
    print("Semantic Compatibility Audit")
    print("=" * 70)
    print()

    print("Loading semantic proximity audit:")
    print(INPUT_CSV)

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    print()
    print(f"Injections loaded: {len(rows)}")
    print("-" * 70)

    if not rows:
        raise RuntimeError(
            "No injections found."
        )

    # ----------------------------------------------------------------------
    # Output records
    # ----------------------------------------------------------------------

    output_rows = []

    global_counts = Counter()
    vulnerability_counts = defaultdict(Counter)

    nearest_distribution = defaultdict(Counter)
    incompatible_examples = []
    unresolved_examples = []

    total_score = 0.0

    for row in rows:

        category = classify(row)

        score = compatibility_score(
            category
        )

        total_score += score

        global_counts[category] += 1

        vulnerability = row[
            "vulnerability"
        ]

        vulnerability_counts[
            vulnerability
        ][category] += 1

        nearest = (
            row.get(
                "nearest_semantic_type",
                ""
            )
            or ""
        )

        if nearest:
            nearest_distribution[
                vulnerability
            ][nearest] += 1

        output_row = dict(row)

        output_row[
            "compatibility"
        ] = category

        output_row[
            "compatibility_score"
        ] = score

        output_rows.append(
            output_row
        )

        if (
            category == "INCOMPATIBLE"
            and len(incompatible_examples) < 25
        ):
            incompatible_examples.append(
                row
            )

        if (
            category == "UNRESOLVED"
            and len(unresolved_examples) < 25
        ):
            unresolved_examples.append(
                row
            )

    # ----------------------------------------------------------------------
    # Write CSV
    # ----------------------------------------------------------------------

    fieldnames = list(
        output_rows[0].keys()
    )

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
        writer.writerows(
            output_rows
        )

    # ----------------------------------------------------------------------
    # Vulnerability report
    # ----------------------------------------------------------------------

    vulnerability_report = {}

    for vulnerability in sorted(
        vulnerability_counts
    ):

        counts = vulnerability_counts[
            vulnerability
        ]

        total = sum(counts.values())

        compatible = counts[
            "COMPATIBLE"
        ]

        partial = counts[
            "PARTIALLY_COMPATIBLE"
        ]

        incompatible = counts[
            "INCOMPATIBLE"
        ]

        unresolved = counts[
            "UNRESOLVED"
        ]

        vulnerability_report[
            vulnerability
        ] = {
            "injections": total,
            "compatible": compatible,
            "partially_compatible": partial,
            "incompatible": incompatible,
            "unresolved": unresolved,
            "compatible_rate": (
                compatible / total * 100
                if total else 0.0
            ),
            "partial_rate": (
                partial / total * 100
                if total else 0.0
            ),
            "incompatible_rate": (
                incompatible / total * 100
                if total else 0.0
            ),
            "unresolved_rate": (
                unresolved / total * 100
                if total else 0.0
            ),
            "nearest_semantic_types": dict(
                nearest_distribution[
                    vulnerability
                ]
            ),
        }

    # ----------------------------------------------------------------------
    # Global metrics
    # ----------------------------------------------------------------------

    total = len(rows)

    report = {
        "dataset": "SolidiFI-benchmark",
        "step": "06.3_semantic_compatibility_audit",
        "annotations_modified": False,

        "total_injections": total,

        "compatible": global_counts[
            "COMPATIBLE"
        ],

        "partially_compatible":
            global_counts[
                "PARTIALLY_COMPATIBLE"
            ],

        "incompatible": global_counts[
            "INCOMPATIBLE"
        ],

        "unresolved": global_counts[
            "UNRESOLVED"
        ],

        "compatible_rate": (
            global_counts["COMPATIBLE"]
            / total * 100
            if total else 0.0
        ),

        "partial_rate": (
            global_counts[
                "PARTIALLY_COMPATIBLE"
            ]
            / total * 100
            if total else 0.0
        ),

        "incompatible_rate": (
            global_counts["INCOMPATIBLE"]
            / total * 100
            if total else 0.0
        ),

        "unresolved_rate": (
            global_counts["UNRESOLVED"]
            / total * 100
            if total else 0.0
        ),

        "weighted_compatibility_score": (
            total_score / total
            if total else 0.0
        ),

        "vulnerabilities":
            vulnerability_report,

        "incompatible_examples":
            incompatible_examples,

        "unresolved_examples":
            unresolved_examples,

        "files": {
            "audit_csv":
                os.path.basename(
                    OUTPUT_CSV
                ),
            "audit_text":
                os.path.basename(
                    OUTPUT_TXT
                ),
        },
    }

    # ----------------------------------------------------------------------
    # JSON
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Text report
    # ----------------------------------------------------------------------

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "SecureGAT-Agent Step 06.3\n"
        )

        f.write(
            "Semantic Compatibility Audit\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "IMPORTANT: This audit does NOT modify "
            "graph annotations.\n\n"
        )

        for vulnerability in sorted(
            vulnerability_report
        ):

            data = vulnerability_report[
                vulnerability
            ]

            f.write(
                f"{vulnerability}\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            f.write(
                f"Injections: "
                f"{data['injections']}\n"
            )

            f.write(
                f"Compatible: "
                f"{data['compatible']} "
                f"({data['compatible_rate']:.2f}%)\n"
            )

            f.write(
                f"Partially compatible: "
                f"{data['partially_compatible']} "
                f"({data['partial_rate']:.2f}%)\n"
            )

            f.write(
                f"Incompatible: "
                f"{data['incompatible']} "
                f"({data['incompatible_rate']:.2f}%)\n"
            )

            f.write(
                f"Unresolved: "
                f"{data['unresolved']} "
                f"({data['unresolved_rate']:.2f}%)\n"
            )

            f.write(
                "Nearest semantic types:\n"
            )

            for semantic_type, count in sorted(
                data[
                    "nearest_semantic_types"
                ].items(),
                key=lambda x: (-x[1], x[0])
            ):

                f.write(
                    f"  {semantic_type:<30}"
                    f"{count}\n"
                )

            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("GLOBAL SUMMARY\n")
        f.write("=" * 70 + "\n")

        f.write(
            f"Total injections: {total}\n"
        )

        f.write(
            f"Compatible: "
            f"{global_counts['COMPATIBLE']} "
            f"({report['compatible_rate']:.2f}%)\n"
        )

        f.write(
            f"Partially compatible: "
            f"{global_counts['PARTIALLY_COMPATIBLE']} "
            f"({report['partial_rate']:.2f}%)\n"
        )

        f.write(
            f"Incompatible: "
            f"{global_counts['INCOMPATIBLE']} "
            f"({report['incompatible_rate']:.2f}%)\n"
        )

        f.write(
            f"Unresolved: "
            f"{global_counts['UNRESOLVED']} "
            f"({report['unresolved_rate']:.2f}%)\n"
        )

        f.write(
            f"Weighted compatibility score: "
            f"{report['weighted_compatibility_score']:.4f}\n"
        )

        f.write("\n")

        if unresolved_examples:

            f.write(
                "=" * 70 + "\n"
            )

            f.write(
                "UNRESOLVED EXAMPLES\n"
            )

            f.write(
                "=" * 70 + "\n"
            )

            for row in unresolved_examples:

                f.write(
                    f"Vulnerability: "
                    f"{row['vulnerability']}\n"
                )

                f.write(
                    f"Contract: "
                    f"{row['contract_id']}\n"
                )

                f.write(
                    f"Injection: "
                    f"{row['injection_id']}\n"
                )

                f.write(
                    f"Location: "
                    f"{row['loc']}\n"
                )

                f.write(
                    f"Length: "
                    f"{row['length']}\n"
                )

                f.write(
                    f"Relation: "
                    f"{row['semantic_relation']}\n"
                )

                f.write(
                    f"Nearest type: "
                    f"{row['nearest_semantic_type']}\n"
                )

                f.write("-" * 70 + "\n")

    # ----------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 06.3 SUMMARY")
    print("=" * 70)

    print(
        f"Total injections: {total}"
    )

    print(
        f"Compatible: "
        f"{global_counts['COMPATIBLE']} "
        f"({report['compatible_rate']:.2f} %)"
    )

    print(
        f"Partially compatible: "
        f"{global_counts['PARTIALLY_COMPATIBLE']} "
        f"({report['partial_rate']:.2f} %)"
    )

    print(
        f"Incompatible: "
        f"{global_counts['INCOMPATIBLE']} "
        f"({report['incompatible_rate']:.2f} %)"
    )

    print(
        f"Unresolved: "
        f"{global_counts['UNRESOLVED']} "
        f"({report['unresolved_rate']:.2f} %)"
    )

    print(
        f"Weighted compatibility score: "
        f"{report['weighted_compatibility_score']:.4f}"
    )

    print()
    print(
        f"CSV: {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        f"Text report: {OUTPUT_TXT}"
    )

    print()
    print("=" * 70)
    print("STEP 06.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()