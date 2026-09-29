#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:27:33 2026

@author: mac
"""


"""
======================================================================
SecureGAT-Agent
AST Dataset Extraction for SolidiFI Benchmark

Version: 1.1

Purpose:
    Convert Solidity smart contracts into Solidity AST JSON files
    for graph-based vulnerability detection.

Input:
    datasets/SolidiFI-benchmark/buggy_contracts/

Output:
    datasets/ast_dataset/

======================================================================
"""


import os
import json
import subprocess
from pathlib import Path
from datetime import datetime



# ============================================================
# PATH CONFIGURATION
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent


SOLIDI_FI_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "SolidiFI-benchmark" /
    "buggy_contracts"
)


AST_OUTPUT_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "ast_dataset"
)



# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================


AST_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ============================================================
# SOLC AST EXTRACTION
# ============================================================


def extract_ast(solidity_file, output_file):

    """
    Extract AST from Solidity contract using solc.

    Uses JSONDecoder.raw_decode because solc may append
    extra compiler information after the JSON object.
    """

    command = [

        "solc",

        "--ast-json",

        str(solidity_file)

    ]


    try:

        result = subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True,

            timeout=120

        )


        stdout = result.stdout.strip()



        if not stdout:


            print(
                "[FAILED EMPTY OUTPUT]",
                solidity_file
            )

            return False



        # ------------------------------------------------
        # Locate first JSON object
        # ------------------------------------------------

        start = stdout.find("{")



        if start == -1:


            print(
                "[FAILED NO JSON]",
                solidity_file
            )


            return False



        json_text = stdout[start:]



        # ------------------------------------------------
        # Parse first JSON object only
        # ------------------------------------------------


        decoder = json.JSONDecoder()


        ast, index = decoder.raw_decode(
            json_text
        )



        # ------------------------------------------------
        # Save AST
        # ------------------------------------------------


        output_file.parent.mkdir(

            parents=True,

            exist_ok=True

        )



        with open(

            output_file,

            "w",

            encoding="utf-8"

        ) as f:


            json.dump(

                ast,

                f,

                indent=2

            )



        return True



    except subprocess.TimeoutExpired:


        print(
            "[TIMEOUT]",
            solidity_file
        )


        return False



    except Exception as e:


        print(

            "[EXCEPTION]",

            solidity_file,

            e

        )


        return False





# ============================================================
# PROCESS DATASET
# ============================================================


def process_dataset():


    print("\n")
    print("=" * 70)
    print("SecureGAT-Agent AST Dataset Extraction")
    print("=" * 70)
    print()


    print(
        "Input:",
        SOLIDI_FI_DIR
    )


    print(
        "Output:",
        AST_OUTPUT_DIR
    )


    print()



    total = 0

    success = 0

    failed = 0



    failed_files = []



    # --------------------------------------------------------
    # Vulnerability categories
    # --------------------------------------------------------


    vulnerability_dirs = sorted(

        [

            x for x in SOLIDI_FI_DIR.iterdir()

            if x.is_dir()

        ]

    )



    print(
        "Vulnerability classes:",
        len(vulnerability_dirs)
    )



    print()



    for vuln_dir in vulnerability_dirs:


        print(
            "\n",
            vuln_dir.name,
            ": scanning"
        )


        contracts = sorted(

            vuln_dir.glob(
                "*.sol"
            )

        )


        print(
            "Contracts:",
            len(contracts)
        )



        for contract in contracts:


            total += 1



            output_dir = (

                AST_OUTPUT_DIR /

                vuln_dir.name

            )



            output_file = (

                output_dir /

                (
                    contract.stem
                    +
                    "_ast.json"
                )

            )



            ok = extract_ast(

                contract,

                output_file

            )



            if ok:


                success += 1



            else:


                failed += 1


                failed_files.append(

                    str(contract)

                )



            if total % 25 == 0:


                print(

                    "Processed:",

                    total

                )



    # ========================================================
    # SUMMARY
    # ========================================================


    print("\n")
    print("=" * 70)
    print("AST EXTRACTION SUMMARY")
    print("=" * 70)


    print(

        "Total contracts:",

        total

    )


    print(

        "Successful AST:",

        success

    )


    print(

        "Failed:",

        failed

    )



    # --------------------------------------------------------
    # Save failure report
    # --------------------------------------------------------


    report = {


        "timestamp":

            datetime.now().isoformat(),


        "total_contracts":

            total,


        "successful_ast":

            success,


        "failed":

            failed,


        "failed_files":

            failed_files


    }



    report_file = (

        AST_OUTPUT_DIR /

        "ast_extraction_report.json"

    )



    with open(

        report_file,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            report,

            f,

            indent=2

        )



    print()


    print(

        "Report saved:",

        report_file

    )


    print()




# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":


    process_dataset()
