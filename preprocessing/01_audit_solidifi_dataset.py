# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:11:18 2021

@author: willie
"""

"""
SecureGAT-Agent:
SolidiFI Dataset Audit

Purpose:
- Scan SolidiFI benchmark
- Extract contract paths
- Create vulnerability labels
- Generate metadata CSV
"""


import os
import pandas as pd


# ===============================
# Dataset location
# ===============================

DATASET_PATH = (
    "datasets/SolidiFI-benchmark/"
    "buggy_contracts"
)


OUTPUT_PATH = (
    "results/"
    "solidifi_dataset_report.csv"
)


# ===============================
# Vulnerability labels
# ===============================

CLASS_MAPPING = {

    "TOD": 0,

    "tx.origin": 1,

    "Timestamp-Dependency": 2,

    "Re-entrancy": 3,

    "Overflow-Underflow": 4,

    "Unhandled-Exceptions": 5,

    "Unchecked-Send": 6
}



def audit_dataset():

    records = []


    print("="*70)
    print("SecureGAT-Agent SolidiFI Dataset Audit")
    print("="*70)


    for folder in os.listdir(DATASET_PATH):

        folder_path = os.path.join(
            DATASET_PATH,
            folder
        )


        if not os.path.isdir(folder_path):
            continue


        if folder not in CLASS_MAPPING:
            print(
                "Skipping unknown folder:",
                folder
            )
            continue



        class_id = CLASS_MAPPING[folder]


        contracts = [
            f for f in os.listdir(folder_path)
            if f.endswith(".sol")
        ]


        print(
            folder,
            ":",
            len(contracts),
            "contracts"
        )


        for contract in contracts:

            records.append(
                {
                    "contract":
                    contract,

                    "vulnerability":
                    folder,

                    "class_id":
                    class_id,

                    "path":
                    os.path.join(
                        folder_path,
                        contract
                    )
                }
            )


    df = pd.DataFrame(records)


    os.makedirs(
        "results",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("\nDataset summary")
    print(df["vulnerability"].value_counts())


    print("\nTotal contracts:")
    print(len(df))


    print(
        "\nSaved:",
        OUTPUT_PATH
    )



if __name__ == "__main__":

    audit_dataset()