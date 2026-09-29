#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:12:22 2026

@author: mac
"""

import os
import pandas as pd


DATASET="datasets/SolidiFI-benchmark/buggy_contracts"


results=[]


for vuln in os.listdir(DATASET):

    vuln_path=os.path.join(DATASET,vuln)

    if not os.path.isdir(vuln_path):
        continue


    for file in os.listdir(vuln_path):

        if file.startswith("BugLog") and file.endswith(".csv"):

            path=os.path.join(
                vuln_path,
                file
            )


            try:
                df=pd.read_csv(
                    path,
                    header=None
                )

                for _,row in df.iterrows():

                    results.append(
                        {
                        "vulnerability":vuln,
                        "buglog":file,
                        "line":row[0],
                        "column":row[1],
                        "type":row[2]
                        }
                    )

            except Exception as e:
                print(
                    "Error:",
                    path,
                    e
                )


df=pd.DataFrame(results)


print("="*60)
print("SolidiFI Injection Audit")
print("="*60)


print(df["vulnerability"].value_counts())


print("\nTotal injections:")
print(len(df))


df.to_csv(
    "results/solidifi_injection_report.csv",
    index=False
)

print("\nSaved:")
print(
"results/solidifi_injection_report.csv"
)