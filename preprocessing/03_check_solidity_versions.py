#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:25:47 2026

@author: mac
"""

import os
import re
from collections import Counter


DATASET_PATH = (
    "datasets/SolidiFI-benchmark/"
    "buggy_contracts"
)


versions=[]


for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".sol"):

            path=os.path.join(root,file)

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                code=f.read()


            match=re.search(
                r"pragma solidity([^;]+);",
                code
            )

            if match:
                versions.append(
                    match.group(1).strip()
                )
            else:
                versions.append(
                    "unknown"
                )


print("="*60)
print("Solidity Version Audit")
print("="*60)

for k,v in Counter(versions).items():
    print(k,":",v)

print("\nTotal:")
print(len(versions))