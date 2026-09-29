#!/bin/bash
set -e
python -m py_compile Model/train_securegat_msa.py
python -m py_compile Model/evaluate_securegat_msa.py
python reproduction/check_reference_protocol.py
python Model/train_securegat_msa.py
python Model/evaluate_securegat_msa.py
