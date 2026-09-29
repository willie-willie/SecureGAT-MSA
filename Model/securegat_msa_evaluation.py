#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:06:33 2026

@author: Willie
"""


"""

SecureGAT-MSA Evaluation


Architecture:
-------------
train_securegat_msa.py
        |
        v
SecureGATMSA
        |
        v
securegat_msa_training/best_model.pt

"""

import csv
import json
import sys

from pathlib import Path
from collections import defaultdict

import numpy as np

import torch

import matplotlib.pyplot as plt


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    average_precision_score,

    matthews_corrcoef,

    confusion_matrix,

)



# ============================================================
# ROOT
# ============================================================


ROOT = Path(__file__).resolve().parents[1]



DATASET_ROOT = (

    ROOT

    /

    "datasets"

    /

    "annotated_graph_dataset"

)



MODEL_READY_ROOT = (

    DATASET_ROOT

    /

    "model_ready"

)



# ============================================================
#  MSA CHECKPOINT
# ============================================================


CHECKPOINT = (

    MODEL_READY_ROOT

    /

    "securegat_msa_training"

    /

    "best_model.pt"

)



# ============================================================
# OUTPUT
# ============================================================


OUTPUT_ROOT = (

    MODEL_READY_ROOT

    /

    "securegat_msa_evaluation"

)



PREDICTION_CSV = (

    OUTPUT_ROOT

    /

    "securegat_msa_predictions.csv"

)



OVERALL_JSON = (

    OUTPUT_ROOT

    /

    "overall_metrics.json"

)



OVERALL_CSV = (

    OUTPUT_ROOT

    /

    "overall_metrics.csv"

)



VULNERABILITY_CSV = (

    OUTPUT_ROOT

    /

    "accuracy_by_vulnerability.csv"

)



NODETYPE_CSV = (

    OUTPUT_ROOT

    /

    "accuracy_by_node_type.csv"

)



VULN_NODETYPE_CSV = (

    OUTPUT_ROOT

    /

    "vulnerability_node_type_accuracy.csv"

)



COMPARISON_CSV = (

    OUTPUT_ROOT

    /

    "baseline_vs_attention.csv"

)



REPORT_TXT = (

    OUTPUT_ROOT

    /

    "securegat_msa_evaluation_report.txt"

)



CONFUSION_FIGURE = (

    OUTPUT_ROOT

    /

    "confusion_matrix.png"

)



ACCURACY_FIGURE = (

    OUTPUT_ROOT

    /

    "accuracy_comparison.png"

)



# ============================================================
# DATA
# ============================================================


SPLIT_CSV = (

    DATASET_ROOT

    /

    "contract_level_splits.csv"

)



BASELINE_METRICS = (

    MODEL_READY_ROOT

    /

    "securegat_training"

    /

    "test_metrics.json"

)



THRESHOLD = 0.50



# ============================================================
# HELPERS
# ============================================================


def fail(message):

    print()

    print("=" * 80)

    print("STEP 24 FAILED")

    print("=" * 80)

    print(message)

    print("=" * 80)

    raise RuntimeError(message)




def ensure_directories():

    OUTPUT_ROOT.mkdir(

        parents=True,

        exist_ok=True

    )




def safe_torch_load(path):

    try:

        return torch.load(

            path,

            map_location="cpu",

            weights_only=False

        )

    except TypeError:

        return torch.load(

            path,

            map_location="cpu"

        )




def save_json(path,data):

    with path.open(

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            data,

            f,

            indent=2

        )


def load_json(path):

    if not path.exists():

        return {}


    with path.open(

        "r",

        encoding="utf-8"

    ) as f:

        return json.load(f)


def save_csv(path,rows):

    if len(rows)==0:

        return


    with path.open(

        "w",

        encoding="utf-8",

        newline=""

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=list(rows[0].keys())

        )


        writer.writeheader()


        writer.writerows(rows)




def get_attribute(graph,name):

    if hasattr(graph,name):

        return getattr(graph,name)


    if isinstance(graph,dict):

        return graph.get(name)


    return None




def load_test_split():

    if not SPLIT_CSV.exists():

        fail(

            f"Missing split file:\n{SPLIT_CSV}"

        )


    records=[]


    with SPLIT_CSV.open(

        "r",

        encoding="utf-8"

    ) as f:


        reader=csv.DictReader(f)


        for row in reader:


            if row["split"]=="test":

                records.append(row)



    return records

# ============================================================
# LOAD EXACT SECUREGAT-MSA IMPLEMENTATION
# ============================================================


def load_msa_module():

    """
    Import the exact Step 23 implementation.

    Do NOT redefine the model.
    """

    import importlib.util


    script = (

        ROOT

        /

        "preprocessing"

        /

        "23_train_securegat_msa.py"

    )


    if not script.exists():

        fail(

            f"Step 23 training script missing:\n{script}"

        )


    spec = importlib.util.spec_from_file_location(

        "securegat_msa_training",

        script

    )


    if spec is None or spec.loader is None:

        fail(

            "Unable to import Step 23 module."

        )


    module = importlib.util.module_from_spec(spec)


    sys.modules[

        "securegat_msa_training"

    ] = module


    spec.loader.exec_module(module)


    return module




# ============================================================
# LOAD MODEL + CHECKPOINT
# ============================================================


def load_model():


    module = load_msa_module()



    if not hasattr(

        module,

        "SecureGATMSA"

    ):


        fail(

            "SecureGATMSA class not found in Step 23."

        )



    model = module.SecureGATMSA()



    if not CHECKPOINT.exists():


        fail(

            f"Checkpoint missing:\n{CHECKPOINT}"

        )



    checkpoint = safe_torch_load(

        CHECKPOINT

    )



    architecture = checkpoint.get(

        "architecture",

        None

    )


    if architecture != "SecureGAT-MSA":


        fail(

            "Wrong checkpoint architecture:\n"

            f"{architecture}"

        )



    model.load_state_dict(

        checkpoint["model_state_dict"]

    )



    model.eval()



    print()

    print(

        "Checkpoint loaded:"

    )

    print(

        CHECKPOINT

    )

    print(

        "Epoch:",

        checkpoint["epoch"]

    )

    print(

        "Validation accuracy:",

        checkpoint["validation_accuracy"]

    )

    print(

        "Architecture:",

        architecture

    )



    return model





# ============================================================
# GRAPH PREPARATION
# ============================================================


def prepare_graph(graph):


    node_type = get_attribute(

        graph,

        "node_type"

    )


    node_depth = get_attribute(

        graph,

        "node_depth"

    )


    edge_index = get_attribute(

        graph,

        "edge_index"

    )


    labels = get_attribute(

        graph,

        "injection_label"

    )

    injection_ids = get_attribute(
        graph,
        "injection_id"
    )

        
    if (
        node_type is None
        or node_depth is None
        or edge_index is None
        or labels is None
        or injection_ids is None
    ):        

        fail(

            "Graph missing required fields."

        )



    node_type = torch.as_tensor(

        node_type,

        dtype=torch.long

    )


    node_depth = torch.as_tensor(

        node_depth,

        dtype=torch.float32

    )


    edge_index = torch.as_tensor(

        edge_index,

        dtype=torch.long

    )

    labels = torch.as_tensor(

        labels,

        dtype=torch.long

    )

    injection_ids = torch.as_tensor(
        injection_ids,
        dtype=torch.long
    )    


    # Same normalization used during training

    max_depth = node_depth.max()



    if max_depth > 0:

        node_depth = (

            node_depth

            /

            max_depth

        )



    return (
        node_type,
        node_depth,
        edge_index,
        labels,
        injection_ids
    )





# ============================================================
# INFERENCE
# ============================================================


@torch.no_grad()

def collect_predictions(

    model,

    records

):


    model.eval()



    rows=[]


    y_true=[]

    y_prob=[]



    for counter,row in enumerate(records,1):


        graph_path = ROOT / row["graph_path"]



        if not graph_path.exists():

            fail(

                f"Graph missing:\n{graph_path}"

            )



        graph = safe_torch_load(

            graph_path

        )


        (
            node_type,
            node_depth,
            edge_index,
            labels,
            injection_ids
        ) = prepare_graph(graph)


        logits = model(

            node_type,

            node_depth,

            edge_index

        )



        # SecureGAT-MSA returns logits

        probability = torch.sigmoid(

            logits

        )



        labels_np = (

            labels

            .cpu()

            .numpy()

        )



        probability_np = (

            probability

            .cpu()

            .numpy()

        )



        original_depth = get_attribute(

            graph,

            "node_depth"

        )



        for idx in range(

            len(labels_np)

        ):


            rows.append(

                {

                    "vulnerability":

                        row["vulnerability"],


                    "contract_id":

                        row["contract_id"],


                    "graph":

                        row["graph"],


                    "node_index":

                        idx,


                    "true_label":

                        int(labels_np[idx]),


                    "probability":

                        float(probability_np[idx]),


                    "prediction":

                        int(

                            probability_np[idx]

                            >= THRESHOLD

                        ),


                    "node_type":

                        int(

                            node_type[idx]

                        ),
                        
                    "node_depth":
                        float(
                            original_depth[idx]
                        ),
                    
                    "injection_label":
                        int(
                            labels_np[idx]
                        ),
                    
                    "injection_id":
                        int(
                            injection_ids[idx]
                        )
                        

                }

            )



        y_true.extend(

            labels_np.tolist()

        )


        y_prob.extend(

            probability_np.tolist()

        )



        if counter % 10 == 0 or counter == len(records):

            print(

                f"Evaluated {counter}/{len(records)} graphs"

            )



    return (

        rows,

        np.asarray(y_true),

        np.asarray(y_prob)

    )





# ============================================================
# METRICS
# ============================================================


def calculate_metrics(

    labels,

    probabilities

):


    predictions = (

        probabilities

        >= THRESHOLD

    ).astype(int)



    return {


        "accuracy":

            float(

                accuracy_score(

                    labels,

                    predictions

                )

            ),


        "precision":

            float(

                precision_score(

                    labels,

                    predictions,

                    zero_division=0

                )

            ),


        "recall":

            float(

                recall_score(

                    labels,

                    predictions,

                    zero_division=0

                )

            ),


        "f1":

            float(

                f1_score(

                    labels,

                    predictions,

                    zero_division=0

                )

            ),


        "roc_auc":

            float(

                roc_auc_score(

                    labels,

                    probabilities

                )

            ),


        "pr_auc":

            float(

                average_precision_score(

                    labels,

                    probabilities

                )

            ),


        "mcc":

            float(

                matthews_corrcoef(

                    labels,

                    predictions

                )

            )

    }
        

# ============================================================
# ACCURACY BY VULNERABILITY
# ============================================================


def accuracy_by_vulnerability(rows):

    groups = defaultdict(list)


    for row in rows:

        groups[
            row["vulnerability"]
        ].append(row)



    output=[]


    for vuln,data in groups.items():

        y_true=[

            r["true_label"]

            for r in data

        ]


        y_pred=[

            r["prediction"]

            for r in data

        ]



        output.append(

            {

                "vulnerability":

                    vuln,


                "nodes":

                    len(data),


                "accuracy":

                    float(

                        accuracy_score(

                            y_true,

                            y_pred

                        )

                    )

            }

        )


    return sorted(

        output,

        key=lambda x:x["accuracy"]

    )





# ============================================================
# ACCURACY BY AST NODE TYPE
# ============================================================


def accuracy_by_node_type(rows):


    groups=defaultdict(list)



    for row in rows:


        groups[

            row["node_type"]

        ].append(row)



    output=[]



    for node_type,data in groups.items():


        y_true=[

            r["true_label"]

            for r in data

        ]


        y_pred=[

            r["prediction"]

            for r in data

        ]



        output.append(

            {

                "node_type":

                    int(node_type),


                "nodes":

                    len(data),


                "accuracy":

                    float(

                        accuracy_score(

                            y_true,

                            y_pred

                        )

                    )

            }

        )


    return sorted(

        output,

        key=lambda x:x["accuracy"]

    )





# ============================================================
# VULNERABILITY × NODE TYPE
# ============================================================


def vulnerability_node_accuracy(rows):


    groups=defaultdict(list)



    for row in rows:


        key=(

            row["vulnerability"],

            row["node_type"]

        )


        groups[key].append(row)



    output=[]



    for key,data in groups.items():


        y_true=[

            r["true_label"]

            for r in data

        ]


        y_pred=[

            r["prediction"]

            for r in data

        ]



        output.append(

            {

                "vulnerability":

                    key[0],


                "node_type":

                    int(key[1]),


                "nodes":

                    len(data),


                "accuracy":

                    float(

                        accuracy_score(

                            y_true,

                            y_pred

                        )

                    )

            }

        )


    return sorted(

        output,

        key=lambda x:x["accuracy"]

    )





# ============================================================
# BASELINE COMPARISON
# ============================================================


def create_comparison(msa_metrics):


    baseline={}

    sa={}


    if BASELINE_METRICS.exists():

        baseline_json = load_json(

            BASELINE_METRICS

        )

        baseline = (

            baseline_json

            .get("test_metrics",{})

        )



    sa_path=(

        MODEL_READY_ROOT

        /

        "securegat_structural_attention"

        /

        "test_metrics.json"

    )


    if sa_path.exists():

        sa_json=load_json(sa_path)

        sa=(

            sa_json

            .get("test_metrics",{})

        )



    rows=[


        {

            "model":

                "SecureGAT",


            "accuracy":

                baseline.get(

                    "accuracy",

                    None

                ),


            "roc_auc":

                baseline.get(

                    "roc_auc",

                    None

                )

        },


        {

            "model":

                "SecureGAT-SA",


            "accuracy":

                sa.get(

                    "accuracy",

                    None

                ),


            "roc_auc":

                sa.get(

                    "roc_auc",

                    None

                )

        },


        {

            "model":

                "SecureGAT-MSA",


            "accuracy":

                msa_metrics["accuracy"],


            "roc_auc":

                msa_metrics["roc_auc"]

        }

    ]


    return rows





# ============================================================
# FIGURES
# ============================================================


def plot_confusion(

    labels,

    probabilities

):


    predictions=(

        probabilities

        >= THRESHOLD

    ).astype(int)



    matrix=confusion_matrix(

        labels,

        predictions

    )



    plt.figure(

        figsize=(6,5)

    )


    plt.imshow(matrix)


    plt.title(

        "SecureGAT-MSA Confusion Matrix"

    )


    plt.xlabel(

        "Predicted"

    )


    plt.ylabel(

        "True"

    )


    plt.colorbar()


    for i in range(2):

        for j in range(2):

            plt.text(

                j,

                i,

                str(matrix[i,j]),

                ha="center",

                va="center"

            )



    plt.tight_layout()


    plt.savefig(

        CONFUSION_FIGURE,

        dpi=300

    )


    plt.close()





def plot_comparison(rows):


    names=[

        r["model"]

        for r in rows

    ]


    values=[

        r["accuracy"]

        if r["accuracy"] is not None

        else 0

        for r in rows

    ]



    plt.figure(

        figsize=(7,5)

    )


    plt.bar(

        names,

        values

    )


    plt.ylabel(

        "Accuracy"

    )


    plt.title(

        "SecureGAT Attention Comparison"

    )


    plt.xticks(

        rotation=30

    )


    plt.tight_layout()


    plt.savefig(

        ACCURACY_FIGURE,

        dpi=300

    )


    plt.close()





# ============================================================
# REPORT
# ============================================================


def write_report(

    metrics,

    vuln,

    comparison

):


    with REPORT_TXT.open(

        "w",

        encoding="utf-8"

    ) as f:


        f.write(

            "SecureGAT-MSA Evaluation Report\n"

        )

        f.write(

            "="*60+"\n\n"

        )


        f.write(

            "Overall metrics\n"

        )


        for k,v in metrics.items():

            f.write(

                f"{k}: {v}\n"

            )



        f.write(

            "\nAccuracy by vulnerability\n"

        )


        for row in vuln:

            f.write(

                f"{row}\n"

            )



        f.write(

            "\nModel comparison\n"

        )


        for row in comparison:

            f.write(

                f"{row}\n"

            )





# ============================================================
# MAIN
# ============================================================


def main():


    ensure_directories()


    print("="*80)

    print(

        "SecureGAT-Agent Step 24"

    )

    print(

        "SecureGAT-MSA Evaluation"

    )

    print("="*80)



    records=load_test_split()



    print(

        "Test graphs:",

        len(records)

    )



    model=load_model()



    rows,labels,probabilities = collect_predictions(

        model,

        records

    )



    save_csv(

        PREDICTION_CSV,

        rows

    )



    metrics=calculate_metrics(

        labels,

        probabilities

    )



    print()

    print("OVERALL RESULTS")

    print("-"*60)


    for k,v in metrics.items():

        print(

            f"{k:15s}: {v:.6f}"

        )



    save_json(

        OVERALL_JSON,

        metrics

    )


    save_csv(

        OVERALL_CSV,

        [metrics]

    )



    vuln=accuracy_by_vulnerability(

        rows

    )


    save_csv(

        VULNERABILITY_CSV,

        vuln

    )


    print()

    print(

        "ACCURACY BY VULNERABILITY"

    )


    for row in vuln:

        print(

            row["vulnerability"],

            f"{row['accuracy']:.6f}"

        )



    node=accuracy_by_node_type(

        rows

    )


    save_csv(

        NODETYPE_CSV,

        node

    )


    vuln_node=vulnerability_node_accuracy(

        rows

    )


    save_csv(

        VULN_NODETYPE_CSV,

        vuln_node

    )



    comparison=create_comparison(

        metrics

    )


    save_csv(

        COMPARISON_CSV,

        comparison

    )



    plot_confusion(

        labels,

        probabilities

    )


    plot_comparison(

        comparison

    )


    write_report(

        metrics,

        vuln,

        comparison

    )



    print()

    print("="*80)

    print(

        "STEP 24 COMPLETE"

    )

    print("="*80)



if __name__=="__main__":

    main()