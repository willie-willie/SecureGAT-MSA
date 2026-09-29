#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:22:02 2026

@author: Willie
"""


"""
SecureGAT-Agent 

SecureGAT-MSA:
Multi-head Structural Attention Graph Attention Network

Purpose
-------
Train a structural multi-head attention extension of SecureGAT-SA.


"""

import csv
import json
import math
import random
import hashlib
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ============================================================
# PATHS
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


OUTPUT_ROOT = (
    MODEL_READY_ROOT
    /
    "securegat_msa_training"
)


FIGURE_ROOT = (
    OUTPUT_ROOT
    /
    "figures"
)


SPLIT_CSV = (
    DATASET_ROOT
    /
    "contract_level_splits.csv"
)


STEP09_REPORT = (
    MODEL_READY_ROOT
    /
    "model_ready_report.json"
)


BEST_MODEL = (
    OUTPUT_ROOT
    /
    "best_model.pt"
)


FINAL_MODEL = (
    OUTPUT_ROOT
    /
    "final_model.pt"
)


TRAINING_HISTORY_CSV = (
    OUTPUT_ROOT
    /
    "training_history.csv"
)


TEST_METRICS_JSON = (
    OUTPUT_ROOT
    /
    "test_metrics.json"
)


TRAINING_REPORT_JSON = (
    OUTPUT_ROOT
    /
    "training_report.json"
)



# ============================================================
# CONFIGURATION
# ============================================================

PROTOCOL_VERSION = (
    "step23_securegat_msa_v1"
)


NUM_NODE_TYPES = 41


NODE_EMBEDDING_DIM = 32


GAT_HIDDEN_DIM = 64


GAT_HEADS = 4


GAT_DROPOUT = 0.20


CLASSIFIER_DROPOUT = 0.20


STRUCTURAL_ATTENTION_HIDDEN = 32


NUM_STRUCTURAL_HEADS = 4


LEARNING_RATE = 0.003


WEIGHT_DECAY = 1e-6


MAX_EPOCHS = 2000


EARLY_STOPPING_PATIENCE = 300


CHECKPOINT_SELECTION_METRIC = (
    "validation_accuracy"
)


THRESHOLD = 0.50


TARGET_FIELD = (
    "injection_label"
)

# ============================================================
# FAILURE HANDLER
# ============================================================


def fail(message):

    print()
    print("=" * 80)
    print("STEP 23 FAILED")
    print("=" * 80)
    print(message)
    print("=" * 80)

    raise RuntimeError(message)



# ============================================================
# DIRECTORY
# ============================================================


def ensure_directories():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )



# ============================================================
# JSON
# ============================================================


def load_json(path):

    if not path.exists():

        fail(
            f"Missing JSON:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)



def save_json(path, data):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )



# ============================================================
# SPLIT LOADING
# ============================================================


def load_split():

    if not SPLIT_CSV.exists():

        fail(
            f"Missing split file:\n{SPLIT_CSV}"
        )


    records=[]


    with SPLIT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:


        reader = csv.DictReader(f)


        required = {

            "split",
            "vulnerability",
            "contract_id",
            "graph",
            "graph_key",
            "graph_path",

        }


        missing = (
            required
            -
            set(reader.fieldnames)
        )


        if missing:

            fail(
                "Split CSV missing:\n"
                +
                str(missing)
            )


        for row in reader:

            records.append(row)


    return records



# ============================================================
# TORCH LOAD
# ============================================================


def safe_torch_load(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )



# ============================================================
# GRAPH ATTRIBUTE
# ============================================================


def get_attribute(graph, name):

    if hasattr(
        graph,
        name
    ):

        return getattr(
            graph,
            name
        )


    if isinstance(
        graph,
        dict
    ):

        return graph.get(name)


    return None



# ============================================================
# AUTHORITATIVE GRAPH RESOLUTION
# ============================================================


def resolve_graph(row):

    """
    Step 08 graph_path is authoritative.
    Never search by basename.
    """


    graph_path = Path(
        row["graph_path"]
    )


    if not graph_path.is_absolute():

        graph_path = (
            ROOT
            /
            graph_path
        )


    graph_path = graph_path.resolve()


    if not graph_path.exists():

        fail(
            f"Graph missing:\n{graph_path}"
        )


    return graph_path



# ============================================================
# GRAPH PREPARATION
# ============================================================


def prepare_graph(graph, device):


    node_type = get_attribute(
        graph,
        "node_type",
    )


    node_depth = get_attribute(
        graph,
        "node_depth",
    )


    edge_index = get_attribute(
        graph,
        "edge_index",
    )


    labels = get_attribute(
        graph,
        TARGET_FIELD,
    )


    if (
        node_type is None
        or node_depth is None
        or edge_index is None
        or labels is None
    ):

        fail(
            "Graph missing required fields."
        )



    node_type = torch.as_tensor(
        node_type,
        dtype=torch.long,
        device=device,
    )


    node_depth = torch.as_tensor(
        node_depth,
        dtype=torch.float32,
        device=device,
    )


    edge_index = torch.as_tensor(
        edge_index,
        dtype=torch.long,
        device=device,
    )


    labels = torch.as_tensor(
        labels,
        dtype=torch.float32,
        device=device,
    )


    # normalize AST depth

    max_depth = (
        node_depth.max()
        .item()
    )


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

    )



# ============================================================
# SECUREGAT-MSA MODEL
# ============================================================


class SecureGATMSA(nn.Module):

    """
    SecureGAT with Multi-head Structural Attention.

    Backbone:
        same as SecureGAT-SA

    Modification:
        single structural gate
        replaced by
        multi-head structural attention

    """


    def __init__(

        self,

        num_node_types=NUM_NODE_TYPES,

        embedding_dim=NODE_EMBEDDING_DIM,

        hidden_dim=GAT_HIDDEN_DIM,

        heads=GAT_HEADS,

        dropout=GAT_DROPOUT,

        classifier_dropout=CLASSIFIER_DROPOUT,

        attention_hidden=
            STRUCTURAL_ATTENTION_HIDDEN,

        num_attention_heads=
            NUM_STRUCTURAL_HEADS,

    ):


        super().__init__()



        self.node_embedding = nn.Embedding(

            num_node_types,

            embedding_dim,

        )



        # -----------------------------
        # Same GAT backbone
        # -----------------------------


        self.gat1 = GATConv(

            embedding_dim + 1,

            hidden_dim,

            heads=heads,

            dropout=dropout,

            concat=True,

        )


        self.gat2 = GATConv(

            hidden_dim * heads,

            hidden_dim,

            heads=1,

            dropout=dropout,

            concat=False,

        )



        # -----------------------------
        # Multi-head structural attention
        # -----------------------------


        attention_input_dim = (

            hidden_dim

            +

            embedding_dim

            +

            1

        )



        self.structural_heads = nn.ModuleList(

            [

                nn.Sequential(

                    nn.Linear(

                        attention_input_dim,

                        attention_hidden,

                    ),

                    nn.ELU(),

                    nn.Linear(

                        attention_hidden,

                        1,

                    ),

                    nn.Sigmoid(),

                )

                for _ in range(
                    num_attention_heads
                )

            ]

        )



        # combine attention heads

        self.attention_projection = nn.Sequential(

            nn.Linear(

                num_attention_heads,

                1,

            ),

            nn.Sigmoid(),

        )



        self.dropout = nn.Dropout(

            classifier_dropout

        )



        self.classifier = nn.Linear(

            hidden_dim,

            1,

        )



    def forward(

        self,

        node_type,

        node_depth,

        edge_index,

    ):



        # node semantics

        type_embedding = (

            self.node_embedding(

                node_type

            )

        )


        depth = node_depth.view(

            -1,

            1,

        )


        x = torch.cat(

            [

                type_embedding,

                depth,

            ],

            dim=1,

        )



        # GAT propagation


        x = self.gat1(

            x,

            edge_index,

        )


        x = F.elu(x)


        x = self.dropout(x)



        x = self.gat2(

            x,

            edge_index,

        )


        x = F.elu(x)



        # =====================================================
        # MULTI HEAD STRUCTURAL ATTENTION
        # =====================================================


        attention_input = torch.cat(

            [

                x,

                type_embedding,

                depth,

            ],

            dim=1,

        )



        heads=[]



        for attention_head in self.structural_heads:


            heads.append(

                attention_head(

                    attention_input

                )

            )



        multi_attention = torch.cat(

            heads,

            dim=1,

        )



        attention = (

            self.attention_projection(

                multi_attention

            )

        )



        # residual structural modulation


        x = x * (

            1.0

            +

            attention

        )



        x = self.dropout(x)



        logits = self.classifier(

            x

        ).view(-1)



        return logits
    
# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    labels,
    probabilities,
    threshold=THRESHOLD,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )


    predictions = (
        probabilities >= threshold
    ).astype(
        np.int64
    )


    metrics = {}


    metrics["accuracy"] = float(
        accuracy_score(
            labels,
            predictions,
        )
    )


    metrics["precision"] = float(
        precision_score(
            labels,
            predictions,
            zero_division=0,
        )
    )


    metrics["recall"] = float(
        recall_score(
            labels,
            predictions,
            zero_division=0,
        )
    )


    metrics["f1"] = float(
        f1_score(
            labels,
            predictions,
            zero_division=0,
        )
    )


    try:

        metrics["roc_auc"] = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

    except ValueError:

        metrics["roc_auc"] = float("nan")



    try:

        metrics["pr_auc"] = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

    except ValueError:

        metrics["pr_auc"] = float("nan")



    try:

        metrics["mcc"] = float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        )

    except ValueError:

        metrics["mcc"] = float("nan")


    return metrics



# ============================================================
# TRAINING EPOCH
# ============================================================


def run_epoch(

    model,

    records,

    optimizer,

    device,

):


    model.train()


    criterion = nn.BCEWithLogitsLoss()


    total_loss = 0.0

    total_nodes = 0


    all_labels=[]

    all_probabilities=[]



    for row in records:


        graph_path = resolve_graph(
            row
        )


        graph = safe_torch_load(
            graph_path
        )



        (
            node_type,

            node_depth,

            edge_index,

            labels,

        ) = prepare_graph(

            graph,

            device,

        )



        optimizer.zero_grad()



        logits = model(

            node_type,

            node_depth,

            edge_index,

        )



        loss = criterion(

            logits,

            labels,

        )



        loss.backward()


        optimizer.step()



        probabilities = torch.sigmoid(

            logits

        )



        node_count = int(

            labels.numel()

        )



        total_loss += (

            float(loss.item())

            *

            node_count

        )


        total_nodes += node_count



        all_labels.extend(

            labels.detach()

            .cpu()

            .numpy()

            .astype(np.int64)

            .tolist()

        )



        all_probabilities.extend(

            probabilities.detach()

            .cpu()

            .numpy()

            .tolist()

        )



    average_loss = (

        total_loss

        /

        total_nodes

    )


    metrics = calculate_metrics(

        all_labels,

        all_probabilities,

    )


    return (

        average_loss,

        metrics,

    )



# ============================================================
# EVALUATION
# ============================================================


@torch.no_grad()

def evaluate(

    model,

    records,

    device,

):


    model.eval()


    criterion = nn.BCEWithLogitsLoss()


    total_loss = 0.0

    total_nodes = 0


    all_labels=[]

    all_probabilities=[]



    for row in records:


        graph_path = resolve_graph(
            row
        )


        graph = safe_torch_load(
            graph_path
        )


        (

            node_type,

            node_depth,

            edge_index,

            labels,

        ) = prepare_graph(

            graph,

            device,

        )



        logits = model(

            node_type,

            node_depth,

            edge_index,

        )



        loss = criterion(

            logits,

            labels,

        )



        probabilities = torch.sigmoid(

            logits

        )



        node_count = int(

            labels.numel()

        )


        total_loss += (

            float(loss.item())

            *

            node_count

        )


        total_nodes += node_count



        all_labels.extend(

            labels.cpu()

            .numpy()

            .astype(np.int64)

            .tolist()

        )


        all_probabilities.extend(

            probabilities.cpu()

            .numpy()

            .tolist()

        )



    average_loss = (

        total_loss

        /

        total_nodes

    )



    metrics = calculate_metrics(

        all_labels,

        all_probabilities,

    )


    return (

        average_loss,

        metrics,

        np.asarray(

            all_labels,

            dtype=np.int64,

        ),

        np.asarray(

            all_probabilities,

            dtype=np.float64,

        ),

    )



# ============================================================
# TRAINING HISTORY
# ============================================================


def save_training_history(history):


    fields = [

        "epoch",

        "train_loss",

        "validation_loss",

        "train_accuracy",

        "validation_accuracy",

        "train_precision",

        "validation_precision",

        "train_recall",

        "validation_recall",

        "train_f1",

        "validation_f1",

        "train_roc_auc",

        "validation_roc_auc",

        "train_pr_auc",

        "validation_pr_auc",

        "train_mcc",

        "validation_mcc",

    ]



    with TRAINING_HISTORY_CSV.open(

        "w",

        newline="",

        encoding="utf-8",

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=fields,

        )


        writer.writeheader()


        writer.writerows(history)



# ============================================================
# CHECKPOINT SAVE
# ============================================================


def save_checkpoint(

    model,

    optimizer,

    epoch,

    validation_accuracy,

):


    torch.save(

        {

            "model_state_dict":

                model.state_dict(),


            "optimizer_state_dict":

                optimizer.state_dict(),


            "epoch":

                epoch,


            "validation_accuracy":

                validation_accuracy,


            "protocol_version":

                PROTOCOL_VERSION,


            "architecture":

                "SecureGAT-MSA",


            "seed":

                SEED,

        },

        BEST_MODEL,

    )
        
# ============================================================
# MAIN
# ============================================================


def main():


    set_seed()


    print("=" * 80)

    print(
        "SecureGAT-Agent Step 23"
    )

    print(
        "SecureGAT-MSA Structural Multi-head Attention Training"
    )

    print("=" * 80)



    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print()

    print(

        f"Device: {device}"

    )


    print(

        "Dataset annotations: FROZEN"

    )

    print(

        "Step 08 split: SOURCE OF TRUTH"

    )

    print(

        "Architecture: SecureGAT-MSA"

    )


    ensure_directories()



    # --------------------------------------------------------
    # LOAD SPLIT
    # --------------------------------------------------------


    records = load_split()



    train_records = [

        r

        for r in records

        if r["split"] == "train"

    ]


    validation_records = [

        r

        for r in records

        if r["split"] == "validation"

    ]


    test_records = [

        r

        for r in records

        if r["split"] == "test"

    ]



    print()

    print(

        f"Train graphs      : {len(train_records)}"

    )

    print(

        f"Validation graphs : {len(validation_records)}"

    )

    print(

        f"Test graphs       : {len(test_records)}"

    )



    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------


    train_contracts = {

        r["contract_id"]

        for r in train_records

    }


    validation_contracts = {

        r["contract_id"]

        for r in validation_records

    }


    test_contracts = {

        r["contract_id"]

        for r in test_records

    }



    if train_contracts & validation_contracts:

        fail(

            "Train-validation contract leakage detected"

        )


    if train_contracts & test_contracts:

        fail(

            "Train-test contract leakage detected"

        )


    if validation_contracts & test_contracts:

        fail(

            "Validation-test contract leakage detected"

        )


    print()

    print(

        "Contract leakage check: PASS"

    )



    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------


    model = SecureGATMSA().to(

        device

    )



    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )



    print()

    print(

        "Model initialized"

    )



    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------


    print()

    print("=" * 80)

    print("TRAINING")

    print("=" * 80)



    history = []


    best_validation_accuracy = -1.0


    best_epoch = None


    epochs_without_improvement = 0



    for epoch in range(

        1,

        MAX_EPOCHS + 1,

    ):



        train_loss, train_metrics = run_epoch(

            model,

            train_records,

            optimizer,

            device,

        )



        validation_loss, validation_metrics, _, _ = evaluate(

            model,

            validation_records,

            device,

        )



        row = {


            "epoch":

                epoch,


            "train_loss":

                train_loss,


            "validation_loss":

                validation_loss,


            "train_accuracy":

                train_metrics["accuracy"],


            "validation_accuracy":

                validation_metrics["accuracy"],


            "train_precision":

                train_metrics["precision"],


            "validation_precision":

                validation_metrics["precision"],


            "train_recall":

                train_metrics["recall"],


            "validation_recall":

                validation_metrics["recall"],


            "train_f1":

                train_metrics["f1"],


            "validation_f1":

                validation_metrics["f1"],


            "train_roc_auc":

                train_metrics["roc_auc"],


            "validation_roc_auc":

                validation_metrics["roc_auc"],


            "train_pr_auc":

                train_metrics["pr_auc"],


            "validation_pr_auc":

                validation_metrics["pr_auc"],


            "train_mcc":

                train_metrics["mcc"],


            "validation_mcc":

                validation_metrics["mcc"],

        }



        history.append(row)



        print(

            f"Epoch {epoch:04d} | "

            f"Train Loss={train_loss:.5f} | "

            f"Val Loss={validation_loss:.5f} | "

            f"Val Acc={validation_metrics['accuracy']:.5f} | "

            f"Val F1={validation_metrics['f1']:.5f}"

        )



        current_accuracy = validation_metrics["accuracy"]



        # ----------------------------------------------------
        # PRIMARY CHECKPOINT METRIC
        #
        # IMPORTANT:
        # Validation Accuracy
        #
        # F1 is reported only.
        # ----------------------------------------------------


        if current_accuracy > best_validation_accuracy:



            best_validation_accuracy = current_accuracy


            best_epoch = epoch


            epochs_without_improvement = 0



            save_checkpoint(

                model,

                optimizer,

                epoch,

                current_accuracy,

            )



        else:


            epochs_without_improvement += 1



        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:


            print()

            print(

                "Early stopping triggered."

            )

            break



    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------


    save_training_history(

        history

    )



    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------


    torch.save(

        {

            "model_state_dict":

                model.state_dict(),


            "epoch":

                history[-1]["epoch"],


            "architecture":

                "SecureGAT-MSA",


            "protocol_version":

                PROTOCOL_VERSION,


            "seed":

                SEED,

        },

        FINAL_MODEL,

    )



    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------


    if not BEST_MODEL.exists():

        fail(

            "Best model checkpoint missing."

        )


    checkpoint = safe_torch_load(

        BEST_MODEL

    )


    model.load_state_dict(

        checkpoint["model_state_dict"]

    )


    model.to(

        device

    )



    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------


    print()

    print("=" * 80)

    print(

        "FINAL TEST EVALUATION"

    )

    print("=" * 80)



    (

        test_loss,

        test_metrics,

        test_labels,

        test_probabilities,

    ) = evaluate(

        model,

        test_records,

        device,

    )



    for key, value in test_metrics.items():

        print(

            f"{key:<15}: {value:.6f}"

        )



    test_report = {


        "architecture":

            "SecureGAT-MSA",


        "protocol_version":

            PROTOCOL_VERSION,


        "best_epoch":

            best_epoch,


        "best_validation_accuracy":

            best_validation_accuracy,


        "test_loss":

            test_loss,


        "test_metrics":

            test_metrics,


        "graphs":

            {

                "train":

                    len(train_records),

                "validation":

                    len(validation_records),

                "test":

                    len(test_records),

            },


    }



    save_json(

        TEST_METRICS_JSON,

        test_report,

    )



    training_report = {


        "architecture":

            "SecureGAT-MSA",


        "device":

            str(device),


        "seed":

            SEED,


        "best_epoch":

            best_epoch,


        "best_validation_accuracy":

            best_validation_accuracy,


        "train_graphs":

            len(train_records),


        "validation_graphs":

            len(validation_records),


        "test_graphs":

            len(test_records),


    }



    save_json(

        TRAINING_REPORT_JSON,

        training_report,

    )



    print()

    print("=" * 80)

    print(

        "STEP 23 COMPLETE"

    )

    print("=" * 80)



    print(

        f"Best epoch: {best_epoch}"

    )

    print(

        f"Best validation accuracy: {best_validation_accuracy:.6f}"

    )

    print()

    print(

        f"Output directory:\n{OUTPUT_ROOT}"

    )




if __name__ == "__main__":

    main()