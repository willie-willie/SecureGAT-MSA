SecureGAT-MSA REFERENCE RESULT REPRODUCTION

Run from the SecureGAT-Agent project root:

python -m py_compile Model/train_securegat_msa.py
python Model/train_securegat_msa.py

python -m py_compile Model/evaluate_securegat_msa.py
python Model/evaluate_securegat_msa.py

Step configuration:
NUM_NODE_TYPES=41
NODE_EMBEDDING_DIM=32
GAT_HIDDEN_DIM=64
GAT_HEADS=4
GAT_DROPOUT=0.20
CLASSIFIER_DROPOUT=0.20
STRUCTURAL_ATTENTION_HIDDEN=32
NUM_STRUCTURAL_HEADS=4
LEARNING_RATE=0.003
WEIGHT_DECAY=1e-6
MAX_EPOCHS=2000
EARLY_STOPPING_PATIENCE=300
CHECKPOINT_SELECTION_METRIC=validation_accuracy
THRESHOLD=0.50
TARGET_FIELD=injection_label


