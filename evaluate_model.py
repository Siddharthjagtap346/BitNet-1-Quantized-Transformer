import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
    classification_report
)

from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset
from data.collate import collate_fn

# ---------------- SETTINGS ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./checkpoints/checkpoint_step382500.pth"
BATCH_SIZE = 8

# ---------------- LOAD DATASET ----------------
dataset = UniProtDataset(
    tsv_path="data/uniprot_annotations.tsv",
    max_len=128,
    max_samples=10000
)

# ---------------- SPLIT DATASET ----------------
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------- LOAD MODEL ----------------
model = BitNetDecoder(
    vocab_size=dataset.tokenizer.vocab_size,
    d_model=96,
    n_layers=4,
    n_heads=4,
    max_seq_len=256
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- GET LOGITS ON VALIDATION ----------------
val_logits = []
val_true = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits, hidden_states, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        func_logits = model.predict_function(hidden_states, batch["attention_mask"])
        val_logits.append(func_logits.cpu())
        val_true.append(batch["function_labels"].cpu())

val_logits = torch.cat(val_logits).numpy()
val_true = torch.cat(val_true).numpy()
num_classes = val_true.shape[1]

# ---------------- PER-CLASS THRESHOLD TUNING ----------------
per_class_thresholds = []

for i in range(num_classes):
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 17):  # 0.1,0.15,...0.9
        preds_t = (torch.sigmoid(torch.tensor(val_logits[:, i])) > t).int().numpy()
        f1 = f1_score(val_true[:, i], preds_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    per_class_thresholds.append(best_thresh)

print("\n========== PER-CLASS THRESHOLDS (from validation) ==========")
for i, t in enumerate(per_class_thresholds):
    print(f"Class {i}: Threshold={t:.2f}")

# ---------------- EVALUATE ON TEST ----------------
test_logits = []
test_true = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits, hidden_states, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        func_logits = model.predict_function(hidden_states, batch["attention_mask"])
        test_logits.append(func_logits.cpu())
        test_true.append(batch["function_labels"].cpu())

test_logits = torch.cat(test_logits).numpy()
y_true = torch.cat(test_true).numpy()

# Apply the thresholds tuned on validation
y_pred = np.zeros_like(y_true, dtype=int)
for i, t in enumerate(per_class_thresholds):
    y_pred[:, i] = (torch.sigmoid(torch.tensor(test_logits[:, i])) > t).int().numpy()

# ---------------- PER-CLASS SUPPORT ----------------
print("\n========== PER-CLASS SUPPORT ==========")
for i in range(num_classes):
    print(f"Class {i}: {y_true[:, i].sum()} positive samples, Threshold={per_class_thresholds[i]:.2f}")

# ---------------- OVERALL METRICS ----------------
print("\n========== OVERALL METRICS ==========")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Micro Precision:", precision_score(y_true, y_pred, average="micro", zero_division=0))
print("Micro Recall:", recall_score(y_true, y_pred, average="micro", zero_division=0))
print("Micro F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))
print("Macro Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
print("Macro Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))

# ---------------- MAIN CONFUSION MATRIX ----------------
mcm = multilabel_confusion_matrix(y_true, y_pred)
TP = mcm[:,1,1].sum()
FP = mcm[:,0,1].sum()
FN = mcm[:,1,0].sum()
TN = mcm[:,0,0].sum()
print("\n========== MAIN CONFUSION MATRIX (Micro) ==========")
print("TN:", TN, "FP:", FP, "FN:", FN, "TP:", TP)
print("Matrix:\n", np.array([[TN, FP], [FN, TP]]))

# ---------------- PER-CLASS CONFUSION MATRICES ----------------
print("\n========== PER-CLASS CONFUSION MATRICES ==========")
FUNCTION_NAMES = getattr(dataset, "function_labels_names", [f"Class_{i}" for i in range(num_classes)])
for i, name in enumerate(FUNCTION_NAMES):
    tn, fp, fn, tp = mcm[i].ravel()
    print(f"\n--- {name} ---")
    print("Support (positives):", y_true[:, i].sum())
    print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
    print("F1:", f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

# ---------------- FULL CLASSIFICATION REPORT ----------------
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_true, y_pred, target_names=FUNCTION_NAMES, zero_division=0))
