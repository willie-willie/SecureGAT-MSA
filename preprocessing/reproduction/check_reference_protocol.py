import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "datasets" / "annotated_graph_dataset" / "contract_level_splits.csv"

if not p.exists():
    raise SystemExit(f"Missing: {p}")

with p.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

counts = {}
contracts = {}
for r in rows:
    s = r["split"]
    counts[s] = counts.get(s, 0) + 1
    contracts.setdefault(s, set()).add(r["contract_id"])

expected = {"train":245, "validation":49, "test":56}
print("Reference split:", counts)

if counts != expected:
    raise SystemExit(f"Expected {expected}, found {counts}")

for a,b in [("train","validation"),("train","test"),("validation","test")]:
    overlap = contracts[a] & contracts[b]
    if overlap:
        raise SystemExit(f"Contract leakage: {a}/{b}: {sorted(overlap)[:10]}")

print("245/49/56 split: PASS")
print("Contract leakage: PASS")
