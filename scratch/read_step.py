import json
from pathlib import Path

p = Path("wardrobe_ml/outputs/comprehensive_dataset_audit_report.json")
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"File size: {p.stat().st_size:,} bytes")
print(f"Top-level keys ({len(data)}):")
for k in data.keys():
    v = data[k]
    if isinstance(v, dict):
        print(f"  - {k} (dict with {len(v)} keys)")
    elif isinstance(v, list):
        print(f"  - {k} (list with {len(v)} items)")
    else:
        print(f"  - {k}: {v}")
