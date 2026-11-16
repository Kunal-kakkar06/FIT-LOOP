import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
p = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report_v2.json")
with open(p, 'r', encoding='utf-8') as f:
    text = f.read()

for i, line in enumerate(text.splitlines(), 1):
    if "product" in line.lower() or "identity" in line.lower() or "identities" in line.lower():
        print(f"Line {i}: {line}")
