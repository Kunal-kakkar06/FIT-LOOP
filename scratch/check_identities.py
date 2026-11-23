import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
nb_path = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\notebooks\01_dataset_exploration.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for idx in [36, 37, 38]:
    src = "".join(nb['cells'][idx]['source'])
    print(f"=== Cell {idx} ===")
    for line in src.splitlines():
        if "identit" in line.lower() or "pair" in line.lower():
            print(f"  {line}")
    print()
