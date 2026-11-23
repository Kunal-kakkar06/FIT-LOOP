import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"wardrobe_ml/notebooks/01_dataset_exploration.ipynb"
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_30 = nb['cells'][30]
print("Cell 30 outputs:")
for out in cell_30.get('outputs', []):
    print("".join(out.get('text', [])))
