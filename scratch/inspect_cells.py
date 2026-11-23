import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"wardrobe_ml/notebooks/01_dataset_exploration.ipynb"
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("".join(nb['cells'][6]['source']))
