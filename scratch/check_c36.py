import json
from pathlib import Path
nb_path = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\notebooks\01_dataset_exploration.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cell_36_src = "".join(nb['cells'][36]['source'])
for line in cell_36_src.splitlines():
    if "Training Pair Clusters" in line:
        print(line)
