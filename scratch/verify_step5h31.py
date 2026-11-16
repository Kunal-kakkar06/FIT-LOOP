import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

nb_path = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\notebooks\01_dataset_exploration.ipynb")
print(f"Checking notebook: {nb_path}")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

assert len(nb['cells']) == 39, f"Expected 39 cells, got {len(nb['cells'])}"

# 1. Check all execution counts
expected_counts = {
    0: ( 'markdown', None ),
    1: ( 'markdown', None ),
    2: ( 'code', 1 ),
    3: ( 'code', 2 ),
    4: ( 'code', 3 ),
    5: ( 'markdown', None ),
    6: ( 'code', 4 ),
    7: ( 'code', 5 ),
    8: ( 'code', 6 ),
    9: ( 'markdown', None ),
    10: ( 'code', 7 ),
    11: ( 'code', 8 ),
    12: ( 'code', 10 ),
    13: ( 'code', 11 ),
    14: ( 'code', 12 ),
    15: ( 'code', 13 ),
    16: ( 'code', 9 ),
    17: ( 'markdown', None ),
    18: ( 'code', 14 ),
    19: ( 'code', 15 ),
    20: ( 'code', 16 ),
    21: ( 'code', 17 ),
    22: ( 'markdown', None ),
    23: ( 'code', 18 ),
    24: ( 'code', 19 ),
    25: ( 'code', 20 ),
    26: ( 'code', 21 ),
    27: ( 'code', 22 ),
    28: ( 'code', 23 ),
    29: ( 'markdown', None ),
    30: ( 'code', 24 ),
    31: ( 'code', 25 ),
    32: ( 'code', 26 ),
    33: ( 'markdown', None ),
    34: ( 'code', 27 ),
    35: ( 'code', 28 ),
    36: ( 'markdown', None ),
    37: ( 'code', 29 ),
    38: ( 'code', 30 ),
}

for idx, (exp_type, exp_cnt) in expected_counts.items():
    cell = nb['cells'][idx]
    actual_type = cell['cell_type']
    actual_cnt = cell.get('execution_count')
    assert actual_type == exp_type, f"Cell {idx} type mismatch: expected {exp_type}, got {actual_type}"
    assert actual_cnt == exp_cnt, f"Cell {idx} exec_count mismatch: expected {exp_cnt}, got {actual_cnt}"

print("✅ All 39 cells have verified types and execution counts!")

# 2. Check Cell 36, 37, 38 sources and outputs
cell_36_src = "".join(nb['cells'][36]['source'])
assert "0$=not labeled / invisible, $1$=occluded, $2$=visible" in cell_36_src, "Cell 36 landmark visibility semantics missing or incorrect"
assert "mean $13.19$ images/pair, median $11$, range $2$ to $233$" in cell_36_src, "Cell 36 pair statistics missing or incorrect"

# Ensure value 2 is NOT described as truncated in landmark visibility anywhere
for line in cell_36_src.splitlines():
    if "Landmark" in line:
        assert "truncated" not in line.lower(), f"Line mentions truncated in landmarks: {line}"

cell_37_src = "".join(nb['cells'][37]['source'])
cell_37_outs = "".join([o.get('text', [''])[0] if isinstance(o.get('text'), list) else o.get('text', '') for o in nb['cells'][37]['outputs']])
assert "comprehensive_dataset_audit_report_v2.json" in cell_37_src
assert "comprehensive_dataset_audit_report_v2.json" in cell_37_outs

cell_38_src = "".join(nb['cells'][38]['source'])
cell_38_outs = "\n".join(["".join(o.get('text', [])) for o in nb['cells'][38]['outputs']])
assert "Visibility coded as 0=not labeled / invisible, 1=occluded, 2=visible" in cell_38_outs, "Cell 38 landmark visibility output incorrect"
assert "mean 13.19, median 11.0, range 2 to 233 images/pair" in cell_38_outs, "Cell 38 pair statistics output incorrect"

for line in cell_38_outs.splitlines():
    if "Landmark" in line or "Visibility" in line:
        assert "truncated" not in line.lower(), f"Line mentions truncated in landmarks: {line}"

assert "comprehensive_dataset_audit_report_v2.json" in cell_38_outs

print("✅ Cells 36, 37, 38 sources and outputs verified with exact corrected semantics!")

# 3. Check v2 JSON report
report_v2_path = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report_v2.json")
assert report_v2_path.exists(), f"v2 report does not exist at {report_v2_path}"
with open(report_v2_path, 'r', encoding='utf-8') as f:
    report_v2 = json.load(f)

# Ensure v1 report still exists
report_v1_path = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report.json")
assert report_v1_path.exists(), f"Original report missing at {report_v1_path}"

# Verify v2 contents
vis = report_v2['4_segmentation_and_landmarks']['landmarks']['visibility_states']
assert vis == {'0': 'not labeled / invisible', '1': 'occluded', '2': 'visible'}, f"Incorrect visibility states in v2 JSON: {vis}"

pair_stats = report_v2['8_pair_cluster_statistics']['images_per_cluster']
assert pair_stats == {'min': 2, 'median': 11.0, 'mean': 13.19, 'max': 233}, f"Incorrect pair stats in v2 JSON: {pair_stats}"

assert report_v2['6_multi_label_statistics']['actual_unique_training_images'] == 191961
assert report_v2['6_multi_label_statistics']['sum_of_category_specific_unique_images'] == 309174

assert report_v2['10_leakage_prevention']['pair_grouped_split_governance']['pair_leakage_rate'] == 0.0
assert 'NEVER called' in report_v2['10_leakage_prevention']['pair_grouped_split_governance']['pair_grouped_split_utility_status']

print(f"✅ Corrected v2 audit report verified ({report_v2_path.stat().st_size:,} bytes)!")
print("✅ Original v1 audit report verified intact!")
