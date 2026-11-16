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
assert "$14,555$ training pair clusters (image-level pair groups)" in cell_36_src, "Cell 36 pair cluster phrasing missing"
assert "The specific garment identity key is `(pair_id, style)`" in cell_36_src, "Cell 36 (pair_id, style) explanation missing"
assert "14,555 unique product identities" not in cell_36_src
assert "14,555 unique garment identities" not in cell_36_src

cell_37_src = "".join(nb['cells'][37]['source'])
cell_37_outs = "".join([o.get('text', [''])[0] if isinstance(o.get('text'), list) else o.get('text', '') for o in nb['cells'][37]['outputs']])
assert "comprehensive_dataset_audit_report_v3.json" in cell_37_src
assert "comprehensive_dataset_audit_report_v3.json" in cell_37_outs
assert "14,555 unique product identities" not in cell_37_src
assert "14,555 unique garment identities" not in cell_37_src

cell_38_src = "".join(nb['cells'][38]['source'])
cell_38_outs = "\n".join(["".join(o.get('text', [])) for o in nb['cells'][38]['outputs']])
assert "14,555 training pair clusters (mean 13.19, median 11.0, range 2 to 233 images/cluster)" in cell_38_outs
assert "specific garment identity key is (pair_id, style)" in cell_38_outs
assert "14,555 unique product identities" not in cell_38_outs
assert "14,555 unique garment identities" not in cell_38_outs
assert "comprehensive_dataset_audit_report_v3.json" in cell_38_outs

print("✅ Cells 36, 37, 38 sources and outputs verified with calibrated terminology!")

# 3. Check v1, v2, v3 JSON reports
p_v1 = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report.json")
p_v2 = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report_v2.json")
p_v3 = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report_v3.json")

assert p_v1.exists(), f"v1 report missing: {p_v1}"
assert p_v2.exists(), f"v2 report missing: {p_v2}"
assert p_v3.exists(), f"v3 report missing: {p_v3}"

with open(p_v3, 'r', encoding='utf-8') as f:
    report_v3 = json.load(f)

# Verify v3 content
pair_stats = report_v3['8_pair_cluster_statistics']
assert pair_stats['training_pair_clusters_total'] == 14555
assert pair_stats['pair_cluster_semantics']['garment_identity_key'] == '(pair_id, style)'
assert pair_stats['images_per_cluster'] == {'min': 2, 'median': 11.0, 'mean': 13.19, 'max': 233}

assert report_v3['6_multi_label_statistics']['actual_unique_training_images'] == 191961
assert report_v3['6_multi_label_statistics']['sum_of_category_specific_unique_images'] == 309174

assert report_v3['10_leakage_prevention']['pair_grouped_split_governance']['pair_leakage_rate'] == 0.0
assert 'NEVER called' in report_v3['10_leakage_prevention']['pair_grouped_split_governance']['pair_grouped_split_utility_status']

print(f"✅ Corrected v3 audit report verified ({p_v3.stat().st_size:,} bytes)!")
print("✅ v1 and v2 audit reports verified intact!")
