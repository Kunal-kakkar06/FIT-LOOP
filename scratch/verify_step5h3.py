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

# 2. Check Cell 36, 37, 38 outputs and sources
cell_36_src = "".join(nb['cells'][36]['source'])
assert "Stage 8" in cell_36_src
assert "101338.json" in cell_36_src
assert "309,174" in cell_36_src
assert "Dataset Readiness Assessment" in cell_36_src

cell_37_src = "".join(nb['cells'][37]['source'])
cell_37_outs = "".join([o.get('text', [''])[0] if isinstance(o.get('text'), list) else o.get('text', '') for o in nb['cells'][37]['outputs']])
assert "comprehensive_dataset_audit_report.json" in cell_37_src
assert "comprehensive_dataset_audit_report.json" in cell_37_outs

cell_38_src = "".join(nb['cells'][38]['source'])
cell_38_outs = "\n".join(["".join(o.get('text', [])) for o in nb['cells'][38]['outputs']])
assert "DATASET READINESS ASSESSMENT" in cell_38_outs
assert "[A] Object Detection & Instance Segmentation Preparation" in cell_38_outs
assert "[B] Attribute-Label Preparation" in cell_38_outs
assert "[C] Image Preprocessing & Standardization" in cell_38_outs
assert "[D] Model-Specific Data Filtering & Curation" in cell_38_outs

print("✅ Cells 36, 37, 38 sources and outputs verified!")

# 3. Check JSON report
report_path = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\comprehensive_dataset_audit_report.json")
assert report_path.exists(), f"Report file does not exist at {report_path}"
with open(report_path, 'r', encoding='utf-8') as f:
    report = json.load(f)

expected_sections = [
    '1_dataset_inventory',
    '2_annotation_stats',
    '3_bbox_validity',
    '4_segmentation_and_landmarks',
    '5_category_distribution',
    '6_multi_label_statistics',
    '7_metadata_distributions',
    '8_pair_cluster_statistics',
    '9_official_partitions',
    '10_leakage_prevention',
    '11_known_limitations',
    '12_artifact_inventory'
]

for sec in expected_sections:
    assert sec in report, f"Missing section {sec} in audit report!"

print(f"✅ All 12 audit report sections present in {report_path.name} ({report_path.stat().st_size:,} bytes)!")

# Verify key metrics
assert report['1_dataset_inventory']['benchmark_total_images'] == 253879
assert report['1_dataset_inventory']['train']['image_count'] == 191961
assert report['1_dataset_inventory']['validation']['image_count'] == 32153
assert report['1_dataset_inventory']['test']['image_count'] == 29765
assert report['1_dataset_inventory']['test']['withheld_gt_count'] == 29765

assert report['2_annotation_stats']['total_garment_instances'] == 312186
assert report['3_bbox_validity']['annotation_present_count'] == 312186
assert report['3_bbox_validity']['strictly_positive_dimensions_count'] == 312185
assert report['3_bbox_validity']['degenerate_bbox_count'] == 1
assert report['3_bbox_validity']['degenerate_bbox_instances'][0]['file'] == '101338.json'

assert report['4_segmentation_and_landmarks']['segmentation_polygons']['instances_with_segmentation'] == 312186
assert report['4_segmentation_and_landmarks']['landmarks']['instances_with_landmarks'] == 312186

assert report['6_multi_label_statistics']['actual_unique_training_images'] == 191961
assert report['6_multi_label_statistics']['sum_of_category_specific_unique_images'] == 309174

assert report['8_pair_cluster_statistics']['training_pair_clusters_total'] == 14555
assert report['8_pair_cluster_statistics']['cross_domain_coexistence']['pairs_with_both_shop_and_user'] == 14555

assert report['10_leakage_prevention']['pair_grouped_split_governance']['pair_leakage_rate'] == 0.0
assert 'NEVER called' in report['10_leakage_prevention']['pair_grouped_split_governance']['pair_grouped_split_utility_status']

print("✅ All key numerical metrics and governance constraints verified!")
