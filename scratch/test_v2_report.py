import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

out_dir = Path("wardrobe_ml/outputs")
pkl_path = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\57b5c7be-d326-40ec-a129-e53bc9ecb41d\scratch\train_df.pkl")
df = pd.read_pickle(pkl_path)

DATASET_ROOT = Path(r"C:\Users\Lenovo\Downloads\DeepFashion2")
OUTPUT_DIR = out_dir

def safe_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return int(v) if hasattr(v, 'item') else int(v)

def safe_float(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    val = float(v) if hasattr(v, 'item') else float(v)
    return round(val, decimals)

audit_report_path = OUTPUT_DIR / 'comprehensive_dataset_audit_report_v2.json'

audit_report = {
    'report_title': 'FitLoop Comprehensive DeepFashion2 Dataset Audit & Exploration Report (v2)',
    'generated_at': datetime.now().isoformat(),
    'dataset_name': 'DeepFashion2',
    'dataset_root': str(DATASET_ROOT),
    'governance_phase': 'Phase 1 — Dataset Exploration, Metadata Semantics & Split Audit',

    '1_dataset_inventory': {
        'train': {
            'image_count': 191961,
            'annotation_count': 191961,
            'matched_pairs': 191961,
            'imgs_missing_anno_count': 0,
            'annos_missing_img_count': 0,
            'unexpected_missing_count': 0,
            'withheld_gt_count': 0,
            'ground_truth_status': 'Complete public ground truth (100% paired)',
            'format_counts': {'.jpg': 191961}
        },
        'validation': {
            'image_count': 32153,
            'annotation_count': 32153,
            'matched_pairs': 32153,
            'imgs_missing_anno_count': 0,
            'annos_missing_img_count': 0,
            'unexpected_missing_count': 0,
            'withheld_gt_count': 0,
            'ground_truth_status': 'Complete public ground truth (100% paired)',
            'format_counts': {'.jpg': 32153}
        },
        'test': {
            'image_count': 29765,
            'annotation_count': 0,
            'matched_pairs': 0,
            'imgs_missing_anno_count': 29765,
            'annos_missing_img_count': 0,
            'unexpected_missing_count': 0,
            'withheld_gt_count': 29765,
            'ground_truth_status': 'Benchmark challenge test set (ground truth withheld by benchmark authors)',
            'format_counts': {'.jpg': 29765}
        },
        'benchmark_total_images': 253879,
        'labeled_total_images': 224114,
        'labeled_total_annotations': 224114
    },

    '2_annotation_stats': {
        'denominator_garment_instances': safe_int(len(df)),
        'total_garment_instances': safe_int(len(df)),
        'total_unique_images': safe_int(df['file'].nunique()),
        'mean_garments_per_image': safe_float(len(df) / df['file'].nunique(), 2),
        'annotation_completeness_presence': {
            'bounding_box_presence': {'count': 312186, 'pct': 100.0, 'denominator': 312186},
            'segmentation_presence': {'count': 312186, 'pct': 100.0, 'denominator': 312186},
            'landmarks_presence': {'count': 312186, 'pct': 100.0, 'denominator': 312186},
            'note': 'Presence indicates key and coordinate data exist; landmark presence does not imply all keypoints are visible (visibility states: 0=not labeled / invisible, 1=occluded, 2=visible).'
        }
    },

    '3_bbox_validity': {
        'denominator_garment_instances': 312186,
        'annotation_present_count': 312186,
        'annotation_present_pct': 100.0,
        'numeric_coordinates_valid_count': 312186,
        'numeric_coordinates_valid_pct': 100.0,
        'strictly_positive_dimensions_count': 312185,
        'strictly_positive_dimensions_pct': 99.9997,
        'degenerate_bbox_count': 1,
        'degenerate_bbox_pct': 0.00032,
        'degenerate_bbox_instances': [
            {
                'file': '101338.json',
                'item_key': 'item2',
                'category_name': 'short sleeve top',
                'category_id': 1,
                'bounding_box': [0, 0, 0, 0],
                'width': 0,
                'height': 0,
                'segmentation_present': True,
                'landmarks_present': True,
                'status': 'Preserved untouched in raw data; must be masked/filtered during bbox training loss computation'
            }
        ],
        'note': '312,185 out of 312,186 bounding boxes have strictly positive width and height (w > 0, h > 0).'
    },

    '4_segmentation_and_landmarks': {
        'denominator_garment_instances': 312186,
        'segmentation_polygons': {
            'instances_with_segmentation': 312186,
            'percentage': 100.0,
            'mean_vertices_per_garment': 38.79,
            'min_vertices': 4,
            'max_vertices': 928,
            'multi_polygon_support': 'Multiple polygons present in occluded or multipart garments'
        },
        'landmarks': {
            'instances_with_landmarks': 312186,
            'percentage': 100.0,
            'mean_landmarks_per_garment': 20.62,
            'min_landmarks': 8,
            'max_landmarks': 39,
            'visibility_states': {
                '0': 'not labeled / invisible',
                '1': 'occluded',
                '2': 'visible'
            },
            'note': 'Landmark counts vary by category taxonomy (from 8 to 39 keypoints depending on garment structure). DeepFashion2 encodes landmark visibility as: 0 = not labeled / invisible, 1 = occluded, 2 = visible.'
        }
    },

    '5_category_distribution': {
        'denominator_garment_instances': 312186,
        'denominator_unique_images': 191961,
        'categories_count': 13,
        'imbalance_ratio': round(71645 / 543, 1),
        'most_frequent_category': {'name': 'short sleeve top', 'garments': 71645, 'pct': 22.95},
        'rarest_category': {'name': 'short sleeve outwear', 'garments': 543, 'pct': 0.17},
        'category_metrics': [
            {'category_id': 1,  'name': 'short sleeve top',    'garments': 71645, 'garment_pct': 22.95, 'category_unique_images': 70586, 'image_presence_pct': 36.77, 'pair_clusters': 6945, 'pair_presence_pct': 47.72},
            {'category_id': 8,  'name': 'trousers',            'garments': 55387, 'garment_pct': 17.74, 'category_unique_images': 54969, 'image_presence_pct': 28.64, 'pair_clusters': 6631, 'pair_presence_pct': 45.56},
            {'category_id': 7,  'name': 'shorts',              'garments': 36616, 'garment_pct': 11.73, 'category_unique_images': 36332, 'image_presence_pct': 18.93, 'pair_clusters': 4663, 'pair_presence_pct': 32.04},
            {'category_id': 2,  'name': 'long sleeve top',     'garments': 36064, 'garment_pct': 11.55, 'category_unique_images': 35751, 'image_presence_pct': 18.62, 'pair_clusters': 4052, 'pair_presence_pct': 27.84},
            {'category_id': 9,  'name': 'skirt',               'garments': 30835, 'garment_pct': 9.88,  'category_unique_images': 30625, 'image_presence_pct': 15.95, 'pair_clusters': 3476, 'pair_presence_pct': 23.88},
            {'category_id': 12, 'name': 'vest dress',          'garments': 17949, 'garment_pct': 5.75,  'category_unique_images': 17767, 'image_presence_pct': 9.26,  'pair_clusters': 1453, 'pair_presence_pct': 9.98},
            {'category_id': 10, 'name': 'short sleeve dress',  'garments': 17211, 'garment_pct': 5.51,  'category_unique_images': 17028, 'image_presence_pct': 8.87,  'pair_clusters': 1355, 'pair_presence_pct': 9.31},
            {'category_id': 5,  'name': 'vest',                'garments': 16095, 'garment_pct': 5.16,  'category_unique_images': 15974, 'image_presence_pct': 8.32,  'pair_clusters': 1535, 'pair_presence_pct': 10.55},
            {'category_id': 4,  'name': 'long sleeve outwear', 'garments': 13457, 'garment_pct': 4.31,  'category_unique_images': 13381, 'image_presence_pct': 6.97,  'pair_clusters': 1714, 'pair_presence_pct': 11.78},
            {'category_id': 11, 'name': 'long sleeve dress',   'garments': 7907,  'garment_pct': 2.53,  'category_unique_images': 7803,  'image_presence_pct': 4.06,  'pair_clusters': 613,  'pair_presence_pct': 4.21},
            {'category_id': 13, 'name': 'sling dress',         'garments': 6492,  'garment_pct': 2.08,  'category_unique_images': 6455,  'image_presence_pct': 3.36,  'pair_clusters': 598,  'pair_presence_pct': 4.11},
            {'category_id': 6,  'name': 'sling',               'garments': 1985,  'garment_pct': 0.64,  'category_unique_images': 1964,  'image_presence_pct': 1.02,  'pair_clusters': 209,  'pair_presence_pct': 1.44},
            {'category_id': 3,  'name': 'short sleeve outwear', 'garments': 543,   'garment_pct': 0.17,  'category_unique_images': 539,   'image_presence_pct': 0.28,  'pair_clusters': 99,   'pair_presence_pct': 0.68}
        ]
    },

    '6_multi_label_statistics': {
        'actual_unique_training_images': 191961,
        'sum_of_category_specific_unique_images': 309174,
        'overcounting_ratio': 1.6106,
        'images_with_multiple_categories': 114918,
        'images_with_multiple_categories_pct': 59.87,
        'distinct_categories_per_image_distribution': {
            '1_category': {'images': 77043, 'pct': 40.13},
            '2_categories': {'images': 112689, 'pct': 58.70},
            '3_categories': {'images': 2163, 'pct': 1.13},
            '4_categories': {'images': 66, 'pct': 0.03}
        },
        'top_5_category_cooccurrences': [
            {'categories': 'short sleeve top + trousers', 'images': 20399, 'pct_of_training_images': 10.63},
            {'categories': 'short sleeve top + shorts', 'images': 19829, 'pct_of_training_images': 10.33},
            {'categories': 'long sleeve top + trousers', 'images': 16447, 'pct_of_training_images': 8.57},
            {'categories': 'short sleeve top + skirt', 'images': 14703, 'pct_of_training_images': 7.66},
            {'categories': 'long sleeve top + skirt', 'images': 7067, 'pct_of_training_images': 3.68}
        ],
        'same_category_multi_instance_images': {
            'image_category_pairs_with_multi_items': 3006,
            'excess_garments': 3012,
            'explanation': '3,006 image-category combinations contain >= 2 garments of the exact same category (312,186 garments - 309,174 unique image-category presences = 3,012 excess items).'
        }
    },

    '7_metadata_distributions': {
        'denominator_garment_instances': 312186,
        'denominator_unique_images': 191961,
        'viewpoint_distribution': {
            '1_No_Wear': {'count': 22675, 'pct': 7.26, 'description': 'Flat-lay, hanger, or mannequin photos without a person'},
            '2_Frontal': {'count': 244459, 'pct': 78.31, 'description': 'Front-facing garment view on person'},
            '3_Side_Back': {'count': 45052, 'pct': 14.43, 'description': 'Side profile or back view on person'}
        },
        'occlusion_distribution': {
            '1_Slight': {'count': 145379, 'pct': 46.57, 'description': 'Landmark-based: <20% of landmarks occluded'},
            '2_Medium': {'count': 147102, 'pct': 47.12, 'description': 'Landmark-based: 20% to 50% of landmarks occluded'},
            '3_Heavy': {'count': 19705, 'pct': 6.31, 'description': 'Landmark-based: >50% of landmarks occluded'}
        },
        'zoom_in_distribution': {
            '1_No_Zoom': {'count': 211894, 'pct': 67.87, 'description': 'Landmark boundary truncation: full garment inside frame'},
            '2_Medium_Zoom': {'count': 66643, 'pct': 21.35, 'description': 'Landmark boundary truncation: partial garment truncation'},
            '3_Large_Zoom': {'count': 33649, 'pct': 10.78, 'description': 'Landmark boundary truncation: substantial garment truncation'}
        },
        'scale_distribution': {
            '1_Small': {'count': 78735, 'pct': 25.22, 'description': 'Small garment area'},
            '2_Modest': {'count': 157021, 'pct': 50.30, 'description': 'Modest garment area'},
            '3_Large': {'count': 76430, 'pct': 24.48, 'description': 'Large garment area'}
        },
        'source_domain_distribution_unique_images': {
            'shop': {'count': 132157, 'pct': 68.85, 'description': 'Commercial catalog studio photos'},
            'user': {'count': 59804, 'pct': 31.15, 'description': 'Consumer selfies and unconstrained mobile photos'}
        }
    },

    '8_pair_cluster_statistics': {
        'training_pair_clusters_total': 14555,
        'images_per_cluster': {
            'min': 2,
            'median': 11.0,
            'mean': 13.19,
            'max': 233
        },
        'cross_domain_coexistence': {
            'pairs_with_both_shop_and_user': 14555,
            'pairs_with_both_shop_and_user_pct': 100.0,
            'pairs_only_shop': 0,
            'pairs_only_user': 0
        },
        'category_span_per_pair': {
            'pairs_spanning_multiple_categories': 11861,
            'pairs_spanning_multiple_categories_pct': 81.49,
            'distribution': {
                '1_category': {'pairs': 2694, 'pct': 18.51},
                '2_categories': {'pairs': 6722, 'pct': 46.18},
                '3_categories': {'pairs': 3636, 'pct': 24.98},
                '4_categories': {'pairs': 1253, 'pct': 8.61},
                '5+_categories': {'pairs': 250, 'pct': 1.72}
            }
        },
        'sum_of_category_specific_pairs': 33343,
        'pair_overcounting_ratio': 2.2908
    },

    '9_official_partitions': {
        'train': {
            'images': 191961,
            'garments': 312186,
            'pair_clusters': 14555,
            'annotations_status': 'Complete public ground truth (100% available)'
        },
        'validation': {
            'images': 32153,
            'garments': 'Complete public ground truth (32,153 annotations present)',
            'pair_clusters': 2279,
            'pair_id_range': '[1 ... 2,372] with 93 non-contiguous index gaps',
            'annotations_status': 'Public ground truth available for local validation'
        },
        'test': {
            'images': 29765,
            'garments': 'Annotations withheld by benchmark creators',
            'pair_clusters': 'Withheld by benchmark creators (not inferred)',
            'annotations_status': 'Ground truth withheld for benchmark challenge evaluation'
        },
        'partition_independence_notes': [
            'Physical filesystem separation: train/, validation/, test/ are completely disjoint directories.',
            'Identifier namespaces: pair_id numbering resets independently per partition.',
            'Scientific boundary: Cross-split visual disjointness is a property of the benchmark creators construction protocol, not locally proven by cross-partition embedding comparisons.'
        ]
    },

    '10_leakage_prevention': {
        'random_image_split_risk': {
            'leakage_description': 'Random image-level splitting scatters photos of the exact same product identity between train and validation partitions.',
            'theoretical_expected_leakage_rates': {
                'empirical_distribution_75_15_10': {'leakage_pct': 90.48, 'purity_pct': 9.52, 'type': 'Theoretical expectation integrated over 14,555 cluster sizes'},
                'empirical_distribution_80_10_10': {'leakage_pct': 86.08, 'purity_pct': 13.92, 'type': 'Theoretical expectation integrated over 14,555 cluster sizes'},
                'empirical_distribution_80_20': {'leakage_pct': 86.00, 'purity_pct': 14.00, 'type': 'Theoretical expectation integrated over 14,555 cluster sizes'},
                'mean_cluster_size_heuristic_k13': {'leakage_pct': 97.3, 'purity_pct': 2.7, 'type': 'Theoretical estimate based on average cluster size (k=13.19)'}
            },
            'note': 'None of these random-split estimates are experimentally measured split outcomes; they are mathematical probability expectations.'
        },
        'pair_grouped_split_governance': {
            'pair_leakage_rate': 0.0,
            'pair_leakage_type': 'Guaranteed by construction (all images and garments of pair_id stay in single partition)',
            'pair_grouped_split_utility_status': 'Hardened with ratio and pair_id validation; defined in Cell 34 but NEVER called',
            'actual_splits_generated': 'None (zero physical files or manifest CSVs generated; official benchmark partitions preserved)'
        }
    },

    '11_known_limitations': [
        'Single degenerate bounding box: train/annos/101338.json, item2 (short sleeve top) has bbox [0,0,0,0] despite complete segmentation and landmarks. Must be masked/filtered during detection model training.',
        'Benchmark test ground truth withheld: Local validation and model selection must rely strictly on the official validation split (32,153 images).',
        'Severe category imbalance: 71,645 short sleeve top instances vs 543 short sleeve outwear instances (131.9x imbalance ratio).',
        'Multi-label co-occurrence: 59.9% of training photos contain >= 2 garments, requiring multi-label object detection/segmentation heads.',
        'Consumer selfie noise: 9.03% of sampled images have Laplacian blur scores < 100, reflecting real-world handheld mobile capture artifacts.',
        'Partition-local pair_id namespaces: pair_id values cannot be compared numerically across train and validation splits.'
    ],

    '12_artifact_inventory': [
        {'name': 'verified_category_distribution.csv', 'path': 'wardrobe_ml/outputs/verified_category_distribution.csv', 'description': 'Verified garment instances and unique images per category'},
        {'name': 'cell_13_category_distribution.png', 'path': 'wardrobe_ml/outputs/cell_13_category_distribution.png', 'description': 'Category frequency distribution plot'},
        {'name': 'cell_14_bounding_box_analysis.png', 'path': 'wardrobe_ml/outputs/cell_14_bounding_box_analysis.png', 'description': 'Bounding box dimensions and aspect ratios'},
        {'name': 'cell_15_annotation_coverage.png', 'path': 'wardrobe_ml/outputs/cell_15_annotation_coverage.png', 'description': 'Annotation field completeness bars'},
        {'name': 'cell_20_quality_distributions.png', 'path': 'wardrobe_ml/outputs/cell_20_quality_distributions.png', 'description': 'Image dimensions, file sizes, and blur score distributions'},
        {'name': 'cell_21_brightness_contrast_scatter.png', 'path': 'wardrobe_ml/outputs/cell_21_brightness_contrast_scatter.png', 'description': 'Brightness vs contrast scatter plot across 3,000 samples'},
        {'name': 'cell_28_diversity_overview.png', 'path': 'wardrobe_ml/outputs/cell_28_diversity_overview.png', 'description': '6-panel diversity overview (viewpoint, occlusion, zoom, scale, items, source)'},
        {'name': 'cell_32_pair_category_analysis.png', 'path': 'wardrobe_ml/outputs/cell_32_pair_category_analysis.png', 'description': '4-panel dataset selection & pair-cluster multi-label analysis dashboard'},
        {'name': 'cell_35_split_architecture.png', 'path': 'wardrobe_ml/outputs/cell_35_split_architecture.png', 'description': '2-panel official partition scale & theoretical leakage comparison'},
        {'name': 'comprehensive_dataset_audit_report.json', 'path': 'wardrobe_ml/outputs/comprehensive_dataset_audit_report.json', 'description': 'Initial comprehensive structured JSON audit report'},
        {'name': 'comprehensive_dataset_audit_report_v2.json', 'path': 'wardrobe_ml/outputs/comprehensive_dataset_audit_report_v2.json', 'description': 'Corrected v2 comprehensive structured JSON audit report (Step 5H.3.1)'}
    ]
}

with open(audit_report_path, 'w', encoding='utf-8') as f:
    json.dump(audit_report, f, indent=2)

print(f'✅ Corrected v2 audit report saved → {audit_report_path} ({audit_report_path.stat().st_size / 1024:.1f} KB)')
