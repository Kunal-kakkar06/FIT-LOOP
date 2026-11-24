"""
FitLoop — Stage 6.8A: Google Colab Dataset Transfer Preparation
==============================================================
Hardened, production-grade utility to audit local Windows DeepFashion2 sources,
validate YOLO manifests and labels, and safely package the dataset into verified ZIP
archives for transfer to Google Drive (/content/drive/MyDrive/FitLoop/staging_sources/).

Safety & Integrity Implementation (Stage 6.8A Final Hardening):
1. Safe AST Configuration Extraction (Zero Import Side Effects & Mutation Detection):
   - Extracts YOLO_CLASS_NAMES, DF2_TO_YOLO_CLASS_MAP, and DF2_CATEGORIES via ast.parse()
     and ast.literal_eval() only.
   - Never executes, imports, or dynamically evaluates wardrobe_ml/config.py.
   - Requires exactly one top-level literal definition (Assign or AnnAssign).
   - Rejects multiple definitions, reassignment, augmented assignment (+=, |=),
     deletion (del), and any subsequent mutation.
   - Detects subscript mutations (var[k] = ...) and attribute mutations (var.attr = ...).
   - Detects mutating method calls (append, extend, insert, update, pop, popitem,
     remove, clear, sort, reverse, setdefault).
   - Detects mutations through statically identifiable aliases (e.g. alias = var; alias.append(...)).
   - Rejects ambiguous or unsupported AST patterns (fails closed).
   - Aborts safely with a clear, actionable exception if any constant is missing, multiply
     defined, mutated, or not a literal.
2. Duplicate-Key-Rejecting YAML Parser (UniqueKeySafeLoader):
   - Custom SafeLoader inheriting from yaml.SafeLoader overriding construct_mapping.
   - Rejects duplicate keys at every mapping level, identifying duplicate key in error message.
   - Preserves safe YAML construction (no arbitrary code execution).
   - Requires 'type(yaml_nc) is int' and 'yaml_nc == 13'.
   - Requires 'names' to be a dictionary with exactly 13 entries.
   - Requires every key to satisfy 'type(key) is int' and matches exact set(range(13)).
   - Rejects boolean keys, string keys, mixed types, missing IDs, extra IDs, and empty values.
   - Compares every class name against YOLO_CLASS_NAMES[class_id] using exact string equality.
   - 'path' key rejects null, None, empty string, or whitespace-only values; requires '.' or './' when present.
   - Rejects POSIX absolute paths, Windows drive letters, backslashes, UNC paths, and '../' traversal.
3. TOCTOU-Immune Atomic No-Replace Publication:
   - Uses os.path.lexists() to detect existing destinations including files, directories,
     junctions, and dangling/broken symlinks.
   - Win32: Employs kernel32.MoveFileExW with MOVEFILE_WRITE_THROUGH and WITHOUT
     MOVEFILE_REPLACE_EXISTING, failing atomically if destination exists (ERROR_ALREADY_EXISTS / ERROR_FILE_EXISTS).
   - Linux: Employs libc.renameat2 with RENAME_NOREPLACE (flags=1). Safely aborts with
     RuntimeError if libc.renameat2 is unavailable; never hard-codes architecture-specific
     syscall numbers and never uses ordinary rename as a fallback.
   - Never replaces an existing directory, including an empty directory.
   - Strictly aborts and preserves temporary staging if safe atomic no-replace cannot be guaranteed.
   - Never uses ordinary Path.rename() as the sole no-overwrite guarantee.
4. Complete POSIX Archive Member Path-Set Equality:
   - Compares 100% of complete relative POSIX member paths against expected path sets.
   - Rejects nested subdirectories and unexpected archive structures.
   - Enforces exact set equality for all 4 partitions:
     * images/train/<filename>.jpg (191,958 members)
     * images/val/<filename>.jpg   (32,153 members)
     * labels/train/<filename>.txt (191,958 members)
     * labels/val/<filename>.txt   (32,153 members)
     * train.txt, val.txt, data.yaml (3 members)
5. Robust Failure Preservation:
   - Generates temporary staging directory in caller scope using tempfile.mkdtemp() / UUID.
   - If archive creation fails at any point, the exact temporary directory is logged and preserved.
   - Failed artifacts are NEVER automatically deleted.
6. Comprehensive & Exhaustive SHA-256 Verification:
   - Full-archive SHA-256 calculation for all 4 ZIP files.
   - Bitwise SHA-256 verification of train.txt, val.txt, and data.yaml against source files.
   - Multi-partition sampling (50 train imgs, 50 val imgs, 50 train lbls, 50 val lbls, 3 manifests = 203 files).
   - Optional exhaustive mode (--verify-all-hashes) genuinely verifying all 448,225 members.
7. Zero Python Optimization Bypass:
   - Zero 'assert' statements; all checks use explicit conditionals and typed exceptions.
8. Original Inviolable Safety Restrictions Preserved:
   - Zero source modifications, zero file deletions, zero overwriting.
   - Zero test split packaging, zero weight downloads, zero inference, zero training, zero Kaggle uploads.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# 1. Authoritative Source and Target Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("C:/Users/Lenovo/Downloads/Fitloop")
ML_ROOT = PROJECT_ROOT / "wardrobe_ml"
CONFIG_FILE = ML_ROOT / "config.py"
DATASET_ROOT = Path("C:/Users/Lenovo/Downloads/DeepFashion2")

TRAIN_IMG_DIR = DATASET_ROOT / "train" / "image"
VAL_IMG_DIR = DATASET_ROOT / "validation" / "image"
TRAIN_LBL_DIR = ML_ROOT / "data" / "processed" / "deepfashion2_yolo" / "labels" / "train"
VAL_LBL_DIR = ML_ROOT / "data" / "processed" / "deepfashion2_yolo" / "labels" / "val"
MANIFEST_DIR = ML_ROOT / "data" / "processed" / "deepfashion2_yolo"

TRAIN_MANIFEST_FILE = MANIFEST_DIR / "train.txt"
VAL_MANIFEST_FILE = MANIFEST_DIR / "val.txt"
DATA_YAML_FILE = MANIFEST_DIR / "data.yaml"

TRANSFER_OUTPUT_DIR = ML_ROOT / "outputs" / "colab_transfer_staging"
INVENTORY_JSON_FILENAME = "colab_archive_inventory.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Authoritative Configuration Extraction & DeepFashion2 Mapping Validation
# ─────────────────────────────────────────────────────────────────────────────

MUTATING_METHODS: Set[str] = {
    "append",
    "extend",
    "insert",
    "update",
    "pop",
    "popitem",
    "remove",
    "clear",
    "sort",
    "reverse",
    "setdefault",
}


def validate_class_mapping(
    yolo_class_names: Any,
    df2_to_yolo_map: Any,
    df2_categories: Any,
) -> None:
    """
    Rigorously validates the DeepFashion2-to-YOLO class mapping:
    1. DF2_CATEGORIES:
       - Must be a dictionary of exactly 13 categories.
       - Keys must be strictly of type int (type(k) is int; booleans rejected).
       - Keys must be integer IDs from 1 through 13.
       - Values must be non-empty strings.
    2. DF2_TO_YOLO_CLASS_MAP:
       - Must be a dictionary of exactly 13 mappings.
       - Keys and values must be strictly of type int:
         'type(k) is int' and 'type(v) is int'. Booleans (e.g. True/False) are strictly rejected.
       - Keys must exactly cover all 13 source categories in DF2_CATEGORIES (set(1..13)).
       - Values must strictly be in range [0, 12].
       - Values must cover all 13 YOLO class IDs (set(range(13))).
    3. YOLO_CLASS_NAMES:
       - Must be a list of exactly 13 string class names.
       - For every source category k in DF2_CATEGORIES, YOLO_CLASS_NAMES[df2_to_yolo_map[k]]
         must exactly equal df2_categories[k].
    """
    # 1. Validate DF2_CATEGORIES
    if not isinstance(df2_categories, dict):
        raise TypeError(f"DF2_CATEGORIES in {CONFIG_FILE} must be a dict, got {type(df2_categories).__name__}")
    if len(df2_categories) != 13:
        raise ValueError(f"DF2_CATEGORIES count mismatch: expected 13 categories, got {len(df2_categories)}")
    for k, v in df2_categories.items():
        if type(k) is not int:
            raise TypeError(
                f"DF2_CATEGORIES key {k!r} must be strictly of type int (booleans rejected), got {type(k).__name__}"
            )
        if not isinstance(v, str) or not v.strip():
            raise TypeError(f"DF2_CATEGORIES value for key {k} must be a non-empty string, got {v!r}")
    expected_df2_keys = set(range(1, 14))
    if set(df2_categories.keys()) != expected_df2_keys:
        raise ValueError(
            f"DF2_CATEGORIES key mismatch: expected keys {expected_df2_keys}, got {set(df2_categories.keys())}"
        )

    # 2. Validate DF2_TO_YOLO_CLASS_MAP key and value types, source coverage, and YOLO range
    if not isinstance(df2_to_yolo_map, dict):
        raise TypeError(f"DF2_TO_YOLO_CLASS_MAP in {CONFIG_FILE} must be a dict, got {type(df2_to_yolo_map).__name__}")
    if len(df2_to_yolo_map) != 13:
        raise ValueError(f"DF2_TO_YOLO_CLASS_MAP count mismatch: expected 13 mappings, got {len(df2_to_yolo_map)}")

    for k, v in df2_to_yolo_map.items():
        if type(k) is not int:
            raise TypeError(
                f"DF2_TO_YOLO_CLASS_MAP key {k!r} must be strictly of type int (booleans rejected), got {type(k).__name__}"
            )
        if type(v) is not int:
            raise TypeError(
                f"DF2_TO_YOLO_CLASS_MAP value {v!r} for key {k} must be strictly of type int (booleans rejected), got {type(v).__name__}"
            )
        if not (0 <= v <= 12):
            raise ValueError(
                f"DF2_TO_YOLO_CLASS_MAP mapped YOLO class ID {v} for DF2 category {k} is out of bounds [0, 12]"
            )

    # Validate expected source-category key coverage using DF2_CATEGORIES
    if set(df2_to_yolo_map.keys()) != set(df2_categories.keys()):
        raise ValueError(
            f"DF2_TO_YOLO_CLASS_MAP keys do not match DF2_CATEGORIES! "
            f"Diff: {set(df2_to_yolo_map.keys()) ^ set(df2_categories.keys())}"
        )

    # Validate that all 13 YOLO class IDs 0 through 12 are covered
    mapped_yolo_ids = set(df2_to_yolo_map.values())
    expected_yolo_ids = set(range(13))
    if mapped_yolo_ids != expected_yolo_ids:
        raise ValueError(
            f"DF2_TO_YOLO_CLASS_MAP values do not cover all 13 YOLO class IDs [0..12]! "
            f"Missing: {expected_yolo_ids - mapped_yolo_ids}, Extra: {mapped_yolo_ids - expected_yolo_ids}"
        )

    # 3. Validate YOLO_CLASS_NAMES and correspondence
    if not isinstance(yolo_class_names, list) or len(yolo_class_names) != 13:
        raise ValueError(f"Invalid YOLO_CLASS_NAMES in {CONFIG_FILE}: expected 13 items, got {yolo_class_names}")

    for idx, name in enumerate(yolo_class_names):
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"YOLO_CLASS_NAMES[{idx}] must be a non-empty string, got {name!r}")

    for df2_id, yolo_id in df2_to_yolo_map.items():
        expected_cat_name = df2_categories[df2_id]
        actual_yolo_name = yolo_class_names[yolo_id]
        if actual_yolo_name != expected_cat_name:
            raise ValueError(
                f"Class mapping mismatch between DF2 category {df2_id} ('{expected_cat_name}') "
                f"and YOLO class ID {yolo_id} ('{actual_yolo_name}') in {CONFIG_FILE}"
            )


def extract_ast_constant(tree: ast.Module, var_name: str) -> Any:
    """
    Extracts the value of a constant assignment from an AST tree using ast.literal_eval().
    Does not execute, import, or evaluate any code.

    Guarantees:
    - Inspects all top-level statements in tree.body.
    - Requires exactly one top-level literal definition (Assign or AnnAssign).
    - Strictly rejects multiple definitions, reassignment, augmented assignment (+=, |=),
      deletion (del), and any subsequent mutation.
    - Detects subscript and attribute mutations (var[k] = ..., var.attr = ...).
    - Detects mutating method calls (append, extend, insert, update, pop, popitem,
      remove, clear, sort, reverse, setdefault).
    - Detects mutations through statically identifiable aliases (e.g. alias = var; alias.append(...)).
    - Rejects ambiguous or unsupported AST patterns (fails closed).
    - Evaluates strictly via ast.literal_eval(); if not a literal or safety cannot be established,
      aborts with a clear, actionable exception.
    """
    definitions: List[Tuple[int, ast.AST, ast.AST]] = []

    for node in tree.body:
        # Check standard assignment: var_name = <value>
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    definitions.append((node.lineno, node.value, node))
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name) and elt.id == var_name:
                            definitions.append((node.lineno, node.value, node))

        # Check annotated assignment: var_name: <type> = <value>
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var_name:
                if node.value is None:
                    raise ValueError(
                        f"Required constant '{var_name}' at line {node.lineno} in {CONFIG_FILE} "
                        f"is type-annotated but has no assigned value."
                    )
                definitions.append((node.lineno, node.value, node))

    # 1. Reject if no assignment was found
    if not definitions:
        raise KeyError(
            f"Required constant '{var_name}' was not found in top-level assignments of {CONFIG_FILE}."
        )

    # 2. Reject if multiple assignments exist
    if len(definitions) > 1:
        line_numbers = [d[0] for d in definitions]
        raise ValueError(
            f"Multiple top-level assignments detected for constant '{var_name}' in {CONFIG_FILE} "
            f"(found {len(definitions)} assignments at lines {line_numbers}). Duplicate definitions "
            f"or reassignments are strictly prohibited; expected exactly one literal definition."
        )

    # Exactly one definition found
    def_lineno, def_value_node, def_node = definitions[0]

    # 3. Exactly one literal assignment found: evaluate using ast.literal_eval()
    try:
        extracted_value = ast.literal_eval(def_value_node)
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"Required constant '{var_name}' at line {def_lineno} in {CONFIG_FILE} cannot be safely "
            f"extracted as a literal (AST expression type: {type(def_value_node).__name__}). "
            f"Expected a literal assignment (e.g. dict or list literal) to allow safe static extraction without execution: {e}"
        )

    # 4. Statically track aliases across all scopes (top-level, functions, nested blocks)
    tracked_names: Set[str] = {var_name}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if node is def_node:
                continue
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Name) and node.value.id in tracked_names:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id not in tracked_names:
                            tracked_names.add(target.id)
                            changed = True
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name) and elt.id not in tracked_names:
                                    tracked_names.add(elt.id)
                                    changed = True
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.value, ast.Name) and node.value.id in tracked_names:
                    if isinstance(node.target, ast.Name) and node.target.id not in tracked_names:
                        tracked_names.add(node.target.id)
                        changed = True

    # Helper to resolve root name of an expression (e.g. obj.attr, arr[i], arr[i][j])
    def get_root_name(n: ast.AST) -> str | None:
        curr = n
        while isinstance(curr, (ast.Subscript, ast.Attribute)):
            curr = curr.value
        if isinstance(curr, ast.Name):
            return curr.id
        return None

    # Build parent relationships so assignments can be checked by scope.
    parent_map = {
    child: parent
    for parent in ast.walk(tree)
    for child in ast.iter_child_nodes(parent)
}

    # 5. Full AST traversal to detect mutations, reassignments, deletions, and mutating calls
    for node in ast.walk(tree):

        if node is def_node:
            continue
        # Check augmented assignments (+=, |=, etc.)
        if isinstance(node, ast.AugAssign):
            root = get_root_name(node.target)
            if root in tracked_names:
                raise ValueError(
                    f"Constant '{var_name}' (or alias '{root}') is modified by augmented assignment "
                    f"(AugAssign at line {node.lineno}) in {CONFIG_FILE}. Dynamic mutations are strictly prohibited."
                )

        # Check deletions (del x, del x[k], del x.attr)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                root = get_root_name(target)
                if root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') is deleted/modified by 'del' statement "
                        f"at line {node.lineno} in {CONFIG_FILE}. Deletion is strictly prohibited."
                    )

        # Check assignments to subscripts or attributes (e.g. x[k] = ..., x.attr = ...)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                root = get_root_name(target)
                if isinstance(target, (ast.Subscript, ast.Attribute)) and root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') is mutated via assignment at line "
                        f"{node.lineno} in {CONFIG_FILE}. Subscript and attribute mutations are strictly prohibited."
                    )
        # Subsequent reassignment to var_name itself.
                elif isinstance(target, ast.Name) and target.id == var_name:

        # Allow a direct class-level reference in _Config:
        # YOLO_CLASS_NAMES = YOLO_CLASS_NAMES
                    parent = parent_map.get(node)

                    is_config_class_reference = (
                        isinstance(parent, ast.ClassDef)
                        and parent.name == "_Config"
                        and len(node.targets) == 1
                        and isinstance(node.value, ast.Name)
                        and node.value.id == var_name
                    )

                    if not is_config_class_reference:
                       raise ValueError(
            f"Constant '{var_name}' is reassigned at line {node.lineno} in {CONFIG_FILE}. "
            f"Multiple definitions or reassignments are strictly prohibited."
        )

        elif isinstance(node, ast.AnnAssign):
            root = get_root_name(node.target)
            if isinstance(node.target, (ast.Subscript, ast.Attribute)) and root in tracked_names:
                raise ValueError(
                    f"Constant '{var_name}' (or alias '{root}') is mutated via annotated assignment at line "
                    f"{node.lineno} in {CONFIG_FILE}. Subscript and attribute mutations are strictly prohibited."
                )
            elif isinstance(node.target, ast.Name) and node.target.id == var_name:
                raise ValueError(
                    f"Constant '{var_name}' is reassigned at line {node.lineno} in {CONFIG_FILE}. "
                    f"Multiple definitions or reassignments are strictly prohibited."
                )

        # Check mutating method calls (e.g. x.append(), x.update(), alias.pop())
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in MUTATING_METHODS:
                root = get_root_name(node.func.value)
                if root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') is mutated via method call '{node.func.attr}()' "
                        f"at line {node.lineno} in {CONFIG_FILE}. Mutating method calls are strictly prohibited."
                    )

            # Check ambiguous argument escapes into function calls
            for arg in node.args:
                root = get_root_name(arg)
                if root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') is passed as an argument to a call "
                        f"at line {node.lineno} in {CONFIG_FILE}. Ambiguous escape pattern detected; "
                        f"mutation safety cannot be established."
                    )
            for kw in node.keywords:
                root = get_root_name(kw.value)
                if root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') is passed as a keyword argument to a call "
                        f"at line {node.lineno} in {CONFIG_FILE}. Ambiguous escape pattern detected; "
                        f"mutation safety cannot be established."
                    )

        elif isinstance(node, ast.Return):
            if node.value is not None:
                root = get_root_name(node.value)
                if root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') escapes via return statement "
                        f"at line {node.lineno} in {CONFIG_FILE}. Mutation safety cannot be established."
                    )

        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            if node.value is not None:
                root = get_root_name(node.value)
                if root in tracked_names:
                    raise ValueError(
                        f"Constant '{var_name}' (or alias '{root}') escapes via yield "
                        f"at line {node.lineno} in {CONFIG_FILE}. Mutation safety cannot be established."
                    )

    return extracted_value


def load_authoritative_config() -> Tuple[List[str], Dict[int, int], Dict[int, str]]:
    """
    Safely extracts and validates authoritative class definitions from wardrobe_ml/config.py
    using static AST analysis and literal evaluation.
    Does not execute, import, or evaluate config.py, preventing arbitrary code execution
    and eliminating __pycache__ bytecode file generation.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Authoritative configuration file missing: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code, filename=str(CONFIG_FILE))
    except SyntaxError as e:
        raise SyntaxError(f"Failed to parse {CONFIG_FILE} into AST: {e}") from e

    yolo_class_names = extract_ast_constant(tree, "YOLO_CLASS_NAMES")
    df2_to_yolo_map = extract_ast_constant(tree, "DF2_TO_YOLO_CLASS_MAP")
    df2_categories = extract_ast_constant(tree, "DF2_CATEGORIES")

    validate_class_mapping(yolo_class_names, df2_to_yolo_map, df2_categories)

    return yolo_class_names, df2_to_yolo_map, df2_categories


YOLO_CLASS_NAMES, DF2_TO_YOLO_CLASS_MAP, DF2_CATEGORIES = load_authoritative_config()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Canonical Dataset Invariants and Constants
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED_TRAIN_IMAGES: Set[str] = {"028225.jpg", "115063.jpg", "101338.jpg"}
EXCLUDED_TRAIN_LABELS: Set[str] = {Path(f).stem + ".txt" for f in EXCLUDED_TRAIN_IMAGES}

EXPECTED_COUNTS: Dict[str, int] = {
    "train_images_raw": 191961,
    "train_images_eligible": 191958,
    "val_images": 32153,
    "train_labels_raw": 191961,
    "train_labels_eligible": 191958,
    "val_labels": 32153,
    "train_instances_eligible": 312181,
    "val_instances": 52490,
    "num_classes": 13,
}

ARCHIVE_NAMES: Dict[str, str] = {
    "labels": "fitloop_yolo_labels.zip",
    "manifests": "fitloop_yolo_manifests.zip",
    "val_images": "fitloop_val_images.zip",
    "train_images": "fitloop_train_images.zip",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Atomic No-Replace Publication & Overwrite Protection
# ─────────────────────────────────────────────────────────────────────────────

def atomic_publish_no_replace(src_dir: Path, dst_dir: Path) -> None:
    """
    Publishes a temporary staging directory to the destination path using a
    strictly non-replacing, atomic filesystem operation.

    Guarantees:
    1. Detects any existing destination (files, directories, junctions, symlinks,
       including broken/dangling symlinks) using os.path.lexists.
    2. Never replaces an existing directory, including an empty directory.
    3. On Windows (Win32): Invokes kernel32.MoveFileExW with MOVEFILE_WRITE_THROUGH (0x8)
       and WITHOUT MOVEFILE_REPLACE_EXISTING (0x1). If the destination exists, the kernel
       atomically refuses the move (ERROR_ALREADY_EXISTS / ERROR_FILE_EXISTS).
    4. On POSIX (Linux): Invokes libc.renameat2 with RENAME_NOREPLACE (flags=1).
       If destination exists, the kernel atomically refuses the rename (EEXIST).
       If libc.renameat2 is unavailable, aborts safely with RuntimeError.
       Never hard-codes architecture-specific syscall numbers and never falls back
       to an overwriting rename.
    5. On any error or conflict, the temporary staging directory is preserved untouched.
    """
    src_str = str(src_dir)
    dst_str = str(dst_dir)

    # 1. Pre-check: Detect existing destinations, including dangling symlinks or junctions
    if os.path.lexists(dst_str):
        raise FileExistsError(
            f"Atomic publish aborted: destination already exists (directory, file, junction, "
            f"or dangling symlink): {dst_dir}. Refusing to overwrite. Temporary staging preserved at: {src_dir}"
        )

    # 2. Platform-appropriate atomic no-replace operation
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        # Win32 MoveFileExW flags:
        # MOVEFILE_REPLACE_EXISTING = 0x00000001 (MUST NOT BE PASSED)
        # MOVEFILE_COPY_ALLOWED     = 0x00000002
        # MOVEFILE_DELAY_UNTIL_REBOOT = 0x00000004
        # MOVEFILE_WRITE_THROUGH    = 0x00000008 (flushes file metadata to disk)
        MOVEFILE_WRITE_THROUGH = 0x00000008

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        MoveFileExW = kernel32.MoveFileExW
        MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        MoveFileExW.restype = wintypes.BOOL

        success = MoveFileExW(src_str, dst_str, MOVEFILE_WRITE_THROUGH)
        if not success:
            err_code = ctypes.get_last_error()
            # 80 = ERROR_FILE_EXISTS, 183 = ERROR_ALREADY_EXISTS, 5 = ERROR_ACCESS_DENIED
            if err_code in (80, 183):
                raise FileExistsError(
                    f"Win32 atomic MoveFileExW refused move: destination already exists "
                    f"(Win32 error {err_code}): {dst_dir}. Staging preserved at: {src_dir}"
                )
            raise OSError(
                f"Win32 atomic MoveFileExW failed with error code {err_code}. "
                f"Destination: {dst_dir}. Staging preserved at: {src_dir}"
            )

    elif sys.platform.startswith("linux"):
        import ctypes

        AT_FDCWD = -100
        RENAME_NOREPLACE = 1  # flags: 1 = RENAME_NOREPLACE (since Linux kernel 3.15)

        try:
            libc = ctypes.CDLL(None, use_errno=True)
            if not hasattr(libc, "renameat2"):
                raise RuntimeError(
                    "libc.renameat2 is unavailable on this Linux system. Safe atomic no-replace publication "
                    "cannot be guaranteed without guessing architecture-specific syscall numbers. "
                    f"Refusing to publish using standard rename to prevent accidental overwriting. "
                    f"Staging directory preserved at: {src_dir}"
                )

            libc.renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            libc.renameat2.restype = ctypes.c_int
            ret = libc.renameat2(
                AT_FDCWD,
                src_str.encode("utf-8"),
                AT_FDCWD,
                dst_str.encode("utf-8"),
                RENAME_NOREPLACE,
            )
            if ret != 0:
                errno_val = ctypes.get_errno()
                if errno_val == 17:  # EEXIST
                    raise FileExistsError(
                        f"Linux renameat2(RENAME_NOREPLACE) refused move: destination already exists "
                        f"(EEXIST): {dst_dir}. Staging preserved at: {src_dir}"
                    )
                raise OSError(
                    errno_val,
                    f"Linux renameat2 failed with errno {errno_val}. Destination: {dst_dir}. Staging preserved at: {src_dir}",
                )
        except (AttributeError, OSError) as e:
            raise RuntimeError(
                f"Safe atomic no-replace publication cannot be guaranteed on this platform/kernel ({e}). "
                f"Ordinary rename is strictly prohibited to prevent accidental replacement. "
                f"Staging directory preserved at: {src_dir}"
            ) from e
    else:
        raise RuntimeError(
            f"Safe atomic no-replace publication is not implemented or guaranteed for platform '{sys.platform}'. "
            f"Ordinary rename is strictly prohibited to prevent accidental replacement. "
            f"Staging directory preserved at: {src_dir}"
        )


def verify_overwrite_protection() -> None:
    """
    Enforces strict overwrite protection:
    - Refuses to execute if the final output path already exists
      (including files, directories, junctions, or dangling symlinks via os.path.lexists).
    - Never automatically deletes, moves, or replaces existing artifacts.
    - Aborts immediately before staging if any destination conflict exists.
    """
    print("\n[SECURITY CHECK 1/7] Enforcing Overwrite Protection...")

    if os.path.lexists(TRANSFER_OUTPUT_DIR):
        print(f"  [CONFLICT DETECTED] Target destination already exists on disk: {TRANSFER_OUTPUT_DIR}")
        if TRANSFER_OUTPUT_DIR.is_dir():
            existing_items = list(TRANSFER_OUTPUT_DIR.glob("*"))
            print(f"  Directory contains {len(existing_items)} existing items:")
            for item in existing_items[:5]:
                print(f"    - {item.name}")
            if len(existing_items) > 5:
                print(f"    ... and {len(existing_items) - 5} more.")
        elif TRANSFER_OUTPUT_DIR.is_symlink():
            print("  Target is an existing symlink or junction.")
        else:
            print("  Target is an existing file.")

        print("\n  [ABORT] Overwrite protection triggered! Overwriting existing paths is strictly forbidden.")
        print("  To proceed, manually back up or move the existing destination:")
        print(f"    Rename-Item '{TRANSFER_OUTPUT_DIR}' '{TRANSFER_OUTPUT_DIR}_backup'")
        raise FileExistsError(f"Refusing to execute: output path already exists: {TRANSFER_OUTPUT_DIR}")

    if not TRANSFER_OUTPUT_DIR.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {TRANSFER_OUTPUT_DIR.parent}")

    print("  [PASS] Overwrite protection verified: Output directory is clear.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Authoritative data.yaml & Class Mapping Validation
# ─────────────────────────────────────────────────────────────────────────────

class UniqueKeySafeLoader(yaml.SafeLoader):
    """
    A strictly safe PyYAML loader that rejects duplicate keys at every mapping level.
    Inherits from yaml.SafeLoader to preserve safe YAML construction (no arbitrary code execution).
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        mapping: Dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found unhashable key: {key!r}",
                    key_node.start_mark,
                )
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key '{key}' in YAML mapping",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def validate_data_yaml_and_classes() -> None:
    """
    Rigorously validates data.yaml:
    - Parses YAML using UniqueKeySafeLoader, strictly rejecting duplicate keys at every mapping level.
    - Validates 'nc': requires type(yaml_nc) is int AND yaml_nc == 13.
    - Validates 'names':
      * Must be a dictionary with exactly 13 entries.
      * Every key must satisfy 'type(key) is int'.
      * Key set must strictly equal set(range(13)).
      * Every value must be a non-empty string.
      * Compares every value against YOLO_CLASS_NAMES[cid] using exact string equality.
      * Rejects boolean keys, string keys, mixed types, missing IDs, extra IDs, and empty values.
      * Does not silently normalize or coerce keys.
    - Validates 'path': rejects null and empty values whenever key is present;
      requires string and accepts strictly '.' or './'.
    - Rejects POSIX absolute paths, Windows drive paths, backslashes, UNC paths, and '../' traversal.
    - Enforces intended dataset layout (train='train.txt', val='val.txt', nc=13).
    - Prohibits substring matching; enforces exact string and integer equality.
    """
    print("\n[SECURITY CHECK 2/7] Validating data.yaml Structure and Authoritative 13-Class Mapping...")

    if not DATA_YAML_FILE.exists():
        raise FileNotFoundError(f"data.yaml missing: {DATA_YAML_FILE}")

    try:
        with open(DATA_YAML_FILE, "r", encoding="utf-8") as f:
            parsed_yaml = yaml.load(f, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse data.yaml with duplicate key check: {e}") from e

    if not isinstance(parsed_yaml, dict):
        raise ValueError("Malformed data.yaml: failed to parse dictionary.")

    def check_forbidden_path_constructs(field_name: str, path_val: str) -> None:
        p = path_val.strip()
        # Reject POSIX absolute paths
        if p.startswith("/"):
            raise ValueError(f"POSIX absolute path rejected in data.yaml '{field_name}': '{p}'")
        # Reject Windows drive letters
        if ":\\" in p or ":/" in p or (len(p) > 1 and p[1] == ":" and p[0].isalpha()):
            raise ValueError(f"Windows drive path rejected in data.yaml '{field_name}': '{p}'")
        # Reject UNC paths or backslash roots
        if p.startswith("//") or p.startswith("\\\\") or p.startswith("\\"):
            raise ValueError(f"UNC or root path rejected in data.yaml '{field_name}': '{p}'")
        # Reject '../' directory traversal
        if ".." in p:
            raise ValueError(f"Path traversal '..' rejected in data.yaml '{field_name}': '{p}'")
        # Reject Windows backslashes
        if "\\" in p:
            raise ValueError(f"Backslash rejected in data.yaml '{field_name}': '{p}'")
        # Reject test split references
        if "test" in p.lower():
            raise ValueError(f"Withheld test split reference rejected in data.yaml '{field_name}': '{p}'")

    # 1. Validate 'path' against intended dataset layout
    # Rejects null/None and empty string whenever present. If present, requires string in ('.', './').
    if "path" in parsed_yaml:
        yaml_path = parsed_yaml["path"]
        if yaml_path is None:
            raise ValueError(
                "data.yaml contains 'path' key with null/None value; null path entries are strictly rejected."
            )
        if not isinstance(yaml_path, str):
            raise ValueError(
                f"data.yaml 'path' must be a string, got {type(yaml_path).__name__}: '{yaml_path}'"
            )
        path_str = yaml_path.strip()
        if not path_str:
            raise ValueError(
                "data.yaml contains empty 'path' value; empty string paths are strictly rejected."
            )
        check_forbidden_path_constructs("path", path_str)
        if path_str not in (".", "./"):
            raise ValueError(
                f"data.yaml 'path' must be '.' or './' for portable staging, got '{path_str}'"
            )

    # 2. Validate 'train' and 'val' entries against intended dataset layout
    for key, expected_options in [("train", ("train.txt", "./train.txt")), ("val", ("val.txt", "./val.txt"))]:
        val = parsed_yaml.get(key)
        if not val or not isinstance(val, str):
            raise ValueError(f"data.yaml missing required string entry '{key}'")
        val_str = val.strip()
        check_forbidden_path_constructs(key, val_str)
        if val_str not in expected_options:
            raise ValueError(
                f"data.yaml '{key}' layout mismatch: expected one of {expected_options}, got '{val_str}'"
            )

    # 3. Validate nc strictly (must satisfy type(yaml_nc) is int AND yaml_nc == 13)
    yaml_nc = parsed_yaml.get("nc")
    if type(yaml_nc) is not int or yaml_nc != 13:
        raise ValueError(
            f"data.yaml 'nc' must be strictly of type int with value 13 (booleans/floats/strings rejected), "
            f"got {type(yaml_nc).__name__}: {yaml_nc!r}"
        )

    # 4. Strictly validate names dictionary against authoritative mapping
    yaml_names = parsed_yaml.get("names")
    if not isinstance(yaml_names, dict):
        raise TypeError(f"data.yaml 'names' must be a dictionary, got {type(yaml_names).__name__}")

    if len(yaml_names) != 13:
        raise ValueError(f"data.yaml 'names' count mismatch: expected exactly 13 entries, got {len(yaml_names)}")

    # Require every key to satisfy type(key) is int (reject booleans, strings, floats, mixed types)
    for k in yaml_names.keys():
        if type(k) is not int:
            raise TypeError(
                f"data.yaml 'names' contains non-int key {k!r} of type {type(k).__name__}. "
                f"All keys must satisfy 'type(key) is int'; string/boolean/coerced keys are strictly rejected."
            )

    # Require the exact key set set(range(13)) (reject missing IDs, extra IDs)
    actual_keys = set(yaml_names.keys())
    expected_keys = set(range(13))
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise ValueError(
            f"data.yaml 'names' key set mismatch! Expected exact keys 0..12: missing={missing}, extra={extra}"
        )

    # Require every value to be a non-empty string and compare against YOLO_CLASS_NAMES[cid] using exact string equality
    for cid in range(13):
        val = yaml_names[cid]
        if not isinstance(val, str) or not val.strip():
            raise TypeError(
                f"data.yaml 'names' value for class ID {cid} must be a non-empty string, got {val!r} ({type(val).__name__})"
            )

        expected_name = YOLO_CLASS_NAMES[cid]
        if val != expected_name:
            raise ValueError(
                f"Class mapping mismatch for class ID {cid}: expected exact string '{expected_name}', got '{val}'"
            )

    # Re-verify authoritative class mapping definitions from config.py
    validate_class_mapping(YOLO_CLASS_NAMES, DF2_TO_YOLO_CLASS_MAP, DF2_CATEGORIES)

    print(f"  Exact 13-class mapping verified ({len(YOLO_CLASS_NAMES)} classes, 0 to 12).")
    print("  [PASS] data.yaml paths and authoritative classes 100% verified.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Source Dataset Validation & Full YOLO Label Parsing
# ─────────────────────────────────────────────────────────────────────────────

def validate_sources_and_labels() -> Dict[str, Any]:
    """
    Validates local Windows source images and conducts exhaustive line-by-line
    validation of every single YOLO label file:
    - Confirms presence of all 3 excluded images and label files.
    - Verifies 1:1 image-label correspondence across all raw files.
    - Parses every label line for exactly 5 numeric fields.
    - Validates integer class IDs in [0, 12].
    - Validates finite, normalized coordinates in [0, 1] and positive dimensions (0, 1].
    - Validates absence of NaN/Inf.
    - Independently counts garment instances (expected: 312,181 train / 52,490 val).
    - Reports any unexpected empty label files.
    """
    print("\n[SECURITY CHECK 3/7] Validating Source Images and Exhaustively Parsing YOLO Labels...")

    for label, p in [
        ("Train Images", TRAIN_IMG_DIR),
        ("Val Images", VAL_IMG_DIR),
        ("Train Labels", TRAIN_LBL_DIR),
        ("Val Labels", VAL_LBL_DIR),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required source directory does not exist: {p}")

    # Confirm all 3 excluded training images and labels exist on disk
    for img_name in sorted(EXCLUDED_TRAIN_IMAGES):
        img_p = TRAIN_IMG_DIR / img_name
        lbl_p = TRAIN_LBL_DIR / (Path(img_name).stem + ".txt")
        if not img_p.exists():
            raise FileNotFoundError(f"Expected excluded training image missing from disk: {img_p}")
        if not lbl_p.exists():
            raise FileNotFoundError(f"Expected excluded training label missing from disk: {lbl_p}")
        print(f"  Confirmed excluded target present on disk: {img_name} ({img_p.stat().st_size:,} bytes) and {lbl_p.name}")

    def scan_dir(p: Path, ext: str) -> Tuple[int, int, Set[str]]:
        count = 0
        total_bytes = 0
        names = set()
        with os.scandir(p) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(ext):
                    count += 1
                    total_bytes += entry.stat().st_size
                    names.add(entry.name)
        return count, total_bytes, names

    print("\n  Scanning raw source directories...")
    t0 = time.time()
    c_ti, b_ti, n_ti = scan_dir(TRAIN_IMG_DIR, ".jpg")
    c_vi, b_vi, n_vi = scan_dir(VAL_IMG_DIR, ".jpg")
    c_tl, b_tl, n_tl = scan_dir(TRAIN_LBL_DIR, ".txt")
    c_vl, b_vl, n_vl = scan_dir(VAL_LBL_DIR, ".txt")
    t_scan = time.time() - t0
    print(f"  Scanned 448,228 files in {t_scan:.2f}s.")

    if c_ti != EXPECTED_COUNTS["train_images_raw"]:
        raise ValueError(f"Raw train image mismatch: expected {EXPECTED_COUNTS['train_images_raw']}, got {c_ti}")
    if c_vi != EXPECTED_COUNTS["val_images"]:
        raise ValueError(f"Val image mismatch: expected {EXPECTED_COUNTS['val_images']}, got {c_vi}")
    if c_tl != EXPECTED_COUNTS["train_labels_raw"]:
        raise ValueError(f"Raw train label mismatch: expected {EXPECTED_COUNTS['train_labels_raw']}, got {c_tl}")
    if c_vl != EXPECTED_COUNTS["val_labels"]:
        raise ValueError(f"Val label mismatch: expected {EXPECTED_COUNTS['val_labels']}, got {c_vl}")

    # 1:1 Correspondence
    train_img_stems = {Path(f).stem for f in n_ti}
    train_lbl_stems = {Path(f).stem for f in n_tl}
    val_img_stems = {Path(f).stem for f in n_vi}
    val_lbl_stems = {Path(f).stem for f in n_vl}

    if train_img_stems != train_lbl_stems:
        raise ValueError(f"Train image-label stem mismatch! Difference: {train_img_stems ^ train_lbl_stems}")
    if val_img_stems != val_lbl_stems:
        raise ValueError(f"Val image-label stem mismatch! Difference: {val_img_stems ^ val_lbl_stems}")
    print("  1:1 image-label correspondence verified across all raw source files.")

    # Filter eligible
    eligible_ti = n_ti - EXCLUDED_TRAIN_IMAGES
    eligible_tl = n_tl - EXCLUDED_TRAIN_LABELS

    if len(eligible_ti) != EXPECTED_COUNTS["train_images_eligible"]:
        raise ValueError(f"Eligible train image mismatch: expected {EXPECTED_COUNTS['train_images_eligible']}, got {len(eligible_ti)}")
    if len(eligible_tl) != EXPECTED_COUNTS["train_labels_eligible"]:
        raise ValueError(f"Eligible train label mismatch: expected {EXPECTED_COUNTS['train_labels_eligible']}, got {len(eligible_tl)}")

    # Exhaustive YOLO label validation
    print("\n  Exhaustively parsing every eligible YOLO label file...")
    def parse_partition_labels(label_dir: Path, label_files: Set[str], split_name: str) -> Tuple[int, List[str], Dict[int, int]]:
        instance_count = 0
        empty_files: List[str] = []
        class_dist = {i: 0 for i in range(13)}
        t_parse_0 = time.time()

        for idx, fname in enumerate(sorted(label_files), 1):
            fpath = label_dir / fname
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            stripped_lines = [l.strip() for l in lines if l.strip()]
            if not stripped_lines:
                empty_files.append(fname)
                continue

            for line_no, line in enumerate(stripped_lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Malformed label in {split_name}/{fname}:{line_no}: expected exactly 5 fields, got {len(parts)}: '{line}'"
                    )

                # 1. Validate integer class ID
                try:
                    cid = int(parts[0])
                except ValueError:
                    raise ValueError(f"Non-integer class ID in {split_name}/{fname}:{line_no}: '{parts[0]}'")

                if not (0 <= cid <= 12):
                    raise ValueError(f"Class ID {cid} out of bounds [0, 12] in {split_name}/{fname}:{line_no}")

                # 2. Validate finite numeric coordinates
                try:
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    raise ValueError(f"Non-numeric coordinate in {split_name}/{fname}:{line_no}: '{line}'")

                if any(math.isnan(v) or math.isinf(v) for v in (xc, yc, w, h)):
                    raise ValueError(f"Non-finite coordinate (NaN/Inf) in {split_name}/{fname}:{line_no}: '{line}'")

                # 3. Validate normalized bounds
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    raise ValueError(f"Bounding box center out of bounds [0, 1] in {split_name}/{fname}:{line_no}: ({xc}, {yc})")

                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    raise ValueError(f"Bounding box dimensions out of bounds (0, 1] in {split_name}/{fname}:{line_no}: w={w}, h={h}")

                class_dist[cid] += 1
                instance_count += 1

            if idx % 50000 == 0 or idx == len(label_files):
                print(f"    Parsed {idx:,}/{len(label_files):,} {split_name} label files...")

        print(f"  Parsed {len(label_files):,} {split_name} label files in {time.time() - t_parse_0:.2f}s.")
        return instance_count, empty_files, class_dist

    train_instances, train_empty, train_classes = parse_partition_labels(TRAIN_LBL_DIR, eligible_tl, "train")
    val_instances, val_empty, val_classes = parse_partition_labels(VAL_LBL_DIR, n_vl, "val")

    if train_empty:
        raise ValueError(f"Unexpected empty label files in eligible train split ({len(train_empty)} files): {train_empty[:5]}")
    if val_empty:
        raise ValueError(f"Unexpected empty label files in validation split ({len(val_empty)} files): {val_empty[:5]}")

    print(f"  Eligible train instances: {train_instances:,} (expected {EXPECTED_COUNTS['train_instances_eligible']:,})")
    print(f"  Validation instances:     {val_instances:,} (expected {EXPECTED_COUNTS['val_instances']:,})")
    if train_instances != EXPECTED_COUNTS["train_instances_eligible"]:
        raise ValueError(f"Train instance count mismatch: expected {EXPECTED_COUNTS['train_instances_eligible']}, got {train_instances}")
    if val_instances != EXPECTED_COUNTS["val_instances"]:
        raise ValueError(f"Val instance count mismatch: expected {EXPECTED_COUNTS['val_instances']}, got {val_instances}")

    # Class coverage
    for cid in range(13):
        if train_classes[cid] <= 0:
            raise ValueError(f"Class {cid} ({YOLO_CLASS_NAMES[cid]}) missing from train split!")
        if val_classes[cid] <= 0:
            raise ValueError(f"Class {cid} ({YOLO_CLASS_NAMES[cid]}) missing from val split!")

    print("  All 13 garment categories confirmed present across both splits.")

    print("  [PASS] Source files, excluded targets, and YOLO label syntax 100% verified.")

    return {
        "train_images": eligible_ti,
        "val_images": n_vi,
        "train_labels": eligible_tl,
        "val_labels": n_vl,
        "train_img_bytes": b_ti,
        "val_img_bytes": b_vi,
        "train_lbl_bytes": b_tl,
        "val_lbl_bytes": b_vl,
        "train_instances": train_instances,
        "val_instances": val_instances,
        "train_classes": train_classes,
        "val_classes": val_classes,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Manifest Path & Physical Resolution Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_full_manifests_and_resolution(audit: Dict[str, Any]) -> None:
    """
    Validates every single manifest entry in train.txt and val.txt:
    - Resolves every relative path against its intended source partition directory.
    - Verifies that each exact physical file exists on disk.
    - Rejects unexpected nested paths (must be flat filenames under partition).
    - Compares complete relative path sets against eligible source inventory.
    - Rejects absolute paths, Windows drive letters, backslashes, and '../' traversal.
    - Rejects duplicate entries.
    """
    print("\n[SECURITY CHECK 4/7] Performing Full Manifest Path & Physical Resolution Validation...")

    for fpath in [TRAIN_MANIFEST_FILE, VAL_MANIFEST_FILE]:
        if not fpath.exists():
            raise FileNotFoundError(f"Required manifest file missing: {fpath}")

    def validate_and_resolve_manifest(
        manifest_file: Path,
        expected_count: int,
        source_dir: Path,
        eligible_set: Set[str],
        expected_prefix: str,
        split_name: str,
    ) -> Set[str]:
        with open(manifest_file, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        clean_lines = [l.strip() for l in raw_lines if l.strip()]

        # 1. Entry count check
        if len(clean_lines) != expected_count:
            raise ValueError(f"Entry count mismatch in {manifest_file.name}: expected {expected_count}, got {len(clean_lines)}")

        # 2. Duplicate entries check
        unique_lines = set(clean_lines)
        if len(clean_lines) != len(unique_lines):
            raise ValueError(f"Duplicate entries detected in {manifest_file.name}! Unique count: {len(unique_lines)}, Total: {len(clean_lines)}")

        resolved_filenames: Set[str] = set()
        normalized_relative_paths: Set[str] = set()

        for line_no, line in enumerate(clean_lines, 1):
            # Reject Windows drive letters
            if ":\\" in line or ":/" in line or (len(line) > 1 and line[1] == ":" and line[0].isalpha()):
                raise ValueError(f"Absolute Windows path detected in {manifest_file.name}:{line_no}: '{line}'")

            # Reject leading root slashes
            if line.startswith("/") or line.startswith("\\"):
                raise ValueError(f"Absolute root path detected in {manifest_file.name}:{line_no}: '{line}'")

            # Reject Windows backslashes
            if "\\" in line:
                raise ValueError(f"Windows backslash detected in {manifest_file.name}:{line_no}: '{line}'")

            # Reject '../' directory traversal
            if ".." in line:
                raise ValueError(f"Path traversal '..' detected in {manifest_file.name}:{line_no}: '{line}'")

            # Reject withheld test split references
            if "test" in line.lower():
                raise ValueError(f"Withheld test split reference detected in {manifest_file.name}:{line_no}: '{line}'")

            # Normalize prefix to check relative path
            prefix_with_dot = f"./{expected_prefix}/"
            prefix_without_dot = f"{expected_prefix}/"

            if line.startswith(prefix_with_dot):
                rel_remainder = line[len(prefix_with_dot):]
            elif line.startswith(prefix_without_dot):
                rel_remainder = line[len(prefix_without_dot):]
            else:
                raise ValueError(f"Invalid path prefix in {manifest_file.name}:{line_no}: expected '{expected_prefix}/...', got '{line}'")

            # Reject unexpected nested subdirectories (must be a flat filename)
            if "/" in rel_remainder or "\\" in rel_remainder:
                raise ValueError(f"Unexpected nested path detected in {manifest_file.name}:{line_no}: '{line}'")

            fname = rel_remainder
            if fname in resolved_filenames:
                raise ValueError(f"Duplicate filename resolved in {manifest_file.name}:{line_no}: '{fname}'")
            resolved_filenames.add(fname)
            normalized_relative_paths.add(f"{expected_prefix}/{fname}")

            # Excluded target check
            if split_name == "train" and fname in EXCLUDED_TRAIN_IMAGES:
                raise ValueError(f"Excluded image {fname} found in {manifest_file.name}:{line_no}!")

            # Verify physical file existence on disk
            target_file_path = source_dir / fname
            if not target_file_path.is_file():
                raise FileNotFoundError(
                    f"Manifest entry does not resolve to an existing physical file: {manifest_file.name}:{line_no} -> {target_file_path}"
                )

        # Compare complete relative sets against eligible set
        expected_relative_set = {f"{expected_prefix}/{f}" for f in eligible_set}
        if normalized_relative_paths != expected_relative_set:
            diff_missing = expected_relative_set - normalized_relative_paths
            diff_extra = normalized_relative_paths - expected_relative_set
            raise ValueError(
                f"Relative path inventory does not equal eligible source set in {manifest_file.name}! Missing: {len(diff_missing)}, Extra: {len(diff_extra)}"
            )

        print(f"  {manifest_file.name}: Verified all {len(clean_lines):,} entries physically exist on disk (zero nested paths, complete relative set match).")
        return resolved_filenames

    train_resolved = validate_and_resolve_manifest(
        TRAIN_MANIFEST_FILE, EXPECTED_COUNTS["train_images_eligible"], TRAIN_IMG_DIR, audit["train_images"], "images/train", "train"
    )
    val_resolved = validate_and_resolve_manifest(
        VAL_MANIFEST_FILE, EXPECTED_COUNTS["val_images"], VAL_IMG_DIR, audit["val_images"], "images/val", "val"
    )

    print("  [PASS] Full manifest path validation and physical source resolution succeeded on 100% of entries.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Safe Archive Creation (inside pre-tracked temporary staging directory)
# ─────────────────────────────────────────────────────────────────────────────

def create_archives_in_temp_dir(audit: Dict[str, Any], temp_dir: Path) -> Dict[str, Any]:
    """
    Creates the 4 portable ZIP archives inside the designated temporary staging directory:
    - Never writes directly to the final destination before verification.
    - Preserves ZIP_STORED for JPEG images to avoid slow recompression.
    - Preserves ZIP_DEFLATED for text labels and manifests.
    - Never modifies, moves, or deletes any source files.
    """
    print(f"\n[SECURITY CHECK 5/7] Assembling Archives in Temporary Staging Directory...")
    print(f"  Temporary Staging Path: {temp_dir}")

    timings: Dict[str, float] = {}

    # 1. Labels ZIP (ZIP_DEFLATED)
    labels_zip_path = temp_dir / ARCHIVE_NAMES["labels"]
    print(f"\n  [1/4] Archiving {labels_zip_path.name} (ZIP_DEFLATED, 224,111 labels)...")
    t0 = time.time()
    with zipfile.ZipFile(labels_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(audit["train_labels"]):
            zf.write(TRAIN_LBL_DIR / fname, arcname=f"labels/train/{fname}")
        for fname in sorted(audit["val_labels"]):
            zf.write(VAL_LBL_DIR / fname, arcname=f"labels/val/{fname}")
    timings["labels"] = time.time() - t0
    print(f"    Completed in {timings['labels']:.2f}s ({labels_zip_path.stat().st_size / (1024**2):.2f} MB).")

    # 2. Manifests ZIP (ZIP_DEFLATED)
    manifests_zip_path = temp_dir / ARCHIVE_NAMES["manifests"]
    print(f"\n  [2/4] Archiving {manifests_zip_path.name} (ZIP_DEFLATED, 3 manifest files)...")
    t0 = time.time()
    with zipfile.ZipFile(manifests_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(TRAIN_MANIFEST_FILE, arcname="train.txt")
        zf.write(VAL_MANIFEST_FILE, arcname="val.txt")
        zf.write(DATA_YAML_FILE, arcname="data.yaml")
    timings["manifests"] = time.time() - t0
    print(f"    Completed in {timings['manifests']:.2f}s ({manifests_zip_path.stat().st_size / (1024**2):.2f} MB).")

    # 3. Validation Images ZIP (ZIP_STORED)
    val_img_zip_path = temp_dir / ARCHIVE_NAMES["val_images"]
    print(f"\n  [3/4] Archiving {val_img_zip_path.name} (ZIP_STORED, 32,153 images)...")
    t0 = time.time()
    with zipfile.ZipFile(val_img_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, fname in enumerate(sorted(audit["val_images"]), 1):
            zf.write(VAL_IMG_DIR / fname, arcname=f"images/val/{fname}")
            if idx % 10000 == 0 or idx == len(audit["val_images"]):
                print(f"    Packed {idx:,}/{len(audit['val_images']):,} validation images...")
    timings["val_images"] = time.time() - t0
    print(f"    Completed in {timings['val_images']:.2f}s ({val_img_zip_path.stat().st_size / (1024**3):.2f} GB).")

    # 4. Training Images ZIP (ZIP_STORED)
    train_img_zip_path = temp_dir / ARCHIVE_NAMES["train_images"]
    print(f"\n  [4/4] Archiving {train_img_zip_path.name} (ZIP_STORED, 191,958 images)...")
    t0 = time.time()
    with zipfile.ZipFile(train_img_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, fname in enumerate(sorted(audit["train_images"]), 1):
            zf.write(TRAIN_IMG_DIR / fname, arcname=f"images/train/{fname}")
            if idx % 25000 == 0 or idx == len(audit["train_images"]):
                print(f"    Packed {idx:,}/{len(audit['train_images']):,} training images...")
    timings["train_images"] = time.time() - t0
    print(f"    Completed in {timings['train_images']:.2f}s ({train_img_zip_path.stat().st_size / (1024**3):.2f} GB).")

    return {
        "paths": {
            "labels": labels_zip_path,
            "manifests": manifests_zip_path,
            "val_images": val_img_zip_path,
            "train_images": train_img_zip_path,
        },
        "timings": timings,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 9. Complete POSIX Archive Member Path-Set Equality & Multi-Tier SHA-256
# ─────────────────────────────────────────────────────────────────────────────

def verify_staged_archives(
    archives: Dict[str, Any],
    audit: Dict[str, Any],
    verify_all_hashes: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """
    Rigorously verifies all 4 archives in temporary staging before publishing:
    - Runs ZipFile.testzip() CRC32 checksums on all members.
    - Compares complete POSIX archive member paths against expected sets (not just filenames).
    - Rejects any nested directories or unexpected archive structures.
    - Verifies 1:1 image-to-label correspondence inside archives.
    - Verifies bitwise SHA-256 parity for all 3 manifest files (train.txt, val.txt, data.yaml).
    - Verifies bitwise SHA-256 parity for image and label members (sampled or exhaustive).
    """
    print("\n[SECURITY CHECK 6/7] Performing Comprehensive Archive Verification (CRC32 & Complete Path Equality)...")

    archive_stats: Dict[str, Any] = {}

    # Complete expected POSIX relative path sets
    expected_train_img_paths: Set[str] = {f"images/train/{f}" for f in audit["train_images"]}
    expected_val_img_paths: Set[str] = {f"images/val/{f}" for f in audit["val_images"]}
    expected_train_lbl_paths: Set[str] = {f"labels/train/{f}" for f in audit["train_labels"]}
    expected_val_lbl_paths: Set[str] = {f"labels/val/{f}" for f in audit["val_labels"]}
    expected_manifest_paths: Set[str] = {"train.txt", "val.txt", "data.yaml"}

    for key, archive_path in archives["paths"].items():
        print(f"\n  Auditing archive: {archive_path.name}...")
        t0 = time.time()

        with zipfile.ZipFile(archive_path, "r") as zf:
            # 1. CRC32 Check
            bad_member = zf.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(f"CRC check failed in {archive_path.name}! Corrupt member: {bad_member}")
            print(f"    testzip() CRC32 check: PASSED (all members intact).")

            namelist = zf.namelist()
            member_set = set(namelist)

            # 2. Duplicate rejection
            if len(namelist) != len(member_set):
                raise ValueError(f"Duplicate members detected inside {archive_path.name}!")

            # 3. Path structure & Nested subdirectory rejection
            for m in namelist:
                if "\\" in m or ":" in m or m.startswith("/"):
                    raise ValueError(f"Unsafe member path in {archive_path.name}: '{m}'")
                if ".." in m:
                    raise ValueError(f"Path traversal '..' detected in {archive_path.name}: '{m}'")

                # Structure check: images and labels must be exactly 3 components ('type/split/file')
                parts = m.split("/")
                if key in ("train_images", "val_images", "labels"):
                    if len(parts) != 3:
                        raise ValueError(
                            f"Unexpected nested or non-flat POSIX member path in {archive_path.name}: '{m}' (expected 3 parts)"
                        )
                elif key == "manifests":
                    if len(parts) != 1:
                        raise ValueError(f"Unexpected nested path in manifests archive: '{m}'")

            # 4. Complete POSIX Path-Set Equality Checks
            if key == "train_images":
                if member_set != expected_train_img_paths:
                    diff_missing = expected_train_img_paths - member_set
                    diff_extra = member_set - expected_train_img_paths
                    raise ValueError(
                        f"train_images POSIX path set mismatch! Missing: {len(diff_missing)}, Extra: {len(diff_extra)}"
                    )
                print(f"    Complete POSIX member path set equality confirmed: exactly {len(member_set):,} 'images/train/*.jpg' members.")

            elif key == "val_images":
                if member_set != expected_val_img_paths:
                    diff_missing = expected_val_img_paths - member_set
                    diff_extra = member_set - expected_val_img_paths
                    raise ValueError(
                        f"val_images POSIX path set mismatch! Missing: {len(diff_missing)}, Extra: {len(diff_extra)}"
                    )
                print(f"    Complete POSIX member path set equality confirmed: exactly {len(member_set):,} 'images/val/*.jpg' members.")

            elif key == "labels":
                expected_all_labels = expected_train_lbl_paths | expected_val_lbl_paths
                if member_set != expected_all_labels:
                    diff_missing = expected_all_labels - member_set
                    diff_extra = member_set - expected_all_labels
                    raise ValueError(
                        f"labels POSIX path set mismatch! Missing: {len(diff_missing)}, Extra: {len(diff_extra)}"
                    )
                # Verify separate partition subsets
                train_sub = {m for m in member_set if m.startswith("labels/train/")}
                val_sub = {m for m in member_set if m.startswith("labels/val/")}
                if train_sub != expected_train_lbl_paths:
                    raise ValueError(f"train_labels POSIX path set mismatch inside labels archive!")
                if val_sub != expected_val_lbl_paths:
                    raise ValueError(f"val_labels POSIX path set mismatch inside labels archive!")
                print(f"    Complete POSIX member path set equality confirmed: {len(train_sub):,} train and {len(val_sub):,} val labels.")

            elif key == "manifests":
                if member_set != expected_manifest_paths:
                    raise ValueError(f"manifests member path mismatch: expected {expected_manifest_paths}, got {member_set}")
                print(f"    Complete POSIX member path set equality confirmed: train.txt, val.txt, data.yaml.")

            # 5. Excluded target absence
            for excl in EXCLUDED_TRAIN_IMAGES:
                if f"images/train/{excl}" in member_set:
                    raise ValueError(f"Excluded image {excl} leaked into {archive_path.name}!")
            for excl_lbl in EXCLUDED_TRAIN_LABELS:
                if f"labels/train/{excl_lbl}" in member_set:
                    raise ValueError(f"Excluded label {excl_lbl} leaked into {archive_path.name}!")

        elapsed = time.time() - t0
        archive_stats[key] = {
            "path": str(archive_path),
            "filename": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "member_count": len(namelist),
            "crc_passed": True,
            "audit_duration_seconds": round(elapsed, 2),
        }

    # 6. Exact 1:1 image-to-label correspondence inside archives
    print("\n  Verifying 1:1 image-label correspondence inside archives...")
    train_img_stems = {Path(m).stem for m in expected_train_img_paths}
    train_lbl_stems = {Path(m).stem for m in expected_train_lbl_paths}
    val_img_stems = {Path(m).stem for m in expected_val_img_paths}
    val_lbl_stems = {Path(m).stem for m in expected_val_lbl_paths}

    if train_img_stems != train_lbl_stems:
        raise ValueError("Archived train images and labels do not have 1:1 stem correspondence!")
    if val_img_stems != val_lbl_stems:
        raise ValueError("Archived val images and labels do not have 1:1 stem correspondence!")
    print("    1:1 correspondence inside archives verified.")

    # 7. Manifests Bitwise SHA-256 Parity
    print("\n  Verifying bitwise SHA-256 parity for manifest files (train.txt, val.txt, data.yaml)...")
    with zipfile.ZipFile(archives["paths"]["manifests"], "r") as zf_man:
        for m_file, src_path in [
            ("train.txt", TRAIN_MANIFEST_FILE),
            ("val.txt", VAL_MANIFEST_FILE),
            ("data.yaml", DATA_YAML_FILE),
        ]:
            src_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()
            arch_hash = hashlib.sha256(zf_man.read(m_file)).hexdigest()
            if src_hash != arch_hash:
                raise ValueError(f"Bitwise SHA-256 mismatch for manifest file: {m_file}")
    print("    Bitwise SHA-256 parity verified for all 3 manifest files: 100% match.")

    # 8. Multi-Partition Bitwise SHA-256 Parity Verification (Sampled or Exhaustive)
    print("\n  Multi-Partition Bitwise SHA-256 Source-to-Archive Verification...")
    rng = random.Random(42)

    sample_size = 50
    if verify_all_hashes:
        print("    [EXHAUSTIVE MODE] Verifying SHA-256 on 100% of source files and archive members...")
        check_ti = sorted(list(audit["train_images"]))
        check_vi = sorted(list(audit["val_images"]))
        check_tl = sorted(list(audit["train_labels"]))
        check_vl = sorted(list(audit["val_labels"]))
    else:
        print(f"    [SAMPLED MODE] Verifying SHA-256 on {sample_size} random samples per partition...")
        check_ti = rng.sample(sorted(list(audit["train_images"])), sample_size)
        check_vi = rng.sample(sorted(list(audit["val_images"])), sample_size)
        check_tl = rng.sample(sorted(list(audit["train_labels"])), sample_size)
        check_vl = rng.sample(sorted(list(audit["val_labels"])), sample_size)

    # Train Images
    with zipfile.ZipFile(archives["paths"]["train_images"], "r") as zf:
        for fname in check_ti:
            src_bytes = (TRAIN_IMG_DIR / fname).read_bytes()
            arch_bytes = zf.read(f"images/train/{fname}")
            if hashlib.sha256(src_bytes).hexdigest() != hashlib.sha256(arch_bytes).hexdigest():
                raise ValueError(f"Bitwise SHA-256 parity mismatch for train image: {fname}")

    # Val Images
    with zipfile.ZipFile(archives["paths"]["val_images"], "r") as zf:
        for fname in check_vi:
            src_bytes = (VAL_IMG_DIR / fname).read_bytes()
            arch_bytes = zf.read(f"images/val/{fname}")
            if hashlib.sha256(src_bytes).hexdigest() != hashlib.sha256(arch_bytes).hexdigest():
                raise ValueError(f"Bitwise SHA-256 parity mismatch for val image: {fname}")

    # Train and Val Labels
    with zipfile.ZipFile(archives["paths"]["labels"], "r") as zf:
        for fname in check_tl:
            src_bytes = (TRAIN_LBL_DIR / fname).read_bytes()
            arch_bytes = zf.read(f"labels/train/{fname}")
            if hashlib.sha256(src_bytes).hexdigest() != hashlib.sha256(arch_bytes).hexdigest():
                raise ValueError(f"Bitwise SHA-256 parity mismatch for train label: {fname}")

        for fname in check_vl:
            src_bytes = (VAL_LBL_DIR / fname).read_bytes()
            arch_bytes = zf.read(f"labels/val/{fname}")
            if hashlib.sha256(src_bytes).hexdigest() != hashlib.sha256(arch_bytes).hexdigest():
                raise ValueError(f"Bitwise SHA-256 parity mismatch for val label: {fname}")

    total_verified_members = (
        (len(check_ti) + len(check_vi) + len(check_tl) + len(check_vl) + 3)
    )
    mode_label = "100% of all members (448,225 files)" if verify_all_hashes else f"{sample_size} samples per partition + 3 manifests ({total_verified_members} files total)"
    print(f"    Bitwise SHA-256 source-to-archive parity verified on {mode_label}: 100% match.")
    print("  [PASS] Comprehensive archive verification succeeded on all 4 archives.")

    return archive_stats, total_verified_members

# ─────────────────────────────────────────────────────────────────────────────
# 10. Checksum Calculation, Inventory Generation & Atomic Publishing
# ─────────────────────────────────────────────────────────────────────────────

def calculate_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):  # 8 MB buffer
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_inventory_and_publish_atomically(
    audit: Dict[str, Any],
    archives: Dict[str, Any],
    archive_stats: Dict[str, Any],
    temp_dir: Path,
    verify_all_hashes: bool,
    total_verified_members: int,
) -> Path:
    """
    Computes full-archive SHA-256 hashes of all 4 ZIP files, serializes
    colab_archive_inventory.json, and atomizes publication by safely moving temp_dir
    to TRANSFER_OUTPUT_DIR via platform-appropriate atomic no-replace operations.
    """
    print("\n[SECURITY CHECK 7/7] Computing Full Archive SHA-256 Checksums and Publishing Atomically...")

    manifest_entries: Dict[str, Any] = {}
    total_archive_bytes = 0

    for key, archive_path in archives["paths"].items():
        print(f"  Computing Archive-Level SHA-256: {archive_path.name} ({archive_path.stat().st_size / (1024**2):.1f} MB)...")
        t0 = time.time()
        archive_sha256 = calculate_file_sha256(archive_path)
        elapsed = time.time() - t0
        print(f"    Archive SHA-256: {archive_sha256} ({elapsed:.1f}s)")

        total_archive_bytes += archive_path.stat().st_size
        manifest_entries[key] = {
            "filename": archive_path.name,
            "relative_path": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "size_mb": round(archive_path.stat().st_size / (1024**2), 2),
            "size_gb": round(archive_path.stat().st_size / (1024**3), 3),
            "archive_sha256": archive_sha256,
            "member_count": archive_stats[key]["member_count"],
            "crc32_checksum_passed": True,
            "creation_time_seconds": round(archives["timings"][key], 2),
            "audit_time_seconds": archive_stats[key]["audit_duration_seconds"],
        }

    total_source_bytes = (
        audit["train_img_bytes"] + audit["val_img_bytes"] + audit["train_lbl_bytes"] + audit["val_lbl_bytes"]
    )

    full_inventory = {
        "stage": "6.8A",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "title": "FitLoop DeepFashion2 YOLO Google Colab Transfer Staging Manifest",
        "verification_status": "VERIFIED_PASS",
        "verification_tier": {
            "archive_level_sha256": "EXHAUSTIVE (All 4 complete ZIP files)",
            "member_level_crc32": "EXHAUSTIVE (100% of 448,225 members verified via ZipFile.testzip)",
            "member_level_inventory": "EXHAUSTIVE (100% of complete POSIX relative paths verified against source sets)",
            "member_level_bitwise_sha256": "EXHAUSTIVE (448,225 files)" if verify_all_hashes else f"SAMPLED ({total_verified_members} files total)",
            "verified_member_sha256_count": total_verified_members,
        },
        "storage_metrics": {
            "total_source_bytes": total_source_bytes,
            "total_source_gb": round(total_source_bytes / (1024**3), 3),
            "total_archive_bytes": total_archive_bytes,
            "total_archive_gb": round(total_archive_bytes / (1024**3), 3),
        },
        "dataset_invariants": {
            "eligible_train_images": EXPECTED_COUNTS["train_images_eligible"],
            "validation_images": EXPECTED_COUNTS["val_images"],
            "eligible_train_labels": EXPECTED_COUNTS["train_labels_eligible"],
            "validation_labels": EXPECTED_COUNTS["val_labels"],
            "eligible_train_instances": audit["train_instances"],
            "validation_instances": audit["val_instances"],
            "num_classes": EXPECTED_COUNTS["num_classes"],
            "class_names": YOLO_CLASS_NAMES,
            "excluded_train_images": list(sorted(EXCLUDED_TRAIN_IMAGES)),
        },
        "archives": manifest_entries,
        "transfer_instructions": {
            "destination_drive_path": "/content/drive/MyDrive/FitLoop/staging_sources/",
            "colab_notebook": "wardrobe_ml/notebooks/stage_6_8a_colab_packaging.ipynb",
            "unpack_command": "unzip -q <archive_name>.zip -d /content/fitloop_temp",
        },
        "safety_rules_compliance": {
            "atomic_staging_enforced": True,
            "atomic_no_replace_publication": True,
            "overwrite_protection_enforced": True,
            "zero_source_files_modified": True,
            "zero_models_trained": True,
            "zero_weights_downloaded": True,
            "zero_inference_runs": True,
            "zero_kaggle_uploads": True,
            "python_optimization_safe": True,
        }
    }

    temp_inventory_path = temp_dir / INVENTORY_JSON_FILENAME
    with open(temp_inventory_path, "w", encoding="utf-8") as f:
        json.dump(full_inventory, f, indent=2)
    print(f"\n  Inventory manifest written: {temp_inventory_path.name}")

    # Atomic No-Replace Publication
    print(f"\n  Publishing staging directory via atomic no-replace operation:")
    print(f"    From: {temp_dir}")
    print(f"    To:   {TRANSFER_OUTPUT_DIR}")

    atomic_publish_no_replace(temp_dir, TRANSFER_OUTPUT_DIR)
    print(f"  [PASS] Staging published atomically to: {TRANSFER_OUTPUT_DIR}")
    return TRANSFER_OUTPUT_DIR / INVENTORY_JSON_FILENAME

# ─────────────────────────────────────────────────────────────────────────────
# 11. Main Orchestration & Failure Handling
# ─────────────────────────────────────────────────────────────────────────────

def run_preparation(create_archives: bool = False, verify_all_hashes: bool = False) -> None:
    print("=" * 70)
    print("FITLOOP STAGE 6.8A: FINAL ARCHIVE VERIFICATION HARDENING")
    print("=" * 70)

    # 1. Overwrite Protection check
    if create_archives:
        verify_overwrite_protection()
    else:
        print("\n[NOTE] Dry-run audit mode active. Destination overwrite check skipped.")

    # 2. Authoritative data.yaml & 13-Class Mapping check
    validate_data_yaml_and_classes()

    # 3. Source & Exhaustive YOLO Label validation
    audit = validate_sources_and_labels()

    # 4. Full Manifest Path & Physical Resolution validation
    validate_full_manifests_and_resolution(audit)

    if not create_archives:
        print("\n" + "=" * 70)
        print("DRY-RUN AUDIT COMPLETED SUCCESSFULLY [PASS]")
        print("=" * 70)
        print("All source images, labels, manifests, YAML mapping, and invariants verified.")
        print("Zero files were created, modified, moved, or deleted.")
        print("\nTo execute actual archive creation with atomic staging and SHA-256 verification:")
        print("  python wardrobe_ml/src/prepare_colab_transfer.py --create-archives")
        return

    # Atomic staging directory tracked in caller scope
    temp_dir = None
    archives = None

    try:
        # Create unique temporary directory before any archive creation begins
        temp_dir_prefix = f"colab_transfer_staging_{uuid.uuid4().hex[:8]}_"
        temp_dir_str = tempfile.mkdtemp(prefix=temp_dir_prefix, dir=str(TRANSFER_OUTPUT_DIR.parent))
        temp_dir = Path(temp_dir_str)
        print(f"\n[ATOMIC STAGING] Created temporary directory: {temp_dir}")

        # 5. Create safe archives inside tracked temporary staging directory
        archives = create_archives_in_temp_dir(audit, temp_dir)

        # 6. Verify staged archives (CRC32, complete POSIX path sets, bitwise SHA-256)
        archive_stats, total_verified_members = verify_staged_archives(
            archives, audit, verify_all_hashes=verify_all_hashes
        )

        # 7. Checksum calculation, manifest creation, and atomic publish
        final_manifest = generate_inventory_and_publish_atomically(
            audit, archives, archive_stats, temp_dir, verify_all_hashes=verify_all_hashes, total_verified_members=total_verified_members
        )

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"[FATAL FAILURE] Stage 6.8A archive preparation aborted: {e}")
        print("Original source files and existing directories remain completely untouched.")
        if temp_dir is not None and temp_dir.exists():
            print(f"[PRESERVED] Temporary artifacts preserved at: {temp_dir}")
            print("Failed artifacts are NOT automatically deleted to facilitate forensic inspection.")
        print("!" * 70)
        raise

    print("\n" + "=" * 70)
    print("STAGE 6.8A ARCHIVE CREATION AND VERIFICATION COMPLETED [VERIFIED_PASS]")
    print("=" * 70)
    print(f"Staged Directory: {TRANSFER_OUTPUT_DIR}")
    for k, filename in ARCHIVE_NAMES.items():
        p = TRANSFER_OUTPUT_DIR / filename
        print(f"  - {filename:<30} ({p.stat().st_size / (1024**2):>8.2f} MB)")
    print(f"  - {INVENTORY_JSON_FILENAME:<30}")
    print("\nNext Step:")
    print("  Upload the 4 ZIP archives in that directory to Google Drive under:")
    print("  /content/drive/MyDrive/FitLoop/staging_sources/")


def main():
    parser = argparse.ArgumentParser(description="FitLoop Stage 6.8A — Safe Colab Dataset Transfer Preparation")
    parser.add_argument(
        "--create-archives",
        action="store_true",
        help="Create and verify the 4 hardened ZIP archives for Google Drive transfer using atomic staging"
    )
    parser.add_argument(
        "--verify-all-hashes",
        action="store_true",
        help="Exhaustively verify bitwise SHA-256 parity for 100% of archive members against source files (slower)"
    )
    args = parser.parse_args()

    run_preparation(create_archives=args.create_archives, verify_all_hashes=args.verify_all_hashes)


if __name__ == "__main__":
    main()
