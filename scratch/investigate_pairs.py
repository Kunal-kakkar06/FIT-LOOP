import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Let's check train_df.pkl
pkl_path = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\57b5c7be-d326-40ec-a129-e53bc9ecb41d\scratch\train_df.pkl")
print(f"Checking train_df.pkl at: {pkl_path}")
if pkl_path.exists():
    df = pd.read_pickle(pkl_path)
    print(f"Loaded train_df: {len(df):,} rows, columns: {list(df.columns)}")
    
    # 1. Unique images per pair_id
    imgs_per_pair = df.groupby('pair_id')['file'].nunique()
    print("\n--- df.groupby('pair_id')['file'].nunique() ---")
    print(f"Unique pair_ids: {len(imgs_per_pair):,}")
    print(f"Min: {imgs_per_pair.min()}")
    print(f"Median: {imgs_per_pair.median()}")
    print(f"Mean: {imgs_per_pair.mean():.4f}")
    print(f"Max: {imgs_per_pair.max()}")
    print(f"Total unique images sum across pairs: {imgs_per_pair.sum():,}")
    print(f"Total unique images in df: {df['file'].nunique():,}")

    # 2. What if someone grouped by something else or did garments per pair?
    garments_per_pair = df.groupby('pair_id').size()
    print("\n--- df.groupby('pair_id').size() (garments per pair) ---")
    print(f"Min: {garments_per_pair.min()}")
    print(f"Median: {garments_per_pair.median()}")
    print(f"Mean: {garments_per_pair.mean():.4f}")
    print(f"Max: {garments_per_pair.max()}")

# Let's check what was in audit_pair_style_relationships.py or other scratch files
brain_scratch = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\57b5c7be-d326-40ec-a129-e53bc9ecb41d\scratch")
for py_file in brain_scratch.glob("*pair*.py"):
    print(f"Found scratch script: {py_file.name}")
