import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
pkl_path = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\57b5c7be-d326-40ec-a129-e53bc9ecb41d\scratch\train_df.pkl")
df = pd.read_pickle(pkl_path)

# Let's check shop only
shop_df = df[df['source'] == 'shop']
s_imgs = shop_df.groupby('pair_id')['file'].nunique()
print(f"Shop only: min={s_imgs.min()}, median={s_imgs.median()}, mean={s_imgs.mean():.2f}, max={s_imgs.max()}")

# User only
user_df = df[df['source'] == 'user']
u_imgs = user_df.groupby('pair_id')['file'].nunique()
print(f"User only: min={u_imgs.min()}, median={u_imgs.median()}, mean={u_imgs.mean():.2f}, max={u_imgs.max()}")

# What about category-specific pair sizes?
for cat, grp in df.groupby('category_name'):
    c_imgs = grp.groupby('pair_id')['file'].nunique()
    if c_imgs.max() == 127 or c_imgs.median() == 10.0:
        print(f"Category {cat}: min={c_imgs.min()}, median={c_imgs.median()}, mean={c_imgs.mean():.2f}, max={c_imgs.max()}")
