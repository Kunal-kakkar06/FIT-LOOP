from pathlib import Path
p = Path("wardrobe_ml/outputs/verified_category_distribution.csv")
print(p.read_text(encoding='utf-8'))
