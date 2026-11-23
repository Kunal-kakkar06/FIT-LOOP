import os, sys, json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

annos_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\annos"
images_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\image"

found_3plus = []
category_representatives = {}

for i in range(1, 1000):
    s_id = f"{i:06d}"
    ap = os.path.join(annos_dir, f"{s_id}.json")
    ip = os.path.join(images_dir, f"{s_id}.jpg")
    if not os.path.exists(ap) or not os.path.exists(ip):
        continue
    with open(ap, "r", encoding="utf-8") as f:
        d = json.load(f)
    items = [k for k in d if k.startswith("item")]
    cats = [d[k].get("category_id") for k in items]
    
    if len(items) >= 3 and len(found_3plus) < 5:
        with Image.open(ip) as im:
            found_3plus.append((s_id, len(items), cats, im.size))
            
    for c in cats:
        if c not in category_representatives:
            with Image.open(ip) as im:
                category_representatives[c] = (s_id, len(items), im.size)
                
    if len(category_representatives) >= 13 and len(found_3plus) >= 3:
        break

print("3+ garment images found:")
for f in found_3plus:
    print(" ", f)

print("\nCategory representatives found:")
for c in sorted(category_representatives):
    print(f"  Category {c:2d}: {category_representatives[c]}")
