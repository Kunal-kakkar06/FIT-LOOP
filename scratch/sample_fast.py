import os, sys, json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

annos_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\annos"
images_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\image"

# We know the known degenerate image: 101338
target_ids = ["101338"]

# Let's inspect sequentially: 1, 2, 3, ... up to 100
selected = [
    ("101338", "known degenerate [0,0,0,0] box + 1 valid skirt")
]

for i in range(1, 200):
    s_id = f"{i:06d}"
    ap = os.path.join(annos_dir, f"{s_id}.json")
    ip = os.path.join(images_dir, f"{s_id}.jpg")
    if not os.path.exists(ap) or not os.path.exists(ip):
        continue
    with open(ap, "r", encoding="utf-8") as f:
        d = json.load(f)
    items = [k for k in d if k.startswith("item")]
    cats = [d[k].get("category_id") for k in items]
    has_discon = any(len(d[k].get("segmentation", [])) > 1 for k in items)
    with Image.open(ip) as im:
        w, h = im.size
    
    # We want a variety of garment counts: 1, 2, 3+
    # and disconnected seg
    selected.append((s_id, len(items), cats, (w, h), has_discon))
    if len(selected) >= 25:
        break

print(f"Sampled {len(selected)} images:")
for s in selected:
    print(" ", s)
