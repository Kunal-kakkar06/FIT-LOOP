import os, sys, json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

annos_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\annos"
images_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\image"

# We want 20 sample images:
# - Must include 101338
# - Single-garment images
# - Two-garment images
# - 3+-garment images
# - Disconnected segmentation polygons (len(seg) > 1)
# - Diverse categories (as many of 1..13 as possible)
# - Diverse resolutions

target_ids = ["101338"]

# Scan sequential annotations to find candidates
candidates = []
categories_seen = set()

# First check 101338
with open(os.path.join(annos_dir, "101338.json"), "r") as f:
    d = json.load(f)
for k, v in d.items():
    if k.startswith("item") and "category_id" in v:
        categories_seen.add(v["category_id"])

for fname in sorted(os.listdir(annos_dir)):
    if not fname.endswith(".json"):
        continue
    base_id = fname[:-5]
    if base_id == "101338":
        continue
    anno_path = os.path.join(annos_dir, fname)
    img_path = os.path.join(images_dir, base_id + ".jpg")
    if not os.path.exists(img_path):
        continue
    try:
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    items = [k for k in data if k.startswith("item")]
    item_count = len(items)
    cats = [data[k].get("category_id") for k in items if "category_id" in data[k]]
    has_disconnected_seg = any(len(data[k].get("segmentation", [])) > 1 for k in items)
    
    # Prioritize new categories, multi-garment, disconnected seg
    new_cats = set(cats) - categories_seen
    if new_cats or (item_count >= 3 and len([c for c in candidates if c["item_count"] >= 3]) < 3) or (has_disconnected_seg and len([c for c in candidates if c["has_disconnected_seg"]]) < 3):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        candidates.append({
            "image_id": base_id,
            "item_count": item_count,
            "categories": cats,
            "has_disconnected_seg": has_disconnected_seg,
            "size": (w, h)
        })
        categories_seen.update(cats)
    if len(candidates) >= 19 and len(categories_seen) >= 13:
        break

selected_ids = ["101338"] + [c["image_id"] for c in candidates[:19]]
print(f"Selected {len(selected_ids)} image IDs:")
for s_id in selected_ids:
    ap = os.path.join(annos_dir, s_id + ".json")
    ip = os.path.join(images_dir, s_id + ".jpg")
    with open(ap, "r") as f:
        d = json.load(f)
    with Image.open(ip) as im:
        w, h = im.size
    items = [k for k in d if k.startswith("item")]
    cats = [d[k].get("category_id") for k in items]
    discon = any(len(d[k].get("segmentation", [])) > 1 for k in items)
    print(f"  {s_id}: items={len(items)}, cats={cats}, size=({w},{h}), disconnected_seg={discon}")

all_cats = set()
for s_id in selected_ids:
    ap = os.path.join(annos_dir, s_id + ".json")
    with open(ap, "r") as f:
        d = json.load(f)
    for k in d:
        if k.startswith("item") and "category_id" in d[k]:
            all_cats.add(d[k]["category_id"])
print(f"\nTotal categories covered in 20 samples: {len(all_cats)}/13 -> {sorted(all_cats)}")
