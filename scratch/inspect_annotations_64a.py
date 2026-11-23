import os, sys, json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

annos_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\annos"
images_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\image"

# 1. Inspect known degenerate annotation: 101338.json
deg_anno_path = os.path.join(annos_dir, "101338.json")
deg_img_path = os.path.join(images_dir, "101338.jpg")
print("=== 101338.json (Known Degenerate) ===")
with open(deg_anno_path, "r", encoding="utf-8") as f:
    deg_data = json.load(f)
print(json.dumps(deg_data, indent=2))

if os.path.exists(deg_img_path):
    with Image.open(deg_img_path) as img:
        print(f"101338.jpg size (width, height): {img.size}")

# 2. Inspect a single-garment, 2-garment, and 3+-garment image
print("\n=== Finding representative annotations ===")
sample_annos = {}
for fname in os.listdir(annos_dir):
    if not fname.endswith(".json"):
        continue
    p = os.path.join(annos_dir, fname)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = [k for k in data.keys() if k.startswith("item")]
    count = len(items)
    
    # Check if disconnected segmentation polygons exist
    has_disconnected_seg = False
    for k in items:
        seg = data[k].get("segmentation", [])
        if len(seg) > 1:
            has_disconnected_seg = True
            
    if count == 1 and "1_garment" not in sample_annos:
        sample_annos["1_garment"] = (fname, data)
    elif count == 2 and "2_garment" not in sample_annos:
        sample_annos["2_garment"] = (fname, data)
    elif count >= 3 and "3plus_garment" not in sample_annos:
        sample_annos["3plus_garment"] = (fname, data)
    if has_disconnected_seg and "disconnected_seg" not in sample_annos:
        sample_annos["disconnected_seg"] = (fname, data)
        
    if len(sample_annos) >= 4:
        break

for label, (fname, data) in sample_annos.items():
    print(f"\n--- {label}: {fname} ---")
    img_p = os.path.join(images_dir, fname.replace(".json", ".jpg"))
    if os.path.exists(img_p):
        with Image.open(img_p) as img:
            print(f"Image size (width, height): {img.size}, mode: {img.mode}")
    items = [k for k in data.keys() if k.startswith("item")]
    print(f"Items count: {len(items)}")
    for k in items:
        cat_id = data[k].get("category_id")
        cat_name = data[k].get("category_name")
        bbox = data[k].get("bounding_box")
        seg = data[k].get("segmentation")
        seg_poly_count = len(seg) if isinstance(seg, list) else 0
        print(f"  {k}: category_id={cat_id} ({cat_name}), bbox={bbox}, seg_polygons={seg_poly_count}")
