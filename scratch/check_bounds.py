import os, sys, json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

annos_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\annos"
images_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\image"

# Check 100 random/sequential annotations to inspect coordinates vs image bounds
sample_files = sorted(os.listdir(annos_dir))[:100]

boundary_issues = []
for fname in sample_files:
    if not fname.endswith(".json"):
        continue
    anno_path = os.path.join(annos_dir, fname)
    img_path = os.path.join(images_dir, fname.replace(".json", ".jpg"))
    if not os.path.exists(img_path):
        continue
    with Image.open(img_path) as img:
        w, h = img.size
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        if not k.startswith("item"):
            continue
        bbox = v.get("bounding_box", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
            boundary_issues.append((fname, k, bbox, (w, h)))

print(f"Checked 100 files. Boundary issues found: {len(boundary_issues)}")
for issue in boundary_issues[:10]:
    print(" ", issue)
