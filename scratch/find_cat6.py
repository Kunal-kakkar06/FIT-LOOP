import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
annos_dir = r"C:\Users\Lenovo\Downloads\DeepFashion2\train\annos"
for i in range(1, 2000):
    s_id = f"{i:06d}"
    ap = os.path.join(annos_dir, f"{s_id}.json")
    if not os.path.exists(ap):
        continue
    with open(ap, "r") as f:
        d = json.load(f)
    for k in d:
        if k.startswith("item") and d[k].get("category_id") == 6:
            print(f"Category 6 found in: {s_id}")
            sys.exit(0)
