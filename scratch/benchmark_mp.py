import os
import sys
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml")
sys.path.insert(0, r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\src")

from annotation_converter import DeepFashion2ToYOLOConverter
import config

annos_dir = config.TRAIN_ANNOTATION_DIR
images_dir = config.TRAIN_IMAGE_DIR

test_ids = [f"{i:06d}" for i in range(1, 2001)]
out_test = Path(r"C:\Users\Lenovo\Downloads\Fitloop\scratch\bench_mp_labels")
out_test.mkdir(parents=True, exist_ok=True)

def worker_task(batch_ids):
    c = DeepFashion2ToYOLOConverter()
    valid = 0
    for s_id in batch_ids:
        ap = annos_dir / f"{s_id}.json"
        ip = images_dir / f"{s_id}.jpg"
        lp = out_test / f"{s_id}.txt"
        res = c.convert_single_annotation(ap, ip, lp)
        valid += res.valid_count
    return valid

if __name__ == "__main__":
    t0 = time.time()
    n_workers = 6
    chunk_size = 200
    chunks = [test_ids[i:i + chunk_size] for i in range(0, len(test_ids), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(worker_task, chunks))
    
    elapsed = time.time() - t0
    rate = len(test_ids) / elapsed
    print(f"Multi-process ({n_workers} workers) 2,000 files took: {elapsed:.2f}s ({rate:.1f} files/sec)")
    print(f"Estimated time for 224,114 files: {224114 / rate / 60:.1f} minutes")
    shutil.rmtree(out_test, ignore_errors=True)
