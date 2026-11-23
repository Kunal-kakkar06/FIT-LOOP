import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
root = Path(r"C:\Users\Lenovo\Downloads\Fitloop")

print("="*60)
print("PROJECT TREE INSPECTION")
print("="*60)

for p in sorted(root.glob("*")):
    if p.name in [".venv", ".git", "__pycache__", ".gemini"]:
        print(f"[DIR - Ignored detail] {p.name}")
        continue
    if p.is_dir():
        print(f"[DIR] {p.name}/")
        for sub in sorted(p.glob("*")):
            if sub.is_dir():
                print(f"    [DIR] {sub.name}/")
                for sub2 in sorted(sub.glob("*")):
                    print(f"        {sub2.name}")
            else:
                print(f"    {sub.name}")
    else:
        print(f"[FILE] {p.name}")

print("\n" + "="*60)
print("CHECKING FOR REQUIREMENTS, CONFIGS, MODELS, CHECKPOINTS")
print("="*60)

for p in root.rglob("*"):
    if any(part in [".venv", ".git", "__pycache__", ".gemini"] for part in p.parts):
        continue
    name_lower = p.name.lower()
    if any(kw in name_lower for kw in ["req", "environment", "pip", "setup", "pyproject", "conda"]):
        print(f"Config/Requirements: {p.relative_to(root)}")
    if any(p.suffix.lower() == ext for ext in [".pt", ".pth", ".onnx", ".tflite", ".bin", ".engine"]):
        print(f"Model Checkpoint: {p.relative_to(root)}")
    if any(kw in name_lower for kw in ["detect", "yolo", "segment", "model", "train"]):
        if p.is_file() and p.suffix.lower() in [".py", ".ipynb", ".sh", ".yaml", ".yml"]:
            print(f"Code referencing ML/YOLO/detect: {p.relative_to(root)}")
